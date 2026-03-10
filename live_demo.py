import argparse
import json
import os
import pickle
import re
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import whisper

try:
    import sounddevice as sd
except Exception:
    sd = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gloss_converter import get_prediction_context_tokens, text_to_tokens, to_gloss

VALID_PREDICTION_REGEX = re.compile(r"^[a-z]+(?:'[a-z]+)?$")


def check_dependencies() -> list[str]:
    missing = []

    try:
        import tensorflow  # noqa: F401
    except Exception:
        missing.append("tensorflow")

    if sd is None:
        missing.append("sounddevice (and system PortAudio)")

    return missing


def write_wav(audio: np.ndarray, sample_rate: int) -> str:
    fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="live_chunk_")
    os.close(fd)

    clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return temp_path


def _load_lstm_tokenizer(tokenizer_path: Path) -> Any:
    suffix = tokenizer_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with tokenizer_path.open("rb") as f:
            return pickle.load(f)

    if suffix == ".json":
        payload = tokenizer_path.read_text(encoding="utf-8")
        try:
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
        except Exception as exc:
            raise RuntimeError(
                "Failed to import tokenizer_from_json from TensorFlow Keras."
            ) from exc

        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and "config" in parsed and "class_name" in parsed:
                return tokenizer_from_json(payload)
        except json.JSONDecodeError:
            pass

        # Fallback: plain mapping JSON of {word: index}
        mapping = json.loads(payload)
        if not isinstance(mapping, dict):
            raise RuntimeError(f"Unsupported tokenizer JSON structure in: {tokenizer_path}")
        return {"word_index": mapping}

    raise RuntimeError(
        f"Unsupported tokenizer extension '{suffix}' for file: {tokenizer_path}"
    )


def _resolve_tokenizer_path(explicit_path: str | None, lstm_model_path: str) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_file():
            raise RuntimeError(f"Tokenizer file not found: {candidate}")
        return candidate

    model_dir = Path(lstm_model_path).resolve().parent
    candidates = [
        model_dir / "tokenizer.pkl",
        model_dir / "tokenizer.pickle",
        model_dir / "tokenizer.json",
        model_dir / "word_index.json",
        model_dir / "vocab.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise RuntimeError(
        "Tokenizer/vocabulary artifact not found. Place one of "
        "`tokenizer.pkl`, `tokenizer.pickle`, `tokenizer.json`, `word_index.json`, "
        "or `vocab.json` next to the LSTM model, or pass --tokenizer-path explicitly."
    )


def _to_word_index_mapping(tokenizer: Any) -> dict[str, int]:
    if isinstance(tokenizer, dict):
        if "word_index" in tokenizer and isinstance(tokenizer["word_index"], dict):
            return tokenizer["word_index"]
        # Allow direct dict mapping {word: index}
        if tokenizer and all(isinstance(v, int) for v in tokenizer.values()):
            return tokenizer
        raise RuntimeError("Tokenizer dictionary does not contain a valid word-index mapping.")

    if hasattr(tokenizer, "word_index") and isinstance(tokenizer.word_index, dict):
        return tokenizer.word_index

    raise RuntimeError("Loaded tokenizer does not expose a valid `word_index` mapping.")


def predict_next_word(
    text: str,
    lstm_model: Any,
    word_index: dict[str, int],
    sequence_len: int,
) -> str:
    if not text.strip():
        return ""

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `pad_sequences` from TensorFlow Keras."
        ) from exc

    index_word = {idx: word for word, idx in word_index.items()}

    tokens = [word_index.get(tok.lower(), 0) for tok in text.split()]
    if not tokens:
        return ""

    padded = pad_sequences([tokens], maxlen=sequence_len, padding="pre")
    probs = lstm_model.predict(padded, verbose=0)[0]
    next_idx = int(np.argmax(probs))

    return index_word.get(next_idx, "")


def _merge_tokens_with_overlap(
    existing: list[str],
    incoming: list[str],
    min_overlap_to_append: int = 2,
) -> tuple[list[str], int]:
    """
    Merge incoming tokens into existing buffer while avoiding repeated overlap.
    Example: existing=[i,hello], incoming=[hello,how,are] -> [i,hello,how,are]
    """
    if not incoming:
        return existing[:], 0
    if not existing:
        return incoming[:], 0

    max_overlap = min(len(existing), len(incoming), 12)
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if existing[-size:] == incoming[:size]:
            overlap = size
            break

    # If overlap is too weak, assume this is a fresh phrase and reset the buffer.
    if overlap < min_overlap_to_append:
        return incoming[:], overlap

    return existing + incoming[overlap:], overlap


def _sanitize_prediction(prediction: str, token_buffer: list[str]) -> str:
    token = (prediction or "").strip().lower()
    if not token:
        return ""
    if not VALID_PREDICTION_REGEX.match(token):
        return ""
    if token_buffer and token == token_buffer[-1]:
        return ""
    return token


def run_live_demo(
    whisper_model_name: str,
    lstm_model_path: str,
    tokenizer_path: str | None,
    sequence_len: int | None,
    sample_rate: int,
    chunk_seconds: float,
    language: str,
    mic_device: str | int | None,
    preferred_context_tokens: int,
    recording_event: threading.Event | None = None,
    gloss_callback: Callable[[str], None] | None = None,
) -> None:
    missing = check_dependencies()
    if missing:
        print("[ERROR] Missing dependencies:")
        for item in missing:
            print(f"  - {item}")
        print("\nInstall Python packages:")
        print("  pip install tensorflow sounddevice")
        print("System package required for sounddevice (Linux):")
        print("  sudo apt install portaudio19-dev")
        raise SystemExit(1)

    print(f"[INFO] Loading Whisper model: {whisper_model_name}")
    stt_model = whisper.load_model(whisper_model_name)

    print(f"[INFO] Loading LSTM model from: {lstm_model_path}")
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TensorFlow Keras. Install tensorflow first."
        ) from exc

    lstm_model = load_model(lstm_model_path)

    resolved_tokenizer_path = _resolve_tokenizer_path(tokenizer_path, lstm_model_path)
    print(f"[INFO] Loading tokenizer/vocab from: {resolved_tokenizer_path}")
    tokenizer = _load_lstm_tokenizer(resolved_tokenizer_path)
    word_index = _to_word_index_mapping(tokenizer)

    if sequence_len is None:
        input_shape = getattr(lstm_model, "input_shape", None)
        if isinstance(input_shape, tuple) and len(input_shape) >= 2 and input_shape[1]:
            sequence_len = int(input_shape[1])
        else:
            sequence_len = 20
            print(
                "[WARN] Could not infer model sequence length. Falling back to --sequence-len=20."
            )

    print("\n[READY] Live mode started. Press Ctrl+C to stop.")
    print(f"[INFO] Listening in chunks of {chunk_seconds:.1f}s at {sample_rate}Hz.\n")

    audio_buffer = []
    was_recording = False
    
    # Override samples_per_chunk to be small enough (e.g. 0.5s) to stay responsive to stop events
    check_interval = 0.5
    samples_per_chunk = int(sample_rate * check_interval)

    try:
        while True:
            if recording_event is not None:
                is_recording = recording_event.is_set()
                
                if not is_recording:
                    if was_recording and len(audio_buffer) > 0:
                        # Transitioned from recording to stopped. Process all accumulated audio.
                        print("[INFO] Processing recorded audio...")
                        chunk_audio = np.concatenate(audio_buffer, axis=0)
                        
                        temp_wav = write_wav(chunk_audio, sample_rate)
                        try:
                            result = stt_model.transcribe(
                                temp_wav,
                                task="translate",
                                language=language,
                                verbose=False,
                                fp16=torch.cuda.is_available(),
                            )
                        finally:
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)

                        chunk_text = (result.get("text") or "").strip()
                        if chunk_text:
                            chunk_tokens = text_to_tokens(chunk_text)
                            if chunk_tokens:
                                full_text = " ".join(chunk_tokens)
                                
                                context_tokens = get_prediction_context_tokens(
                                    chunk_tokens,
                                    preferred_n=preferred_context_tokens,
                                    min_n=2,
                                )
                                context_text = " ".join(context_tokens)
                                raw_prediction = predict_next_word(
                                    context_text,
                                    lstm_model=lstm_model,
                                    word_index=word_index,
                                    sequence_len=sequence_len,
                                )
                                prediction = _sanitize_prediction(raw_prediction, chunk_tokens)
                                
                                gloss = to_gloss(full_text)
                                
                                current_gloss_words = gloss.split() if gloss else []
                                if current_gloss_words and gloss_callback is not None:
                                    for word in current_gloss_words:
                                        gloss_callback(word)

                                print("RECORDED OUTPUT")
                                print(f"Text: {chunk_text}")
                                print(f"Prediction: {prediction}")
                                print(f"Gloss: {gloss}")
                                print("-" * 80)
                        
                        audio_buffer.clear()
                    
                    was_recording = False
                    time.sleep(0.1)
                    continue
                else:
                    was_recording = True

            audio = sd.rec(
                samples_per_chunk,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
                device=mic_device,
            )
            audio_buffer.append(audio[:, 0])

            # If recording_event is None (standalone mode), process every chunk_seconds
            if recording_event is None:
                if len(audio_buffer) * check_interval >= chunk_seconds:
                    chunk_audio = np.concatenate(audio_buffer, axis=0)
                    temp_wav = write_wav(chunk_audio, sample_rate)
                    try:
                        result = stt_model.transcribe(
                            temp_wav,
                            task="translate",
                            language=language,
                            verbose=False,
                            fp16=torch.cuda.is_available(),
                        )
                    finally:
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)

                    chunk_text = (result.get("text") or "").strip()
                    if chunk_text:
                        chunk_tokens = text_to_tokens(chunk_text)
                        if chunk_tokens:
                            full_text = " ".join(chunk_tokens)
                            
                            context_tokens = get_prediction_context_tokens(
                                chunk_tokens,
                                preferred_n=preferred_context_tokens,
                                min_n=2,
                            )
                            context_text = " ".join(context_tokens)
                            raw_prediction = predict_next_word(
                                context_text,
                                lstm_model=lstm_model,
                                word_index=word_index,
                                sequence_len=sequence_len,
                            )
                            prediction = _sanitize_prediction(raw_prediction, chunk_tokens)
                            gloss = to_gloss(full_text)
                            
                            print("LIVE OUTPUT")
                            print(f"Chunk: {chunk_text}")
                            print(f"Prediction: {prediction}")
                            print(f"Gloss: {gloss}")
                            print("-" * 80)
                    audio_buffer.clear()

    except KeyboardInterrupt:
        print("\n[STOPPED] Live session ended.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Live pipeline:\n"
            "Speech (mic) -> Whisper text -> Text buffer -> "
            "LSTM next-word prediction (last 2/3 tokens) -> Gloss conversion"
        )
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper model name.")
    parser.add_argument(
        "--lstm-model-path",
        "--lm-model-path",
        dest="lstm_model_path",
        default=str(Path(__file__).resolve().parent / "model" / "language_model.h5"),
        help="Path to your trained LSTM model (.h5).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional tokenizer/vocab artifact path. If omitted, auto-detects next to model "
            "(tokenizer.pkl/json, word_index.json, vocab.json)."
        ),
    )
    parser.add_argument(
        "--sequence-len",
        type=int,
        default=None,
        help="Optional model input sequence length override.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Audio chunk duration in seconds.",
    )
    parser.add_argument(
        "--preferred-context-tokens",
        type=int,
        default=3,
        choices=[2, 3],
        help="Use last N buffered tokens for prediction context (2 or 3).",
    )
    parser.add_argument("--language", default="en", help="Whisper language code.")
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Optional microphone device name/id for sounddevice.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    mic_device: str | int | None = args.mic_device
    if isinstance(mic_device, str) and mic_device.isdigit():
        mic_device = int(mic_device)

    run_live_demo(
        whisper_model_name=args.whisper_model,
        lstm_model_path=args.lstm_model_path,
        tokenizer_path=args.tokenizer_path,
        sequence_len=args.sequence_len,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        language=args.language,
        mic_device=mic_device,
        preferred_context_tokens=args.preferred_context_tokens,
    )


if __name__ == "__main__":
    main()
