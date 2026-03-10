import argparse
import json
from pathlib import Path
from typing import List

import whisper

from gloss_converter import to_gloss
from gloss_to_video import gloss_to_video_sequence
from keypoint_extractor import extract_keypoints
from keypoint_retarget import retarget


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline:\n"
            "AUDIO -> TEXT (Whisper) -> GLOSS -> VIDEOS -> "
            "MEDIAPIPE KEYPOINTS -> AVATAR GLTF ANIMATION"
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input audio file (e.g. testcase1.mp3).",
    )
    parser.add_argument(
        "--avatar",
        default="avatar.gltf",
        help="Base avatar glTF file to animate. Default: avatar.gltf",
    )
    parser.add_argument(
        "--output",
        default="avatar_with_anim.gltf",
        help="Output glTF file with embedded animation. Default: avatar_with_anim.gltf",
    )
    parser.add_argument(
        "--animation-name",
        default="AutoGesture",
        help="Name for the animation clip stored in the glTF.",
    )
    parser.add_argument(
        "--keypoints-dir",
        default="keypoints_out",
        help="Directory where extracted keypoints JSON files will be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS used when computing keyframe times.",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model size to use (tiny, base, small, medium, large, etc.).",
    )
    return parser


def _select_keypoints_file(json_paths: List[str]) -> Path:
    """
    Pick one keypoints JSON to retarget. For now we use the last one in the list,
    which corresponds to the last video in the gloss sequence.

    This keeps the implementation simple while still supporting multi-word
    glosses where each word maps to one video.
    """
    if not json_paths:
        raise RuntimeError("No keypoints JSON files were produced; cannot retarget avatar.")
    return Path(json_paths[-1])


def run_pipeline(
    audio_path: Path,
    avatar_path: Path,
    output_path: Path,
    animation_name: str,
    keypoints_dir: Path,
    fps: float,
    whisper_model_name: str,
) -> None:
    # 1) AUDIO -> TEXT
    print(f"[INFO] Loading Whisper model: {whisper_model_name}")
    model = whisper.load_model(whisper_model_name)

    print(f"[INFO] Transcribing audio: {audio_path}")
    result = model.transcribe(str(audio_path), task="transcribe", language="en", verbose=False)
    text = (result.get("text") or "").strip()
    if not text:
        raise RuntimeError("Whisper transcription returned empty text.")

    print("[INFO] Final text output:")
    print(text)

    # 2) TEXT -> GLOSS
    gloss = to_gloss(text)
    if not gloss:
        raise RuntimeError("Gloss converter returned empty gloss string.")

    print("[INFO] Final gloss output:")
    print(gloss)

    # 3) GLOSS -> VIDEO SEQUENCE
    video_list = gloss_to_video_sequence(gloss)
    print("[INFO] Video sequence:", video_list)

    if not video_list:
        raise RuntimeError("No videos found for the produced gloss; cannot continue.")

    # 4) VIDEOS -> KEYPOINTS (Mediapipe)
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting keypoints into: {keypoints_dir}")
    keypoint_files = extract_keypoints(
        video_paths=[str(v) for v in video_list],
        output_dir=str(keypoints_dir),
        fps=int(fps),
    )

    if not keypoint_files:
        raise RuntimeError("Keypoint extraction produced no JSON files.")

    print("[INFO] Keypoint data saved at:", keypoint_files)

    keypoints_path = _select_keypoints_file(keypoint_files)

    # 5) KEYPOINTS -> AVATAR GLTF ANIMATION
    print(f"[INFO] Retargeting keypoints from {keypoints_path} onto avatar {avatar_path}")
    retarget(
        gltf_path=avatar_path,
        keypoints_path=keypoints_path,
        mapping_path=None,
        output_path=output_path,
        animation_name=animation_name,
        fps=fps,
    )

    print(f"[SUCCESS] Animation '{animation_name}' embedded into {output_path}")
    print("[HINT] Serve this folder (e.g. `python3 -m http.server 8000`) and open index.html in a browser.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"[ERROR] Audio file not found: {audio_path}")

    avatar_path = Path(args.avatar)
    if not avatar_path.is_file():
        raise SystemExit(f"[ERROR] Avatar glTF not found: {avatar_path}")

    output_path = Path(args.output)
    keypoints_dir = Path(args.keypoints_dir)

    run_pipeline(
        audio_path=audio_path,
        avatar_path=avatar_path,
        output_path=output_path,
        animation_name=args.animation_name,
        keypoints_dir=keypoints_dir,
        fps=args.fps,
        whisper_model_name=args.whisper_model,
    )


if __name__ == "__main__":
    main()


