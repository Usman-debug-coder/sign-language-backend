# PSL Sign Language Translation — Backend API

Audio-to-Sign-Language translation pipeline exposed as a REST API. Accepts an MP3 audio file and produces an animated 3D avatar (GLTF) performing Pakistan Sign Language (PSL) gestures.

## Pipeline

```
Audio (MP3) → Text (Whisper) → Gloss (NLP) → Videos (PSL lookup) → Keypoints (MediaPipe) → Avatar Animation (GLTF)
```

## Live URL

| Service | URL |
|---------|-----|
| **Backend API** | https://psl-sign-language-api.onrender.com |
| **Frontend UI** | https://frontend-ten-black-54.vercel.app |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload an MP3 file. Returns `{ job_id }` |
| `GET` | `/status/<job_id>` | Poll processing progress (0–100%) and intermediate outputs |
| `GET` | `/gltf/<job_id>` | Serve the animated GLTF for Three.js loading |
| `GET` | `/download/<job_id>` | Download the GLTF as a file |
| `GET` | `/health` | Health check — returns `{ "status": "ok" }` |

## Project Structure

```
backend/
├── app.py                   # Flask API server (entry point)
├── gloss_converter.py       # English text → Gloss (NLP, NLTK)
├── gloss_to_video.py        # Gloss words → PSL video file lookup
├── keypoint_extractor.py    # Video → MediaPipe pose/hand keypoints (JSON)
├── keypoint_retarget.py     # Keypoints → GLTF avatar animation (quaternions)
├── run_full_pipeline.py     # CLI orchestrator for all 5 stages
├── psl_gloss_mapper.py      # PSL dictionary index loader
├── psl_dictionary_items.json# 5000+ PSL terms from psl.org.pk
├── model/                   # LSTM language model for live prediction
│   ├── language_model_best.h5
│   ├── max_len.txt
│   └── tokenizer.pkl
├── requirements.txt         # Python dependencies
├── build.sh                 # Render build script (installs ffmpeg + pip)
├── render.yaml              # Render deployment config
├── runtime.txt              # Python version pin (3.10.12)
├── Procfile                 # Gunicorn start command
├── Dockerfile               # Docker build (for HF Spaces or local Docker)
└── .env.example             # Environment variable reference
```

## Local Development

```bash
# Clone
git clone https://github.com/Usman-debug-coder/sign-language-backend.git
cd sign-language-backend

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Required data files (not in repo — add manually):
#   avatar.gltf   → base 3D avatar model
#   videos/       → PSL gesture MP4s (hello.mp4, how.mp4, you.mp4, etc.)

# Run
python app.py
# Server starts at http://localhost:5000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port |
| `WHISPER_MODEL` | `medium` | Whisper model size (`tiny`, `base`, `small`, `medium`) |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |

> **Free tier note:** Use `WHISPER_MODEL=tiny` on Render free tier to fit in memory.

## Deployment (Render — Free)

This repo is deployed on [Render](https://render.com) with auto-deploy on push to `main`.

- **Runtime:** Python 3.10
- **Build:** `build.sh` installs FFmpeg + pip packages + NLTK data
- **Start:** Gunicorn with 600s timeout (for long Whisper transcriptions)
- **Service ID:** `srv-d6o79bn5r7bs73a7s1q0`

To deploy your own:
1. Fork this repo
2. Go to [render.com](https://render.com) → New → Web Service → connect your fork
3. Render auto-detects `render.yaml` and configures everything
4. Add `avatar.gltf` and `videos/` to your repo

## Tech Stack

- **Flask** — REST API framework
- **OpenAI Whisper** — Speech-to-text
- **NLTK** — Text-to-Gloss NLP processing
- **MediaPipe** — Pose & hand landmark extraction
- **pygltflib** — GLTF avatar animation embedding
- **Gunicorn** — Production WSGI server
