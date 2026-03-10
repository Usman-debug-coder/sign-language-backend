#!/usr/bin/env bash
# Render build script — installs system dependencies + Python packages

set -e

# Install FFmpeg (needed by whisper and ffmpeg-python)
apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data at build time
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"
