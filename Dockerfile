FROM python:3.11-slim

# Install system dependencies for OpenCV, MediaPipe, FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploaded_audio videos keypoints_out

EXPOSE 5000

ENV PORT=5000
ENV WHISPER_MODEL=tiny
ENV ALLOWED_ORIGINS=*

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "600", "--workers", "1", "--threads", "4"]
