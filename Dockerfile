FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default PulseAudio socket path when bridging from host/WSL
    PULSE_SERVER=unix:/run/pulse/native

# PortAudio + ALSA + sndfile for sounddevice; ca-certs for HTTPS/WSS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1 \
    libasound2 \
    libasound2-plugins \
    alsa-utils \
    pulseaudio-utils \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY agent/ ./agent/
COPY main.py .

# Non-root is nicer, but /dev/snd often needs group access; stay root for audio devices.
CMD ["python", "main.py"]
