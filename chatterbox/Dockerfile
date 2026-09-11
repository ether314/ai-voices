# Chatterbox Nano/Turbo — local voice cloning API
# Nano (~110M) fits a 4GB GPU; Turbo (~350M) needs more VRAM or use CPU.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    CHATTERBOX_MODEL=nano \
    CHATTERBOX_DEVICE=auto \
    GIT_HTTP_LOW_SPEED_LIMIT=0 \
    GIT_HTTP_LOW_SPEED_TIME=999999

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch/torchaudio from the chosen index first.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
        "numpy>=1.24.0,<2.0.0" \
        torch==2.6.0 torchaudio==2.6.0 \
        --index-url "${TORCH_INDEX}"

# Perth is a git dependency that often flakes — install with retries first.
RUN pip install --no-cache-dir --retries 10 \
        "git+https://github.com/resemble-ai/Perth.git@master"

# Chatterbox from GitHub (PyPI 0.1.7 lacks Nano / nano=True).
RUN pip install --no-cache-dir --retries 10 \
        "git+https://github.com/resemble-ai/chatterbox.git" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.32.0" \
        "python-multipart>=0.0.12"

# Gradio is pulled in by chatterbox but unused by this API.
RUN pip uninstall -y gradio || true

COPY app.py .

EXPOSE 8090
VOLUME ["/cache/huggingface", "/voices"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=8 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
