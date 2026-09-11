# Chatterbox Nano / Turbo (Docker)

Local voice cloning API using [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox).

| Model | Params | When to use |
|-------|--------|-------------|
| **nano** (default) | ~110M | Best for 4GB GPUs / CPU |
| **turbo** | ~350M | Higher quality if you have VRAM |

## Quick start

```bash
cd chatterbox
# GPU (recommended with NVIDIA):
docker compose --profile gpu up -d --build
# CPU-only:
docker compose --profile cpu up -d --build
```

Defaults to **Nano**. GPU uses CUDA PyTorch; first GPU build is a large download.

First start also downloads model weights into the `hf_cache` volume.

- API docs: http://localhost:8090/docs  
- Health: http://localhost:8090/health  

Put a clean **~10 second** recording of your voice at `voices/me.wav` (WAV preferred).
Chatterbox clones **zero-shot** (no fine-tuning) — that file *is* your voice profile.

### Speak with your default voice

```bash
curl -X POST http://localhost:8090/tts ^
  -F "text=Hi, this is my local Chatterbox voice." ^
  -o out.wav
```

Optional: pass `-F "audio_prompt=@voices/other.wav"` to override the default.
### Switch to Turbo

```bash
set CHATTERBOX_MODEL=turbo
docker compose up -d
```

### Profiles

- `--profile gpu` → `chatterbox-local-gpu` (CUDA; large first download)
- `--profile cpu` → `chatterbox-local` (CPU torch)

Prefer **nano** on a 4GB GTX 1650.
