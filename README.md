# Local mic voice agent

Speak into your PC mic → live transcript (Cartesia **Ink-2**) → reply (Cursor **composer-2.5**) → spoken aloud (Cartesia **Sonic**, Joe Marazzo voice).

## Quick start (Docker — recommended)

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS/Linux)
- API keys:
  - `CARTESIA_API_KEY` from [play.cartesia.ai/keys](https://play.cartesia.ai/keys)
  - `CURSOR_API_KEY` from [Cursor Dashboard → Integrations](https://cursor.com/dashboard/integrations)

### 2. Configure env

```bash
cp .env.example .env
# Edit .env and paste your keys
```

### 3. Build

```bash
docker compose build
```

### 4. Run

```bash
docker compose run --rm voice-agent
```

Or:

```bash
docker compose up --build
```

Stop with `Ctrl+C`.

> Note: the image already contains the app. Compose does not bind-mount the project folder by default (Windows paths with spaces break Docker Desktop mounts). Rebuild after code changes: `docker compose build`.

---

## Windows audio (Docker Desktop)

Docker Desktop on Windows runs Linux containers in a VM. **True mic/speaker passthrough is limited** — `/dev/snd` from Windows does not map cleanly the way it does on a native Linux host.

### Practical options

**(A) Prefer host Python for live mic/speakers (simplest on Windows)**  
Use Docker to *build/install* a reproducible environment, but for day-to-day conversation with your real devices, run on the host (see [Local Python fallback](#local-python-fallback) below). This is the most reliable path for mic + speakers on Windows.

**(B) WSL2 + PulseAudio / WSLg bridge**  
If you use WSL2 with [WSLg](https://github.com/microsoft/wslg) (Windows 11), PulseAudio is often available inside WSL. You can point the container at that server:

1. In WSL, confirm Pulse is up (`pactl info`).
2. Uncomment the Pulse volume in `docker-compose.yml` and set something like:

```yaml
volumes:
  - /mnt/wslg/runtime-dir/pulse:/run/pulse:ro
environment:
  PULSE_SERVER: unix:/run/pulse/native
```

3. Run the compose service from a context that can see that socket (often: Docker backend set to WSL2, compose invoked from WSL).

**(C) Linux Docker host**  
On a real Linux machine, uncomment in `docker-compose.yml`:

```yaml
devices:
  - /dev/snd:/dev/snd
group_add:
  - audio
```

Then `docker compose run --rm voice-agent` can use ALSA devices via PortAudio.

`docker compose build` and `docker compose run` always work for installing Python deps inside the image even when host audio is not attached; without a working sound device, `sounddevice` will error at runtime when opening the mic/speakers.

---

## Local Python fallback

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # if you have not already
python main.py
```

List audio devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Optional `.env` overrides:

| Variable | Default | Purpose |
|---|---|---|
| `CARTESIA_VOICE_ID` | Joe Marazzo `d2b2c34b-bfd0-4c44-a85a-1d7ab8f3bbcc` | TTS voice |
| `CURSOR_MODEL` | `composer-2.5` | Cursor agent model |
| `AUDIO_INPUT_DEVICE` | system default | Mic device index |
| `AUDIO_OUTPUT_DEVICE` | system default | Speaker device index |

---

## How it works

1. Capture mono PCM at **16 kHz / pcm_s16le** (~100 ms frames).
2. Stream to Cartesia **Ink-2** `auto_finalize` WebSocket; print `turn.update` live; on `turn.end` send transcript **verbatim**.
3. Long-lived Cursor SDK agent (`composer-2.5`) replies with voice-only prompts (no tools / file edits). Text streams via `run.iter_text()`; always `run.wait()`.
4. Stream text into Cartesia **Sonic** TTS WebSocket continuations; play PCM on speakers.
5. **Echo guard:** mute STT while the agent speaks, then ~300 ms cooldown before listening again.

### Phone tip

Put the phone on speaker between you and the PC, keep volumes moderate. Headphones on the PC make “coach me” mode cleaner if only you need to hear replies.

---

## Project layout

```
AI voices/
  main.py
  agent/
    audio.py      # mic + speakers
    stt.py        # Ink-2
    tts.py        # Sonic
    llm.py        # Cursor SDK
    session.py    # loop + echo guard
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```
