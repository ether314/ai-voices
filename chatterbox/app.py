"""Minimal Chatterbox Nano/Turbo HTTP API for local voice cloning."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

MODEL_NAME = os.getenv("CHATTERBOX_MODEL", "nano").strip().lower()  # nano | turbo
DEVICE_ENV = os.getenv("CHATTERBOX_DEVICE", "auto").strip().lower()
DEFAULT_VOICE = Path(
    os.getenv("CHATTERBOX_VOICE_PATH", "/voices/me.wav")
).expanduser()


def _resolve_device() -> str:
    if DEVICE_ENV in ("cuda", "cpu", "mps"):
        if DEVICE_ENV == "cuda" and not torch.cuda.is_available():
            print("[chatterbox] CUDA requested but unavailable — falling back to CPU", flush=True)
            return "cpu"
        return DEVICE_ENV
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _resolve_device()
_model = None


def _load_model():
    global _model
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    nano = MODEL_NAME != "turbo"
    print(
        f"[chatterbox] loading model={MODEL_NAME} nano={nano} device={DEVICE}",
        flush=True,
    )
    _model = ChatterboxTurboTTS.from_pretrained(device=DEVICE, nano=nano)
    print(f"[chatterbox] ready (sr={_model.sr})", flush=True)
    if DEFAULT_VOICE.is_file():
        print(f"[chatterbox] default voice: {DEFAULT_VOICE}", flush=True)
    else:
        print(f"[chatterbox] no default voice at {DEFAULT_VOICE}", flush=True)
    return _model


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Chatterbox local TTS", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ok": _model is not None,
        "model": MODEL_NAME,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "sample_rate": getattr(_model, "sr", None),
        "default_voice": str(DEFAULT_VOICE) if DEFAULT_VOICE.is_file() else None,
    }


@app.get("/voice")
def voice_info() -> dict:
    if not DEFAULT_VOICE.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No default voice at {DEFAULT_VOICE}. Place me.wav in /voices.",
        )
    info = ta.info(str(DEFAULT_VOICE))
    dur = None
    if info.sample_rate and info.num_frames:
        dur = round(info.num_frames / info.sample_rate, 2)
    return {
        "path": str(DEFAULT_VOICE),
        "sample_rate": info.sample_rate,
        "channels": info.num_channels,
        "duration_s": dur,
        "bytes": DEFAULT_VOICE.stat().st_size,
        "note": "Chatterbox is zero-shot; this clip conditions the clone (no fine-tune).",
    }


def _generate(
    text: str,
    ref_path: str,
    temperature: Optional[float],
    exaggeration: Optional[float],
):
    kwargs: dict = {"audio_prompt_path": ref_path}
    if exaggeration is not None:
        kwargs["exaggeration"] = float(exaggeration)
    try:
        if temperature is not None:
            return _model.generate(text, temperature=float(temperature), **kwargs)
        return _model.generate(text, **kwargs)
    except TypeError:
        # Older signatures without temperature/exaggeration.
        kwargs.pop("exaggeration", None)
        try:
            if temperature is not None:
                return _model.generate(text, temperature=float(temperature), audio_prompt_path=ref_path)
            return _model.generate(text, audio_prompt_path=ref_path)
        except TypeError:
            return _model.generate(text, audio_prompt_path=ref_path)


def _apply_speed(wav, speed: float):
    """Time-compress/stretch with torchaudio (factor>1 = faster)."""
    if speed is None or abs(float(speed) - 1.0) < 0.01:
        return wav
    factor = max(0.6, min(1.5, float(speed)))
    sped, _ = ta.functional.speed(wav, _model.sr, factor)
    return sped


@app.post("/tts")
async def tts(
    text: str = Form(..., description="Text to speak"),
    audio_prompt: Optional[UploadFile] = File(
        None,
        description="Optional reference clip; defaults to /voices/me.wav",
    ),
    temperature: Optional[float] = Form(None),
    exaggeration: Optional[float] = Form(None),
    speed: Optional[float] = Form(None, description="Playback speed 0.6–1.5 (1.0 = normal)"),
) -> Response:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    tmp_path: Optional[str] = None
    try:
        if audio_prompt is not None and audio_prompt.filename:
            suffix = os.path.splitext(audio_prompt.filename or "ref.wav")[1] or ".wav"
            raw = await audio_prompt.read()
            if not raw:
                raise HTTPException(status_code=400, detail="audio_prompt is empty")
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            ref_path = tmp_path
            voice_src = "upload"
        else:
            if not DEFAULT_VOICE.is_file():
                raise HTTPException(
                    status_code=400,
                    detail="No audio_prompt and no default voice at /voices/me.wav",
                )
            ref_path = str(DEFAULT_VOICE)
            voice_src = "default"

        try:
            wav = _generate(text, ref_path, temperature, exaggeration)
            wav = _apply_speed(wav, speed if speed is not None else 1.0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    buf = io.BytesIO()
    ta.save(buf, wav, _model.sr, format="wav")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Chatterbox-Model": MODEL_NAME,
            "X-Chatterbox-Device": DEVICE,
            "X-Chatterbox-Voice": voice_src,
            "X-Chatterbox-Speed": str(speed if speed is not None else 1.0),
        },
    )
