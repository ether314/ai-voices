"""Local Whisper VAD should finalize turns on relative pauses (Stereo Mix)."""

from __future__ import annotations

import asyncio
import struct

import pytest

from agent.local_stt import LocalWhisperSTT


def _pcm_chunk(rms_target: float, samples: int = 1600) -> bytes:
    """16-bit mono PCM with approx target RMS (constant amplitude)."""
    amp = max(-32767, min(32767, int(rms_target)))
    return struct.pack("<" + "h" * samples, *([amp] * samples))


@pytest.mark.asyncio
async def test_relative_pause_requests_finalize() -> None:
    """Bed noise above absolute gate but below peak*drop must end the turn."""
    stt = LocalWhisperSTT(end_silence_ms=250)
    stt._model = object()
    stt._closed = asyncio.Event()
    # Prevent send_audio from treating a missing worker as failure.
    stt._worker = asyncio.create_task(asyncio.sleep(3600), name="dummy")

    for _ in range(8):
        await stt.send_audio(_pcm_chunk(4000))
    assert stt._in_speech is True
    assert stt._peak_rms >= 3900

    # ~300ms of audio at RMS 1200: above start gate (350) but < 0.42*4000.
    for _ in range(4):
        await stt.send_audio(_pcm_chunk(1200))

    assert stt._finalize_requested is True
    assert stt._pending_transcribe is True

    stt._closed.set()
    stt._worker.cancel()
    try:
        await stt._worker
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_absolute_quiet_still_finalizes() -> None:
    stt = LocalWhisperSTT(end_silence_ms=250)
    stt._model = object()
    stt._closed = asyncio.Event()
    stt._worker = asyncio.create_task(asyncio.sleep(3600), name="dummy")

    for _ in range(6):
        await stt.send_audio(_pcm_chunk(2000))
    for _ in range(4):
        await stt.send_audio(_pcm_chunk(40))

    assert stt._finalize_requested is True

    stt._closed.set()
    stt._worker.cancel()
    try:
        await stt._worker
    except asyncio.CancelledError:
        pass
