"""Local Chatterbox TTS (Docker GPU/CPU) via HTTP /tts."""

from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import numpy as np

from agent.audio import SAMPLE_RATE, _resample_mono
from agent.expression import should_flush_phrase, smart_append, speech_for_tts
from agent.voice_style import DEFAULT_SPEED, DEFAULT_TONALITY, local_tts_params

OnAudioChunk = Callable[[bytes], Awaitable[None]]

DEFAULT_CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://127.0.0.1:8090").rstrip("/")


def _should_flush(text: str, *, first: bool) -> bool:
    return should_flush_phrase(text, first=first)

def wav_bytes_to_pcm16le(wav_bytes: bytes, *, target_rate: int = SAMPLE_RATE) -> bytes:
    """Decode a WAV response into mono pcm_s16le at target_rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"unsupported WAV sample width {sample_width}")
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != target_rate:
        audio = _resample_mono(audio, rate, target_rate)
    return audio.tobytes()


class ChatterboxTTS:
    """Pipelined local TTS: LLM text in, GPU synth overlaps with playback."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_CHATTERBOX_URL,
        sample_rate: int = SAMPLE_RATE,
        on_audio: Optional[OnAudioChunk] = None,
        timeout_s: float = 120.0,
        speed: float = DEFAULT_SPEED,
        tonality: str = DEFAULT_TONALITY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sample_rate = sample_rate
        self.on_audio = on_audio
        self.timeout_s = timeout_s
        self.speed = speed
        self.tonality = tonality
        self._http: aiohttp.ClientSession | None = None

    def set_base_url(self, url: str) -> None:
        self.base_url = (url or DEFAULT_CHATTERBOX_URL).rstrip("/")

    def set_delivery(self, *, speed: Optional[float] = None, tonality: Optional[str] = None) -> None:
        if speed is not None:
            self.speed = float(speed)
        if tonality is not None:
            self.tonality = tonality

    async def _session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None

    async def health(self) -> dict:
        url = urljoin(self.base_url + "/", "health")
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {
                            "ok": False,
                            "reachable": True,
                            "error": f"HTTP {resp.status}: {text[:200]}",
                            "url": self.base_url,
                        }
                    data = await resp.json(content_type=None)
                    data["ok"] = bool(data.get("ok", True))
                    data["reachable"] = True
                    data["url"] = self.base_url
                    return data
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "reachable": False,
                "error": str(exc),
                "url": self.base_url,
            }

    async def _synth_pcm(
        self,
        text: str,
        *,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> bytes:
        if not text.strip():
            return b""
        if should_stop and should_stop():
            return b""
        url = urljoin(self.base_url + "/", "tts")
        params = local_tts_params(speed=self.speed, tonality=self.tonality)
        form = aiohttp.FormData()
        form.add_field("text", text.strip())
        form.add_field("speed", f"{params['speed']:g}")
        form.add_field("temperature", f"{params['temperature']:g}")
        form.add_field("exaggeration", f"{params['exaggeration']:g}")
        session = await self._session()
        t0 = asyncio.get_running_loop().time()
        async with session.post(url, data=form) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise RuntimeError(
                    f"Chatterbox TTS failed ({resp.status}): {detail[:300]}"
                )
            wav_bytes = await resp.read()
        if should_stop and should_stop():
            return b""
        pcm = await asyncio.to_thread(
            wav_bytes_to_pcm16le, wav_bytes, target_rate=self.sample_rate
        )
        ms = (asyncio.get_running_loop().time() - t0) * 1000
        print(
            f"[tts/local] synth {len(text.strip())} chars -> "
            f"{len(pcm)} bytes in {ms:.0f}ms",
            flush=True,
        )
        return pcm

    async def _play_pcm(
        self,
        pcm: bytes,
        *,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        if not pcm or self.on_audio is None:
            return
        frame = self.sample_rate // 10 * 2  # 100ms of int16 mono
        for i in range(0, len(pcm), frame):
            if should_stop and should_stop():
                return
            chunk = pcm[i : i + frame]
            if chunk:
                await self.on_audio(chunk)

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Three-stage pipeline: LLM text -> GPU synth -> play (overlapped)."""

        def _stop() -> bool:
            return bool(should_stop and should_stop())

        phrase_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        # Keep one phrase synthesizing ahead of the speaker.
        pcm_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2)

        async def consume_llm() -> None:
            joined = ""
            first = True
            try:
                async for piece in text_chunks:
                    if _stop():
                        break
                    if piece:
                        joined = smart_append(joined, piece)
                    if _should_flush(joined, first=first):
                        phrase = speech_for_tts(joined, backend="local")
                        if phrase:
                            await phrase_q.put(phrase)
                        joined = ""
                        first = False
                leftover = speech_for_tts(joined, backend="local")
                if leftover and not _stop():
                    await phrase_q.put(leftover)
            finally:
                await phrase_q.put(None)

        async def synthesizer() -> None:
            try:
                while True:
                    phrase = await phrase_q.get()
                    if phrase is None:
                        break
                    if _stop():
                        break
                    # Last gate before GPU synth — adapt Cartesia markup / spacing.
                    phrase = speech_for_tts(phrase, backend="local")
                    if not phrase:
                        continue
                    pcm = await self._synth_pcm(phrase, should_stop=should_stop)
                    if _stop():
                        break
                    if pcm:
                        await pcm_q.put(pcm)
            finally:
                await pcm_q.put(None)

        async def player() -> None:
            while True:
                pcm = await pcm_q.get()
                if pcm is None:
                    break
                if _stop():
                    break
                await self._play_pcm(pcm, should_stop=should_stop)

        try:
            await asyncio.gather(consume_llm(), synthesizer(), player())
        except asyncio.CancelledError:
            raise
