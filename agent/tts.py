"""Cartesia Sonic TTS WebSocket continuations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Optional

from cartesia import AsyncCartesia

from agent.expression import normalize_speech_for_tts, should_flush_phrase, smart_append
from agent.voice_style import (
    DEFAULT_SPEED,
    DEFAULT_TONALITY,
    cartesia_generation_config,
)

OnAudioChunk = Callable[[bytes], Awaitable[None]]

DEFAULT_VOICE_ID = "55e8e671-7e46-4a15-9b53-4955c92aec0c"
DEFAULT_MODEL_ID = "sonic-3.6"

# Curated Cartesia voices for the UI dropdown (id must be a real Cartesia voice UUID).
CARTESIA_VOICES: tuple[dict[str, str], ...] = (
    {"id": DEFAULT_VOICE_ID, "label": "Joe Marazzo (default)"},
    {"id": "4e65cba0-b17c-4c6c-a985-df33894014d7", "label": "Hong"},
    {"id": "47c38ca4-5f35-497b-b1a3-415245fb35e1", "label": "Daniel — Modern Assistant"},
    {"id": "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4", "label": "Skylar — Friendly Guide"},
    {"id": "30894953-bcce-41fe-892c-15ce19c843ff", "label": "Parker — Supportive Pal"},
    {"id": "ef191366-f52f-447a-a398-ed8c0f2943a1", "label": "Archie — Approachable Mate"},
)


class SonicTTS:
    """Stream LLM text chunks into Sonic and forward PCM to a playback callback."""

    def __init__(
        self,
        client: AsyncCartesia,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        sample_rate: int = 16_000,
        on_audio: Optional[OnAudioChunk] = None,
        speed: float = DEFAULT_SPEED,
        tonality: str = DEFAULT_TONALITY,
    ) -> None:
        self._client = client
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.on_audio = on_audio
        self.speed = speed
        self.tonality = tonality

    def set_delivery(self, *, speed: Optional[float] = None, tonality: Optional[str] = None) -> None:
        if speed is not None:
            self.speed = float(speed)
        if tonality is not None:
            self.tonality = tonality

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Push streamed text into one TTS context and play audio as it arrives."""
        output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }
        gen_cfg = cartesia_generation_config(speed=self.speed, tonality=self.tonality)

        def _stop() -> bool:
            return bool(should_stop and should_stop())

        async with self._client.tts.websocket_connect() as ws:
            ctx = ws.context(
                model_id=self.model_id,
                voice=self.voice_id,
                output_format=output_format,
                language="en",
                generation_config=gen_cfg,
            )

            async def push_text() -> None:
                # Phrase-buffer so we can self-check spacing before Sonic speaks.
                joined = ""
                first = True
                async for chunk in text_chunks:
                    if _stop():
                        return
                    if not chunk:
                        continue
                    joined = smart_append(joined, chunk)
                    spaced = normalize_speech_for_tts(joined)
                    if should_flush_phrase(spaced, first=first):
                        phrase = spaced.strip()
                        if phrase:
                            await ctx.push(phrase + " ")
                        joined = ""
                        first = False
                leftover = normalize_speech_for_tts(joined).strip()
                if leftover and not _stop():
                    await ctx.push(leftover)
                if _stop():
                    return
                await ctx.no_more_inputs()

            async def receive_audio() -> None:
                async for response in ctx.receive():
                    if _stop():
                        return
                    rtype = getattr(response, "type", None)
                    if rtype == "chunk":
                        audio = getattr(response, "audio", None)
                        if audio and self.on_audio is not None and not _stop():
                            await self.on_audio(audio)
                    elif rtype == "error":
                        message = (
                            getattr(response, "message", None)
                            or getattr(response, "title", None)
                            or str(response)
                        )
                        print(f"[tts] error: {message}", flush=True)

            try:
                await asyncio.gather(push_text(), receive_audio())
            except asyncio.CancelledError:
                # Stop button — exit the websocket context immediately.
                raise

    async def speak(self, text: str) -> None:
        async def _one() -> AsyncIterator[str]:
            yield text

        await self.speak_stream(_one())
