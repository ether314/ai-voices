"""Cartesia Sonic TTS WebSocket continuations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Optional

from cartesia import AsyncCartesia

OnAudioChunk = Callable[[bytes], Awaitable[None]]

DEFAULT_VOICE_ID = "55e8e671-7e46-4a15-9b53-4955c92aec0c"
DEFAULT_MODEL_ID = "sonic-3.6"


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
    ) -> None:
        self._client = client
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.on_audio = on_audio

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

        def _stop() -> bool:
            return bool(should_stop and should_stop())

        async with self._client.tts.websocket_connect() as ws:
            ctx = ws.context(
                model_id=self.model_id,
                voice=self.voice_id,
                output_format=output_format,
                language="en",
            )

            async def push_text() -> None:
                async for chunk in text_chunks:
                    if _stop():
                        return
                    if chunk:
                        await ctx.push(chunk)
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
