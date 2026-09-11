"""Cartesia Ink-2 auto_finalize STT WebSocket."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Optional

from cartesia import AsyncCartesia

OnTurnEnd = Callable[[str], Awaitable[None]]
OnPartial = Callable[[str], Awaitable[None]]


class InkSTT:
    """Stream mic PCM into Ink-2 and surface turn updates / turn ends."""

    def __init__(
        self,
        client: AsyncCartesia,
        *,
        model: str = "ink-2",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16_000,
        on_partial: Optional[OnPartial] = None,
        on_turn_end: Optional[OnTurnEnd] = None,
    ) -> None:
        self._client = client
        self.model = model
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.on_partial = on_partial
        self.on_turn_end = on_turn_end
        self._connection: Any = None
        self._recv_task: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()

    @property
    def alive(self) -> bool:
        return (
            self._connection is not None
            and not self._closed.is_set()
            and self._recv_task is not None
            and not self._recv_task.done()
        )

    async def __aenter__(self) -> "InkSTT":
        # Allow reuse after close() — previous reconnects left _closed set forever.
        self._closed = asyncio.Event()
        self._connection = await self._client.stt.auto_finalize.websocket(
            encoding=self.encoding,
            model=self.model,
            sample_rate=self.sample_rate,
        ).__aenter__()
        self._recv_task = asyncio.create_task(self._receive_loop(), name="stt-receive")
        print("[stt] websocket connected", flush=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def send_audio(self, pcm: bytes) -> None:
        if self._connection is None or not pcm:
            return
        if self._closed.is_set() or (
            self._recv_task is not None and self._recv_task.done()
        ):
            raise ConnectionError("STT websocket is closed")
        await self._connection.send_raw(pcm)

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._connection is not None:
            try:
                await self._connection.send({"type": "close"})
            except Exception:
                pass
        if self._recv_task is not None:
            try:
                await asyncio.wait_for(self._recv_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._recv_task.cancel()
            self._recv_task = None
        if self._connection is not None:
            try:
                await self._connection.__aexit__(None, None, None)
            except Exception:
                pass
            self._connection = None
        print("[stt] websocket closed", flush=True)

    async def _receive_loop(self) -> None:
        assert self._connection is not None
        try:
            async for event in self._connection:
                etype = getattr(event, "type", None)
                if etype == "turn.update":
                    transcript = getattr(event, "transcript", "") or ""
                    if self.on_partial is not None:
                        await self.on_partial(transcript)
                elif etype == "turn.end":
                    # Verbatim — do not strip/normalize (Cartesia STT pitfall).
                    transcript = getattr(event, "transcript", "") or ""
                    if self.on_turn_end is not None:
                        await self.on_turn_end(transcript)
                elif etype == "error":
                    message = getattr(event, "message", None) or str(event)
                    print(f"[stt] error: {message}", flush=True)
        except Exception as exc:  # noqa: BLE001
            if not self._closed.is_set():
                print(f"[stt] receive ended: {exc}", flush=True)
        # Leave cleanup to close()/__aexit__; feed detects recv_task.done().
