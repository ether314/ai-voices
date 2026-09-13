"""Cartesia Ink-2 auto_finalize STT WebSocket.

HARD GATE: Ink is disabled unless the session explicitly opts in with
``allow_cloud_stt=True``. Constructing ``AsyncCartesia`` for TTS alone does
not open this path and must not bill Speech-to-Text tokens.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Optional

from cartesia import AsyncCartesia

OnTurnEnd = Callable[[str], Awaitable[None]]
OnPartial = Callable[[str], Awaitable[None]]

# Module latch — flipped only by VoiceSession when Cartesia STT is intentionally enabled.
_CLOUD_STT_PERMITTED = False


def permit_cloud_stt(enabled: bool) -> None:
    """Allow or forbid InkSTT websocket opens (session-owned)."""
    global _CLOUD_STT_PERMITTED
    _CLOUD_STT_PERMITTED = bool(enabled)
    if enabled:
        print(
            "[stt] WARNING: Cartesia Ink STT PERMITTED — this bills Speech-to-Text tokens",
            flush=True,
        )
    else:
        print("[stt] Cartesia Ink STT forbidden (local / disabled)", flush=True)


def cloud_stt_permitted() -> bool:
    return _CLOUD_STT_PERMITTED


class InkSTTBlockedError(RuntimeError):
    """Raised when code tries to open Cartesia Ink without an explicit permit."""


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
        allow_cloud_stt: bool = False,
    ) -> None:
        if not allow_cloud_stt or not _CLOUD_STT_PERMITTED:
            msg = (
                "REFUSING Cartesia Ink STT — cloud STT is hard-disabled. "
                "Use local Whisper, or set STT_BACKEND=cartesia, set "
                "ALLOW_CARTESIA_STT=1, and Apply STT with confirm_cost in the UI."
            )
            print(f"[stt] {msg}", flush=True)
            raise InkSTTBlockedError(msg)
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
        if not _CLOUD_STT_PERMITTED:
            msg = "REFUSING Ink websocket — Cartesia STT not permitted"
            print(f"[stt] {msg}", flush=True)
            raise InkSTTBlockedError(msg)
        # Allow reuse after close() — previous reconnects left _closed set forever.
        self._closed = asyncio.Event()
        print(
            "[stt] opening Cartesia Ink auto_finalize websocket (BILLS STT TOKENS)",
            flush=True,
        )
        self._connection = await self._client.stt.auto_finalize.websocket(
            encoding=self.encoding,
            model=self.model,
            sample_rate=self.sample_rate,
        ).__aenter__()
        self._recv_task = asyncio.create_task(self._receive_loop(), name="stt-receive")
        print("[stt] websocket connected (Cartesia Ink)", flush=True)
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
