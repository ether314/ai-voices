"""In-process pub/sub for live transcript events (terminal + browser)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


from agent.expression import smart_append, strip_tags


Role = Literal["you", "agent", "status", "error"]


@dataclass
class TranscriptEvent:
    role: Role
    text: str
    partial: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TranscriptHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[TranscriptEvent]] = set()
        self._history: list[TranscriptEvent] = []
        self._lock = asyncio.Lock()
        self._live_you: str = ""

    @property
    def live_you(self) -> str:
        return self._live_you

    async def publish(self, event: TranscriptEvent) -> None:
        async with self._lock:
            if event.role == "you":
                self._live_you = event.text if event.partial else ""
                if event.partial:
                    if (
                        self._history
                        and self._history[-1].role == "you"
                        and self._history[-1].partial
                    ):
                        self._history[-1] = event
                    else:
                        self._history.append(event)
                else:
                    if (
                        self._history
                        and self._history[-1].role == "you"
                        and self._history[-1].partial
                    ):
                        self._history[-1] = event
                    else:
                        self._history.append(event)
            elif event.role == "agent" and event.partial:
                replaced = False
                for i in range(len(self._history) - 1, -1, -1):
                    prev = self._history[i]
                    if prev.role == "agent" and prev.partial:
                        self._history[i] = TranscriptEvent(
                            role="agent",
                            text=smart_append(prev.text, event.text),
                            partial=True,
                        )
                        event = self._history[i]
                        replaced = True
                        break
                    if prev.role == "agent" and not prev.partial:
                        break
                if not replaced:
                    self._history.append(event)
            elif event.role == "agent" and not event.partial:
                replaced = False
                for i in range(len(self._history) - 1, -1, -1):
                    prev = self._history[i]
                    if prev.role == "agent" and prev.partial:
                        self._history[i] = event
                        replaced = True
                        break
                    if prev.role == "agent" and not prev.partial:
                        break
                if not replaced:
                    self._history.append(event)
            else:
                self._history.append(event)

            if len(self._history) > 200:
                self._history = self._history[-200:]

            for q in list(self._subscribers):
                self._enqueue(q, event)

    def _enqueue(
        self, q: asyncio.Queue[TranscriptEvent], event: TranscriptEvent
    ) -> None:
        # Coalesce: never drop the subscriber; keep the newest event.
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        # For rapid partials, replace any queued partial of the same role.
        if event.partial:
            kept: list[TranscriptEvent] = []
            while True:
                try:
                    pending = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if pending.partial and pending.role == event.role:
                    continue
                kept.append(pending)
            for item in kept:
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    break
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue[TranscriptEvent]:
        q: asyncio.Queue[TranscriptEvent] = asyncio.Queue(maxsize=64)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[TranscriptEvent]) -> None:
        self._subscribers.discard(q)

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._history]

    def conversation_text(self) -> str:
        """Build a prompt-friendly transcript from finalized + live speech."""
        lines: list[str] = []
        for event in self._history:
            if event.role == "you":
                # Interlocutor on the call — identity comes from Saved Context script.
                lines.append(f"Caller/room: {event.text}")
            elif event.role == "agent":
                lines.append(f"Joe: {strip_tags(event.text)}")
        live = self._live_you.strip()
        if live:
            # Avoid duplicating if the last history line already matches live partial.
            if not lines or not lines[-1].endswith(live):
                lines.append(f"Caller/room (live): {live}")
        return "\n".join(lines)

    async def clear(self) -> None:
        """Wipe live transcript history (does not touch saved pinned context)."""
        async with self._lock:
            self._history.clear()
            self._live_you = ""
        event = TranscriptEvent(
            role="status",
            text="CONTEXT_CLEARED — cleared",
        )
        async with self._lock:
            self._history.append(event)
            for q in list(self._subscribers):
                self._enqueue(q, event)


hub = TranscriptHub()
