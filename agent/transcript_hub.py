"""In-process pub/sub for live transcript events (terminal + browser)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agent.expression import smart_append, strip_tags


Role = Literal["you", "agent", "status", "error"]

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_PATH = _DATA_DIR / "transcript_log.jsonl"
_MAX_HISTORY = 200


@dataclass
class TranscriptEvent:
    role: Role
    text: str
    partial: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptEvent | None:
        role = data.get("role")
        text = data.get("text")
        if role not in ("you", "agent", "status", "error"):
            return None
        if not isinstance(text, str):
            return None
        return cls(
            role=role,  # type: ignore[arg-type]
            text=text,
            partial=bool(data.get("partial", False)),
        )


class TranscriptHub:
    def __init__(self, *, log_path: Path | None = None) -> None:
        self._subscribers: set[asyncio.Queue[TranscriptEvent]] = set()
        self._history: list[TranscriptEvent] = []
        self._lock = asyncio.Lock()
        self._live_you: str = ""
        self._log_path = log_path or _LOG_PATH
        self._load_from_disk()

    @property
    def live_you(self) -> str:
        return self._live_you

    def _load_from_disk(self) -> None:
        path = self._log_path
        if not path.is_file():
            return
        loaded: list[TranscriptEvent] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    event = TranscriptEvent.from_dict(data)
                    if event is not None:
                        loaded.append(event)
        except OSError as exc:
            print(f"[transcript] could not load log: {exc}", flush=True)
            return
        if len(loaded) > _MAX_HISTORY:
            loaded = loaded[-_MAX_HISTORY:]
        self._history = loaded
        if loaded:
            print(
                f"[transcript] restored {len(loaded)} event(s) from {path.name}",
                flush=True,
            )

    def _rewrite_disk(self) -> None:
        path = self._log_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for event in self._history:
                    fh.write(event.to_json())
                    fh.write("\n")
            tmp.replace(path)
        except OSError as exc:
            print(f"[transcript] could not save log: {exc}", flush=True)

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

            if len(self._history) > _MAX_HISTORY:
                self._history = self._history[-_MAX_HISTORY:]

            self._rewrite_disk()

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
            self._rewrite_disk()
        event = TranscriptEvent(
            role="status",
            text="CONTEXT_CLEARED — cleared",
        )
        async with self._lock:
            self._history.append(event)
            self._rewrite_disk()
            for q in list(self._subscribers):
                self._enqueue(q, event)


hub = TranscriptHub()
