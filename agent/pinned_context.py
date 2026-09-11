"""Persistent background context for Joe (survives conversation refresh)."""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "pinned_context.txt"


class PinnedContextStore:
    """In-memory cache backed by a local text file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self._text = ""
        self.load()

    def load(self) -> str:
        try:
            if self.path.is_file():
                self._text = self.path.read_text(encoding="utf-8")
            else:
                self._text = ""
        except OSError:
            self._text = ""
        return self._text

    def get(self) -> str:
        """Return the latest text from disk (never a stale in-memory-only copy)."""
        return self.load()

    def hash(self) -> str:
        text = self.get().strip()
        if not text:
            return ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def save(self, text: str) -> str:
        cleaned = (text or "").strip()
        self._text = cleaned
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit flush+fsync so a reply that reloads mid-write never sees a partial file.
        with self.path.open("w", encoding="utf-8", newline="\n") as f:
            if cleaned:
                f.write(cleaned + "\n")
            f.flush()
            try:
                import os

                os.fsync(f.fileno())
            except OSError:
                pass
        return self._text
