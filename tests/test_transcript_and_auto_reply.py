"""Transcript hub persistence + auto-reply preference defaults."""

from __future__ import annotations

import json
from pathlib import Path

from agent.session import _load_auto_reply, _save_auto_reply
from agent.transcript_hub import TranscriptEvent, TranscriptHub


def test_transcript_hub_persists_and_reloads(tmp_path: Path) -> None:
    log_path = tmp_path / "transcript_log.jsonl"
    hub = TranscriptHub(log_path=log_path)

    import asyncio

    async def _run() -> None:
        await hub.publish(
            TranscriptEvent(role="you", text="hello there", partial=False)
        )
        await hub.publish(
            TranscriptEvent(role="agent", text="hi back", partial=False)
        )

    asyncio.run(_run())
    assert log_path.is_file()
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "hello there"

    hub2 = TranscriptHub(log_path=log_path)
    snap = hub2.snapshot()
    assert len(snap) == 2
    assert snap[0]["role"] == "you"
    assert snap[1]["role"] == "agent"
    assert snap[1]["text"] == "hi back"


def test_transcript_hub_clear_rewrites_disk(tmp_path: Path) -> None:
    log_path = tmp_path / "transcript_log.jsonl"
    hub = TranscriptHub(log_path=log_path)

    import asyncio

    async def _run() -> None:
        await hub.publish(TranscriptEvent(role="you", text="gone", partial=False))
        await hub.clear()

    asyncio.run(_run())
    snap = hub.snapshot()
    assert len(snap) == 1
    assert "CONTEXT_CLEARED" in snap[0]["text"]
    reloaded = TranscriptHub(log_path=log_path).snapshot()
    assert len(reloaded) == 1
    assert "CONTEXT_CLEARED" in reloaded[0]["text"]


def test_auto_reply_defaults_off(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "auto_reply.txt"
    monkeypatch.setattr("agent.session._AUTO_REPLY_PATH", path)
    assert _load_auto_reply(default=False) is False
    _save_auto_reply(True)
    assert path.read_text(encoding="utf-8").strip() == "1"
    assert _load_auto_reply(default=False) is True
    _save_auto_reply(False)
    assert _load_auto_reply(default=False) is False
