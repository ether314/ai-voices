"""Cursor SDK agent wrapper for voice-only replies."""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from typing import Optional

from cursor_sdk import AsyncClient, LocalAgentOptions

DEFAULT_MODEL = "gpt-5.6-luna"

VOICE_SYSTEM = (
    "You are Joe on a live phone call. "
    "Reply with short spoken dialogue only — one thought at a time. "
    "Always write normal English with a space between every word. "
    "Never concatenate words (wrong: 'Howareyou'; right: 'How are you'). "
    "Do not use markdown, lists, code fences, bullet points, stage directions, "
    "emotion tags, or delivery headers. "
    "Do not use tools, edit files, run commands, or mention being an agent. "
    "Keep replies natural for text-to-speech."
)


def _snippet(text: str, n: int = 160) -> str:
    one = " ".join((text or "").split())
    if len(one) <= n:
        return one
    return one[: n - 1] + "…"


def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def wrap_voice_prompt(user_transcript: str) -> str:
    """Wrap a user turn so the coding agent stays in voice-only mode."""
    return (
        f"{VOICE_SYSTEM}\n\n"
        f"User just said (transcribed verbatim):\n"
        f"{user_transcript}\n\n"
        f"Respond with spoken dialogue only."
    )


def build_pinned_block(pinned_context: str) -> str:
    """Mandatory scenario block — placed first in the turn prompt."""
    pinned = (pinned_context or "").strip()
    if not pinned:
        return ""
    if len(pinned) > 2500:
        pinned = pinned[:2500] + "…"
    return (
        "### MUST FOLLOW — SCENARIO SCRIPT (HIGHEST PRIORITY)\n"
        "This block overrides any earlier messages in this agent chat, any default "
        "coding-assistant habits, and any generic 'be helpful' instincts.\n"
        "It is the standing script for THIS call:\n"
        "- Who you are talking to (the Caller/room person)\n"
        "- The goal of the call\n"
        "- Facts you must remember\n"
        "- How you should behave / stay in character\n"
        "Role-play accordingly on EVERY line. Do not invent a conflicting premise. "
        "Do not soft-pedal, summarize away, or ignore this script. "
        "Scenario compliance beats generic Joe blandness. "
        "Still speak as short natural phone-call dialogue (not a briefing).\n"
        "----- SCRIPT START -----\n"
        f"{pinned}\n"
        "----- SCRIPT END -----\n\n"
    )


def wrap_history_prompt(
    conversation: str, *, pinned_context: str = ""
) -> str:
    """Ask for a spoken reply grounded in saved context + call transcript."""
    # Keep prompts short so Cursor stays responsive.
    max_chars = 4500
    pinned = (pinned_context or "").strip()
    body = (conversation or "").strip()
    if len(body) > max_chars:
        body = "…\n" + body[-max_chars:]
    pinned_block = build_pinned_block(pinned)
    # Scenario FIRST so it is not buried under style rules or a long transcript.
    return (
        f"{pinned_block}"
        f"{VOICE_SYSTEM}\n\n"
        "You are Joe, speaking live on a phone call. "
        "Caller/room in the transcript IS the person described in the scenario script "
        "(when a script is present). "
        "When the scenario script is present above, your next line MUST follow it. "
        "Say the next natural line — in character, in the flow of the call. "
        "Do not summarize the whole call unless asked. Do not narrate; just speak.\n\n"
        f"Transcript so far:\n{body or '(none yet — open using the scenario script)'}\n\n"
        "Respond with spoken dialogue only."
    )


class CursorVoiceLLM:
    """Cursor agent for voice; prefers fresh agents when a scenario script is active.

    The hub already injects the full call transcript every turn, so durable multi-turn
    agent memory mostly duplicates history and can bury a newly saved scenario.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        cwd: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.cwd = cwd or os.getcwd()
        self._client: AsyncClient | None = None
        self._agent = None
        self._cm_client = None
        self._cm_agent = None
        self._last_pinned_hash: str = ""

    async def start(self) -> None:
        self._cm_client = await AsyncClient.launch_bridge(workspace=self.cwd)
        self._client = await self._cm_client.__aenter__()
        self._cm_agent = await self._client.agents.create(
            model=self.model,
            api_key=self.api_key,
            local=LocalAgentOptions(cwd=self.cwd),
        )
        self._agent = await self._cm_agent.__aenter__()

    async def close(self) -> None:
        if self._cm_agent is not None:
            await self._cm_agent.__aexit__(None, None, None)
            self._cm_agent = None
            self._agent = None
        if self._cm_client is not None:
            await self._cm_client.__aexit__(None, None, None)
            self._cm_client = None
            self._client = None

    async def reset(self) -> None:
        """Start a fresh Cursor agent so prior turns leave the context window."""
        if self._client is None:
            raise RuntimeError("CursorVoiceLLM.start() was not called")
        if self._cm_agent is not None:
            await self._cm_agent.__aexit__(None, None, None)
            self._cm_agent = None
            self._agent = None
        self._cm_agent = await self._client.agents.create(
            model=self.model,
            api_key=self.api_key,
            local=LocalAgentOptions(cwd=self.cwd),
        )
        self._agent = await self._cm_agent.__aenter__()
        print("[llm] context purged — new Cursor agent", flush=True)

    def invalidate_pinned_cache(self) -> None:
        """Force the next history reply to recreate the agent (after Save context)."""
        self._last_pinned_hash = "__invalidate__"

    async def _ensure_fresh_for_pinned(self, pinned: str) -> None:
        """Recreate the agent so SDK multi-turn memory cannot bury the scenario.

        Hub already injects the full call transcript every turn, so durable agent
        chat is redundant. Without a reset, earlier bland 'assistant' turns keep
        winning over a newly saved (or repeatedly injected) script.
        """
        digest = (
            hashlib.sha256(pinned.encode("utf-8")).hexdigest()[:12] if pinned else ""
        )
        # Fresh agent on every history reply (pinned or not after a script was used /
        # invalidated). Cheap relative to ignoring the scenario.
        await self.reset()
        self._last_pinned_hash = digest

    async def stream_reply(self, user_transcript: str) -> AsyncIterator[str]:
        if self._agent is None:
            raise RuntimeError("CursorVoiceLLM.start() was not called")
        prompt = wrap_voice_prompt(user_transcript)
        run = await self._agent.send(prompt)
        try:
            async for chunk in run.iter_text():
                if chunk:
                    yield chunk
        finally:
            # Always wait so the run settles and resources are released.
            await run.wait()

    async def stream_history_reply(
        self, conversation: str, *, pinned_context: str = ""
    ) -> AsyncIterator[str]:
        if self._agent is None:
            raise RuntimeError("CursorVoiceLLM.start() was not called")
        pinned = (pinned_context or "").strip()
        await self._ensure_fresh_for_pinned(pinned)
        prompt = wrap_history_prompt(conversation, pinned_context=pinned)
        fp = _prompt_fingerprint(prompt)
        if pinned:
            print(
                f"[llm] pinned ACTIVE chars={len(pinned)} "
                f"hash={hashlib.sha256(pinned.encode()).hexdigest()[:12]} "
                f"snippet={_snippet(pinned)!r}",
                flush=True,
            )
            print(
                f"[llm] prompt proof ({len(prompt)} chars, fp={fp}): "
                f"{_snippet(prompt, 500)!r}",
                flush=True,
            )
            if "MUST FOLLOW — SCENARIO SCRIPT" not in prompt or pinned[:40] not in prompt:
                print(
                    "[llm] WARNING: pinned text missing from outgoing prompt!",
                    flush=True,
                )
        else:
            print(
                f"[llm] pinned EMPTY — no scenario script on this reply "
                f"(prompt {len(prompt)} chars, fp={fp})",
                flush=True,
            )
        print(f"[llm] sending history reply ({len(prompt)} chars)…", flush=True)
        run = await self._agent.send(prompt)
        yielded = False
        try:
            async for chunk in run.iter_text():
                if chunk:
                    yielded = True
                    yield chunk
        finally:
            result = await run.wait()
            status = getattr(result, "status", None)
            print(f"[llm] run finished status={status!r} yielded={yielded}", flush=True)
            if not yielded:
                # Fallback if streaming produced nothing.
                text = getattr(result, "result", None) or ""
                if not text and hasattr(run, "text"):
                    try:
                        text = await run.text()
                    except Exception:  # noqa: BLE001
                        text = ""
                if text:
                    yield str(text)
                else:
                    raise RuntimeError(
                        f"Cursor returned no spoken text (status={status!r}). "
                        "Try Stop, then Respond as Joe again."
                    )
