"""Cursor SDK agent wrapper for voice-only replies."""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from typing import Literal, Optional

from cursor_sdk import AsyncClient, LocalAgentOptions

from agent.expression import cartesia_paralinguistics_prompt_block

DEFAULT_MODEL = "composer-2.5"
# Cursor SDK: pass `local=` for on-machine runtime; `cloud=` would spawn a VM.
# Joe always uses local — never silently fall back to cloud.
DEFAULT_CURSOR_RUNTIME: Literal["local"] = "local"


def resolve_cursor_runtime() -> Literal["local"]:
    """Return the Cursor agent runtime. Only ``local`` is allowed for Joe.

    Set ``CURSOR_RUNTIME=local`` (default). Any other value fails closed —
    we never fall back to a Cursor cloud / background-composer VM.
    """
    raw = (os.getenv("CURSOR_RUNTIME") or DEFAULT_CURSOR_RUNTIME).strip().lower()
    if not raw:
        raw = DEFAULT_CURSOR_RUNTIME
    if raw != "local":
        raise RuntimeError(
            f"CURSOR_RUNTIME={raw!r} is not supported for Joe. "
            "Only local Cursor runtime is allowed (set CURSOR_RUNTIME=local). "
            "Cloud agents are disabled — no silent fallback to a cloud VM."
        )
    return "local"


def _is_cloud_agent_id(agent_id: str) -> bool:
    """Cloud / background-composer agent IDs are prefixed ``bc-``."""
    return (agent_id or "").strip().lower().startswith("bc-")

VOICE_SYSTEM = (
    "You are Joe on a live phone call. "
    "Reply with short spoken dialogue only — one thought at a time. "
    "Always write normal English with a space between every word. "
    "Never concatenate words (wrong: 'Howareyou'; right: 'How are you'). "
    "Keep replies phone-call length (not essays). "
    "Do not use markdown, lists, code fences, bullet points, or DELIVERY headers. "
    "Do not use tools, edit files, run commands, or mention being an agent.\n"
    + cartesia_paralinguistics_prompt_block()
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
        f"Respond with spoken dialogue only (Cartesia markup allowed as above)."
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


def _trim_transcript(conversation: str, *, max_chars: int = 3000, max_lines: int = 28) -> str:
    """Keep recent dialogue for speed; full pinned scenario is injected separately."""
    body = (conversation or "").strip()
    if not body:
        return ""
    lines = body.splitlines()
    if len(lines) > max_lines:
        body = "…\n" + "\n".join(lines[-max_lines:])
    if len(body) > max_chars:
        body = "…\n" + body[-max_chars:]
    return body


def wrap_history_prompt(
    conversation: str, *, pinned_context: str = ""
) -> str:
    """Ask for a spoken reply grounded in saved context + call transcript."""
    # Trim dialogue only — pinned/script stays full via build_pinned_block.
    pinned = (pinned_context or "").strip()
    body = _trim_transcript(conversation)
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
        "Do not summarize the whole call unless asked. Do not narrate; just speak. "
        "Vary delivery with the scenario using Cartesia markup when it helps sound human.\n\n"
        f"Transcript so far:\n{body or '(none yet — open using the scenario script)'}\n\n"
        "Respond with spoken dialogue only (Cartesia markup allowed as above)."
    )


class CursorVoiceLLM:
    """Cursor agent for voice; prefers fresh agents when a scenario script is active.

    Always uses the Cursor SDK **local** runtime (on-machine bridge + cwd), never
    a cloud / ``bc-`` background agent. The hub already injects the full call
    transcript every turn, so durable multi-turn agent memory mostly duplicates
    history and can bury a newly saved scenario.
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
        self.runtime: Literal["local"] = resolve_cursor_runtime()
        self._client: AsyncClient | None = None
        self._agent = None
        self._cm_client = None
        self._cm_agent = None
        self._last_pinned_hash: str = ""
        self._last_agent_id: str = ""

    def _local_options(self) -> LocalAgentOptions:
        """Explicit local runtime options — never pass ``cloud=``."""
        return LocalAgentOptions(cwd=self.cwd)

    async def _assert_local_agent(self, agent) -> None:
        """Fail closed if the SDK somehow handed us a cloud agent."""
        agent_id = str(getattr(agent, "agent_id", "") or "")
        self._last_agent_id = agent_id
        if _is_cloud_agent_id(agent_id):
            raise RuntimeError(
                f"Cursor created a cloud agent (id={agent_id!r}) but Joe requires "
                "local runtime. Ensure CURSOR_RUNTIME=local, Cursor is installed, "
                "and the local bridge can start — refusing cloud fallback."
            )
        if self._client is None:
            return
        try:
            info = await self._client.agents.get(
                agent_id, cwd=self.cwd, api_key=self.api_key
            )
        except Exception as exc:  # noqa: BLE001
            # Local store lookup can fail on some bridge builds; ID prefix is enough.
            print(
                f"[llm] could not verify agent runtime via get_agent: {exc}",
                flush=True,
            )
            return
        runtime = getattr(info, "runtime", None)
        if runtime and runtime != "local":
            raise RuntimeError(
                f"Cursor agent runtime is {runtime!r} (id={agent_id!r}); "
                "Joe requires local. Refusing to continue with cloud."
            )

    async def _create_local_agent(self):
        if self._client is None:
            raise RuntimeError("CursorVoiceLLM.start() was not called")
        try:
            cm = await self._client.agents.create(
                model=self.model,
                api_key=self.api_key,
                local=self._local_options(),
                # Never pass cloud= — omit so the SDK cannot select a VM.
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to create a local Cursor agent. "
                "Keep Cursor installed/running so the SDK local bridge can start, "
                "set CURSOR_RUNTIME=local, and do not use cloud agents. "
                f"Underlying error: {exc}"
            ) from exc
        agent = await cm.__aenter__()
        try:
            await self._assert_local_agent(agent)
        except Exception:
            await cm.__aexit__(None, None, None)
            raise
        return cm, agent

    async def start(self) -> None:
        # Re-resolve in case env changed between construct and start.
        self.runtime = resolve_cursor_runtime()
        print(f"Cursor runtime: {self.runtime}", flush=True)
        try:
            self._cm_client = await AsyncClient.launch_bridge(workspace=self.cwd)
            self._client = await self._cm_client.__aenter__()
        except Exception as exc:  # noqa: BLE001
            await self.close()
            raise RuntimeError(
                "Failed to launch the Cursor local bridge. "
                "Joe uses Cursor on this machine (not a cloud VM). "
                "Install/open Cursor, ensure the cursor-sdk bridge can start, "
                f"and retry. Underlying error: {exc}"
            ) from exc
        try:
            self._cm_agent, self._agent = await self._create_local_agent()
        except Exception:
            await self.close()
            raise
        print(
            f"[llm] Cursor agent ready runtime={self.runtime} "
            f"id={self._last_agent_id or '?'}",
            flush=True,
        )

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
        self._cm_agent, self._agent = await self._create_local_agent()
        print(
            f"[llm] context purged — new Cursor agent "
            f"(runtime={self.runtime}, id={self._last_agent_id or '?'})",
            flush=True,
        )

    def invalidate_pinned_cache(self) -> None:
        """Force the next history reply to recreate the agent (after Save context)."""
        self._last_pinned_hash = "__invalidate__"

    async def _ensure_fresh_for_pinned(self, pinned: str) -> None:
        """Recreate the agent so SDK multi-turn memory cannot bury the scenario.

        Hub already injects the call transcript every turn, so durable agent chat
        is redundant. Without a reset, earlier bland 'assistant' turns keep
        winning over a newly saved (or repeatedly injected) script.
        Fidelity > speed: always fresh-per-reply + MUST-FOLLOW pinned block.
        """
        digest = (
            hashlib.sha256(pinned.encode("utf-8")).hexdigest()[:12] if pinned else ""
        )
        # Fresh agent on every history reply. Reuse would risk bland drift even
        # when the pinned hash is unchanged — keep reset + re-inject every turn.
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
