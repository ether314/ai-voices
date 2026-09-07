"""Cursor SDK agent wrapper for voice-only replies."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Optional

from cursor_sdk import AsyncClient, LocalAgentOptions

DEFAULT_MODEL = "gpt-5.6-luna"

VOICE_SYSTEM = (
    "You are a warm, concise voice assistant in a live spoken conversation. "
    "Reply with short spoken dialogue only — one thought at a time. "
    "Do not use markdown, lists, code fences, bullet points, or stage directions. "
    "Do not use tools, edit files, run commands, or mention being an agent. "
    "Keep replies friendly and natural for text-to-speech."
)


def wrap_voice_prompt(user_transcript: str) -> str:
    """Wrap a user turn so the coding agent stays in voice-only mode."""
    return (
        f"{VOICE_SYSTEM}\n\n"
        f"User just said (transcribed verbatim):\n"
        f"{user_transcript}\n\n"
        f"Respond with spoken dialogue only."
    )


def wrap_history_prompt(conversation: str) -> str:
    """Ask for a spoken reply grounded in the full call transcript."""
    # Keep prompts short so Cursor stays responsive.
    max_chars = 4500
    if len(conversation) > max_chars:
        conversation = "…\n" + conversation[-max_chars:]
    return (
        f"{VOICE_SYSTEM}\n\n"
        "You are Joe, speaking live on a phone call with Cartesia voice. "
        "You will keep talking in a continuous loop until the user stops you, "
        "so say the next natural stretch out loud — about 2 to 4 short sentences. "
        "Stay in the flow of the call using the transcript. Do not summarize the "
        "whole call. Do not narrate. Do not say you will wait — just keep talking.\n\n"
        f"Transcript so far:\n{conversation}\n\n"
        "Respond with spoken dialogue only."
    )


class CursorVoiceLLM:
    """Long-lived Cursor agent; streams assistant text via run.iter_text()."""

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

    async def stream_history_reply(self, conversation: str) -> AsyncIterator[str]:
        if self._agent is None:
            raise RuntimeError("CursorVoiceLLM.start() was not called")
        prompt = wrap_history_prompt(conversation)
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
