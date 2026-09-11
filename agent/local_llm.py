"""OpenAI-compatible local chat LLM (Ollama, LM Studio, etc.)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

import aiohttp

from agent.llm import (
    _prompt_fingerprint,
    _snippet,
    wrap_history_prompt,
    wrap_voice_prompt,
)

DEFAULT_LOCAL_LLM_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_LLM_MODEL = "llama3.2"


class LocalVoiceLLM:
    """Stateless voice replies via OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LOCAL_LLM_URL,
        model: str = DEFAULT_LOCAL_LLM_MODEL,
        api_key: str = "ollama",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_LOCAL_LLM_URL).rstrip("/")
        self.model = (model or DEFAULT_LOCAL_LLM_MODEL).strip() or DEFAULT_LOCAL_LLM_MODEL
        self.api_key = api_key or "ollama"
        self.timeout_s = timeout_s
        self._session: aiohttp.ClientSession | None = None
        self._last_pinned_hash: str = ""

    def set_base_url(self, url: str) -> None:
        self.base_url = (url or DEFAULT_LOCAL_LLM_URL).rstrip("/")

    def set_model(self, model: str) -> None:
        self.model = (model or DEFAULT_LOCAL_LLM_MODEL).strip() or DEFAULT_LOCAL_LLM_MODEL

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def reset(self) -> None:
        """No durable agent memory — hub injects full transcript each turn."""
        self._last_pinned_hash = ""
        print("[llm/local] context purged (stateless)", flush=True)

    def invalidate_pinned_cache(self) -> None:
        self._last_pinned_hash = "__invalidate__"

    def _chat_url(self) -> str:
        return urljoin(self.base_url + "/", "chat/completions")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def health(self) -> dict[str, Any]:
        """Probe OpenAI-compatible /models (or fall back to a tiny completion)."""
        await self.start()
        assert self._session is not None
        models_url = urljoin(self.base_url + "/", "models")
        try:
            async with self._session.get(
                models_url, headers=self._headers()
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    return {
                        "ok": False,
                        "reachable": True,
                        "error": f"HTTP {resp.status}: {body[:200]}",
                        "url": self.base_url,
                        "model": self.model,
                    }
                data: Any = {}
                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    data = {}
                ids: list[str] = []
                for item in data.get("data") or []:
                    mid = item.get("id") if isinstance(item, dict) else None
                    if mid:
                        ids.append(str(mid))
                known = self.model in ids if ids else True
                return {
                    "ok": True,
                    "reachable": True,
                    "url": self.base_url,
                    "model": self.model,
                    "models": ids[:40],
                    "model_listed": known if ids else None,
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "reachable": False,
                "error": str(exc),
                "url": self.base_url,
                "model": self.model,
            }

    async def _stream_chat(self, prompt: str) -> AsyncIterator[str]:
        await self.start()
        assert self._session is not None
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
        }
        print(
            f"[llm/local] POST {self._chat_url()} model={self.model!r} "
            f"({len(prompt)} chars)…",
            flush=True,
        )
        yielded = False
        try:
            async with self._session.post(
                self._chat_url(),
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    raise RuntimeError(
                        f"Local LLM failed ({resp.status}): {detail[:300]}"
                    )
                buffer = ""
                async for raw in resp.content:
                    if not raw:
                        continue
                    buffer += raw.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content")
                        if not piece:
                            # Some servers send non-streaming-shaped chunks.
                            piece = (choices[0].get("message") or {}).get("content")
                        if piece:
                            yielded = True
                            yield str(piece)
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Local LLM unreachable at {self.base_url}: {exc}") from exc
        if not yielded:
            raise RuntimeError(
                f"Local LLM returned no spoken text (model={self.model!r} "
                f"@ {self.base_url}). Is Ollama/LM Studio running?"
            )

    async def stream_reply(self, user_transcript: str) -> AsyncIterator[str]:
        prompt = wrap_voice_prompt(user_transcript)
        async for chunk in self._stream_chat(prompt):
            yield chunk

    async def stream_history_reply(
        self, conversation: str, *, pinned_context: str = ""
    ) -> AsyncIterator[str]:
        pinned = (pinned_context or "").strip()
        digest = (
            hashlib.sha256(pinned.encode("utf-8")).hexdigest()[:12] if pinned else ""
        )
        self._last_pinned_hash = digest
        prompt = wrap_history_prompt(conversation, pinned_context=pinned)
        fp = _prompt_fingerprint(prompt)
        if pinned:
            print(
                f"[llm/local] pinned ACTIVE chars={len(pinned)} "
                f"hash={digest} snippet={_snippet(pinned)!r}",
                flush=True,
            )
            print(
                f"[llm/local] prompt proof ({len(prompt)} chars, fp={fp}): "
                f"{_snippet(prompt, 500)!r}",
                flush=True,
            )
            if "MUST FOLLOW — SCENARIO SCRIPT" not in prompt or pinned[:40] not in prompt:
                print(
                    "[llm/local] WARNING: pinned text missing from outgoing prompt!",
                    flush=True,
                )
        else:
            print(
                f"[llm/local] pinned EMPTY — no scenario script on this reply "
                f"(prompt {len(prompt)} chars, fp={fp})",
                flush=True,
            )
        async for chunk in self._stream_chat(prompt):
            yield chunk
