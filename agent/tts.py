"""Cartesia Sonic TTS WebSocket continuations."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Optional

from cartesia import AsyncCartesia

from agent.expression import should_flush_phrase, smart_append, speech_for_tts
from agent.voice_style import (
    DEFAULT_SPEED,
    DEFAULT_TONALITY,
    cartesia_generation_config,
)

OnAudioChunk = Callable[[bytes], Awaitable[None]]

# Private clones on this Cartesia account (verified via GET /voices + TTS smoke).
JOE_VOICE_ID = "55e8e671-7e46-4a15-9b53-4955c92aec0c"
JACK_VOICE_ID = "7e94372f-47be-42b7-ba88-1c1f463da123"
# Stale public/docs ID that was briefly used as "Jack" — 404 on this account.
LEGACY_BAD_JACK_ID = "9a703183-88ce-460f-a019-ee260504ef7e"

DEFAULT_VOICE_ID = JACK_VOICE_ID
DEFAULT_MODEL_ID = "sonic-3.6"

# Offline fallback for the UI when the Cartesia voices list cannot be fetched.
CARTESIA_VOICES: tuple[dict[str, str], ...] = (
    {"id": JOE_VOICE_ID, "label": "Joe Marazzo"},
    {"id": JACK_VOICE_ID, "label": "Jack (default)"},
    {"id": "4e65cba0-b17c-4c6c-a985-df33894014d7", "label": "Hong"},
    {"id": "47c38ca4-5f35-497b-b1a3-415245fb35e1", "label": "Daniel — Modern Assistant"},
    {"id": "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4", "label": "Skylar — Friendly Guide"},
    {"id": "30894953-bcce-41fe-892c-15ce19c843ff", "label": "Parker — Supportive Pal"},
    {"id": "ef191366-f52f-447a-a398-ed8c0f2943a1", "label": "Archie — Approachable Mate"},
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_VOICES_CACHE_PATH = _DATA_DIR / "cartesia_voices.json"
_VOICES_CACHE_TTL_S = 6 * 60 * 60  # refresh file cache at most every 6h
_MEM_CACHE: list[dict[str, str]] | None = None
_MEM_CACHE_AT = 0.0
_MEM_CACHE_TTL_S = 10 * 60  # in-process reuse for /api/tts
_FETCH_LOCK = asyncio.Lock()


def resolve_cartesia_voice_id(voice_id: str, *, fallback: str = DEFAULT_VOICE_ID) -> str:
    """Map known-bad IDs to working ones; otherwise return stripped id or fallback."""
    vid = (voice_id or "").strip()
    if not vid:
        return fallback or DEFAULT_VOICE_ID
    if vid == LEGACY_BAD_JACK_ID:
        return JACK_VOICE_ID
    return vid


def _pin_priority_voices(voices: list[dict[str, str]]) -> list[dict[str, str]]:
    """Put Joe and Jack first when present; keep remaining names A–Z."""
    by_id = {v["id"]: v for v in voices if v.get("id")}
    pinned_ids = (JOE_VOICE_ID, JACK_VOICE_ID)
    pinned: list[dict[str, str]] = []
    for pid in pinned_ids:
        if pid in by_id:
            label = by_id[pid]["label"]
            if pid == JACK_VOICE_ID and "(default)" not in label.lower():
                label = f"{label} (default)" if label else "Jack (default)"
            pinned.append({"id": pid, "label": label})
    rest = sorted(
        (v for v in voices if v.get("id") not in pinned_ids),
        key=lambda v: (v.get("label") or "").lower(),
    )
    return pinned + rest


def _read_voices_file_cache() -> list[dict[str, str]] | None:
    try:
        if not _VOICES_CACHE_PATH.is_file():
            return None
        raw = json.loads(_VOICES_CACHE_PATH.read_text(encoding="utf-8"))
        fetched_at = float(raw.get("fetched_at") or 0)
        if time.time() - fetched_at > _VOICES_CACHE_TTL_S:
            return None
        voices = raw.get("voices")
        if not isinstance(voices, list) or not voices:
            return None
        out: list[dict[str, str]] = []
        for item in voices:
            if not isinstance(item, dict):
                continue
            vid = str(item.get("id") or "").strip()
            label = str(item.get("label") or item.get("name") or "").strip()
            if vid and label:
                out.append({"id": vid, "label": label})
        return out or None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_voices_file_cache(voices: list[dict[str, str]]) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": time.time(),
            "count": len(voices),
            "voices": voices,
        }
        _VOICES_CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[tts] could not cache cartesia voices: {exc}", flush=True)


async def _fetch_voices_from_api(api_key: str) -> list[dict[str, str]]:
    """Page through Cartesia GET /voices (limit≤100); pin Joe/Jack only if present."""
    voices: list[dict[str, str]] = []
    seen: set[str] = set()
    async with AsyncCartesia(api_key=api_key) as client:
        # AsyncPaginator yields every voice across pages.
        async for v in client.voices.list(limit=100):
            vid = getattr(v, "id", None)
            name = getattr(v, "name", None) or "Unnamed"
            if not vid:
                continue
            sid = str(vid)
            if sid in seen:
                continue
            seen.add(sid)
            voices.append({"id": sid, "label": str(name)})
    return _pin_priority_voices(voices)


async def list_cartesia_voices(
    api_key: str,
    *,
    force_refresh: bool = False,
) -> list[dict[str, str]]:
    """Cartesia account voices for the UI.

    Primary source is the live API. Short memory + file caches avoid hammering
    the API; pass force_refresh=True (Refresh voices) to bypass both.
    CARTESIA_VOICES is offline emergency fallback only — never the primary list.
    """
    global _MEM_CACHE, _MEM_CACHE_AT

    if (
        not force_refresh
        and _MEM_CACHE is not None
        and (time.time() - _MEM_CACHE_AT) < _MEM_CACHE_TTL_S
    ):
        return [dict(v) for v in _MEM_CACHE]

    async with _FETCH_LOCK:
        if (
            not force_refresh
            and _MEM_CACHE is not None
            and (time.time() - _MEM_CACHE_AT) < _MEM_CACHE_TTL_S
        ):
            return [dict(v) for v in _MEM_CACHE]

        if not force_refresh:
            cached = _read_voices_file_cache()
            if cached:
                _MEM_CACHE = _pin_priority_voices(cached)
                _MEM_CACHE_AT = time.time()
                return [dict(v) for v in _MEM_CACHE]

        if not (api_key or "").strip():
            print("[tts] no API key — using offline voice fallback", flush=True)
            return [dict(v) for v in CARTESIA_VOICES]

        try:
            voices = await _fetch_voices_from_api(api_key.strip())
        except Exception as exc:  # noqa: BLE001 — UI must still load
            print(f"[tts] voices list failed: {exc}", flush=True)
            cached = _read_voices_file_cache()
            if cached:
                _MEM_CACHE = _pin_priority_voices(cached)
                _MEM_CACHE_AT = time.time()
                return [dict(v) for v in _MEM_CACHE]
            return [dict(v) for v in CARTESIA_VOICES]

        if not voices:
            print("[tts] API returned 0 voices — using offline fallback", flush=True)
            return [dict(v) for v in CARTESIA_VOICES]

        _write_voices_file_cache(voices)
        _MEM_CACHE = voices
        _MEM_CACHE_AT = time.time()
        print(f"[tts] loaded {len(voices)} Cartesia voices", flush=True)
        return [dict(v) for v in voices]


def voice_label_for_id(voice_id: str, voices: list[dict[str, str]] | None = None) -> str:
    vid = resolve_cartesia_voice_id(voice_id)
    pool = voices if voices is not None else list(CARTESIA_VOICES)
    for v in pool:
        if v.get("id") == vid:
            return v.get("label") or vid
    return vid[:8] + "…" if len(vid) > 8 else vid


class SonicTTS:
    """Stream LLM text chunks into Sonic and forward PCM to a playback callback."""

    def __init__(
        self,
        client: AsyncCartesia,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        sample_rate: int = 16_000,
        on_audio: Optional[OnAudioChunk] = None,
        speed: float = DEFAULT_SPEED,
        tonality: str = DEFAULT_TONALITY,
    ) -> None:
        self._client = client
        self.voice_id = resolve_cartesia_voice_id(voice_id)
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.on_audio = on_audio
        self.speed = speed
        self.tonality = tonality

    def set_delivery(self, *, speed: Optional[float] = None, tonality: Optional[str] = None) -> None:
        if speed is not None:
            self.speed = float(speed)
        if tonality is not None:
            self.tonality = tonality

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Push streamed text into one TTS context and play audio as it arrives."""
        output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }
        gen_cfg = cartesia_generation_config(speed=self.speed, tonality=self.tonality)

        def _stop() -> bool:
            return bool(should_stop and should_stop())

        async with self._client.tts.websocket_connect() as ws:
            ctx = ws.context(
                model_id=self.model_id,
                voice=self.voice_id,
                output_format=output_format,
                language="en",
                generation_config=gen_cfg,
            )

            async def push_text() -> None:
                # Phrase-buffer so we can self-check spacing before Sonic speaks.
                joined = ""
                first = True
                async for chunk in text_chunks:
                    if _stop():
                        return
                    if not chunk:
                        continue
                    joined = smart_append(joined, chunk)
                    # Flush on raw buffer so incomplete SSML is never split mid-tag.
                    if should_flush_phrase(joined, first=first):
                        phrase = speech_for_tts(joined, backend="cartesia")
                        if phrase:
                            await ctx.push(phrase + " ")
                        joined = ""
                        first = False
                leftover = speech_for_tts(joined, backend="cartesia")
                if leftover and not _stop():
                    await ctx.push(leftover)
                if _stop():
                    return
                await ctx.no_more_inputs()

            async def receive_audio() -> None:
                async for response in ctx.receive():
                    if _stop():
                        return
                    rtype = getattr(response, "type", None)
                    if rtype == "chunk":
                        audio = getattr(response, "audio", None)
                        if audio and self.on_audio is not None and not _stop():
                            await self.on_audio(audio)
                    elif rtype == "error":
                        message = (
                            getattr(response, "message", None)
                            or getattr(response, "title", None)
                            or str(response)
                        )
                        print(f"[tts] error: {message}", flush=True)

            try:
                await asyncio.gather(push_text(), receive_audio())
            except asyncio.CancelledError:
                # Stop button — exit the websocket context immediately.
                raise

    async def speak(self, text: str) -> None:
        async def _one() -> AsyncIterator[str]:
            yield text

        await self.speak_stream(_one())
