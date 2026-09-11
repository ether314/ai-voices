"""Short cached voice fillers to cover LLM/TTS latency after turn-end."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp

from agent.audio import SAMPLE_RATE, SpeakerPlayback
from agent.local_tts import DEFAULT_CHATTERBOX_URL, wav_bytes_to_pcm16le
from agent.voice_style import (
    DEFAULT_SPEED,
    DEFAULT_TONALITY,
    cartesia_generation_config,
    local_tts_params,
    style_cache_tag,
)

FILLER_PHRASES: tuple[str, ...] = (
    "Yeah.",
    "Mm-hmm.",
    "Right.",
    "Okay.",
    "Got it.",
    "Sure.",
    "Uh-huh.",
    "Yeah, okay.",
)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fillers"


def _slug(phrase: str) -> str:
    return (
        phrase.lower()
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
    )


def _pcm_path(
    phrase: str,
    *,
    backend: str,
    voice_id: str = "",
    style_tag: str = "",
) -> Path:
    parts = [backend]
    if backend == "cartesia" and voice_id:
        parts.append(voice_id[:8])
    if style_tag:
        parts.append(style_tag)
    parts.append(_slug(phrase))
    parts.append(str(SAMPLE_RATE))
    return _CACHE_DIR / ("_".join(parts) + ".pcm")


def _duration_s(pcm: bytes) -> float:
    return (len(pcm) / 2) / float(SAMPLE_RATE)


class FillerBank:
    """Disk-cached ack clips played while the real reply is still generating."""

    def __init__(
        self,
        *,
        chatterbox_url: str = DEFAULT_CHATTERBOX_URL,
        cartesia_api_key: str = "",
        cartesia_voice_id: str = "",
        speed: float = DEFAULT_SPEED,
        tonality: str = DEFAULT_TONALITY,
    ) -> None:
        self.chatterbox_url = chatterbox_url.rstrip("/")
        self.cartesia_api_key = cartesia_api_key
        self.cartesia_voice_id = cartesia_voice_id
        self.speed = speed
        self.tonality = tonality
        self._mem: dict[str, bytes] = {}
        self._lock = asyncio.Lock()
        self._warm_task: asyncio.Task[None] | None = None
        self._rr = 0

    def set_chatterbox_url(self, url: str) -> None:
        self.chatterbox_url = (url or DEFAULT_CHATTERBOX_URL).rstrip("/")

    def set_cartesia_voice_id(self, voice_id: str) -> None:
        vid = (voice_id or "").strip()
        if vid == self.cartesia_voice_id:
            return
        self.cartesia_voice_id = vid
        # Drop in-memory Cartesia clips so the next warm uses the new voice.
        for key in [k for k in self._mem if k.startswith("cartesia:")]:
            del self._mem[key]

    def set_delivery(self, *, speed: Optional[float] = None, tonality: Optional[str] = None) -> None:
        changed = False
        if speed is not None and float(speed) != self.speed:
            self.speed = float(speed)
            changed = True
        if tonality is not None and tonality != self.tonality:
            self.tonality = tonality
            changed = True
        if changed:
            self._mem.clear()

    def start_warm(self, *, backend: str) -> None:
        style = style_cache_tag(speed=self.speed, tonality=self.tonality)
        warm_key = (
            backend,
            self.cartesia_voice_id if backend == "cartesia" else "",
            style,
        )
        if (
            self._warm_task is not None
            and not self._warm_task.done()
            and getattr(self, "_warm_key", None) == warm_key
        ):
            return
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()

        async def _run() -> None:
            try:
                await self.ensure_all(backend=backend)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[filler] warm failed: {exc}", flush=True)

        self._warm_key = warm_key
        self._warm_task = asyncio.create_task(_run(), name="filler-warm")

    async def ensure_all(self, *, backend: str) -> None:
        for phrase in FILLER_PHRASES:
            await self.ensure_one(phrase, backend=backend)

    async def ensure_one(self, phrase: str, *, backend: str) -> bytes:
        voice_tag = self.cartesia_voice_id if backend == "cartesia" else ""
        style = style_cache_tag(speed=self.speed, tonality=self.tonality)
        key = f"{backend}:{voice_tag}:{style}:{phrase}"
        cached = self._mem.get(key)
        if cached:
            return cached
        path = _pcm_path(
            phrase, backend=backend, voice_id=voice_tag, style_tag=style
        )
        if path.is_file() and path.stat().st_size > 0:
            data = path.read_bytes()
            self._mem[key] = data
            return data
        async with self._lock:
            cached = self._mem.get(key)
            if cached:
                return cached
            if path.is_file() and path.stat().st_size > 0:
                data = path.read_bytes()
                self._mem[key] = data
                return data
            print(f"[filler] generating {phrase!r} via {backend}...", flush=True)
            if backend == "local":
                data = await self._gen_local(phrase)
            else:
                data = await self._gen_cartesia(phrase)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self._mem[key] = data
            print(
                f"[filler] cached {phrase!r} ({_duration_s(data):.2f}s)",
                flush=True,
            )
            return data

    async def _gen_local(self, phrase: str) -> bytes:
        url = urljoin(self.chatterbox_url + "/", "tts")
        params = local_tts_params(speed=self.speed, tonality=self.tonality)
        form = aiohttp.FormData()
        form.add_field("text", phrase)
        form.add_field("speed", f"{params['speed']:g}")
        form.add_field("temperature", f"{params['temperature']:g}")
        form.add_field("exaggeration", f"{params['exaggeration']:g}")
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=form) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"filler TTS {resp.status}: {detail[:200]}")
                wav_bytes = await resp.read()
        return await asyncio.to_thread(
            wav_bytes_to_pcm16le, wav_bytes, target_rate=SAMPLE_RATE
        )

    async def _gen_cartesia(self, phrase: str) -> bytes:
        """One-shot Cartesia bytes TTS → pcm_s16le @ SAMPLE_RATE."""
        if not self.cartesia_api_key or not self.cartesia_voice_id:
            raise RuntimeError("Cartesia key/voice missing for fillers")
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "Authorization": f"Bearer {self.cartesia_api_key}",
            "Cartesia-Version": "2025-04-16",
            "Content-Type": "application/json",
        }
        body = {
            "model_id": "sonic-3.6",
            "transcript": phrase,
            "voice": {"mode": "id", "id": self.cartesia_voice_id},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 44100,
            },
            "language": "en",
            "generation_config": cartesia_generation_config(
                speed=self.speed, tonality=self.tonality
            ),
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"cartesia filler {resp.status}: {detail[:200]}")
                wav_bytes = await resp.read()
        return await asyncio.to_thread(
            wav_bytes_to_pcm16le, wav_bytes, target_rate=SAMPLE_RATE
        )

    def pick_phrase(self) -> str:
        # Rotate with light randomness so it doesn't feel robotic.
        self._rr = (self._rr + 1) % len(FILLER_PHRASES)
        if random.random() < 0.35:
            return random.choice(FILLER_PHRASES)
        return FILLER_PHRASES[self._rr]

    async def enqueue(
        self,
        speaker: SpeakerPlayback,
        *,
        backend: str,
        phrase: Optional[str] = None,
    ) -> float:
        """Enqueue one filler; returns approximate duration seconds (0 if skipped)."""
        phrase = phrase or self.pick_phrase()
        try:
            pcm = await self.ensure_one(phrase, backend=backend)
        except Exception as exc:  # noqa: BLE001
            print(f"[filler] skip ({phrase!r}): {exc}", flush=True)
            return 0.0
        if not pcm:
            return 0.0
        await speaker.enqueue(pcm)
        dur = _duration_s(pcm)
        print(f"[filler] play {phrase!r} ({dur:.2f}s)", flush=True)
        return dur
