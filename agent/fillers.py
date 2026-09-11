"""Short cached voice fillers to cover LLM/TTS latency after turn-end."""

from __future__ import annotations

import asyncio
import random
import re
from collections import deque
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

# Vibe → short phone-call acks / hesitations. Keep clips brief for latency.
FILLER_BANK: dict[str, tuple[str, ...]] = {
    "agree": (
        "Yeah.",
        "Mm-hmm.",
        "Right.",
        "Okay.",
        "Got it.",
        "Sure.",
        "Uh-huh.",
        "Yeah, okay.",
        "Yep.",
        "Gotcha.",
        "Makes sense.",
        "Fair enough.",
        "I hear you.",
        "For sure.",
        "Alright.",
        "True.",
        "Totally.",
        "Yeah, yeah.",
        "Okay, cool.",
        "Mhm.",
    ),
    "think": (
        "Hmm.",
        "Uh...",
        "Well...",
        "Okay so...",
        "Hmm, okay.",
        "Right, so...",
        "One sec.",
        "Let me see.",
        "Interesting.",
        "Okay...",
        "Hmm, right.",
        "Yeah, hang on.",
        "So...",
        "Alright, um...",
    ),
    "surprise": (
        "Oh.",
        "Oh wow.",
        "Huh.",
        "Really?",
        "Oh okay.",
        "Whoa.",
        "Oh, huh.",
        "No way.",
        "Oh, right.",
        "Wait, huh.",
    ),
    "pushback": (
        "Hmm, maybe.",
        "I don't know...",
        "Well, hang on.",
        "Hmm.",
        "Okay but...",
        "Not sure.",
        "Eh...",
        "Well...",
        "Kinda.",
        "Maybe.",
    ),
    "neutral": (
        "Yeah.",
        "Okay.",
        "Right.",
        "Mm.",
        "Uh-huh.",
        "Alright.",
        "Gotcha.",
        "Mhm.",
        "Sure.",
        "Yep.",
        "Okay.",
        "Mm-hmm.",
    ),
}

# Flat unique list for warm / ensure_all (order stable).
FILLER_PHRASES: tuple[str, ...] = tuple(
    dict.fromkeys(
        phrase
        for phrases in FILLER_BANK.values()
        for phrase in phrases
    )
)

_VIBES: tuple[str, ...] = ("agree", "think", "surprise", "pushback", "neutral")

_AGREE_KW = (
    "yes",
    "yeah",
    "yep",
    "exactly",
    "agree",
    "true",
    "perfect",
    "great",
    "thanks",
    "thank you",
    "cool",
    "alright",
    "sounds good",
    "of course",
    "absolutely",
    "definitely",
    "correct",
    "right on",
    "love that",
)
_THINK_KW = (
    "how",
    "why",
    "what if",
    "explain",
    "wonder",
    "maybe we",
    "should we",
    "complicated",
    "figure out",
    "think about",
    "not sure",
    "unsure",
    "question",
    "could you",
    "can you",
    "would you",
    "help me",
    "walk me",
    "clarify",
)
_SURPRISE_KW = (
    "wow",
    "crazy",
    "can't believe",
    "cannot believe",
    "guess what",
    "suddenly",
    "oh my",
    "no way",
    "seriously",
    "unexpected",
    "shocking",
    "insane",
    "unbelievable",
    "did you hear",
    "you'll never",
)
_PUSHBACK_KW = (
    "no",
    "nah",
    "nope",
    "wrong",
    "don't",
    "doesn't",
    "never",
    "can't",
    "cannot",
    "disagree",
    "actually",
    "however",
    "but ",
    "not really",
    "i doubt",
    "that's not",
    "problem",
    "issue",
    "wait no",
    "hold on",
)
_PIN_AGREE = ("friendly", "casual", "supportive", "agree", "rapport", "warm")
_PIN_THINK = ("interview", "technical", "plan", "strategy", "analysis", "coach")
_PIN_PUSH = ("negotiat", "debate", "sales", "object", "pushback", "skeptic", "conflict")
_PIN_SURPRISE = ("gossip", "news", "story", "drama", "surprise")

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fillers"
_RECENT_MAX = 8
_WORD_RE = re.compile(r"[a-z0-9']+")


def _slug(phrase: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return s or "filler"


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


def _norm_text(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _kw_hits(text: str, keywords: tuple[str, ...], *, substr: bool = False) -> float:
    if not text:
        return 0.0
    score = 0.0
    for kw in keywords:
        if " " in kw or substr:
            if kw in text:
                score += 1.6 if " " in kw else 1.0
        else:
            # Word-boundary-ish: prefer whole tokens.
            if f" {kw} " in f" {text} ":
                score += 1.0
    return score


def classify_vibe(*, user_text: str = "", pinned: str = "") -> str:
    """Fast keyword heuristic — no LLM. Returns a FILLER_BANK key."""
    user = _norm_text(user_text)
    pin = _norm_text(pinned)
    scores = {v: 0.0 for v in _VIBES}
    scores["neutral"] = 0.35  # mild default bias

    scores["agree"] += _kw_hits(user, _AGREE_KW)
    scores["think"] += _kw_hits(user, _THINK_KW)
    scores["surprise"] += _kw_hits(user, _SURPRISE_KW)
    scores["pushback"] += _kw_hits(user, _PUSHBACK_KW)

    # Questions / longer turns lean "thinking".
    raw = (user_text or "").strip()
    if "?" in raw:
        scores["think"] += 1.2
    if len(user.split()) >= 18:
        scores["think"] += 0.8

    # Soft signal from pinned scenario (stems; weaker than last utterance).
    scores["agree"] += 0.7 * _kw_hits(pin, _PIN_AGREE, substr=True)
    scores["think"] += 0.7 * _kw_hits(pin, _PIN_THINK, substr=True)
    scores["pushback"] += 0.7 * _kw_hits(pin, _PIN_PUSH, substr=True)
    scores["surprise"] += 0.7 * _kw_hits(pin, _PIN_SURPRISE, substr=True)

    best = max(scores.items(), key=lambda kv: kv[1])
    # If nothing distinctive, stay neutral.
    if best[0] != "neutral" and best[1] < scores["neutral"] + 0.45:
        return "neutral"
    return best[0]


def last_caller_utterance(conversation: str) -> str:
    """Extract the last Caller/room line from hub.conversation_text()."""
    last = ""
    for line in (conversation or "").splitlines():
        s = line.strip()
        if s.startswith("Caller/room"):
            _, _, rest = s.partition(":")
            last = rest.strip()
    return last


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
        self._recent: deque[str] = deque(maxlen=_RECENT_MAX)

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

    def _mem_key(self, phrase: str, *, backend: str) -> str:
        voice_tag = self.cartesia_voice_id if backend == "cartesia" else ""
        style = style_cache_tag(speed=self.speed, tonality=self.tonality)
        return f"{backend}:{voice_tag}:{style}:{phrase}"

    def _disk_path(self, phrase: str, *, backend: str) -> Path:
        voice_tag = self.cartesia_voice_id if backend == "cartesia" else ""
        style = style_cache_tag(speed=self.speed, tonality=self.tonality)
        return _pcm_path(
            phrase, backend=backend, voice_id=voice_tag, style_tag=style
        )

    def _is_ready(self, phrase: str, *, backend: str) -> bool:
        key = self._mem_key(phrase, backend=backend)
        if key in self._mem:
            return True
        path = self._disk_path(phrase, backend=backend)
        return path.is_file() and path.stat().st_size > 0

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
        # Warm vibe cores first so context picks hit cache sooner.
        ordered: list[str] = []
        for vibe in ("neutral", "agree", "think", "surprise", "pushback"):
            for phrase in FILLER_BANK[vibe]:
                if phrase not in ordered:
                    ordered.append(phrase)
        for phrase in FILLER_PHRASES:
            if phrase not in ordered:
                ordered.append(phrase)
        for phrase in ordered:
            await self.ensure_one(phrase, backend=backend)

    async def ensure_one(self, phrase: str, *, backend: str) -> bytes:
        key = self._mem_key(phrase, backend=backend)
        cached = self._mem.get(key)
        if cached:
            return cached
        path = self._disk_path(phrase, backend=backend)
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

    def pick_phrase(
        self,
        *,
        user_text: str = "",
        pinned: str = "",
        backend: str = "",
    ) -> str:
        """Context-aware pick with recency penalty; prefer already-cached clips."""
        vibe = classify_vibe(user_text=user_text, pinned=pinned)
        primary = FILLER_BANK.get(vibe, FILLER_BANK["neutral"])
        # Light mix from neutral so vibe banks don't get sticky.
        pool = list(dict.fromkeys([*primary, *FILLER_BANK["neutral"]]))
        recent = set(self._recent)
        scored: list[tuple[float, str]] = []
        for phrase in pool:
            score = 1.0
            if phrase in primary:
                score += 1.4
            if phrase in recent:
                # Strong penalty for last few plays; still allow if pool tiny.
                age = 0
                for i, p in enumerate(reversed(self._recent)):
                    if p == phrase:
                        age = i
                        break
                score -= 3.5 - min(age, _RECENT_MAX) * 0.35
            if backend and self._is_ready(phrase, backend=backend):
                score += 2.5
            elif backend:
                score -= 0.8  # avoid blocking on cold TTS when possible
            # Small jitter so ties don't always pick the same clip.
            score += random.random() * 0.35
            scored.append((score, phrase))
        scored.sort(key=lambda t: t[0], reverse=True)
        # Sample from top few for variety.
        top = scored[: min(5, len(scored))]
        weights = [max(0.05, s) for s, _ in top]
        phrase = random.choices([p for _, p in top], weights=weights, k=1)[0]
        self._recent.append(phrase)
        return phrase

    async def enqueue(
        self,
        speaker: SpeakerPlayback,
        *,
        backend: str,
        phrase: Optional[str] = None,
        user_text: str = "",
        pinned: str = "",
    ) -> float:
        """Enqueue one filler; returns approximate duration seconds (0 if skipped)."""
        phrase = phrase or self.pick_phrase(
            user_text=user_text, pinned=pinned, backend=backend
        )
        # If the pick isn't ready, fall back to any ready clip to avoid TTS wait.
        if not self._is_ready(phrase, backend=backend):
            ready = [
                p
                for p in FILLER_PHRASES
                if p != phrase and self._is_ready(p, backend=backend)
            ]
            if ready:
                # Still warm the intended phrase in the background.
                asyncio.create_task(
                    self.ensure_one(phrase, backend=backend),
                    name="filler-lazy-warm",
                )
                fallback = ready[0]
                for p in ready:
                    if p not in self._recent:
                        fallback = p
                        break
                print(
                    f"[filler] cold {phrase!r} — using cached {fallback!r}",
                    flush=True,
                )
                phrase = fallback
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
