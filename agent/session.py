"""Turn orchestration: mic → STT → button-started continuous Joe replies → TTS."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional, Union

from cartesia import AsyncCartesia

from agent.audio import MicCapture, SpeakerPlayback, list_audio_devices
from agent.expression import (
    join_speech_chunks,
    strip_tags,
)
from agent.fillers import FillerBank
from agent.llm import CursorVoiceLLM
from agent.local_tts import ChatterboxTTS, DEFAULT_CHATTERBOX_URL
from agent.pinned_context import PinnedContextStore
from agent.stt import InkSTT
from agent.transcript_hub import TranscriptEvent, hub
from agent.tts import CARTESIA_VOICES, DEFAULT_VOICE_ID, SonicTTS
from agent.voice_style import (
    DEFAULT_SPEED,
    DEFAULT_TONALITY,
    choices_payload,
    clamp_speed,
    load_speed,
    load_tonality,
    normalize_tonality,
    save_speed,
    save_tonality,
)

# Room mic re-hears Joe from speakers; keep STT deaf until playback + reverb settle.
ECHO_COOLDOWN_S = 0.85
# Ink often fires turn.end on a short breath — brief silence check before Joe speaks.
TURN_CONFIRM_S = 1.0
AUTO_REPLY_COOLDOWN_S = 1.5
FILLER_AFTER_LLM_WAIT_S = 0.9
TTS_BACKENDS = ("cartesia", "local")
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TTS_BACKEND_PATH = _DATA_DIR / "tts_backend.txt"
_CHATTERBOX_URL_PATH = _DATA_DIR / "chatterbox_url.txt"
_CARTESIA_VOICE_PATH = _DATA_DIR / "cartesia_voice_id.txt"


def _load_tts_backend() -> str:
    env = os.getenv("TTS_BACKEND", "").strip().lower()
    if env in TTS_BACKENDS:
        return env
    try:
        if _TTS_BACKEND_PATH.is_file():
            val = _TTS_BACKEND_PATH.read_text(encoding="utf-8").strip().lower()
            if val in TTS_BACKENDS:
                return val
    except OSError:
        pass
    return "cartesia"


def _load_chatterbox_url() -> str:
    env = os.getenv("CHATTERBOX_URL", "").strip()
    if env:
        return env.rstrip("/")
    try:
        if _CHATTERBOX_URL_PATH.is_file():
            val = _CHATTERBOX_URL_PATH.read_text(encoding="utf-8").strip()
            if val:
                return val.rstrip("/")
    except OSError:
        pass
    return DEFAULT_CHATTERBOX_URL


def _load_cartesia_voice_id(fallback: str = DEFAULT_VOICE_ID) -> str:
    # Persisted UI selection wins so the voice dropdown survives restarts.
    try:
        if _CARTESIA_VOICE_PATH.is_file():
            val = _CARTESIA_VOICE_PATH.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    env = os.getenv("CARTESIA_VOICE_ID", "").strip()
    if env:
        return env
    return fallback or DEFAULT_VOICE_ID


def _save_tts_backend(backend: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _TTS_BACKEND_PATH.write_text(backend + "\n", encoding="utf-8")


def _save_chatterbox_url(url: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CHATTERBOX_URL_PATH.write_text(url.rstrip("/") + "\n", encoding="utf-8")


def _save_cartesia_voice_id(voice_id: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CARTESIA_VOICE_PATH.write_text(voice_id.strip() + "\n", encoding="utf-8")


class VoiceSession:
    def __init__(
        self,
        *,
        cartesia_api_key: str,
        cursor_api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        cursor_model: str = "composer-2.5",
        cwd: Optional[str] = None,
        echo_cooldown_s: float = ECHO_COOLDOWN_S,
    ) -> None:
        self.cartesia_api_key = cartesia_api_key
        self.cursor_api_key = cursor_api_key
        # Env > persisted file > constructor/default (same pattern as TTS backend).
        self.voice_id = _load_cartesia_voice_id(voice_id or DEFAULT_VOICE_ID)
        self.cursor_model = cursor_model
        self.cwd = cwd or os.getcwd()
        self.echo_cooldown_s = echo_cooldown_s

        self._mic = MicCapture()
        self._speaker = SpeakerPlayback()
        self._busy = False
        self._talking = False
        self._powered = True
        self._auto_reply = True
        self._reply_after_busy = False
        self._turn_lock = asyncio.Lock()
        self._pending_turns: asyncio.Queue[str] = asyncio.Queue()
        self._stop = asyncio.Event()
        # Feed/recv death or Turn on after idle-timeout — reopen Ink websocket.
        self._stt_restart = asyncio.Event()
        self._active_speak_task: asyncio.Task[None] | None = None
        self._llm: CursorVoiceLLM | None = None
        self._pinned = PinnedContextStore()
        self._tts_backend = _load_tts_backend()
        self._chatterbox_url = _load_chatterbox_url()
        self._tts_speed = load_speed(DEFAULT_SPEED)
        self._tts_tonality = load_tonality(DEFAULT_TONALITY)
        self._sonic: SonicTTS | None = None
        self._local_tts: ChatterboxTTS | None = None
        self._tts: Union[SonicTTS, ChatterboxTTS] | None = None
        self._ignore_barge_until = 0.0
        # After Joe's speakers go quiet, ignore STT (echo) until this time.
        self._echo_gate_until = 0.0
        self._last_auto_start = 0.0
        self._last_partial_at = 0.0
        self._pending_auto_task: asyncio.Task[None] | None = None
        self._turn_confirm_gen = 0
        self._fillers = FillerBank(
            chatterbox_url=self._chatterbox_url,
            cartesia_api_key=cartesia_api_key,
            cartesia_voice_id=self.voice_id,
            speed=self._tts_speed,
            tonality=self._tts_tonality,
        )

    @property
    def talking(self) -> bool:
        return self._talking

    @property
    def powered(self) -> bool:
        return self._powered

    def _listening_blocked(self) -> bool:
        return self._busy or self._speaker.playing or self._turn_lock.locked()

    async def _queue_history_reply(self) -> bool:
        conversation = hub.conversation_text().strip()
        pinned = self._pinned.get().strip()
        if not conversation and not pinned:
            return False
        # Drop stale queued prompts — always use freshest history when we run.
        while not self._pending_turns.empty():
            try:
                self._pending_turns.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._pending_turns.put(conversation)
        return True

    async def turn_off(self) -> dict[str, str | bool]:
        """Soft power off: stop talking, mute mic, keep process alive for Turn on."""
        self._cancel_pending_auto()
        if self._talking or self._busy:
            await self.stop_talking(pause_auto=True)
        self._powered = False
        self._auto_reply = False
        self._mic.set_muted(True)
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="APP_OFF — off",
            )
        )
        print("[session] app turned off (paused)", flush=True)
        return {"ok": True, "powered": False}

    def _request_stt_restart(self, reason: str) -> None:
        if self._stop.is_set():
            return
        if not self._stt_restart.is_set():
            print(f"[stt] restart requested ({reason})", flush=True)
        self._stt_restart.set()

    async def turn_on(self) -> dict[str, str | bool]:
        """Resume listening after a soft power off."""
        self._powered = True
        self._auto_reply = True
        self._mic.set_muted(False)
        # Soft-off often idle-timed the Ink socket; force a fresh websocket.
        self._request_stt_restart("turn-on")
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="APP_ON — listening",
            )
        )
        print("[session] app turned on", flush=True)
        return {"ok": True, "powered": True}

    async def start_talking(self) -> dict[str, str | bool]:
        """Queue a spoken reply (Respond button or auto after other speaker stops)."""
        if not self._powered:
            return {
                "ok": False,
                "talking": False,
                "error": "App is off — click Turn on first.",
            }
        # Manual Respond (or confirmed auto) cancels any pending silence wait.
        self._cancel_pending_auto()
        conversation = hub.conversation_text().strip()
        pinned = self._pinned.get().strip()
        if not conversation and not pinned:
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text="need speech or context",
                )
            )
            return {
                "ok": False,
                "talking": False,
                "error": "No transcript or saved context yet.",
            }

        self._auto_reply = True
        was_talking = self._talking
        self._talking = True
        if self._busy:
            self._reply_after_busy = True
        queued = await self._queue_history_reply()
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=(
                    "TALKING_ON — speaking"
                    if not was_talking
                    else "TALKING_ON — refresh"
                ),
            )
        )
        if not queued:
            if not was_talking:
                self._talking = False
            return {
                "ok": False,
                "talking": self._talking,
                "error": "No transcript or saved context yet.",
            }
        print("[session] reply queued", flush=True)
        return {"ok": True, "talking": True, "auto_reply": True}

    def _cancel_pending_auto(self) -> None:
        task = self._pending_auto_task
        self._pending_auto_task = None
        if task is not None and not task.done():
            task.cancel()

    def _schedule_auto_reply(self) -> None:
        """Wait for a real pause after turn.end before Joe responds or fills."""
        self._cancel_pending_auto()
        self._turn_confirm_gen += 1
        gen = self._turn_confirm_gen

        async def _confirm() -> None:
            try:
                while True:
                    started = asyncio.get_running_loop().time()
                    await asyncio.sleep(TURN_CONFIRM_S)
                    if gen != self._turn_confirm_gen:
                        return
                    if not self._powered or not self._auto_reply:
                        return
                    # Speech resumed during the wait — keep holding.
                    if self._last_partial_at > started:
                        print(
                            "[session] still hearing speech — holding reply",
                            flush=True,
                        )
                        await hub.publish(
                            TranscriptEvent(
                                role="status",
                                text="waiting — still talking",
                            )
                        )
                        continue
                    break
                now = asyncio.get_running_loop().time()
                if now - self._last_auto_start < AUTO_REPLY_COOLDOWN_S:
                    return
                self._last_auto_start = now
                print(
                    f"[session] pause confirmed ({TURN_CONFIRM_S:.1f}s) — auto Respond",
                    flush=True,
                )
                await self.start_talking()
            except asyncio.CancelledError:
                return

        self._pending_auto_task = asyncio.create_task(
            _confirm(), name="turn-confirm"
        )

    async def _cut_audio(self) -> None:
        """Hard-stop playback + TTS without changing auto-reply preference."""
        self._talking = False
        self._reply_after_busy = False
        while not self._pending_turns.empty():
            try:
                self._pending_turns.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._speaker.clear()
        task = self._active_speak_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
        self._active_speak_task = None
        self._speaker.clear()
        self._busy = False
        self._echo_gate_until = (
            asyncio.get_running_loop().time() + self.echo_cooldown_s
        )
        # Never leave echo-mute stuck after Stop / barge-in.
        if self._powered:
            self._mic.set_muted(False)

    async def interrupt_speaking(self, *, reason: str = "barge-in") -> None:
        """Other person started talking — shut Joe up, keep auto-reply on."""
        if not (self._talking or self._busy or self._speaker.playing):
            return
        print(f"[session] interrupt ({reason}) — Joe goes quiet", flush=True)
        self._cancel_pending_auto()
        await self._cut_audio()
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="TALKING_OFF — barge-in",
            )
        )

    async def stop_talking(self, *, pause_auto: bool = True) -> dict[str, str | bool]:
        """Stop button: silence now. pause_auto=True until Respond is clicked."""
        print("[session] Stop — cutting audio now", flush=True)
        if pause_auto:
            self._auto_reply = False
        self._cancel_pending_auto()
        await self._cut_audio()
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=(
                    "TALKING_OFF — stopped"
                    if pause_auto
                    else "TALKING_OFF — quiet"
                ),
            )
        )
        return {
            "ok": True,
            "talking": False,
            "auto_reply": self._auto_reply,
        }

    async def clear_context(self) -> dict[str, str | bool | int]:
        """Refresh live conversation; keep saved (pinned) context on disk."""
        if self._talking:
            await self.stop_talking()
        while not self._pending_turns.empty():
            try:
                self._pending_turns.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._reply_after_busy = False
        await hub.clear()
        if self._llm is not None:
            try:
                await self._llm.reset()
            except Exception as exc:  # noqa: BLE001
                print(f"[session] LLM reset failed: {exc}", flush=True)
                return {"ok": False, "talking": False, "error": str(exc)}
        pinned_len = len(self._pinned.get().strip())
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=(
                    f"CONVERSATION_REFRESHED — cleared · context {pinned_len}"
                    if pinned_len
                    else "CONVERSATION_REFRESHED — cleared"
                ),
            )
        )
        print(
            f"[session] conversation refreshed (pinned={pinned_len} chars)",
            flush=True,
        )
        return {
            "ok": True,
            "talking": False,
            "pinned_chars": pinned_len,
            "pinned": self._pinned.get(),
        }

    async def get_pinned_context(self) -> dict[str, str | bool | int]:
        text = self._pinned.get()
        return {"ok": True, "text": text, "chars": len(text.strip())}

    async def set_pinned_context(self, text: str) -> dict[str, str | bool | int]:
        saved = self._pinned.save(text)
        # Cursor agent.send() keeps prior turns — force a fresh agent so the next
        # reply cannot be steered by pre-save bland conversation memory.
        if self._llm is not None:
            self._llm.invalidate_pinned_cache()
            try:
                await self._llm.reset()
            except Exception as exc:  # noqa: BLE001
                print(f"[session] LLM reset after save failed: {exc}", flush=True)
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=(
                    f"CONTEXT_SAVED — saved ({len(saved)})"
                    if saved
                    else "CONTEXT_SAVED — cleared"
                ),
            )
        )
        digest = self._pinned.hash()
        print(
            f"[session] pinned context saved ({len(saved)} chars"
            f"{f', hash={digest}' if digest else ''}) — LLM agent reset",
            flush=True,
        )
        return {"ok": True, "text": saved, "chars": len(saved), "hash": digest}

    # Back-compat for older UI wiring
    async def request_history_reply(self) -> dict[str, str | bool]:
        return await self.start_talking()

    def get_audio_devices(self) -> dict:
        return list_audio_devices(
            current_input=self._mic.device,
            current_output=self._speaker.device,
        )

    async def set_input_device(self, device: int) -> dict[str, str | bool | int]:
        try:
            await self._mic.set_device(int(device))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=f"mic {self._mic.device}",
            )
        )
        return {"ok": True, "device": int(self._mic.device or device)}

    async def set_output_device(self, device: int) -> dict[str, str | bool | int]:
        try:
            await self._speaker.set_device(int(device))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=f"speakers {self._speaker.device}",
            )
        )
        return {"ok": True, "device": int(self._speaker.device or device)}

    def _select_active_tts(self) -> None:
        if self._tts_backend == "local" and self._local_tts is not None:
            self._tts = self._local_tts
        elif self._sonic is not None:
            self._tts = self._sonic
        else:
            self._tts = self._local_tts

    async def get_tts_settings(self) -> dict[str, Any]:
        health: dict[str, Any] = {"ok": False, "reachable": False}
        if self._local_tts is not None:
            health = await self._local_tts.health()
        else:
            probe = ChatterboxTTS(base_url=self._chatterbox_url)
            health = await probe.health()
        voices = [dict(v) for v in CARTESIA_VOICES]
        known = {v["id"] for v in voices}
        if self.voice_id not in known:
            voices.insert(
                0,
                {"id": self.voice_id, "label": f"Custom ({self.voice_id[:8]}…)"},
            )
        return {
            "ok": True,
            "backend": self._tts_backend,
            "backends": [
                {
                    "id": "cartesia",
                    "label": "Cartesia Sonic (cloud)",
                },
                {
                    "id": "local",
                    "label": "Local GPU Chatterbox (Docker)",
                },
            ],
            "voice_id": self.voice_id,
            "voices": voices,
            "chatterbox_url": self._chatterbox_url,
            "speed": self._tts_speed,
            "tonality": self._tts_tonality,
            "local_health": health,
            **choices_payload(),
        }

    def _apply_delivery_to_engines(self) -> None:
        if self._sonic is not None:
            self._sonic.set_delivery(speed=self._tts_speed, tonality=self._tts_tonality)
        if self._local_tts is not None:
            self._local_tts.set_delivery(
                speed=self._tts_speed, tonality=self._tts_tonality
            )
        self._fillers.set_delivery(speed=self._tts_speed, tonality=self._tts_tonality)

    async def set_tts_settings(
        self,
        *,
        backend: Optional[str] = None,
        chatterbox_url: Optional[str] = None,
        voice_id: Optional[str] = None,
        speed: Optional[float | str] = None,
        tonality: Optional[str] = None,
    ) -> dict[str, Any]:
        delivery_changed = False
        if speed is not None:
            try:
                spd = clamp_speed(float(speed))
            except (TypeError, ValueError):
                return {"ok": False, "error": f"Invalid speed {speed!r}"}
            if spd != self._tts_speed:
                self._tts_speed = spd
                save_speed(spd)
                delivery_changed = True

        if tonality is not None:
            tone = normalize_tonality(str(tonality))
            if tone != self._tts_tonality:
                self._tts_tonality = tone
                save_tonality(tone)
                delivery_changed = True

        if delivery_changed:
            self._apply_delivery_to_engines()
            self._fillers.start_warm(backend=self._tts_backend)
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=(
                        f"Voice delivery — {self._tts_speed:g}×, "
                        f"{self._tts_tonality}"
                    ),
                )
            )
            print(
                f"[session] TTS delivery -> speed={self._tts_speed:g} "
                f"tonality={self._tts_tonality}",
                flush=True,
            )

        if chatterbox_url is not None:
            url = chatterbox_url.strip().rstrip("/")
            if not url:
                return {"ok": False, "error": "Chatterbox URL is empty"}
            self._chatterbox_url = url
            _save_chatterbox_url(url)
            if self._local_tts is not None:
                self._local_tts.set_base_url(url)
            self._fillers.set_chatterbox_url(url)

        if voice_id is not None:
            vid = voice_id.strip()
            if not vid:
                return {"ok": False, "error": "Cartesia voice id is empty"}
            if vid != self.voice_id:
                self.voice_id = vid
                _save_cartesia_voice_id(vid)
                if self._sonic is not None:
                    self._sonic.voice_id = vid
                self._fillers.set_cartesia_voice_id(vid)
                if self._tts_backend == "cartesia":
                    self._fillers.start_warm(backend="cartesia")
                label = next(
                    (v["label"] for v in CARTESIA_VOICES if v["id"] == vid),
                    vid[:8] + "…",
                )
                await hub.publish(
                    TranscriptEvent(
                        role="status",
                        text=f"Cartesia voice — {label}",
                    )
                )
                print(f"[session] Cartesia voice -> {vid} ({label})", flush=True)

        if backend is not None:
            b = backend.strip().lower()
            if b not in TTS_BACKENDS:
                return {
                    "ok": False,
                    "error": f"Unknown backend {backend!r}. Use cartesia or local.",
                }
            if b == "local":
                health = (
                    await self._local_tts.health()
                    if self._local_tts is not None
                    else await ChatterboxTTS(base_url=self._chatterbox_url).health()
                )
                if not health.get("ok"):
                    err = health.get("error") or "Chatterbox not ready"
                    await hub.publish(
                        TranscriptEvent(
                            role="error",
                            text=f"local TTS down: {err}",
                        )
                    )
                    return {
                        "ok": False,
                        "error": err,
                        "local_health": health,
                        "backend": self._tts_backend,
                        "chatterbox_url": self._chatterbox_url,
                    }
            self._tts_backend = b
            _save_tts_backend(b)
            self._select_active_tts()
            self._fillers.start_warm(backend=b)
            label = "local" if b == "local" else "cloud"
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=f"TTS_BACKEND — {label}",
                )
            )
            print(f"[session] TTS backend -> {b} ({self._chatterbox_url})", flush=True)

        return await self.get_tts_settings()

    async def run(self) -> None:
        print(
            "Voice agent ready. Speak near the mic.\n"
            "Joe auto-replies when the other person stops; Stop pauses auto.\n"
            "http://localhost:7860/\n"
            "Ctrl+C to stop.\n",
            flush=True,
        )
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="listening",
            )
        )

        async with AsyncCartesia(api_key=self.cartesia_api_key) as cartesia_stt, AsyncCartesia(
            api_key=self.cartesia_api_key
        ) as cartesia_tts:
            llm = CursorVoiceLLM(
                api_key=self.cursor_api_key,
                model=self.cursor_model,
                cwd=self.cwd,
            )
            await llm.start()
            self._llm = llm

            async def on_tts_audio(pcm: bytes) -> None:
                # Drop late chunks the moment Stop flips talking off.
                if not self._talking:
                    return
                # Open speakers → room mic; mute so STT doesn't treat Joe as barge-in.
                self._mic.set_muted(True)
                await self._speaker.enqueue(pcm)

            tts_sonic = SonicTTS(
                cartesia_tts,
                voice_id=self.voice_id,
                on_audio=on_tts_audio,
                speed=self._tts_speed,
                tonality=self._tts_tonality,
            )
            tts_local = ChatterboxTTS(
                base_url=self._chatterbox_url,
                on_audio=on_tts_audio,
                speed=self._tts_speed,
                tonality=self._tts_tonality,
            )
            self._sonic = tts_sonic
            self._local_tts = tts_local
            self._select_active_tts()
            self._fillers.set_chatterbox_url(self._chatterbox_url)
            self._fillers.start_warm(backend=self._tts_backend)
            print(
                f"TTS backend: {self._tts_backend} "
                f"(local URL {self._chatterbox_url})",
                flush=True,
            )

            last_live_print = 0.0

            async def on_partial(transcript: str) -> None:
                nonlocal last_live_print
                if not self._powered:
                    return
                now = asyncio.get_running_loop().time()
                # Speaker bleed / post-playback reverb — do not treat as the other person.
                if (
                    self._mic.muted
                    or self._speaker.playing
                    or now < self._echo_gate_until
                ):
                    return
                display = transcript.replace("\n", " ")
                self._last_partial_at = now
                if now - last_live_print >= 0.25:
                    print(f"[live] {display}", flush=True)
                    last_live_print = now
                await hub.publish(
                    TranscriptEvent(role="you", text=transcript, partial=True)
                )
                speaking = (
                    self._talking
                    or self._busy
                    or self._speaker.playing
                    or (
                        self._active_speak_task is not None
                        and not self._active_speak_task.done()
                    )
                )
                # They resumed mid-pause — do not fire filler/reply yet.
                if (
                    not speaking
                    and self._pending_auto_task is not None
                    and not self._pending_auto_task.done()
                    and len((transcript or "").strip()) >= 3
                ):
                    self._cancel_pending_auto()
                    print(
                        "[session] speech resumed — cancelled pending reply",
                        flush=True,
                    )
                    await hub.publish(
                        TranscriptEvent(
                            role="status",
                            text="waiting — still talking",
                        )
                    )
                # Barge-in: other person started talking while Joe is mid-reply.
                # Only during silent LLM think — mic is muted once speakers play.
                if not speaking:
                    return
                if now < self._ignore_barge_until:
                    return
                text = (transcript or "").strip()
                # Need a real utterance, not a one-word echo while Joe is starting.
                if len(text) < 18:
                    return
                await self.interrupt_speaking(reason="other-speaker")

            async def on_turn_end(transcript: str) -> None:
                if not self._powered:
                    return
                if transcript is None or transcript == "" or not transcript.strip():
                    return
                now = asyncio.get_running_loop().time()
                if (
                    self._talking
                    or self._busy
                    or self._speaker.playing
                    or self._mic.muted
                    or now < self._echo_gate_until
                ):
                    print(
                        "[session] ignore turn.end during/after Joe playback (echo)",
                        flush=True,
                    )
                    return
                print(f"\n[you] {transcript}", flush=True)
                await hub.publish(
                    TranscriptEvent(role="you", text=transcript, partial=False)
                )
                # Hold for a real pause before auto-reply / filler.
                if not self._auto_reply:
                    return
                if now - self._last_auto_start < AUTO_REPLY_COOLDOWN_S:
                    return
                print(
                    f"[session] turn.end — waiting {TURN_CONFIRM_S:.1f}s "
                    "to confirm pause",
                    flush=True,
                )
                await hub.publish(
                    TranscriptEvent(
                        role="status",
                        text=f"waiting — {TURN_CONFIRM_S:.1f}s pause",
                    )
                )
                self._schedule_auto_reply()

            stt = InkSTT(
                cartesia_stt,
                on_partial=on_partial,
                on_turn_end=on_turn_end,
            )

            await self._speaker.start()
            await self._mic.start()
            # Agent is on at boot — capture must not start muted.
            self._mic.set_muted(False)
            print("[mic] ready (unmuted)", flush=True)

            reply_task = asyncio.create_task(
                self._reply_loop(llm), name="reply-loop"
            )
            try:
                while not self._stop.is_set():
                    try:
                        self._stt_restart.clear()
                        async with stt:
                            feed_task = asyncio.create_task(
                                self._feed_stt(stt), name="feed-stt"
                            )
                            stop_wait = asyncio.create_task(
                                self._stop.wait(), name="stop-wait"
                            )
                            restart_wait = asyncio.create_task(
                                self._stt_restart.wait(), name="stt-restart-wait"
                            )
                            done, _pending = await asyncio.wait(
                                {feed_task, stop_wait, restart_wait},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for task in (stop_wait, restart_wait):
                                if not task.done():
                                    task.cancel()
                            if not feed_task.done():
                                feed_task.cancel()
                            await asyncio.gather(
                                feed_task,
                                stop_wait,
                                restart_wait,
                                return_exceptions=True,
                            )
                            if self._stop.is_set():
                                break
                            print("[stt] reconnecting websocket...", flush=True)
                            await hub.publish(
                                TranscriptEvent(
                                    role="status",
                                    text="STT reconnecting…",
                                )
                            )
                            await asyncio.sleep(0.35)
                            continue
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc)
                        print(f"[stt] connection failed: {msg}", flush=True)
                        await hub.publish(
                            TranscriptEvent(
                                role="error",
                                text=f"STT down: {msg}",
                            )
                        )
                        try:
                            await asyncio.wait_for(self._stop.wait(), timeout=8.0)
                            break
                        except asyncio.TimeoutError:
                            print("[stt] retrying connection...", flush=True)
                            continue
            finally:
                reply_task.cancel()
                await asyncio.gather(reply_task, return_exceptions=True)
                await self._mic.stop()
                await self._speaker.stop()
                self._llm = None
                if self._local_tts is not None:
                    await self._local_tts.close()
                self._sonic = None
                self._local_tts = None
                self._tts = None
                await llm.close()

    async def _feed_stt(self, stt: InkSTT) -> None:
        """Keep STT fed even when TTS/LLM is slow — never block mic forever."""
        failures = 0
        async for chunk in self._mic.chunks():
            if self._stop.is_set():
                break
            if self._stt_restart.is_set():
                return
            if not stt.alive:
                self._request_stt_restart("recv-dead")
                return
            try:
                await asyncio.wait_for(stt.send_audio(chunk), timeout=1.0)
                failures = 0
            except asyncio.TimeoutError:
                print("[stt] send slow — dropping frame to stay live", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                msg = str(exc)
                print(f"[stt] send failed: {msg}", flush=True)
                lower = msg.lower()
                fatal = (
                    failures >= 3
                    or "1001" in lower
                    or "going away" in lower
                    or "idle timeout" in lower
                    or "closed" in lower
                    or "connection" in lower
                )
                if fatal:
                    self._request_stt_restart("send-failed")
                    return
                await asyncio.sleep(0.05)

    async def _reply_loop(self, llm: CursorVoiceLLM) -> None:
        while not self._stop.is_set():
            payload = await self._pending_turns.get()
            if not self._talking:
                continue
            async with self._turn_lock:
                if not self._talking:
                    continue
                self._busy = True
                print("[agent] thinking…", flush=True)
                engine = self._tts_backend
                await hub.publish(
                    TranscriptEvent(
                        role="status",
                        text=f"speaking… ({'local' if engine == 'local' else 'cloud'})",
                    )
                )
                fresh = hub.conversation_text().strip() or payload
                # Always re-read from disk so UI Save is what this reply uses.
                pinned = self._pinned.load().strip()
                pinned_hash = self._pinned.hash()
                if pinned:
                    print(
                        f"[session] reply pinned chars={len(pinned)} "
                        f"hash={pinned_hash} path={self._pinned.path}",
                        flush=True,
                    )
                else:
                    print(
                        f"[session] reply pinned EMPTY path={self._pinned.path}",
                        flush=True,
                    )
                collected: list[str] = []
                tts = self._tts
                if tts is None:
                    await hub.publish(
                        TranscriptEvent(
                            role="error",
                            text="TTS not ready — restart the app.",
                        )
                    )
                    self._busy = False
                    continue

                async def text_stream():
                    if pinned:
                        print(
                            f"[agent] applying saved context ({len(pinned)} chars, "
                            f"hash={pinned_hash})",
                            flush=True,
                        )
                    async for piece in llm.stream_history_reply(
                        fresh, pinned_context=pinned
                    ):
                        if not self._talking:
                            break
                        if not piece:
                            continue
                        print(piece, end="", flush=True)
                        collected.append(piece)
                        caption = strip_tags(piece)
                        if caption:
                            await hub.publish(
                                TranscriptEvent(
                                    role="agent",
                                    text=caption,
                                    partial=True,
                                )
                            )
                        spoken = strip_tags(piece)
                        if spoken:
                            yield spoken
                    print(flush=True)

                speak_task: asyncio.Task[None] | None = None
                reply_attempted = False
                interrupted = False
                try:
                    if self._talking:
                        reply_attempted = True
                        # Mic stays open while LLM thinks (real barge-in). Mute starts
                        # on first TTS/filler PCM — room speakers otherwise re-trigger STT.
                        self._speaker.arm()
                        # Grace while LLM thinks — don't treat room echo/partials as barge-in yet.
                        self._ignore_barge_until = (
                            asyncio.get_running_loop().time()
                            + max(2.2, FILLER_AFTER_LLM_WAIT_S + 1.4)
                        )
                        llm_q: asyncio.Queue[str | None] = asyncio.Queue()

                        async def _prefetch_llm() -> None:
                            try:
                                async for spoken in text_stream():
                                    await llm_q.put(spoken)
                            finally:
                                await llm_q.put(None)

                        prefetch = asyncio.create_task(
                            _prefetch_llm(), name="llm-prefetch"
                        )
                        # Think silently first. Filler only if LLM is still slow
                        # and the other person has stayed quiet.
                        first_held: str | None = None
                        got_first = False
                        filler_s = 0.0
                        try:
                            first_held = await asyncio.wait_for(
                                llm_q.get(), timeout=FILLER_AFTER_LLM_WAIT_S
                            )
                            got_first = True
                        except asyncio.TimeoutError:
                            quiet_for = (
                                asyncio.get_running_loop().time()
                                - self._last_partial_at
                            )
                            if quiet_for >= TURN_CONFIRM_S * 0.75 and self._talking:
                                self._mic.set_muted(True)
                                filler_s = await self._fillers.enqueue(
                                    self._speaker, backend=self._tts_backend
                                )
                                await hub.publish(
                                    TranscriptEvent(
                                        role="status",
                                        text="speaking... filler",
                                    )
                                )
                            else:
                                print(
                                    "[session] skip filler — not sure turn is done",
                                    flush=True,
                                )

                        async def _buffered_speech():
                            if got_first:
                                if first_held is None:
                                    return
                                yield first_held
                            while True:
                                item = await llm_q.get()
                                if item is None:
                                    break
                                yield item

                        self._ignore_barge_until = max(
                            self._ignore_barge_until,
                            asyncio.get_running_loop().time()
                            + max(0.8, filler_s + 0.35),
                        )
                        speak_task = asyncio.create_task(
                            tts.speak_stream(
                                _buffered_speech(),
                                should_stop=lambda: not self._talking,
                            ),
                            name="tts-speak",
                        )
                        self._active_speak_task = speak_task
                        # Keep prefetch referenced so it isn't GC'd mid-turn.
                        speak_task.add_done_callback(
                            lambda _t, p=prefetch: p.cancel()
                            if not p.done()
                            else None
                        )
                        try:
                            await asyncio.wait_for(speak_task, timeout=180.0)
                        except asyncio.CancelledError:
                            interrupted = True
                            print("\n[session] speak cancelled", flush=True)
                            self._speaker.clear()
                        if self._talking:
                            await self._speaker.wait_drained()
                            self._talking = False
                            await hub.publish(
                                TranscriptEvent(
                                    role="status",
                                    text="TALKING_OFF — done",
                                )
                            )
                        else:
                            interrupted = True
                            self._speaker.clear()
                        # After Joe finishes (or is cut), allow barge-in again soon.
                        self._ignore_barge_until = (
                            asyncio.get_running_loop().time() + 0.35
                        )
                    # Always commit Joe's words to the transcript, even after Stop/barge-in.
                    full = strip_tags(join_speech_chunks(collected))
                    if full:
                        await hub.publish(
                            TranscriptEvent(
                                role="agent", text=full, partial=False
                            )
                        )
                    elif reply_attempted and not interrupted:
                        await hub.publish(
                            TranscriptEvent(
                                role="error",
                                text="No audio generated — click Respond as Joe again.",
                            )
                        )
                except asyncio.TimeoutError:
                    print("\n[error] reply timed out", flush=True)
                    self._speaker.clear()
                    await hub.publish(
                        TranscriptEvent(
                            role="error",
                            text="Reply timed out — click Stop, then Respond as Joe.",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[error] reply failed: {exc}", flush=True)
                    self._speaker.clear()
                    await hub.publish(
                        TranscriptEvent(role="error", text=str(exc))
                    )
                finally:
                    self._active_speak_task = None
                    # Hold the echo gate across drain settle so trailing STT is ignored.
                    self._echo_gate_until = (
                        asyncio.get_running_loop().time() + self.echo_cooldown_s
                    )
                    await asyncio.sleep(self.echo_cooldown_s)
                    self._busy = False
                    if self._powered:
                        self._mic.set_muted(False)
                    self._reply_after_busy = False
                    # Turn-taking: do not auto-queue another monologue.
                    # Next reply comes from turn-end auto or Respond.

    def request_stop(self) -> None:
        self._talking = False
        self._stop.set()
