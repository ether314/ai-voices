"""Turn orchestration: mic → STT → button-started continuous Joe replies → TTS."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional, Union

from cartesia import AsyncCartesia

from agent.audio import (
    MicCapture,
    SpeakerPlayback,
    list_audio_devices,
    resolve_device_choice,
)
from agent.expression import (
    join_speech_chunks,
    strip_tags,
)
from agent.fillers import FillerBank, last_caller_utterance
from agent.llm import (
    DEFAULT_MODEL as DEFAULT_CURSOR_MODEL,
    CursorVoiceLLM,
    resolve_cursor_runtime,
)
from agent.local_llm import (
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_LOCAL_LLM_URL,
    LocalVoiceLLM,
)
from agent.local_tts import ChatterboxTTS, DEFAULT_CHATTERBOX_URL
from agent.pinned_context import PinnedContextStore
from agent.local_stt import (
    LocalWhisperSTT,
    detect_stt_device,
    resolve_whisper_model,
)
from agent.stt import InkSTT, permit_cloud_stt
from agent.transcript_hub import TranscriptEvent, hub
from agent.tts import (
    DEFAULT_VOICE_ID,
    SonicTTS,
    list_cartesia_voices,
    resolve_cartesia_voice_id,
    voice_label_for_id,
)
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
# Stereo Mix also needs mute-while-Joe (Joe is in the mix); cooldown can be shorter.
ECHO_COOLDOWN_S = 0.55
# Ink often fires turn.end on a short breath — brief silence check before Joe speaks.
# Loopback / Stereo Mix: jump in earlier (more false starts OK). Room mic: wait longer.
TURN_CONFIRM_LOOPBACK_S = 0.4
TURN_CONFIRM_MIC_S = 0.95
TURN_CONFIRM_S = TURN_CONFIRM_MIC_S  # default / room-mic baseline
AUTO_REPLY_COOLDOWN_S = 0.95
FILLER_AFTER_LLM_WAIT_S = 0.5
TTS_BACKENDS = ("cartesia", "local")
LLM_BACKENDS = ("cursor", "local")
STT_BACKENDS = ("local", "cartesia")
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TTS_BACKEND_PATH = _DATA_DIR / "tts_backend.txt"
_CHATTERBOX_URL_PATH = _DATA_DIR / "chatterbox_url.txt"
_CARTESIA_VOICE_PATH = _DATA_DIR / "cartesia_voice_id.txt"
_LLM_BACKEND_PATH = _DATA_DIR / "llm_backend.txt"
_LOCAL_LLM_URL_PATH = _DATA_DIR / "local_llm_url.txt"
_LOCAL_LLM_MODEL_PATH = _DATA_DIR / "local_llm_model.txt"
_STT_BACKEND_PATH = _DATA_DIR / "stt_backend.txt"
_AUTO_REPLY_PATH = _DATA_DIR / "auto_reply.txt"

STTEngine = Union[InkSTT, LocalWhisperSTT]

VoiceLLM = Union[CursorVoiceLLM, LocalVoiceLLM]


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
    # Known-bad Jack UUID is rewritten to the account's working Jack clone.
    try:
        if _CARTESIA_VOICE_PATH.is_file():
            val = _CARTESIA_VOICE_PATH.read_text(encoding="utf-8").strip()
            if val:
                resolved = resolve_cartesia_voice_id(val, fallback=fallback)
                if resolved != val:
                    try:
                        _save_cartesia_voice_id(resolved)
                    except OSError:
                        pass
                return resolved
    except OSError:
        pass
    env = os.getenv("CARTESIA_VOICE_ID", "").strip()
    if env:
        return resolve_cartesia_voice_id(env, fallback=fallback)
    return resolve_cartesia_voice_id(fallback or DEFAULT_VOICE_ID)


def _save_tts_backend(backend: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _TTS_BACKEND_PATH.write_text(backend + "\n", encoding="utf-8")


def _save_chatterbox_url(url: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CHATTERBOX_URL_PATH.write_text(url.rstrip("/") + "\n", encoding="utf-8")


def _save_cartesia_voice_id(voice_id: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    vid = resolve_cartesia_voice_id(voice_id)
    _CARTESIA_VOICE_PATH.write_text(vid + "\n", encoding="utf-8")


def _load_llm_backend() -> str:
    env = os.getenv("LLM_BACKEND", "").strip().lower()
    if env in LLM_BACKENDS:
        return env
    try:
        if _LLM_BACKEND_PATH.is_file():
            val = _LLM_BACKEND_PATH.read_text(encoding="utf-8").strip().lower()
            if val in LLM_BACKENDS:
                return val
    except OSError:
        pass
    return "cursor"


def _load_local_llm_url() -> str:
    env = os.getenv("LOCAL_LLM_URL", "").strip()
    if env:
        return env.rstrip("/")
    try:
        if _LOCAL_LLM_URL_PATH.is_file():
            val = _LOCAL_LLM_URL_PATH.read_text(encoding="utf-8").strip()
            if val:
                return val.rstrip("/")
    except OSError:
        pass
    return DEFAULT_LOCAL_LLM_URL


def _load_local_llm_model() -> str:
    env = os.getenv("LOCAL_LLM_MODEL", "").strip()
    if env:
        return env
    try:
        if _LOCAL_LLM_MODEL_PATH.is_file():
            val = _LOCAL_LLM_MODEL_PATH.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    return DEFAULT_LOCAL_LLM_MODEL


def _save_llm_backend(backend: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _LLM_BACKEND_PATH.write_text(backend + "\n", encoding="utf-8")


def _save_local_llm_url(url: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _LOCAL_LLM_URL_PATH.write_text(url.rstrip("/") + "\n", encoding="utf-8")


def _save_local_llm_model(model: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _LOCAL_LLM_MODEL_PATH.write_text(model.strip() + "\n", encoding="utf-8")


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def allow_cartesia_stt_env() -> bool:
    """Hard kill-switch. Cartesia Ink requires ALLOW_CARTESIA_STT=1 in addition to backend=cartesia."""
    return _env_truthy("ALLOW_CARTESIA_STT")


def _load_auto_reply(default: bool = False) -> bool:
    """Persisted preference; missing file → default OFF."""
    try:
        if _AUTO_REPLY_PATH.is_file():
            raw = _AUTO_REPLY_PATH.read_text(encoding="utf-8").strip().lower()
            if raw in ("1", "true", "on", "yes"):
                return True
            if raw in ("0", "false", "off", "no"):
                return False
    except OSError:
        pass
    return default


def _save_auto_reply(enabled: bool) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _AUTO_REPLY_PATH.write_text(
            ("1" if enabled else "0") + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"[session] could not save auto_reply: {exc}", flush=True)


def _load_stt_backend() -> str:
    env = os.getenv("STT_BACKEND", "").strip().lower()
    chosen = ""
    if env in STT_BACKENDS:
        chosen = env
    else:
        try:
            if _STT_BACKEND_PATH.is_file():
                val = _STT_BACKEND_PATH.read_text(encoding="utf-8").strip().lower()
                if val in STT_BACKENDS:
                    chosen = val
        except OSError:
            pass
    # Default local — avoid Cartesia Ink STT token burn.
    if not chosen:
        chosen = "local"
    # Hard-disable cloud STT unless both backend=cartesia AND ALLOW_CARTESIA_STT=1.
    if chosen == "cartesia" and not allow_cartesia_stt_env():
        print(
            "[stt] REFUSING Cartesia Ink — STT_BACKEND wants cartesia but "
            "ALLOW_CARTESIA_STT is not set. Forcing local Whisper "
            "(no Speech-to-Text tokens).",
            flush=True,
        )
        try:
            _save_stt_backend("local")
        except OSError:
            pass
        return "local"
    return chosen


def _save_stt_backend(backend: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _STT_BACKEND_PATH.write_text(backend + "\n", encoding="utf-8")


class VoiceSession:
    def __init__(
        self,
        *,
        cartesia_api_key: str,
        cursor_api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        cursor_model: str = DEFAULT_CURSOR_MODEL,
        cwd: Optional[str] = None,
        echo_cooldown_s: float = ECHO_COOLDOWN_S,
    ) -> None:
        self.cartesia_api_key = cartesia_api_key
        self.cursor_api_key = cursor_api_key
        # Env > persisted file > constructor/default (same pattern as TTS backend).
        self.voice_id = _load_cartesia_voice_id(voice_id or DEFAULT_VOICE_ID)
        self.cursor_model = cursor_model or DEFAULT_CURSOR_MODEL
        self.cwd = cwd or os.getcwd()
        self.echo_cooldown_s = echo_cooldown_s

        self._mic = MicCapture()
        self._speaker = SpeakerPlayback()
        self._busy = False
        self._talking = False
        self._powered = True
        # Default OFF; last preference survives restarts via data/auto_reply.txt.
        self._auto_reply = _load_auto_reply(default=False)
        self._reply_after_busy = False
        self._turn_lock = asyncio.Lock()
        self._pending_turns: asyncio.Queue[str] = asyncio.Queue()
        self._stop = asyncio.Event()
        # Feed/recv death or Turn on after idle-timeout — reopen Ink websocket.
        self._stt_restart = asyncio.Event()
        self._stt_restart_reason = ""
        self._active_speak_task: asyncio.Task[None] | None = None
        self._cursor_llm: CursorVoiceLLM | None = None
        self._local_llm: LocalVoiceLLM | None = None
        self._llm: VoiceLLM | None = None
        self._llm_backend = _load_llm_backend()
        self._local_llm_url = _load_local_llm_url()
        self._local_llm_model = _load_local_llm_model()
        self._pinned = PinnedContextStore()
        self._tts_backend = _load_tts_backend()
        self._chatterbox_url = _load_chatterbox_url()
        self._stt_backend = _load_stt_backend()
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
        self._last_auto_off_hint_at = 0.0
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

    def _set_auto_reply(self, enabled: bool) -> None:
        self._auto_reply = bool(enabled)
        _save_auto_reply(self._auto_reply)

    async def set_auto_reply(self, enabled: bool) -> dict[str, Any]:
        """UI Auto-reply toggle — persist preference without starting a reply."""
        was = self._auto_reply
        self._set_auto_reply(enabled)
        if not self._auto_reply:
            self._cancel_pending_auto()
        if was != self._auto_reply:
            label = "on" if self._auto_reply else "off"
            print(f"[session] auto-reply {label} (toggle)", flush=True)
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=(
                        "AUTO_ON — pause will trigger Joe"
                        if self._auto_reply
                        else "AUTO_OFF — enable Auto-reply for pause replies"
                    ),
                )
            )
        return {
            "ok": True,
            "auto_reply": self._auto_reply,
            "powered": self._powered,
            "talking": self._talking,
        }

    async def get_session_state(self) -> dict[str, Any]:
        return {
            "ok": True,
            "powered": self._powered,
            "talking": self._talking,
            "auto_reply": self._auto_reply,
        }

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
        self._set_auto_reply(False)
        self._mic.set_muted(True)
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="APP_OFF — off",
            )
        )
        print("[session] app turned off (paused)", flush=True)
        return {"ok": True, "powered": False, "auto_reply": False}

    def _request_stt_restart(self, reason: str) -> None:
        if self._stop.is_set():
            return
        self._stt_restart_reason = reason or "restart"
        if not self._stt_restart.is_set():
            print(f"[stt] restart requested ({reason})", flush=True)
        self._stt_restart.set()

    async def turn_on(self) -> dict[str, str | bool]:
        """Resume listening after a soft power off."""
        self._powered = True
        # Do not force auto-reply on — keep last preference (usually off after Turn off).
        self._mic.set_muted(False)
        # Soft-off often idle-timed the STT socket; force a fresh session.
        self._request_stt_restart("turn-on")
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="APP_ON — listening",
            )
        )
        print(
            f"[session] app turned on (auto_reply={'on' if self._auto_reply else 'off'})",
            flush=True,
        )
        return {
            "ok": True,
            "powered": True,
            "auto_reply": self._auto_reply,
        }

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

        # Respond as Joe enables auto turn-taking until Stop / Turn off.
        self._set_auto_reply(True)
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
                "auto_reply": self._auto_reply,
                "error": "No transcript or saved context yet.",
            }
        print("[session] reply queued (auto on)", flush=True)
        return {"ok": True, "talking": True, "auto_reply": True}

    def _turn_confirm_s(self) -> float:
        """Shorter pause confirm on Stereo Mix / loopback; longer for room mic."""
        if self._mic.is_loopback:
            return TURN_CONFIRM_LOOPBACK_S
        return TURN_CONFIRM_MIC_S

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
        confirm_s = self._turn_confirm_s()

        async def _confirm() -> None:
            try:
                while True:
                    started = asyncio.get_running_loop().time()
                    await asyncio.sleep(confirm_s)
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
                    f"[session] pause confirmed ({confirm_s:.1f}s) — auto Respond",
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
            self._set_auto_reply(False)
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

    async def set_input_device(
        self,
        device: Optional[int] = None,
        *,
        name: Optional[str] = None,
    ) -> dict[str, str | bool | int]:
        resolved = resolve_device_choice(kind="input", index=device, name=name)
        if resolved is None:
            return {
                "ok": False,
                "error": f"No input device matched index={device!r} name={name!r}",
            }
        try:
            await self._mic.set_device(int(resolved))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=f"mic {self._mic.device}",
            )
        )
        return {"ok": True, "device": int(self._mic.device or resolved)}

    async def set_output_device(
        self,
        device: Optional[int] = None,
        *,
        name: Optional[str] = None,
    ) -> dict[str, str | bool | int]:
        resolved = resolve_device_choice(kind="output", index=device, name=name)
        if resolved is None:
            return {
                "ok": False,
                "error": f"No output device matched index={device!r} name={name!r}",
            }
        try:
            await self._speaker.set_device(int(resolved))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=f"speakers {self._speaker.device}",
            )
        )
        return {"ok": True, "device": int(self._speaker.device or resolved)}

    def _select_active_llm(self) -> None:
        if self._llm_backend == "local" and self._local_llm is not None:
            self._llm = self._local_llm
        elif self._cursor_llm is not None:
            self._llm = self._cursor_llm
        else:
            self._llm = self._local_llm

    def _cursor_runtime(self) -> str:
        if self._cursor_llm is not None:
            return getattr(self._cursor_llm, "runtime", None) or resolve_cursor_runtime()
        return resolve_cursor_runtime()

    def _llm_status_label(self) -> str:
        if self._llm_backend == "local":
            return f"local ({self._local_llm_model})"
        return f"cursor/{self._cursor_runtime()} ({self.cursor_model})"

    async def get_llm_settings(self) -> dict[str, Any]:
        cursor_runtime = self._cursor_runtime()
        health: dict[str, Any] = {
            "ok": False,
            "reachable": False,
            "skipped": self._llm_backend != "local",
        }
        local_models: list[str] = []
        # Only probe the OpenAI-compatible local API when that backend is active.
        # Cursor does not need Ollama; probing it just surfaces connection noise.
        if self._llm_backend == "local":
            if self._local_llm is not None:
                health = await self._local_llm.health()
            else:
                probe = LocalVoiceLLM(
                    base_url=self._local_llm_url,
                    model=self._local_llm_model,
                    api_key=os.getenv("LOCAL_LLM_API_KEY", "ollama").strip()
                    or "ollama",
                )
                try:
                    health = await probe.health()
                finally:
                    await probe.close()
            local_models = list(health.get("models") or [])
            if self._local_llm_model and self._local_llm_model not in local_models:
                local_models = [self._local_llm_model, *local_models]
            hint = (
                f"Local OpenAI-compatible chat at {self._local_llm_url} "
                "(Ollama / LM Studio). Model list from /v1/models when reachable."
            )
        else:
            if self._local_llm_model:
                local_models = [self._local_llm_model]
            hint = (
                f"Reply model: Cursor ({cursor_runtime}) — {self.cursor_model}"
            )
        return {
            "ok": True,
            "backend": self._llm_backend,
            "backends": [
                {
                    "id": "cursor",
                    "label": f"Cursor local ({self.cursor_model})",
                },
                {
                    "id": "local",
                    "label": "Local (OpenAI-compatible)",
                },
            ],
            "cursor_model": self.cursor_model,
            "cursor_runtime": cursor_runtime,
            "local_url": self._local_llm_url,
            "local_model": self._local_llm_model,
            "local_models": local_models,
            "local_health": health,
            "active_label": self._llm_status_label(),
            "hint": hint,
        }

    async def set_llm_settings(
        self,
        *,
        backend: Optional[str] = None,
        local_url: Optional[str] = None,
        local_model: Optional[str] = None,
    ) -> dict[str, Any]:
        if local_url is not None:
            url = local_url.strip().rstrip("/")
            if not url:
                return {"ok": False, "error": "Local LLM URL is empty"}
            self._local_llm_url = url
            _save_local_llm_url(url)
            if self._local_llm is not None:
                self._local_llm.set_base_url(url)

        if local_model is not None:
            model = local_model.strip()
            if not model:
                return {"ok": False, "error": "Local LLM model name is empty"}
            self._local_llm_model = model
            _save_local_llm_model(model)
            if self._local_llm is not None:
                self._local_llm.set_model(model)

        if backend is not None:
            b = backend.strip().lower()
            if b not in LLM_BACKENDS:
                return {
                    "ok": False,
                    "error": f"Unknown LLM backend {backend!r}. Use cursor or local.",
                }
            if b == "local":
                if self._local_llm is None:
                    self._local_llm = LocalVoiceLLM(
                        base_url=self._local_llm_url,
                        model=self._local_llm_model,
                        api_key=os.getenv("LOCAL_LLM_API_KEY", "ollama").strip()
                        or "ollama",
                    )
                    await self._local_llm.start()
                else:
                    self._local_llm.set_base_url(self._local_llm_url)
                    self._local_llm.set_model(self._local_llm_model)
                health = await self._local_llm.health()
                if not health.get("ok"):
                    err = health.get("error") or "Local LLM not ready"
                    await hub.publish(
                        TranscriptEvent(
                            role="error",
                            text=f"local LLM down: {err}",
                        )
                    )
                    return {
                        "ok": False,
                        "error": err,
                        "local_health": health,
                        "backend": self._llm_backend,
                        "local_url": self._local_llm_url,
                        "local_model": self._local_llm_model,
                    }
            elif b == "cursor":
                if self._cursor_llm is None:
                    return {
                        "ok": False,
                        "error": "Cursor LLM not ready — wait for Voice agent ready.",
                    }
                try:
                    await self._cursor_llm.reset()
                except Exception as exc:  # noqa: BLE001
                    print(f"[session] Cursor reset on switch failed: {exc}", flush=True)

            self._llm_backend = b
            _save_llm_backend(b)
            self._select_active_llm()
            if self._llm is not None:
                try:
                    await self._llm.reset()
                except Exception as exc:  # noqa: BLE001
                    print(f"[session] LLM reset after switch failed: {exc}", flush=True)
            label = self._llm_status_label()
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=f"LLM_BACKEND — {label}",
                )
            )
            print(f"[session] LLM backend -> {label}", flush=True)

        return await self.get_llm_settings()

    def _select_active_tts(self) -> None:
        if self._tts_backend == "local" and self._local_tts is not None:
            self._tts = self._local_tts
        elif self._sonic is not None:
            self._tts = self._sonic
        else:
            self._tts = self._local_tts

    async def get_tts_settings(
        self, *, force_refresh_voices: bool = False
    ) -> dict[str, Any]:
        health: dict[str, Any] = {"ok": False, "reachable": False}
        if self._local_tts is not None:
            health = await self._local_tts.health()
        else:
            probe = ChatterboxTTS(base_url=self._chatterbox_url)
            health = await probe.health()
        voices = await list_cartesia_voices(
            self.cartesia_api_key, force_refresh=force_refresh_voices
        )
        # Joe/Jack are pin-to-top only when the API (or cache) already includes them.
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
            "voices_count": len(voices),
            "voices_refreshed": bool(force_refresh_voices),
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
            vid = resolve_cartesia_voice_id(voice_id)
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
                voices = await list_cartesia_voices(self.cartesia_api_key)
                label = voice_label_for_id(vid, voices)
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

    def _stt_status_label(self) -> str:
        if self._stt_backend == "local":
            return f"local ({resolve_whisper_model()})"
        return "cartesia (ink-2)"

    async def get_stt_settings(self) -> dict[str, Any]:
        allow = allow_cartesia_stt_env()
        stt_device, stt_compute = detect_stt_device()
        return {
            "ok": True,
            "backend": self._stt_backend,
            "backends": [
                {
                    "id": "local",
                    "label": f"Local Whisper ({resolve_whisper_model()})",
                },
                {
                    "id": "cartesia",
                    "label": "Cartesia Ink-2 (cloud STT tokens — costs money)",
                    "requires_allow_env": True,
                    "allow_env_set": allow,
                },
            ],
            "whisper_model": resolve_whisper_model(),
            "stt_device": stt_device,
            "stt_compute_type": stt_compute,
            "active_label": self._stt_status_label(),
            "allow_cartesia_stt": allow,
            "hint": (
                "Local STT runs faster-whisper on this PC (no Cartesia Ink "
                "tokens). Detected device: "
                f"{stt_device}/{stt_compute}. "
                "Model size is set via STT_WHISPER_MODEL in .env. "
                "Cartesia Ink is hard-disabled unless ALLOW_CARTESIA_STT=1 "
                "in .env AND you Apply STT with an explicit cost confirmation."
            ),
        }

    async def set_stt_settings(
        self,
        *,
        backend: Optional[str] = None,
        confirm_cost: bool = False,
    ) -> dict[str, Any]:
        if backend is not None:
            b = backend.strip().lower()
            if b not in STT_BACKENDS:
                return {
                    "ok": False,
                    "error": f"Unknown STT backend {backend!r}. Use local or cartesia.",
                }
            if b == "cartesia":
                if not allow_cartesia_stt_env():
                    print(
                        "[stt] REFUSING switch to Cartesia Ink — set "
                        "ALLOW_CARTESIA_STT=1 in .env first",
                        flush=True,
                    )
                    return {
                        "ok": False,
                        "error": (
                            "Cartesia STT is hard-disabled. Add ALLOW_CARTESIA_STT=1 "
                            "to .env, restart the agent, then Apply again with cost confirmation. "
                            "This bills Speech-to-Text tokens."
                        ),
                    }
                if not confirm_cost:
                    print(
                        "[stt] REFUSING switch to Cartesia Ink — missing confirm_cost",
                        flush=True,
                    )
                    return {
                        "ok": False,
                        "error": (
                            "Cartesia Ink bills Speech-to-Text tokens. Confirm in the UI "
                            "(confirm_cost) before enabling."
                        ),
                    }
            prev = self._stt_backend
            self._stt_backend = b
            _save_stt_backend(b)
            # Latch matches backend so a stray InkSTT construct cannot open WS on local.
            permit_cloud_stt(b == "cartesia" and allow_cartesia_stt_env())
            label = self._stt_status_label()
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=f"STT_BACKEND — {label}",
                )
            )
            print(f"[session] STT backend -> {label}", flush=True)
            if prev != b:
                self._request_stt_restart("backend-switch")
        return await self.get_stt_settings()

    async def run(self) -> None:
        print(
            "Voice agent ready. Speak near the mic.\n"
            "Auto-reply defaults OFF — use the Auto-reply toggle (or Respond as Joe).\n"
            "Stop turns auto off; pause replies need Auto-reply ON.\n"
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

        async with AsyncCartesia(api_key=self.cartesia_api_key) as cartesia_tts:
            cursor_llm = CursorVoiceLLM(
                api_key=self.cursor_api_key,
                model=self.cursor_model,
                cwd=self.cwd,
            )
            await cursor_llm.start()
            self._cursor_llm = cursor_llm

            local_llm = LocalVoiceLLM(
                base_url=self._local_llm_url,
                model=self._local_llm_model,
                api_key=os.getenv("LOCAL_LLM_API_KEY", "ollama").strip() or "ollama",
            )
            await local_llm.start()
            self._local_llm = local_llm
            self._select_active_llm()
            print(
                f"LLM backend: {self._llm_status_label()} "
                f"(local URL {self._local_llm_url})",
                flush=True,
            )
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=f"LLM — {self._llm_status_label()}",
                )
            )
            # Default: forbid Ink. Only the cartesia branch below may re-permit.
            permit_cloud_stt(False)
            print(
                f"STT backend: {self._stt_status_label()} "
                f"(Cartesia Ink allowed={allow_cartesia_stt_env()})",
                flush=True,
            )
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text=f"STT — {self._stt_status_label()}",
                )
            )

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
            voice_label = voice_label_for_id(self.voice_id)
            print(
                f"TTS backend: {self._tts_backend} "
                f"(local URL {self._chatterbox_url})",
                flush=True,
            )
            print(f"Cartesia voice: {voice_label} ({self.voice_id})", flush=True)

            # Preload Whisper before opening the mic so download/warmup
            # does not starve the capture queue.
            if self._stt_backend == "local":
                from agent.local_stt import (
                    _load_model,
                    detect_stt_device,
                    resolve_whisper_model,
                )

                _dev, _ctype = detect_stt_device()
                await asyncio.to_thread(
                    _load_model, resolve_whisper_model(), _dev, _ctype
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
                # Hard echo: Joe is in the speakers / mic muted — do not log as "you".
                if self._mic.muted or self._speaker.playing:
                    print(
                        "[session] ignore turn.end during Joe audio (echo)",
                        flush=True,
                    )
                    return
                # Always store the finalized utterance in the transcript log.
                print(f"\n[you] {transcript}", flush=True)
                await hub.publish(
                    TranscriptEvent(role="you", text=transcript, partial=False)
                )
                # Hold auto-reply while Joe is busy or echo gate is still settling.
                if self._talking or self._busy or now < self._echo_gate_until:
                    print(
                        "[session] logged turn.end; skip auto (busy/echo gate)",
                        flush=True,
                    )
                    return
                if not self._auto_reply:
                    print(
                        "[session] turn.end logged — auto off "
                        "(enable Auto-reply for pause replies)",
                        flush=True,
                    )
                    if now - self._last_auto_off_hint_at > 8.0:
                        self._last_auto_off_hint_at = now
                        await hub.publish(
                            TranscriptEvent(
                                role="status",
                                text="auto off — enable Auto-reply",
                            )
                        )
                    return
                if now - self._last_auto_start < AUTO_REPLY_COOLDOWN_S:
                    return
                confirm_s = self._turn_confirm_s()
                print(
                    f"[session] turn.end — waiting {confirm_s:.1f}s "
                    "to confirm pause"
                    + (" (loopback)" if self._mic.is_loopback else ""),
                    flush=True,
                )
                await hub.publish(
                    TranscriptEvent(
                        role="status",
                        text=f"waiting — {confirm_s:.1f}s pause",
                    )
                )
                self._schedule_auto_reply()

            await self._speaker.start()
            await self._mic.start()
            # Agent is on at boot — capture must not start muted.
            self._mic.set_muted(False)
            print("[mic] ready (unmuted)", flush=True)

            reply_task = asyncio.create_task(
                self._reply_loop(), name="reply-loop"
            )
            stt_fail_streak = 0
            try:
                while not self._stop.is_set():
                    try:
                        self._stt_restart.clear()
                        self._stt_restart_reason = ""
                        use_cartesia_stt = (
                            self._stt_backend == "cartesia"
                            and allow_cartesia_stt_env()
                        )
                        active_stt: STTEngine | None = None
                        if use_cartesia_stt:
                            # Explicit opt-in only — burns Cartesia Speech-to-Text tokens.
                            permit_cloud_stt(True)
                            print(
                                "[stt] starting Cartesia Ink (BILLS STT TOKENS)",
                                flush=True,
                            )
                            async with AsyncCartesia(
                                api_key=self.cartesia_api_key
                            ) as cartesia_stt:
                                stt = InkSTT(
                                    cartesia_stt,
                                    on_partial=on_partial,
                                    on_turn_end=on_turn_end,
                                    allow_cloud_stt=True,
                                )
                                active_stt = stt
                                async with stt:
                                    await self._run_stt_session(stt)
                        else:
                            if self._stt_backend == "cartesia":
                                print(
                                    "[stt] REFUSING Cartesia Ink mid-run — "
                                    "ALLOW_CARTESIA_STT missing; falling back to local",
                                    flush=True,
                                )
                                self._stt_backend = "local"
                                _save_stt_backend("local")
                            permit_cloud_stt(False)
                            print(
                                "[stt] starting local Whisper "
                                "(Cartesia Ink unreachable)",
                                flush=True,
                            )
                            stt = LocalWhisperSTT(
                                on_partial=on_partial,
                                on_turn_end=on_turn_end,
                                # Stereo Mix: finalize sooner on short pauses.
                                end_silence_ms=(
                                    400.0 if self._mic.is_loopback else None
                                ),
                            )
                            active_stt = stt
                            async with stt:
                                await self._run_stt_session(stt)
                        if self._stop.is_set():
                            break
                        reason = self._stt_restart_reason or "session-ended"
                        intentional = reason in ("turn-on", "backend-switch")
                        worker_err = getattr(active_stt, "last_error", None)
                        if intentional:
                            stt_fail_streak = 0
                            print(
                                f"[stt] restarting ({reason})",
                                flush=True,
                            )
                            await hub.publish(
                                TranscriptEvent(
                                    role="status",
                                    text="STT restarting…",
                                )
                            )
                            await asyncio.sleep(0.25)
                            continue

                        stt_fail_streak += 1
                        delay = min(
                            20.0, 0.75 * (2 ** min(stt_fail_streak - 1, 4))
                        )
                        detail = worker_err or reason
                        msg = (
                            f"STT dropped ({detail}); retry in {delay:.1f}s"
                            if stt_fail_streak > 1
                            else f"STT dropped ({detail}); reconnecting…"
                        )
                        print(f"[stt] {msg}", flush=True)
                        await hub.publish(
                            TranscriptEvent(
                                role="error" if stt_fail_streak > 1 else "status",
                                text=msg,
                            )
                        )
                        try:
                            await asyncio.wait_for(
                                self._stop.wait(), timeout=delay
                            )
                            break
                        except asyncio.TimeoutError:
                            continue
                    except Exception as exc:  # noqa: BLE001
                        stt_fail_streak += 1
                        msg = str(exc)
                        delay = min(
                            20.0, 1.0 * (2 ** min(stt_fail_streak - 1, 4))
                        )
                        print(
                            f"[stt] connection failed: {msg} "
                            f"(retry in {delay:.1f}s)",
                            flush=True,
                        )
                        await hub.publish(
                            TranscriptEvent(
                                role="error",
                                text=f"STT down: {msg}",
                            )
                        )
                        try:
                            await asyncio.wait_for(
                                self._stop.wait(), timeout=delay
                            )
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
                if self._local_llm is not None:
                    await self._local_llm.close()
                self._local_llm = None
                self._cursor_llm = None
                await cursor_llm.close()

    async def _run_stt_session(self, stt: STTEngine) -> None:
        """Feed mic until stop or restart requested."""
        feed_task = asyncio.create_task(self._feed_stt(stt), name="feed-stt")
        stop_wait = asyncio.create_task(self._stop.wait(), name="stop-wait")
        restart_wait = asyncio.create_task(
            self._stt_restart.wait(), name="stt-restart-wait"
        )
        try:
            await asyncio.wait(
                {feed_task, stop_wait, restart_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
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

    async def _feed_stt(self, stt: STTEngine) -> None:
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
                    or "worker stopped" in lower
                )
                if fatal:
                    self._request_stt_restart("send-failed")
                    return
                await asyncio.sleep(0.05)

    async def _reply_loop(self) -> None:
        while not self._stop.is_set():
            payload = await self._pending_turns.get()
            if not self._talking:
                continue
            async with self._turn_lock:
                if not self._talking:
                    continue
                self._busy = True
                llm = self._llm
                if llm is None:
                    await hub.publish(
                        TranscriptEvent(
                            role="error",
                            text="LLM not ready — restart the app.",
                        )
                    )
                    self._busy = False
                    continue
                print(f"[agent] thinking… ({self._llm_status_label()})", flush=True)
                engine = self._tts_backend
                llm_tag = "local" if self._llm_backend == "local" else "cursor"
                await hub.publish(
                    TranscriptEvent(
                        role="status",
                        text=(
                            f"speaking… ({llm_tag} / "
                            f"{'local' if engine == 'local' else 'cloud'} TTS)"
                        ),
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
                        # Captions: clean speech. Audio path: keep Cartesia SSML / cues.
                        caption = strip_tags(piece)
                        if caption:
                            await hub.publish(
                                TranscriptEvent(
                                    role="agent",
                                    text=caption,
                                    partial=True,
                                )
                            )
                        if piece.strip():
                            yield piece
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
                            if quiet_for >= self._turn_confirm_s() * 0.75 and self._talking:
                                self._mic.set_muted(True)
                                filler_s = await self._fillers.enqueue(
                                    self._speaker,
                                    backend=self._tts_backend,
                                    user_text=last_caller_utterance(fresh),
                                    pinned=pinned,
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
