"""Local faster-whisper STT (no Cartesia Ink / cloud tokens)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import numpy as np

OnTurnEnd = Callable[[str], Awaitable[None]]
OnPartial = Callable[[str], Awaitable[None]]

DEFAULT_WHISPER_MODEL = "base.en"
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_CUDA_UNAVAILABLE_REASON: str | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_whisper_model() -> str:
    return (
        os.getenv("STT_WHISPER_MODEL", "").strip()
        or os.getenv("STT_MODEL", "").strip()
        or DEFAULT_WHISPER_MODEL
    )


def detect_stt_device() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper / CTranslate2."""
    forced = os.getenv("STT_DEVICE", "auto").strip().lower() or "auto"
    if forced == "cpu" or _CUDA_UNAVAILABLE_REASON:
        return "cpu", (os.getenv("STT_COMPUTE_TYPE", "").strip() or "int8")
    if forced == "cuda":
        device = "cuda"
    else:
        device = "cpu"
        # CTranslate2 may list CUDA even when cuBLAS DLLs are missing on Windows.
        cuda_ok = True
        try:
            import ctypes

            ctypes.WinDLL("cublas64_12.dll")
        except Exception:
            try:
                import ctypes

                ctypes.WinDLL("cublas64_11.dll")
            except Exception:
                cuda_ok = False
        if cuda_ok:
            try:
                import ctranslate2

                try:
                    types = ctranslate2.get_supported_compute_types("cuda")
                    if types:
                        device = "cuda"
                except Exception:
                    device = "cpu"
            except Exception:
                device = "cpu"

    if device == "cuda":
        compute = os.getenv("STT_COMPUTE_TYPE", "").strip() or "float16"
    else:
        compute = os.getenv("STT_COMPUTE_TYPE", "").strip() or "int8"
    return device, compute


def _cached_model(model_size: str, device: str | None = None) -> Any | None:
    for (size, dev, _ctype), model in _MODEL_CACHE.items():
        if size != model_size:
            continue
        if device is None or device == dev:
            return model
    return None


def _load_model(model_size: str, device: str, compute_type: str) -> Any:
    global _CUDA_UNAVAILABLE_REASON
    key = (model_size, device, compute_type)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    # Reuse any already-warmed copy of this model size.
    warmed = _cached_model(model_size)
    if warmed is not None:
        return warmed

    from faster_whisper import WhisperModel

    def _try_load(dev: str, ctype: str) -> Any:
        print(
            f"[stt] loading faster-whisper model={model_size} "
            f"device={dev} compute={ctype}",
            flush=True,
        )
        t0 = time.perf_counter()
        model = WhisperModel(model_size, device=dev, compute_type=ctype)
        # Construction can succeed while encode fails (e.g. missing cublas DLL).
        warmup = np.zeros(16_000, dtype=np.float32)
        list(
            model.transcribe(
                warmup,
                language="en",
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
            )[0]
        )
        print(
            f"[stt] model ready in {time.perf_counter() - t0:.1f}s "
            f"({dev}/{ctype})",
            flush=True,
        )
        _MODEL_CACHE[(model_size, dev, ctype)] = model
        return model

    if device == "cuda" and _CUDA_UNAVAILABLE_REASON:
        device, compute_type = "cpu", "int8"

    try:
        return _try_load(device, compute_type)
    except Exception as exc:  # noqa: BLE001
        if device == "cpu":
            raise
        _CUDA_UNAVAILABLE_REASON = str(exc)
        print(
            f"[stt] CUDA unavailable ({exc}); falling back to CPU int8",
            flush=True,
        )
        return _try_load("cpu", "int8")


def _safe_print(msg: str) -> None:
    """Windows consoles are often cp1252 — never let a log char kill the STT worker."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _pcm_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def _pcm_to_float32(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return samples
    return samples / 32768.0


def _transcribe_sync(model: Any, audio: np.ndarray) -> str:
    if audio.size < 1600:  # <100ms @ 16k
        return ""
    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        best_of=1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    parts: list[str] = []
    for seg in segments:
        text = (getattr(seg, "text", None) or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


class LocalWhisperSTT:
    """Chunked local STT with energy VAD → partials + silence turn ends."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        model_size: Optional[str] = None,
        on_partial: Optional[OnPartial] = None,
        on_turn_end: Optional[OnTurnEnd] = None,
        end_silence_ms: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.model_size = model_size or resolve_whisper_model()
        self.on_partial = on_partial
        self.on_turn_end = on_turn_end

        self._model: Any = None
        self._device = "cpu"
        self._compute_type = "int8"
        self._closed = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # VAD / buffering — Stereo Mix often never drops below an absolute gate
        # (bed noise / compression), so we also treat a drop from utterance peak
        # as silence. Default end-silence is shorter so sentence pauses finalize.
        self._speech_rms = _env_float("STT_SPEECH_RMS", 350.0)
        env_silence = _env_float("STT_END_SILENCE_MS", 480.0)
        silence_ms = (
            float(end_silence_ms) if end_silence_ms is not None else env_silence
        )
        self._end_silence_s = max(0.22, silence_ms / 1000.0)
        self._min_speech_s = _env_float("STT_MIN_SPEECH_MS", 280.0) / 1000.0
        self._partial_interval_s = (
            _env_float("STT_PARTIAL_INTERVAL_MS", 500.0) / 1000.0
        )
        self._max_buffer_s = _env_float("STT_MAX_BUFFER_S", 28.0)
        # Pause = RMS below peak * ratio (and above a tiny floor).
        self._silence_drop = _env_float("STT_SILENCE_DROP_RATIO", 0.42)
        self._noise_floor = 80.0
        self._peak_rms = 0.0

        self._buf = bytearray()
        self._in_speech = False
        self._speech_started_at = 0.0
        self._last_voice_at = 0.0
        self._quiet_samples = 0
        self._speech_samples = 0
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._pending_transcribe = False
        self._transcribing = False
        self._finalize_requested = False
        self._lock = asyncio.Lock()
        self.last_error: str | None = None

    @property
    def alive(self) -> bool:
        return (
            self._model is not None
            and not self._closed.is_set()
            and self._worker is not None
            and not self._worker.done()
        )

    @property
    def label(self) -> str:
        return f"local/{self.model_size}@{self._device}"

    async def __aenter__(self) -> "LocalWhisperSTT":
        self._closed = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._device, self._compute_type = detect_stt_device()
        self._model = await asyncio.to_thread(
            _load_model, self.model_size, self._device, self._compute_type
        )
        # Refresh device if CUDA fell back / cache reused a different device.
        for (size, device, compute), model in _MODEL_CACHE.items():
            if model is self._model and size == self.model_size:
                self._device, self._compute_type = device, compute
                break
        self._reset_utt()
        self.last_error = None
        self._worker = asyncio.create_task(self._worker_loop(), name="local-stt")
        _safe_print(f"[stt] local whisper connected ({self.label})")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed.is_set() or not pcm:
            return
        if self._worker is not None and self._worker.done():
            raise ConnectionError("local STT worker stopped")

        rms = _pcm_rms(pcm)
        now = time.monotonic()
        # Track quiet-floor for adaptive gate (Stereo Mix can be quieter).
        if rms < self._speech_rms * 0.55:
            self._noise_floor = (0.95 * self._noise_floor) + (0.05 * rms)
        gate = max(self._speech_rms, self._noise_floor * 2.8)

        async with self._lock:
            n_samples = len(pcm) // 2
            if not self._in_speech:
                if rms < gate:
                    return
                self._in_speech = True
                self._speech_started_at = now
                self._peak_rms = rms
                self._buf.clear()
                self._last_partial_text = ""
                self._last_partial_at = 0.0
                self._last_voice_at = now
                self._quiet_samples = 0
                self._speech_samples = n_samples
                self._finalize_requested = False
                self._buf.extend(pcm)
                return

            self._peak_rms = max(self._peak_rms, rms)
            # Hysteresis: leave "voice" after a drop from this utterance's peak.
            # Stereo Mix / livestream often never falls below the absolute start gate
            # because of bed noise — relative drop is what makes pauses finalize.
            stop_thr = max(
                gate * 0.55,
                self._noise_floor * 1.85,
                self._peak_rms * self._silence_drop,
                90.0,
            )

            if rms >= stop_thr:
                self._last_voice_at = now
                self._quiet_samples = 0
                self._speech_samples += n_samples
                self._finalize_requested = False
                self._buf.extend(pcm)
                max_bytes = int(self._max_buffer_s * self.sample_rate) * 2
                if len(self._buf) > max_bytes:
                    overflow = len(self._buf) - max_bytes
                    del self._buf[:overflow]
                speech_len = self._speech_samples / float(self.sample_rate)
                if (
                    speech_len >= self._min_speech_s
                    and now - self._last_partial_at >= self._partial_interval_s
                ):
                    self._pending_transcribe = True
            else:
                # Pause / quieter bed — count silence by audio duration (not wall
                # clock) so bursty capture still detects real pauses.
                self._buf.extend(pcm)
                self._quiet_samples += n_samples
                speech_len = self._speech_samples / float(self.sample_rate)
                silent_for = self._quiet_samples / float(self.sample_rate)
                if (
                    speech_len >= self._min_speech_s
                    and silent_for >= self._end_silence_s
                ):
                    self._finalize_requested = True
                    self._pending_transcribe = True

    async def flush_turn(self) -> None:
        """Best-effort finalize of the current utterance (e.g. on close)."""
        async with self._lock:
            text = self._last_partial_text
            had_speech = self._in_speech and bool(self._buf)
            pcm = bytes(self._buf) if had_speech else b""
            self._reset_utt()
        if not text and pcm and self._model is not None:
            try:
                text = await asyncio.to_thread(
                    _transcribe_sync, self._model, _pcm_to_float32(pcm)
                )
            except Exception:
                text = ""
        if text and self.on_turn_end is not None:
            await self.on_turn_end(text)

    async def close(self) -> None:
        if self._closed.is_set():
            return
        # Stop worker first so it cannot race a duplicate turn.end.
        self._closed.set()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        try:
            await self.flush_turn()
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"[stt] flush on close failed: {exc}")
        _safe_print("[stt] local whisper closed")

    def _reset_utt(self) -> None:
        self._buf.clear()
        self._in_speech = False
        self._speech_started_at = 0.0
        self._last_voice_at = 0.0
        self._quiet_samples = 0
        self._speech_samples = 0
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._pending_transcribe = False
        self._finalize_requested = False
        self._peak_rms = 0.0

    async def _worker_loop(self) -> None:
        try:
            while not self._closed.is_set():
                await asyncio.sleep(0.05)
                if self._transcribing:
                    continue
                async with self._lock:
                    need = self._pending_transcribe or self._finalize_requested
                    finalize = self._finalize_requested
                    pcm = bytes(self._buf) if need else b""
                    if need:
                        self._pending_transcribe = False
                if not need or not pcm:
                    continue
                self._transcribing = True
                try:
                    audio = _pcm_to_float32(pcm)
                    text = await asyncio.to_thread(
                        _transcribe_sync, self._model, audio
                    )
                except Exception as exc:  # noqa: BLE001
                    _safe_print(f"[stt] local decode error: {exc}")
                    text = ""
                finally:
                    self._transcribing = False

                if self._closed.is_set():
                    break
                if not text and finalize:
                    text = self._last_partial_text
                if not text:
                    if finalize:
                        async with self._lock:
                            self._reset_utt()
                    continue

                if finalize:
                    async with self._lock:
                        self._reset_utt()
                    # ASCII-only: Windows cp1252 consoles raise UnicodeEncodeError on ≥
                    # which previously killed this worker and caused endless reconnects.
                    _safe_print(
                        f"[stt] turn.end ({len(text)} chars) silence>="
                        f"{self._end_silence_s * 1000:.0f}ms"
                    )
                    if self.on_turn_end is not None:
                        try:
                            await self.on_turn_end(text)
                        except Exception as exc:  # noqa: BLE001
                            _safe_print(f"[stt] on_turn_end failed: {exc}")
                else:
                    if text != self._last_partial_text:
                        self._last_partial_text = text
                        self._last_partial_at = time.monotonic()
                        if self.on_partial is not None:
                            try:
                                await self.on_partial(text)
                            except Exception as exc:  # noqa: BLE001
                                _safe_print(f"[stt] on_partial failed: {exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if not self._closed.is_set():
                self.last_error = str(exc)
                _safe_print(f"[stt] local worker ended: {exc}")
