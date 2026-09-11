"""Mic capture and speaker playback at 16 kHz pcm_s16le."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
# ~100 ms frames
FRAME_SAMPLES = SAMPLE_RATE // 10


def _device_index(env_name: str) -> Optional[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _match_device_by_name(needle: str, *, kind: str = "input") -> Optional[int]:
    """Resolve a device by case-insensitive name substring (stable across re-plugs)."""
    needle_l = needle.lower().strip()
    if not needle_l:
        return None
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    scored: list[tuple[int, int]] = []
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get(channel_key) or 0) < 1:
            continue
        name = str(dev.get("name") or "")
        lower = name.lower()
        if "mapper" in lower or "primary sound" in lower:
            continue
        if needle_l not in lower:
            continue
        score = 100 if lower.startswith(needle_l) else 50
        score += len(needle_l)
        # Input: prefer WDM-KS Stereo Mix. Output: prefer MME/DirectSound (16 kHz OK).
        if kind == "output":
            try:
                host = str(
                    sd.query_hostapis()[int(dev.get("hostapi", 0))]["name"]
                ).lower()
            except Exception:  # noqa: BLE001
                host = ""
            if host == "mme":
                score += 40
            elif "directsound" in host:
                score += 25
            elif "wasapi" in host:
                score -= 20
            elif "wdm-ks" in host:
                score -= 30
            if "hdmi" in lower or "nvidia" in lower:
                score -= 100
        else:
            score += _hostapi_bonus(dev)
            if "stereo mix" in lower:
                score += 80
        scored.append((score, i))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def resolve_named_or_index(env_name: str, *, kind: str = "input") -> Optional[int]:
    """AUDIO_*_DEVICE may be a numeric index or a name substring (e.g. Stereo Mix)."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    as_int = _device_index(env_name)
    if as_int is not None:
        try:
            info = sd.query_devices(as_int)
            ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
            if int(info.get(ch_key) or 0) > 0:
                return as_int
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[audio] {env_name}={as_int} is missing or wrong kind; "
            f"falling back to name/auto pick",
            flush=True,
        )
        return None
    matched = _match_device_by_name(raw, kind=kind)
    if matched is not None:
        print(
            f"[audio] {env_name}={raw!r} -> {describe_device(matched, kind=kind)}",
            flush=True,
        )
        return matched
    print(f"[audio] {env_name}={raw!r} matched no {kind} device", flush=True)
    return None


def _input_gain() -> float:
    raw = os.getenv("AUDIO_INPUT_GAIN", "5.0").strip()
    try:
        return max(0.5, min(float(raw), 12.0))
    except ValueError:
        return 5.0


def _output_gain() -> float:
    raw = os.getenv("AUDIO_OUTPUT_GAIN", "2.5").strip()
    try:
        return max(0.5, min(float(raw), 8.0))
    except ValueError:
        return 2.5


def pick_speaker_output_device() -> Optional[int]:
    """Prefer Realtek/PC speakers over GPU HDMI / headset outputs."""
    devices = sd.query_devices()
    scored: list[tuple[int, int]] = []
    for i, dev in enumerate(devices):
        if int(dev.get("max_output_channels") or 0) < 1:
            continue
        name = str(dev.get("name") or "").lower()
        if "mapper" in name or "primary sound" in name:
            continue
        score = 0
        if "speakers" in name and "realtek" in name:
            score += 120
        elif "speakers" in name:
            score += 80
        if "realtek" in name:
            score += 40
        if "hdmi" in name or "nvidia" in name or "display" in name:
            score -= 100
        if "headset" in name or "headphones" in name or "hands-free" in name:
            score -= 40
        if score > 0:
            scored.append((score, i))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def resolve_output_device(explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return explicit
    env = resolve_named_or_index("AUDIO_OUTPUT_DEVICE", kind="output")
    if env is not None:
        return env
    return pick_speaker_output_device()


def _want_external_voice() -> bool:
    raw = os.getenv("AUDIO_EXTERNAL_VOICE", "").strip().lower()
    return raw in ("1", "true", "yes", "on", "loopback", "stereo")


def _prefer_room_mic() -> bool:
    """True when the user explicitly wants a physical mic over system loopback."""
    raw = os.getenv("AUDIO_PREFER_MIC", "").strip().lower()
    if raw in ("1", "true", "yes", "on", "mic", "room"):
        return True
    raw_ext = os.getenv("AUDIO_EXTERNAL_VOICE", "").strip().lower()
    return raw_ext in ("0", "false", "no", "off", "mic", "room")


def _is_loopback_name(name: str) -> bool:
    """True for Stereo Mix / virtual cable / system-audio capture devices."""
    lower = name.lower()
    return (
        "stereo mix" in lower
        or "what u hear" in lower
        or "loopback" in lower
        or "pc speaker" in lower
        or "cable output" in lower
        or "vb-audio" in lower
        or "voicemeeter" in lower
        or ("virtual" in lower and "cable" in lower)
    )


def _hostapi_bonus(dev: dict) -> int:
    try:
        host = str(sd.query_hostapis()[int(dev.get("hostapi", 0))]["name"]).lower()
    except Exception:  # noqa: BLE001
        return 0
    if "wdm-ks" in host:
        return 25
    if "wasapi" in host:
        return 10
    if host == "mme":
        return -15
    return 0


def score_input_device(index: int, *, prefer_system: Optional[bool] = None) -> int:
    """Higher = better default capture. System loopback wins unless room-mic forced."""
    try:
        dev = sd.query_devices(index)
    except Exception:  # noqa: BLE001
        return -10_000
    if int(dev.get("max_input_channels") or 0) < 1:
        return -10_000
    name = str(dev.get("name") or "")
    lower = name.lower()
    if "mapper" in lower or "primary sound capture" in lower:
        return -10_000

    if prefer_system is None:
        prefer_system = not _prefer_room_mic()
    external = _want_external_voice() or prefer_system

    score = 0
    # System / livestream path (TikTok in browser, Discord, etc.).
    if "stereo mix" in lower or "what u hear" in lower:
        score += 320 if prefer_system else (200 if external else 80)
    elif _is_loopback_name(name):
        # PC Speaker / VB-Cable / VoiceMeeter — still system audio, slightly behind Stereo Mix.
        if "pc speaker" in lower:
            score += 260 if prefer_system else 30
        else:
            score += 300 if prefer_system else (180 if external else 60)
    else:
        # Room / laptop mics — only preferred when AUDIO_PREFER_MIC / EXTERNAL_VOICE=0.
        if "microphone array 2" in lower or "mic array 2" in lower:
            score += 80 if prefer_system else 150
        elif "microphone array 1" in lower or "mic array 1" in lower:
            score += 50 if prefer_system else 90
        elif "microphone array" in lower or "mic array" in lower:
            score += 20 if prefer_system else 40
        if "webcam" in lower or "camera" in lower or "onn 4k" in lower:
            score += 25 if prefer_system else 55
        if "realtek" in lower and "mic" in lower:
            score += 15 if prefer_system else 30

    if "headset" in lower or "hands-free" in lower or "bluetooth" in lower:
        score -= 80
    if "wireless headset" in lower:
        score -= 40
    score += _hostapi_bonus(dev)
    return score


def ranked_input_devices(*, prefer_system: Optional[bool] = None) -> list[tuple[int, int]]:
    """Return (score, device_index) for usable inputs, best first."""
    if prefer_system is None:
        prefer_system = not _prefer_room_mic()
    scored: list[tuple[int, int]] = []
    for i, _dev in enumerate(sd.query_devices()):
        s = score_input_device(i, prefer_system=prefer_system)
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    return scored


def pick_room_input_device() -> Optional[int]:
    """Prefer Stereo Mix / loopback for livestream audio; room mic if forced.

    Default: system loopback when present (TikTok / browser / Discord on speakers).
    Force room mic with AUDIO_PREFER_MIC=1 or AUDIO_EXTERNAL_VOICE=0.
    """
    ranked = ranked_input_devices()
    if not ranked:
        return None
    return ranked[0][1]


def resolve_input_device(explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return explicit
    env = resolve_named_or_index("AUDIO_INPUT_DEVICE", kind="input")
    if env is not None:
        return env
    return pick_room_input_device()


def describe_device(index: Optional[int], *, kind: str = "input") -> str:
    if index is None:
        default = sd.default.device
        index = int(default[0] if kind == "input" else default[1])
    info = sd.query_devices(index)
    return f"{index}: {info['name']}"


def find_loopback_devices() -> list[tuple[int, str]]:
    """All input devices that capture system / livestream audio."""
    found: list[tuple[int, str]] = []
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels") or 0) < 1:
            continue
        name = str(dev.get("name") or "")
        lower = name.lower()
        if "mapper" in lower or "primary sound capture" in lower:
            continue
        if _is_loopback_name(name):
            found.append((i, name))
    return found


def print_input_device_summary(selected: Optional[int] = None) -> None:
    """Log selectable inputs and which one is active (startup / switch)."""
    if selected is None:
        selected = resolve_input_device()
    loops = find_loopback_devices()
    print("[audio] input devices:", flush=True)
    for i, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels") or 0) < 1:
            continue
        name = str(dev.get("name") or f"Device {i}")
        lower = name.lower()
        if "mapper" in lower or "primary sound capture" in lower:
            continue
        mark = " <-- selected" if selected is not None and i == selected else ""
        tag = " [loopback / system audio]" if _is_loopback_name(name) else ""
        print(f"  {i}: {name}{tag}{mark}", flush=True)
    if loops:
        stereo = next((p for p in loops if "stereo mix" in p[1].lower()), loops[0])
        print(
            f"[audio] loopback available - for TikTok/browser livestream pick "
            f"Stereo Mix / cable (e.g. {stereo[0]}: {stereo[1]})",
            flush=True,
        )
    else:
        print(
            "[audio] NO loopback device found. Enable Stereo Mix "
            "(Sound settings -> Recording -> Show Disabled Devices -> "
            "enable Stereo Mix) or install VB-Audio Cable / VoiceMeeter "
            "and route TikTok to that virtual input.",
            flush=True,
        )


def _resample_mono(mono: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    if src_rate == dst_rate or mono.size == 0:
        return mono
    n_out = max(1, int(round(mono.shape[0] * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, num=mono.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    out = np.interp(x_new, x_old, mono.astype(np.float32))
    return np.clip(out, -32768, 32767).astype(np.int16)


def list_audio_devices(
    *,
    current_input: Optional[int] = None,
    current_output: Optional[int] = None,
) -> dict:
    """Return selectable input/output devices for the web UI."""
    devices = sd.query_devices()
    default_in, default_out = sd.default.device
    inputs: list[dict] = []
    outputs: list[dict] = []
    for i, dev in enumerate(devices):
        name = str(dev.get("name") or f"Device {i}")
        lower = name.lower()
        # Skip host wrappers that confuse users.
        if "mapper" in lower or "primary sound" in lower:
            continue
        is_loop = _is_loopback_name(name)
        label = f"{name} - loopback / system audio" if is_loop else name
        entry = {
            "index": i,
            "name": label,
            "hostapi": int(dev.get("hostapi", -1)),
            "loopback": is_loop,
        }
        if int(dev.get("max_input_channels") or 0) > 0:
            inputs.append(entry)
        if int(dev.get("max_output_channels") or 0) > 0:
            outputs.append(entry)

    # Loopback / system-audio devices first so livestream capture is obvious.
    inputs.sort(key=lambda e: (0 if e.get("loopback") else 1, e["index"]))

    cur_in = current_input if current_input is not None else resolve_input_device()
    cur_out = current_output if current_output is not None else resolve_output_device()
    if cur_in is None:
        cur_in = int(default_in)
    if cur_out is None:
        cur_out = int(default_out)

    has_loopback = any(e.get("loopback") for e in inputs)
    if has_loopback:
        hint = (
            "TikTok / browser livestream -> select Stereo Mix (or any "
            "'loopback / system audio'). Room phone near laptop -> Microphone Array. "
            "Joe mutes capture while speaking so he does not hear himself."
        )
    else:
        hint = (
            "No loopback / system-audio device found. Windows: Sound settings -> "
            "Recording -> right-click -> Show Disabled Devices -> enable Stereo Mix. "
            "Or install VB-Audio Cable / VoiceMeeter and select that virtual input."
        )

    return {
        "inputs": inputs,
        "outputs": outputs,
        "current_input": cur_in,
        "current_output": cur_out,
        "has_loopback": has_loopback,
        "hint": hint,
    }


class MicCapture:
    """Async iterator of mono pcm_s16le chunks from a room-capable mic."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
        device: Optional[int] = None,
        gain: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = resolve_input_device(device)
        self.gain = gain if gain is not None else _input_gain()
        # ~20s at 100ms frames; drop-oldest when STT falls behind.
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._muted = False
        self._dropped_frames = 0
        self._capture_rate = sample_rate
        self._loopback = False
        self._rms_ema = 0.0
        self._last_level_log = 0.0

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def is_loopback(self) -> bool:
        return self._loopback

    def set_muted(self, muted: bool) -> None:
        was = self._muted
        self._muted = bool(muted)
        if was != self._muted:
            print(
                f"[mic] {'muted' if self._muted else 'unmuted'}",
                flush=True,
            )

    def _enqueue_pcm(self, pcm: bytes) -> None:
        """Drop oldest frames if full so PortAudio never raises on the event loop."""
        q = self._queue
        while q.full():
            try:
                q.get_nowait()
                self._dropped_frames += 1
                if self._dropped_frames % 50 == 1:
                    print(
                        f"[mic] audio backlog - dropped {self._dropped_frames} frames",
                        flush=True,
                    )
            except asyncio.QueueEmpty:
                break
        try:
            q.put_nowait(pcm)
        except asyncio.QueueFull:
            pass

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        del frames, time_info, status
        if self._loop is None:
            return
        # Echo / soft power-off: feed silence so Cartesia STT does not idle-timeout,
        # but never ship speaker bleed into the transcript.
        if self._muted:
            n = int(indata.shape[0])
            if self._capture_rate != self.sample_rate and self._capture_rate > 0:
                n = max(1, int(round(n * self.sample_rate / self._capture_rate)))
            silence = np.zeros(n, dtype=np.int16)
            now = time.monotonic()
            if now - self._last_level_log >= 5.0:
                self._last_level_log = now
                print("[mic] muted — silence keepalive to STT", flush=True)
            self._loop.call_soon_threadsafe(self._enqueue_pcm, silence.tobytes())
            return
        # Average all channels: array ch0 is often beamformed toward the laptop
        # user and will reject a phone / other speaker across the desk.
        if indata.shape[1] <= 1:
            mono = np.ascontiguousarray(indata[:, 0])
        else:
            mixed = np.mean(indata.astype(np.float32), axis=1)
            mono = np.clip(mixed, -32768, 32767).astype(np.int16)
        if self._capture_rate != self.sample_rate:
            mono = _resample_mono(mono, self._capture_rate, self.sample_rate)
        if self.gain != 1.0:
            boosted = mono.astype(np.float32) * self.gain
            mono = np.clip(boosted, -32768, 32767).astype(np.int16)
        level = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2))) if mono.size else 0.0
        self._rms_ema = (0.9 * self._rms_ema) + (0.1 * level)
        now = time.monotonic()
        if now - self._last_level_log >= 5.0:
            self._last_level_log = now
            print(f"[mic] level rms~{self._rms_ema:.0f}", flush=True)
        self._loop.call_soon_threadsafe(self._enqueue_pcm, mono.tobytes())

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        print_input_device_summary(self.device)
        await self._open_stream()

    async def _open_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        candidates: list[Optional[int]] = [self.device]
        for alt in (
            resolve_input_device(),
            pick_room_input_device(),
        ):
            if alt is not None and alt not in candidates:
                candidates.append(alt)
        # Prefer ranked inputs (Stereo Mix first) over stale hardcoded indices.
        for _score, alt in ranked_input_devices():
            if alt not in candidates:
                candidates.append(alt)

        last_err: Exception | None = None
        loop = asyncio.get_running_loop()
        for device in candidates:
            if device is None:
                continue
            try:
                info = sd.query_devices(device)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
            name = str(info.get("name") or "")
            max_in = max(1, min(int(info.get("max_input_channels") or 1), 4))
            native = int(float(info.get("default_samplerate") or SAMPLE_RATE))
            rates: list[int] = []
            for rate in (SAMPLE_RATE, native, 48000, 44100):
                if rate not in rates:
                    rates.append(rate)
            for rate in rates:
                channels = max_in
                while channels >= 1:
                    blocksize = max(int(rate) // 10, 256)

                    def _open(
                        dev: int = device,
                        sr: int = rate,
                        ch: int = channels,
                        bs: int = blocksize,
                    ) -> sd.InputStream:
                        stream = sd.InputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype=DTYPE,
                            blocksize=bs,
                            device=dev,
                            callback=self._callback,
                        )
                        stream.start()
                        return stream

                    try:
                        stream = await asyncio.wait_for(
                            loop.run_in_executor(None, _open),
                            timeout=3.0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        channels -= 1
                        continue

                    self._stream = stream
                    self.device = device
                    self._capture_rate = rate
                    self._loopback = _is_loopback_name(name)
                    print(
                        f"Mic: {describe_device(self.device, kind='input')} "
                        f"(gain={self.gain:g}, ch={channels}, "
                        f"capture={rate}Hz->{self.sample_rate}Hz"
                        f"{', loopback' if self._loopback else ''})",
                        flush=True,
                    )
                    return
        raise RuntimeError(f"Could not open any input device: {last_err}")

    async def set_device(self, device: int) -> None:
        """Switch mic device without ending the chunk iterator."""
        self.device = int(device)
        if self._loop is None:
            return
        await self._open_stream()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def chunks(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


class SpeakerPlayback:
    """Stream pcm_s16le to the default (or configured) output device."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        device: Optional[int] = None,
        gain: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = resolve_output_device(device)
        self.gain = gain if gain is not None else _output_gain()
        self._buffer: Deque[np.ndarray] = deque()
        self._buf_lock = threading.Lock()
        self._playing = False
        self._accept = True
        self._drain = asyncio.Event()
        self._drain.set()
        self._stream: sd.OutputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_samples = 0

    @property
    def playing(self) -> bool:
        with self._buf_lock:
            return self._playing or self._pending_samples > 0

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        del time_info, status
        outdata.fill(0)
        filled = 0
        drained = False
        with self._buf_lock:
            if not self._accept:
                self._playing = False
                drained = True
            while self._accept and filled < frames and self._buffer:
                chunk = self._buffer[0]
                need = frames - filled
                take = min(need, chunk.shape[0])
                outdata[filled : filled + take, 0] = chunk[:take]
                if take < chunk.shape[0]:
                    self._buffer[0] = chunk[take:]
                else:
                    self._buffer.popleft()
                filled += take
                self._pending_samples = max(0, self._pending_samples - take)

            if self._pending_samples == 0 and not self._buffer:
                self._playing = False
                drained = True
        if drained and self._loop is not None:
            self._loop.call_soon_threadsafe(self._drain.set)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._open_stream()

    async def _open_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._buf_lock:
            self._buffer.clear()
            self._pending_samples = 0
            self._playing = False
            self._accept = True
        self._drain.set()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=FRAME_SAMPLES,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        print(
            f"Speakers: {describe_device(self.device, kind='output')} "
            f"(gain={self.gain:g})",
            flush=True,
        )

    async def set_device(self, device: int) -> None:
        """Switch speaker device while the agent keeps running."""
        self.device = int(device)
        if self._loop is None:
            return
        await self._open_stream()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._buf_lock:
            self._buffer.clear()
            self._pending_samples = 0
            self._playing = False
        self._drain.set()

    async def enqueue(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._buf_lock:
            if not self._accept:
                return
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if audio.size == 0:
            return
        if self.gain != 1.0:
            audio = np.clip(audio * self.gain, -32768, 32767)
        audio_i16 = audio.astype(np.int16)
        with self._buf_lock:
            if not self._accept:
                return
            self._buffer.append(audio_i16)
            self._pending_samples += audio_i16.size
            self._playing = True
        self._drain.clear()

    def arm(self) -> None:
        """Allow audio again after a Stop clear (before next speak)."""
        with self._buf_lock:
            self._accept = True

    def clear(self) -> None:
        """Immediately silence playback (Stop button)."""
        with self._buf_lock:
            self._accept = False
            self._buffer.clear()
            self._pending_samples = 0
            self._playing = False
        self._drain.set()
        # Abort the current PortAudio buffer so Stop is truly instant.
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
                stream.start()
            except Exception:  # noqa: BLE001
                pass

    async def wait_drained(self) -> None:
        # Settle so the last PortAudio callback can clear the buffer.
        while self.playing:
            try:
                await asyncio.wait_for(self._drain.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                with self._buf_lock:
                    if self._pending_samples == 0 and not self._buffer:
                        break
        with self._buf_lock:
            self._playing = False
        self._drain.set()
