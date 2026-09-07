"""Mic capture and speaker playback at 16 kHz pcm_s16le."""

from __future__ import annotations

import asyncio
import os
import threading
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
    return int(raw)


def _input_gain() -> float:
    raw = os.getenv("AUDIO_INPUT_GAIN", "3.0").strip()
    try:
        return max(0.5, min(float(raw), 12.0))
    except ValueError:
        return 3.0


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
    env = _device_index("AUDIO_OUTPUT_DEVICE")
    if env is not None:
        return env
    return pick_speaker_output_device()


def pick_room_input_device() -> Optional[int]:
    """Prefer a room/array mic so a phone on speaker can be heard.

    Headset boom mics and default webcam mics often only hear *you*, not the
    other party on a handset/speakerphone across the desk.
    """
    devices = sd.query_devices()
    scored: list[tuple[int, int]] = []
    for i, dev in enumerate(devices):
        if int(dev.get("max_input_channels") or 0) < 1:
            continue
        name = str(dev.get("name") or "").lower()
        # Skip mappers / duplicates that are usually host wrappers.
        if "mapper" in name or "primary sound capture" in name:
            continue
        score = 0
        if "microphone array" in name or "mic array" in name:
            score += 100
        if "realtek" in name:
            score += 40
        if "stereo mix" in name or "what u hear" in name or "loopback" in name:
            # System loopback — not a physical phone; keep as last resort only.
            score -= 50
        if "headset" in name or "hands-free" in name or "bluetooth" in name:
            score -= 80
        if "webcam" in name or "camera" in name or "onn 4k" in name:
            score -= 30
        if score > 0:
            scored.append((score, i))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def resolve_input_device(explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return explicit
    env = _device_index("AUDIO_INPUT_DEVICE")
    if env is not None:
        return env
    return pick_room_input_device()


def describe_device(index: Optional[int], *, kind: str = "input") -> str:
    if index is None:
        default = sd.default.device
        index = int(default[0] if kind == "input" else default[1])
    info = sd.query_devices(index)
    return f"{index}: {info['name']}"


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
        entry = {
            "index": i,
            "name": name,
            "hostapi": int(dev.get("hostapi", -1)),
        }
        if int(dev.get("max_input_channels") or 0) > 0:
            inputs.append(entry)
        if int(dev.get("max_output_channels") or 0) > 0:
            outputs.append(entry)

    cur_in = current_input if current_input is not None else resolve_input_device()
    cur_out = current_output if current_output is not None else resolve_output_device()
    if cur_in is None:
        cur_in = int(default_in)
    if cur_out is None:
        cur_out = int(default_out)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "current_input": cur_in,
        "current_output": cur_out,
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
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._muted = False

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        del frames, time_info, status
        if self._muted or self._loop is None:
            return
        mono = np.ascontiguousarray(indata[:, 0])
        if self.gain != 1.0:
            boosted = mono.astype(np.float32) * self.gain
            mono = np.clip(boosted, -32768, 32767).astype(np.int16)
        pcm = mono.tobytes()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm)
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._open_stream()

    async def _open_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        print(
            f"Mic: {describe_device(self.device, kind='input')} "
            f"(gain={self.gain:g}) — phone on speaker near this mic",
            flush=True,
        )

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
