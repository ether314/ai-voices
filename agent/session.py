"""Turn orchestration: mic → STT → button-started continuous Joe replies → TTS."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from cartesia import AsyncCartesia

from agent.audio import MicCapture, SpeakerPlayback, list_audio_devices
from agent.llm import CursorVoiceLLM
from agent.stt import InkSTT
from agent.transcript_hub import TranscriptEvent, hub
from agent.tts import DEFAULT_VOICE_ID, SonicTTS

ECHO_COOLDOWN_S = 0.15


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
        self.voice_id = voice_id
        self.cursor_model = cursor_model
        self.cwd = cwd or os.getcwd()
        self.echo_cooldown_s = echo_cooldown_s

        self._mic = MicCapture()
        self._speaker = SpeakerPlayback()
        self._busy = False
        self._talking = False
        self._reply_after_busy = False
        self._turn_lock = asyncio.Lock()
        self._pending_turns: asyncio.Queue[str] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._active_speak_task: asyncio.Task[None] | None = None

    @property
    def talking(self) -> bool:
        return self._talking

    def _listening_blocked(self) -> bool:
        return self._busy or self._speaker.playing or self._turn_lock.locked()

    async def _queue_history_reply(self) -> bool:
        conversation = hub.conversation_text().strip()
        if not conversation:
            return False
        # Drop stale queued prompts — always use freshest history when we run.
        while not self._pending_turns.empty():
            try:
                self._pending_turns.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._pending_turns.put(conversation)
        return True

    async def start_talking(self) -> dict[str, str | bool]:
        """Click Respond: enter talk mode and always queue a fresh spoken reply."""
        conversation = hub.conversation_text().strip()
        if not conversation:
            await hub.publish(
                TranscriptEvent(
                    role="status",
                    text="No transcript yet — wait for speech, then click Respond.",
                )
            )
            return {"ok": False, "talking": False, "error": "No transcript yet."}

        was_talking = self._talking
        self._talking = True
        # If a prior reply got stuck, allow a new one to be queued behind it.
        if self._busy:
            self._reply_after_busy = True
        queued = await self._queue_history_reply()
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=(
                    "TALKING_ON — Joe will keep speaking until you click Stop."
                    if not was_talking
                    else "TALKING_ON — keep-talking loop refreshed from latest transcript."
                ),
            )
        )
        if not queued:
            if not was_talking:
                self._talking = False
            return {"ok": False, "talking": self._talking, "error": "No transcript yet."}
        print("[session] Respond clicked — reply queued", flush=True)
        return {"ok": True, "talking": True}

    async def stop_talking(self) -> dict[str, str | bool]:
        """Stop immediately: silence speakers, cancel TTS, clear queue."""
        print("[session] Stop — cutting audio now", flush=True)
        self._talking = False
        self._reply_after_busy = False
        while not self._pending_turns.empty():
            try:
                self._pending_turns.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Hard-cut speaker output right away.
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
        await hub.publish(
            TranscriptEvent(
                role="status",
                text="TALKING_OFF — stopped. Click Respond as Joe to start again.",
            )
        )
        return {"ok": True, "talking": False}

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
                text=f"Input set to device {self._mic.device}",
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
                text=f"Output set to device {self._speaker.device}",
            )
        )
        return {"ok": True, "device": int(self._speaker.device or device)}

    async def run(self) -> None:
        print(
            "Voice agent ready. Speak near the mic.\n"
            "Click Respond as Joe — he keeps talking until you click Stop.\n"
            "http://localhost:7860/\n"
            "Ctrl+C to stop.\n",
            flush=True,
        )
        await hub.publish(
            TranscriptEvent(
                role="status",
                text=(
                    "Listening only. Click “Respond as Joe” and he keeps speaking "
                    "until you click Stop."
                ),
            )
        )

        async with AsyncCartesia(api_key=self.cartesia_api_key) as cartesia:
            llm = CursorVoiceLLM(
                api_key=self.cursor_api_key,
                model=self.cursor_model,
                cwd=self.cwd,
            )
            await llm.start()

            async def on_tts_audio(pcm: bytes) -> None:
                # Drop late chunks the moment Stop flips talking off.
                if not self._talking:
                    return
                await self._speaker.enqueue(pcm)

            tts = SonicTTS(
                cartesia,
                voice_id=self.voice_id,
                on_audio=on_tts_audio,
            )

            last_live_print = 0.0

            async def on_partial(transcript: str) -> None:
                nonlocal last_live_print
                display = transcript.replace("\n", " ")
                now = asyncio.get_running_loop().time()
                if now - last_live_print >= 0.25:
                    print(f"[live] {display}", flush=True)
                    last_live_print = now
                await hub.publish(
                    TranscriptEvent(role="you", text=transcript, partial=True)
                )

            async def on_turn_end(transcript: str) -> None:
                if transcript is None or transcript == "" or not transcript.strip():
                    return
                print(f"\n[you] {transcript}", flush=True)
                await hub.publish(
                    TranscriptEvent(role="you", text=transcript, partial=False)
                )
                if not self._talking:
                    return
                if self._listening_blocked():
                    self._reply_after_busy = True
                    return
                await self._queue_history_reply()

            stt = InkSTT(
                cartesia,
                on_partial=on_partial,
                on_turn_end=on_turn_end,
            )

            await self._speaker.start()
            await self._mic.start()

            try:
                async with stt:
                    feed_task = asyncio.create_task(
                        self._feed_stt(stt), name="feed-stt"
                    )
                    reply_task = asyncio.create_task(
                        self._reply_loop(llm, tts), name="reply-loop"
                    )
                    await self._stop.wait()
                    feed_task.cancel()
                    reply_task.cancel()
                    await asyncio.gather(feed_task, reply_task, return_exceptions=True)
            finally:
                await self._mic.stop()
                await self._speaker.stop()
                await llm.close()

    async def _feed_stt(self, stt: InkSTT) -> None:
        async for chunk in self._mic.chunks():
            if self._stop.is_set():
                break
            await stt.send_audio(chunk)

    async def _reply_loop(self, llm: CursorVoiceLLM, tts: SonicTTS) -> None:
        while not self._stop.is_set():
            payload = await self._pending_turns.get()
            if not self._talking:
                continue
            async with self._turn_lock:
                if not self._talking:
                    continue
                self._busy = True
                print("[agent] thinking…", flush=True)
                await hub.publish(
                    TranscriptEvent(
                        role="status",
                        text="Joe is responding…",
                    )
                )
                fresh = hub.conversation_text().strip() or payload
                collected: list[str] = []

                async def text_stream():
                    async for piece in llm.stream_history_reply(fresh):
                        if not self._talking:
                            break
                        print(piece, end="", flush=True)
                        collected.append(piece)
                        await hub.publish(
                            TranscriptEvent(
                                role="agent", text=piece, partial=True
                            )
                        )
                        yield piece
                    print(flush=True)

                speak_task: asyncio.Task[None] | None = None
                try:
                    if self._talking:
                        self._speaker.arm()
                        speak_task = asyncio.create_task(
                            tts.speak_stream(
                                text_stream(),
                                should_stop=lambda: not self._talking,
                            ),
                            name="tts-speak",
                        )
                        self._active_speak_task = speak_task
                        try:
                            await asyncio.wait_for(speak_task, timeout=90.0)
                        except asyncio.CancelledError:
                            print("\n[session] speak cancelled", flush=True)
                            self._speaker.clear()
                            # Do not re-raise — keep reply loop alive for next Respond.
                        if self._talking:
                            await self._speaker.wait_drained()
                        else:
                            self._speaker.clear()
                    full = "".join(collected)
                    if full and self._talking:
                        await hub.publish(
                            TranscriptEvent(
                                role="agent", text=full, partial=False
                            )
                        )
                    elif self._talking and not full:
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
                    await asyncio.sleep(self.echo_cooldown_s)
                    self._busy = False
                    self._mic.set_muted(False)
                    self._reply_after_busy = False
                    if self._talking:
                        print("[session] keep talking — queue next chunk", flush=True)
                        await self._queue_history_reply()

    def request_stop(self) -> None:
        self._talking = False
        self._stop.set()
