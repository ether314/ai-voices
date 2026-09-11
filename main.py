"""Local mic voice agent: Cartesia STT/TTS + Cursor Composer."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.session import VoiceSession
from agent.tts import DEFAULT_VOICE_ID
from agent.ui import start_ui


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(
            f"Missing {name}. Copy .env.example to .env and set your keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


async def _amain() -> None:
    load_dotenv()
    cartesia_key = _require_env("CARTESIA_API_KEY")
    cursor_key = _require_env("CURSOR_API_KEY")
    voice_id = os.getenv("CARTESIA_VOICE_ID", DEFAULT_VOICE_ID).strip() or DEFAULT_VOICE_ID
    model = os.getenv("CURSOR_MODEL", "gpt-5.6-luna").strip() or "gpt-5.6-luna"
    cwd = str(Path(__file__).resolve().parent)
    print(f"Cursor model: {model}", flush=True)

    session = VoiceSession(
        cartesia_api_key=cartesia_key,
        cursor_api_key=cursor_key,
        voice_id=voice_id,
        cursor_model=model,
        cwd=cwd,
    )

    loop = asyncio.get_running_loop()
    try:
        # SIGINT on Windows still raises KeyboardInterrupt around run_until_complete;
        # also register for Unix / Docker.
        for sig in ("SIGINT", "SIGTERM"):
            import signal

            if hasattr(signal, sig):
                loop.add_signal_handler(getattr(signal, sig), session.request_stop)
    except NotImplementedError:
        # Windows event loop often lacks add_signal_handler.
        pass

    ui_runner = await start_ui(
        host="127.0.0.1",
        port=7860,
        on_start=session.start_talking,
        on_stop=session.stop_talking,
        on_clear=session.clear_context,
        on_power_on=session.turn_on,
        on_power_off=session.turn_off,
        on_respond=session.start_talking,
        get_context=session.get_pinned_context,
        set_context=session.set_pinned_context,
        get_tts=session.get_tts_settings,
        set_tts=session.set_tts_settings,
        get_devices=session.get_audio_devices,
        set_input=session.set_input_device,
        set_output=session.set_output_device,
    )
    try:
        await session.run()
    except KeyboardInterrupt:
        session.request_stop()
    finally:
        await ui_runner.cleanup()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
