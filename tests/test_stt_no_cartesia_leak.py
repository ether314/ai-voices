"""Prove local STT never opens Cartesia Ink (no Speech-to-Text websocket)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force local before importing session helpers.
os.environ["STT_BACKEND"] = "local"
os.environ.pop("ALLOW_CARTESIA_STT", None)


def test_load_stt_backend_defaults_local() -> None:
    from agent.session import _load_stt_backend

    assert _load_stt_backend() == "local"


def test_ink_stt_refuses_without_permit() -> None:
    from agent.stt import InkSTT, InkSTTBlockedError, permit_cloud_stt

    permit_cloud_stt(False)
    client = MagicMock()
    try:
        InkSTT(client, allow_cloud_stt=False)
        raise AssertionError("InkSTT must refuse without allow_cloud_stt")
    except InkSTTBlockedError:
        pass
    try:
        # Even with constructor flag, module latch must be set.
        InkSTT(client, allow_cloud_stt=True)
        raise AssertionError("InkSTT must refuse when latch is False")
    except InkSTTBlockedError:
        pass


def test_set_stt_rejects_cartesia_without_gates() -> None:
    async def _run() -> None:
        from agent.session import VoiceSession

        session = VoiceSession(
            cartesia_api_key="sk_test",
            cursor_api_key="cursor_test",
        )
        denied = await session.set_stt_settings(backend="cartesia", confirm_cost=False)
        assert denied.get("ok") is False
        denied2 = await session.set_stt_settings(backend="cartesia", confirm_cost=True)
        assert denied2.get("ok") is False
        assert session._stt_backend == "local"

    asyncio.run(_run())


def test_local_whisper_never_calls_ink_websocket() -> None:
    async def _run() -> None:
        import agent.stt as stt_mod
        from agent.local_stt import LocalWhisperSTT

        calls = {"ink_enter": 0, "ws": 0}

        async def boom_enter(self):  # noqa: ANN001
            calls["ink_enter"] += 1
            raise AssertionError("InkSTT.__aenter__ must not run under local STT")

        stt_mod.InkSTT.__aenter__ = boom_enter  # type: ignore[method-assign]
        stt_mod.permit_cloud_stt(False)

        # Patch auto_finalize.websocket on any client that might appear.
        from cartesia import AsyncCartesia

        orig_init = AsyncCartesia.__init__

        def patched_init(self, *a, **k):  # noqa: ANN001
            orig_init(self, *a, **k)
            af = self.stt.auto_finalize
            orig_ws = af.websocket

            def boom_ws(*args, **kwargs):  # noqa: ANN001
                calls["ws"] += 1
                raise AssertionError("stt.auto_finalize.websocket must not open")

            af.websocket = boom_ws  # type: ignore[method-assign]
            self._orig_ws = orig_ws

        AsyncCartesia.__init__ = patched_init  # type: ignore[method-assign]

        try:
            # Construct TTS client the way session does — must not touch STT.
            async with AsyncCartesia(api_key=os.getenv("CARTESIA_API_KEY", "sk_x")) as _:
                pass
        except Exception:
            # No real key / network — init patch still counted ws=0 if never called.
            pass

        partials: list[str] = []
        ends: list[str] = []

        async def on_partial(t: str) -> None:
            partials.append(t)

        async def on_turn_end(t: str) -> None:
            ends.append(t)

        stt = LocalWhisperSTT(on_partial=on_partial, on_turn_end=on_turn_end)
        async with stt:
            silence = b"\x00\x00" * 1600  # 100ms @ 16kHz mono s16le
            for _ in range(30):
                await stt.send_audio(silence)
                await asyncio.sleep(0.05)

        assert calls["ink_enter"] == 0, calls
        assert calls["ws"] == 0, calls
        print(
            "PASS local Whisper path: ink_enter=0 websocket=0 "
            f"(partials={len(partials)} ends={len(ends)})"
        )

    asyncio.run(_run())


if __name__ == "__main__":
    test_load_stt_backend_defaults_local()
    print("PASS config local")
    test_ink_stt_refuses_without_permit()
    print("PASS InkSTT refuse")
    test_set_stt_rejects_cartesia_without_gates()
    print("PASS API refuse cartesia without ALLOW+confirm")
    test_local_whisper_never_calls_ink_websocket()
    print("ALL PASS")
