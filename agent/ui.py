"""Live transcript UI at http://localhost:7860/"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional

from aiohttp import web

from agent.transcript_hub import TranscriptHub, hub

RespondHandler = Callable[[], Awaitable[dict[str, Any]]]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice agent</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --you: #7dd3fc;
      --agent: #86efac;
      --muted: #94a3b8;
      --line: #243044;
      --live: #fbbf24;
      --btn: #22c55e;
      --btn-text: #052e16;
      --stop: #ef4444;
      --stop-text: #450a0a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1e3a5f 0%, var(--bg) 55%);
      color: #e2e8f0;
      min-height: 100vh;
    }
    header {
      padding: 1.25rem 1.5rem 0.75rem;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      background: rgba(15, 20, 25, 0.92);
      backdrop-filter: blur(8px);
      z-index: 2;
    }
    .top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }
    header h1 {
      margin: 0;
      font-size: 1.15rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    header p {
      margin: 0.35rem 0 0;
      color: var(--muted);
      font-size: 0.9rem;
      max-width: 36rem;
    }
    #status {
      display: inline-block;
      margin-left: 0.5rem;
      font-size: 0.75rem;
      color: #fbbf24;
    }
    #status.ok { color: #4ade80; }
    #respond {
      border: none;
      border-radius: 999px;
      padding: 0.85rem 1.35rem;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      background: var(--btn);
      color: var(--btn-text);
      box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.35);
      white-space: nowrap;
      user-select: none;
    }
    #respond.talking {
      background: var(--stop);
      color: var(--stop-text);
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.4);
    }
    #respond:hover { filter: brightness(1.06); }
    #btnHint {
      margin-top: 0.35rem;
      font-size: 0.75rem;
      color: var(--muted);
      text-align: right;
    }
    #live {
      margin-top: 0.85rem;
      padding: 0.85rem 1rem;
      border-radius: 12px;
      border: 1px solid #854d0e;
      background: #1c1917;
      min-height: 3.2rem;
    }
    #live .label {
      display: block;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--live);
      margin-bottom: 0.35rem;
    }
    #live .text {
      color: #fef3c7;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    #live.empty .text { color: var(--muted); }
    .devices {
      margin-top: 0.85rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    @media (max-width: 720px) {
      .devices { grid-template-columns: 1fr; }
    }
    .devices label {
      display: block;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }
    .devices select {
      width: 100%;
      background: var(--panel);
      color: #e2e8f0;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.65rem 0.75rem;
      font-size: 0.9rem;
    }
    #deviceHint {
      margin-top: 0.45rem;
      font-size: 0.75rem;
      color: var(--muted);
    }
    #log {
      max-width: 820px;
      margin: 0 auto;
      padding: 1.25rem 1.5rem 3rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .msg {
      padding: 0.85rem 1rem;
      border-radius: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg .who {
      display: block;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.35rem;
      opacity: 0.85;
    }
    .msg.you .who { color: var(--you); }
    .msg.agent .who { color: var(--agent); }
    .msg.status, .msg.error { opacity: 0.9; font-size: 0.9rem; }
    .msg.error { border-color: #7f1d1d; color: #fecaca; }
    .msg.partial { opacity: 0.7; border-style: dashed; }
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div>
        <h1>Voice agent <span id="status">connecting…</span></h1>
        <p>Click once to keep Joe in the call. Click <strong>Stop</strong> when you want silence again.</p>
      </div>
      <div>
        <button id="respond" type="button">Respond as Joe</button>
        <div id="btnHint">Click to start · click Stop to end</div>
      </div>
    </div>
    <div id="live" class="empty">
      <span class="label">Live caption</span>
      <div class="text">Waiting for speech…</div>
    </div>
    <div class="devices">
      <div>
        <label for="inputDevice">Microphone input</label>
        <select id="inputDevice"></select>
      </div>
      <div>
        <label for="outputDevice">Speaker output</label>
        <select id="outputDevice"></select>
      </div>
    </div>
    <div id="deviceHint">Choose which mic listens and which speakers Joe uses.</div>
  </header>
  <main id="log"></main>
  <script>
    const log = document.getElementById("log");
    const statusEl = document.getElementById("status");
    const liveBox = document.getElementById("live");
    const liveText = liveBox.querySelector(".text");
    const respondBtn = document.getElementById("respond");
    const btnHint = document.getElementById("btnHint");
    const inputDevice = document.getElementById("inputDevice");
    const outputDevice = document.getElementById("outputDevice");
    const deviceHint = document.getElementById("deviceHint");
    let lastPartialYou = null;
    let lastPartialAgent = null;
    let talking = false;
    let loadingDevices = false;

    function setLive(text) {
      if (text && text.trim()) {
        liveBox.classList.remove("empty");
        liveText.textContent = text;
      } else {
        liveBox.classList.add("empty");
        liveText.textContent = "Waiting for speech…";
      }
    }

    function fillSelect(sel, items, current) {
      sel.innerHTML = "";
      for (const d of items) {
        const opt = document.createElement("option");
        opt.value = String(d.index);
        opt.textContent = d.index + ": " + d.name;
        if (d.index === current) opt.selected = true;
        sel.appendChild(opt);
      }
    }

    async function loadDevices() {
      loadingDevices = true;
      try {
        const data = await fetch("/api/audio/devices").then((r) => r.json());
        fillSelect(inputDevice, data.inputs || [], data.current_input);
        fillSelect(outputDevice, data.outputs || [], data.current_output);
        deviceHint.textContent = "Choose which mic listens and which speakers Joe uses.";
      } catch (err) {
        deviceHint.textContent = "Could not load audio devices: " + err;
      } finally {
        loadingDevices = false;
      }
    }

    async function setDevice(kind, index) {
      deviceHint.textContent = "Switching " + kind + "…";
      try {
        const res = await fetch("/api/audio/" + kind, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device: Number(index) }),
        });
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          deviceHint.textContent = data.error || ("Failed to set " + kind);
          await loadDevices();
          return;
        }
        deviceHint.textContent = kind === "input"
          ? "Microphone updated."
          : "Speakers updated.";
      } catch (err) {
        deviceHint.textContent = String(err);
      }
    }

    inputDevice.addEventListener("change", () => {
      if (loadingDevices) return;
      setDevice("input", inputDevice.value);
    });
    outputDevice.addEventListener("change", () => {
      if (loadingDevices) return;
      setDevice("output", outputDevice.value);
    });

    function setTalkingUI(on) {
      talking = on;
      if (on) {
        respondBtn.classList.add("talking");
        respondBtn.textContent = "Stop";
        btnHint.textContent = "Click Stop to end continuous replies";
      } else {
        respondBtn.classList.remove("talking");
        respondBtn.textContent = "Respond as Joe";
        btnHint.textContent = "Click to start · click Stop to end";
      }
    }

    async function doStart() {
      btnHint.textContent = "Starting continuous replies…";
      try {
        const res = await fetch("/api/talk/start", { method: "POST" });
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not start";
          setTalkingUI(false);
          return;
        }
        setTalkingUI(true);
      } catch (err) {
        btnHint.textContent = String(err);
        setTalkingUI(false);
      }
    }

    async function doStop() {
      btnHint.textContent = "Stopping…";
      try {
        await fetch("/api/talk/stop", { method: "POST" });
      } catch (err) {
        btnHint.textContent = String(err);
      }
      setTalkingUI(false);
    }

    respondBtn.addEventListener("click", async () => {
      if (talking) {
        // One-click Stop
        await doStop();
        return;
      }
      await doStart();
    });

    function addOrUpdate(ev) {
      const text = ev.text || "";
      if (ev.role === "status" && text.includes("TALKING_ON")) setTalkingUI(true);
      if (ev.role === "status" && text.includes("TALKING_OFF")) setTalkingUI(false);

      if (ev.role === "you" && ev.partial) {
        setLive(ev.text);
        if (!lastPartialYou) {
          lastPartialYou = el("you", ev.text, true);
          log.appendChild(lastPartialYou);
        } else {
          lastPartialYou.querySelector(".body").textContent = ev.text;
        }
        scroll();
        return;
      }
      if (ev.role === "you" && !ev.partial) {
        setLive("");
        if (lastPartialYou) {
          lastPartialYou.classList.remove("partial");
          lastPartialYou.querySelector(".body").textContent = ev.text;
          lastPartialYou = null;
        } else {
          log.appendChild(el("you", ev.text, false));
        }
        scroll();
        return;
      }
      if (ev.role === "agent" && ev.partial) {
        if (!lastPartialAgent) {
          lastPartialAgent = el("agent", ev.text, true);
          log.appendChild(lastPartialAgent);
        } else {
          lastPartialAgent.querySelector(".body").textContent = ev.text;
        }
        scroll();
        return;
      }
      if (ev.role === "agent" && !ev.partial) {
        if (lastPartialAgent) {
          lastPartialAgent.classList.remove("partial");
          lastPartialAgent.querySelector(".body").textContent = ev.text;
          lastPartialAgent = null;
        } else {
          log.appendChild(el("agent", ev.text, false));
        }
        scroll();
        return;
      }
      log.appendChild(el(ev.role, ev.text, false));
      scroll();
    }

    function el(role, text, partial) {
      const div = document.createElement("div");
      div.className = "msg " + role + (partial ? " partial" : "");
      const who = document.createElement("span");
      who.className = "who";
      who.textContent = role === "you" ? "Heard" : role === "agent" ? "Joe" : role;
      const body = document.createElement("div");
      body.className = "body";
      body.textContent = text;
      div.appendChild(who);
      div.appendChild(body);
      return div;
    }

    function scroll() {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }

    async function boot() {
      await loadDevices();
      const hist = await fetch("/api/history").then((r) => r.json());
      for (const ev of hist) addOrUpdate(ev);

      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(proto + "://" + location.host + "/ws");
      ws.onopen = () => {
        statusEl.textContent = "live";
        statusEl.className = "ok";
      };
      ws.onclose = () => {
        statusEl.textContent = "disconnected — reconnecting…";
        statusEl.className = "";
        setTimeout(boot, 1200);
      };
      ws.onmessage = (m) => addOrUpdate(JSON.parse(m.data));
    }
    boot();
  </script>
</body>
</html>
"""


def create_app(
    transcript_hub: TranscriptHub | None = None,
    *,
    on_start: Optional[RespondHandler] = None,
    on_stop: Optional[RespondHandler] = None,
    on_respond: Optional[RespondHandler] = None,
    get_devices: Optional[Callable[[], dict[str, Any]]] = None,
    set_input: Optional[Callable[[int], Awaitable[dict[str, Any]]]] = None,
    set_output: Optional[Callable[[int], Awaitable[dict[str, Any]]]] = None,
) -> web.Application:
    h = transcript_hub or hub
    start_handler = on_start or on_respond
    app = web.Application()

    async def index(_: web.Request) -> web.Response:
        return web.Response(text=HTML, content_type="text/html")

    async def history(_: web.Request) -> web.Response:
        return web.json_response(h.snapshot())

    async def talk_start(_: web.Request) -> web.Response:
        if start_handler is None:
            return web.json_response(
                {"ok": False, "error": "Start handler not ready"}, status=503
            )
        result = await start_handler()
        ok = result.get("ok") is True or result.get("ok") == "true"
        return web.json_response(result, status=200 if ok else 409)

    async def talk_stop(_: web.Request) -> web.Response:
        if on_stop is None:
            return web.json_response(
                {"ok": False, "error": "Stop handler not ready"}, status=503
            )
        result = await on_stop()
        return web.json_response(result)

    async def respond(request: web.Request) -> web.Response:
        return await talk_start(request)

    async def audio_devices(_: web.Request) -> web.Response:
        if get_devices is None:
            return web.json_response(
                {"ok": False, "error": "Audio devices not ready"}, status=503
            )
        return web.json_response(get_devices())

    async def audio_input(request: web.Request) -> web.Response:
        if set_input is None:
            return web.json_response(
                {"ok": False, "error": "Input switch not ready"}, status=503
            )
        body = await request.json()
        result = await set_input(int(body.get("device")))
        ok = result.get("ok") is True
        return web.json_response(result, status=200 if ok else 400)

    async def audio_output(request: web.Request) -> web.Response:
        if set_output is None:
            return web.json_response(
                {"ok": False, "error": "Output switch not ready"}, status=503
            )
        body = await request.json()
        result = await set_output(int(body.get("device")))
        ok = result.get("ok") is True
        return web.json_response(result, status=200 if ok else 400)

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        q = h.subscribe()
        try:
            while not ws.closed:
                event = await q.get()
                await ws.send_str(event.to_json())
        except Exception:  # noqa: BLE001
            pass
        finally:
            h.unsubscribe(q)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/api/history", history)
    app.router.add_post("/api/talk/start", talk_start)
    app.router.add_post("/api/talk/stop", talk_stop)
    app.router.add_post("/api/respond", respond)
    app.router.add_get("/api/audio/devices", audio_devices)
    app.router.add_post("/api/audio/input", audio_input)
    app.router.add_post("/api/audio/output", audio_output)
    app.router.add_get("/ws", ws_handler)
    return app


async def start_ui(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    on_start: Optional[RespondHandler] = None,
    on_stop: Optional[RespondHandler] = None,
    on_respond: Optional[RespondHandler] = None,
    get_devices: Optional[Callable[[], dict[str, Any]]] = None,
    set_input: Optional[Callable[[int], Awaitable[dict[str, Any]]]] = None,
    set_output: Optional[Callable[[int], Awaitable[dict[str, Any]]]] = None,
) -> web.AppRunner:
    runner = web.AppRunner(
        create_app(
            on_start=on_start,
            on_stop=on_stop,
            on_respond=on_respond,
            get_devices=get_devices,
            set_input=set_input,
            set_output=set_output,
        )
    )
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"Transcript UI: http://localhost:{port}/", flush=True)
    return runner
