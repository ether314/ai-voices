"""Live transcript UI at http://localhost:7860/"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from aiohttp import web

from agent.transcript_hub import TranscriptHub, hub


def _json_dumps(obj: Any) -> str:
    """Strict single-value JSON (reject NaN/Inf; stringify unknown types)."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)


def _json_response(data: Any, *, status: int = 200) -> web.Response:
    try:
        return web.json_response(data, status=status, dumps=_json_dumps)
    except (TypeError, ValueError) as exc:
        return web.json_response(
            {"ok": False, "error": f"JSON serialize failed: {exc}"},
            status=500,
        )

RespondHandler = Callable[[], Awaitable[dict[str, Any]]]
AutoReplySetter = Callable[[bool], Awaitable[dict[str, Any]]]
ContextGetter = Callable[[], Awaitable[dict[str, Any]]]
ContextSetter = Callable[[str], Awaitable[dict[str, Any]]]
SessionGetter = Callable[[], Awaitable[dict[str, Any]]]
TtsGetter = Callable[..., Awaitable[dict[str, Any]]]
TtsSetter = Callable[..., Awaitable[dict[str, Any]]]
LlmGetter = Callable[[], Awaitable[dict[str, Any]]]
LlmSetter = Callable[..., Awaitable[dict[str, Any]]]
SttGetter = Callable[[], Awaitable[dict[str, Any]]]
SttSetter = Callable[..., Awaitable[dict[str, Any]]]
AudioDeviceSetter = Callable[..., Awaitable[dict[str, Any]]]

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
    #stopTalk {
      margin-left: 0.45rem;
      border: none;
      border-radius: 999px;
      padding: 0.85rem 1.35rem;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      background: var(--stop);
      color: var(--stop-text);
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.4);
      white-space: nowrap;
      user-select: none;
    }
    #stopTalk:hover { filter: brightness(1.06); }
    #stopTalk:disabled {
      opacity: 0.45;
      cursor: default;
      filter: none;
    }
    #autoReply {
      margin-left: 0.45rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.85rem 1.15rem;
      font-size: 0.9rem;
      font-weight: 700;
      cursor: pointer;
      background: transparent;
      color: var(--muted);
      white-space: nowrap;
      user-select: none;
    }
    #autoReply.on {
      background: #0ea5e9;
      color: #082f49;
      border-color: transparent;
      box-shadow: 0 0 0 1px rgba(14, 165, 233, 0.35);
    }
    #autoReply:hover { filter: brightness(1.06); color: #e2e8f0; }
    #autoReply.on:hover { color: #082f49; }
    #autoReply:disabled {
      opacity: 0.45;
      cursor: default;
      filter: none;
    }
    #clearCtx {
      margin-left: 0.5rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.85rem 1.1rem;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      background: transparent;
      color: var(--muted);
      white-space: nowrap;
      user-select: none;
    }
    #clearCtx:hover { color: #e2e8f0; border-color: #64748b; }
    #quitApp {
      margin-left: 0.5rem;
      border: none;
      border-radius: 999px;
      padding: 0.85rem 1.1rem;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      background: var(--stop);
      color: var(--stop-text);
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.4);
      white-space: nowrap;
      user-select: none;
    }
    #quitApp.off {
      background: var(--btn);
      color: var(--btn-text);
      box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.35);
    }
    #quitApp:hover { filter: brightness(1.06); }
    .context-box {
      margin-top: 0.85rem;
      display: grid;
      gap: 0.5rem;
    }
    .context-box label {
      display: block;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    #pinnedContext {
      width: 100%;
      min-height: 5.5rem;
      resize: vertical;
      background: var(--panel);
      color: #e2e8f0;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 0.75rem 0.85rem;
      font: inherit;
      line-height: 1.45;
    }
    .context-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
    }
    #saveCtx {
      border: none;
      border-radius: 999px;
      padding: 0.65rem 1.1rem;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      background: #334155;
      color: #f8fafc;
    }
    #saveCtx:hover { filter: brightness(1.08); }
    #contextHint {
      font-size: 0.75rem;
      color: var(--muted);
    }
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
    #setupPanel.collapsed {
      display: none;
    }
    #toggleSetup {
      margin-left: 0.45rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.85rem 1.1rem;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      background: transparent;
      color: var(--muted);
      white-space: nowrap;
      user-select: none;
    }
    #toggleSetup:hover { color: #e2e8f0; border-color: #64748b; }
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
    .devices input[type="url"],
    .devices input[type="text"] {
      width: 100%;
      box-sizing: border-box;
      background: var(--panel);
      color: #e2e8f0;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.65rem 0.75rem;
      font-size: 0.9rem;
    }
    #applyTts {
      border: none;
      border-radius: 999px;
      padding: 0.65rem 1.1rem;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      background: #334155;
      color: #f8fafc;
    }
    #applyTts:hover { filter: brightness(1.08); }
    #refreshVoices, #refreshDevices {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.65rem 1.0rem;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      background: transparent;
      color: var(--muted);
    }
    #refreshVoices:hover, #refreshDevices:hover {
      color: #e2e8f0;
      border-color: #64748b;
    }
    #ttsHint {
      font-size: 0.75rem;
      color: var(--muted);
    }
    .field-note {
      display: block;
      margin-top: 0.25rem;
      font-size: 0.65rem;
      letter-spacing: 0.04em;
      text-transform: none;
      color: #64748b;
      font-weight: 400;
    }
    #applyLlm {
      border: none;
      border-radius: 999px;
      padding: 0.65rem 1.1rem;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      background: #334155;
      color: #f8fafc;
    }
    #applyLlm:hover { filter: brightness(1.08); }
    #llmHint {
      font-size: 0.75rem;
      color: var(--muted);
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
    .msg.status {
      padding: 0.15rem 0.15rem 0.15rem 0.85rem;
      background: transparent;
      border: none;
      font-size: 0.75rem;
      color: #64748b;
      line-height: 1.3;
    }
    .msg.status .who { display: none; }
    .msg.status .body { color: #64748b; font-style: italic; }
    .msg.error { border-color: #7f1d1d; color: #fecaca; opacity: 0.9; font-size: 0.9rem; }
    .msg.partial { opacity: 0.7; border-style: dashed; }
    .msg.agent .status-note {
      display: block;
      margin-top: 0.4rem;
      font-size: 0.72rem;
      color: #64748b;
      font-style: italic;
    }
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div>
        <h1>Voice agent <span id="status">connecting…</span></h1>
        <p>Pause replies need <strong>Auto-reply</strong> on · <strong>Respond</strong> speaks now (and turns auto on) · <strong>Stop</strong> silences and turns auto off</p>
      </div>
      <div>
        <button id="respond" type="button">Respond as Joe</button>
        <button id="autoReply" type="button" title="When on, a pause in others' speech starts Joe's reply">Auto-reply: off</button>
        <button id="stopTalk" type="button">Stop</button>
        <button id="clearCtx" type="button" class="secondary">Refresh chat</button>
        <button id="quitApp" type="button">Turn off</button>
        <button id="toggleSetup" type="button">Hide setup</button>
        <div id="btnHint">auto off</div>
      </div>
    </div>
    <div id="setupPanel">
      <div class="context-box">
        <label for="pinnedContext">Saved context for Joe</label>
        <textarea id="pinnedContext" placeholder="Who Joe is talking to, the goal of the call, facts he should remember…"></textarea>
        <div class="context-actions">
          <button id="saveCtx" type="button">Save context</button>
          <span id="contextHint">Saved to disk — Refresh chat clears the transcript but keeps this.</span>
        </div>
      </div>
      <div id="live" class="empty">
        <span class="label" id="liveLabel">Live caption</span>
        <div class="text">Waiting for speech…</div>
      </div>
      <div class="devices">
        <div>
          <label for="inputDevice">Microphone input <span class="field-note">live from this PC</span></label>
          <select id="inputDevice"></select>
        </div>
        <div>
          <label for="outputDevice">Speaker output <span class="field-note">live from this PC</span></label>
          <select id="outputDevice"></select>
        </div>
      </div>
      <div class="context-actions" style="margin-top:0.5rem">
        <button id="refreshDevices" type="button">Refresh devices</button>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="sttBackend">Speech-to-text <span class="field-note">product backends</span></label>
          <select id="sttBackend"></select>
        </div>
      </div>
      <div class="context-actions" style="margin-top:0.5rem">
        <button id="applyStt" type="button">Apply STT</button>
        <span id="sttHint">Local Whisper keeps Cartesia STT tokens at zero. Cartesia Ink needs ALLOW_CARTESIA_STT=1 in .env plus an explicit cost confirmation.</span>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="llmBackend">Reply model <span class="field-note">product backends</span></label>
          <select id="llmBackend"></select>
        </div>
        <div>
          <label for="localLlmModel">Local model name <span class="field-note">live from Ollama /v1/models when reachable</span></label>
          <select id="localLlmModel"></select>
        </div>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="localLlmUrl">Local LLM URL</label>
          <input id="localLlmUrl" type="url" value="http://127.0.0.1:11434/v1" />
        </div>
      </div>
      <div class="context-actions" style="margin-top:0.5rem">
        <button id="applyLlm" type="button">Apply reply model</button>
        <span id="llmHint">Reply model: Cursor (local)</span>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="ttsBackend">Voice engine <span class="field-note">product backends</span></label>
          <select id="ttsBackend"></select>
        </div>
        <div>
          <label for="cartesiaVoice">Cartesia voice <span class="field-note">live from Cartesia API</span></label>
          <select id="cartesiaVoice"></select>
        </div>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="ttsSpeed">Speaking speed <span class="field-note">fixed product enum</span></label>
          <select id="ttsSpeed"></select>
        </div>
        <div>
          <label for="ttsTonality">Tonality / delivery <span class="field-note">fixed product enum</span></label>
          <select id="ttsTonality"></select>
        </div>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="chatterboxUrl">Local Chatterbox URL</label>
          <input id="chatterboxUrl" type="url" value="http://127.0.0.1:8090" />
        </div>
      </div>
      <div class="context-actions" style="margin-top:0.5rem">
        <button id="applyTts" type="button">Apply voice settings</button>
        <button id="refreshVoices" type="button">Refresh voices</button>
        <span id="ttsHint">Speed &amp; tonality apply to Cartesia (Sonic emotion) and local Chatterbox (temperature + time-stretch). Cartesia voice is cloud-only.</span>
      </div>
      <div id="deviceHint">TikTok / browser livestream → Stereo Mix (loopback / system audio). Room phone near laptop → Microphone Array. If Stereo Mix is missing: Sound settings → Recording → Show Disabled Devices → Enable Stereo Mix.</div>
    </div>
  </header>
  <main id="log"></main>
  <script>
    const log = document.getElementById("log");
    const statusEl = document.getElementById("status");
    const liveBox = document.getElementById("live");
    const liveText = liveBox.querySelector(".text");
    const liveLabel = document.getElementById("liveLabel");
    const respondBtn = document.getElementById("respond");
    const autoReplyBtn = document.getElementById("autoReply");
    const stopBtn = document.getElementById("stopTalk");
    const clearBtn = document.getElementById("clearCtx");
    const quitBtn = document.getElementById("quitApp");
    const saveCtxBtn = document.getElementById("saveCtx");
    const pinnedContext = document.getElementById("pinnedContext");
    const contextHint = document.getElementById("contextHint");
    const btnHint = document.getElementById("btnHint");
    const inputDevice = document.getElementById("inputDevice");
    const outputDevice = document.getElementById("outputDevice");
    const deviceHint = document.getElementById("deviceHint");
    const ttsBackend = document.getElementById("ttsBackend");
    const cartesiaVoice = document.getElementById("cartesiaVoice");
    const ttsSpeed = document.getElementById("ttsSpeed");
    const ttsTonality = document.getElementById("ttsTonality");
    const chatterboxUrl = document.getElementById("chatterboxUrl");
    const applyTtsBtn = document.getElementById("applyTts");
    const refreshVoicesBtn = document.getElementById("refreshVoices");
    const refreshDevicesBtn = document.getElementById("refreshDevices");
    const ttsHint = document.getElementById("ttsHint");
    const llmBackend = document.getElementById("llmBackend");
    const localLlmUrl = document.getElementById("localLlmUrl");
    const localLlmModel = document.getElementById("localLlmModel");
    const applyLlmBtn = document.getElementById("applyLlm");
    const llmHint = document.getElementById("llmHint");
    const sttBackend = document.getElementById("sttBackend");
    const applySttBtn = document.getElementById("applyStt");
    const sttHint = document.getElementById("sttHint");
    const setupPanel = document.getElementById("setupPanel");
    const toggleSetupBtn = document.getElementById("toggleSetup");
    let lastPartialYou = null;
    let lastPartialAgent = null;
    let talking = false;
    let autoReply = false;
    let loadingDevices = false;
    let powered = true;
    let setupCollapsed = false;
    let ws = null;
    let histReady = false;
    let pendingEvents = [];
    let bootStarted = false;

    async function readJson(res) {
      const text = await res.text();
      try {
        return JSON.parse(text);
      } catch (err) {
        const snip = String(text || "").slice(0, 160).replace(/\\s+/g, " ");
        const where = res.url || "(unknown url)";
        throw new Error(
          "Bad JSON from " + where + " (HTTP " + res.status + "): " + snip
        );
      }
    }

    function parseWsJson(raw) {
      try {
        return JSON.parse(raw);
      } catch (err) {
        const snip = String(raw || "").slice(0, 160).replace(/\\s+/g, " ");
        console.warn("Bad WebSocket JSON:", snip, err);
        return null;
      }
    }

    function applySetupCollapsed(collapsed) {
      setupCollapsed = !!collapsed;
      setupPanel.classList.toggle("collapsed", setupCollapsed);
      toggleSetupBtn.textContent = setupCollapsed ? "Show setup" : "Hide setup";
      try {
        localStorage.setItem("voiceSetupCollapsed", setupCollapsed ? "1" : "0");
      } catch (e) {}
    }

    try {
      applySetupCollapsed(localStorage.getItem("voiceSetupCollapsed") === "1");
    } catch (e) {
      applySetupCollapsed(false);
    }

    toggleSetupBtn.addEventListener("click", () => {
      applySetupCollapsed(!setupCollapsed);
    });

    function setLive(text, who) {
      if (text && String(text).trim()) {
        liveBox.classList.remove("empty");
        liveText.textContent = text;
        if (liveLabel) {
          liveLabel.textContent = who === "agent" ? "Joe (live)" : "Live caption";
        }
      } else {
        liveBox.classList.add("empty");
        liveText.textContent = "Waiting for speech…";
        if (liveLabel) liveLabel.textContent = "Live caption";
      }
    }

    function fillSelect(sel, items, current) {
      sel.innerHTML = "";
      for (const d of items) {
        const opt = document.createElement("option");
        opt.value = String(d.index);
        opt.textContent = d.name;
        opt.dataset.rawName = d.raw_name || d.name || "";
        if (d.index === current) opt.selected = true;
        sel.appendChild(opt);
      }
    }

    function fillCartesiaVoices(voices, current) {
      cartesiaVoice.innerHTML = "";
      const list = voices && voices.length ? voices : [];
      for (const v of list) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = v.label || v.id;
        if (v.id === current) opt.selected = true;
        cartesiaVoice.appendChild(opt);
      }
      if (current && ![...cartesiaVoice.options].some((o) => o.value === current)) {
        const opt = document.createElement("option");
        opt.value = current;
        opt.textContent = "Custom (" + current.slice(0, 8) + "…)";
        opt.selected = true;
        cartesiaVoice.appendChild(opt);
      }
    }

    function fillSelectChoices(sel, items, current, valueKey, labelKey) {
      const cur = current == null ? "" : String(current);
      sel.innerHTML = "";
      const list = items && items.length ? items : [];
      for (const item of list) {
        const opt = document.createElement("option");
        const val = typeof item === "string" ? item : String(item[valueKey]);
        const lab = typeof item === "string" ? item : (item[labelKey] || val);
        opt.value = val;
        opt.textContent = lab;
        if (val === cur) opt.selected = true;
        sel.appendChild(opt);
      }
      if (cur && ![...sel.options].some((o) => o.value === cur)) {
        const opt = document.createElement("option");
        opt.value = cur;
        opt.textContent = cur;
        opt.selected = true;
        sel.appendChild(opt);
      }
    }

    function fillLocalModels(models, current) {
      const list = (models && models.length)
        ? models.map((id) => ({ id: String(id), label: String(id) }))
        : [];
      if (!list.length && current) {
        list.push({ id: String(current), label: String(current) });
      }
      fillSelectChoices(localLlmModel, list, current, "id", "label");
    }

    function llmHintFrom(data) {
      const h = data.local_health || {};
      const runtime = data.cursor_runtime || "local";
      const model = data.cursor_model || "composer-2.5";
      if (data.backend === "local") {
        const url = data.local_url || "";
        const name = data.local_model || "?";
        if (h.ok) {
          return "Local LLM connected — " + name + " @ " + url;
        }
        return "Local LLM refused — " + (h.error || "unreachable") + " @ " + url;
      }
      return "Reply model: Cursor (" + runtime + ") — " + model;
    }

    function sttHintFrom(data) {
      const label = data.active_label || data.backend || "?";
      const note = data.hint || "";
      const hw = (data.stt_device && data.stt_compute_type)
        ? (" Device: " + data.stt_device + "/" + data.stt_compute_type + ".")
        : "";
      if (data.backend === "local") {
        return "Using local Whisper (" + (data.whisper_model || "?") + ") on this PC — no Cartesia STT tokens." + hw + " " + note;
      }
      return "Using Cartesia Ink-2 — this bills Cartesia STT tokens. Active: " + label + ". " + note;
    }

    async function loadSttSettings() {
      try {
        const data = await fetch("/api/stt").then((r) => readJson(r));
        if (data.backends && data.backends.length) {
          fillSelectChoices(sttBackend, data.backends, data.backend, "id", "label");
        } else if (data.backend) {
          sttBackend.value = data.backend;
        }
        sttHint.textContent = sttHintFrom(data);
      } catch (err) {
        sttHint.textContent = "Could not load STT settings: " + err;
      }
    }

    applySttBtn.addEventListener("click", async () => {
      const chosen = sttBackend.value;
      let confirmCost = false;
      if (chosen === "cartesia") {
        const ok = window.confirm(
          "Cartesia Ink Speech-to-Text BILLS STT TOKENS on your Cartesia account.\\n\\n" +
          "Also required: ALLOW_CARTESIA_STT=1 in .env (then restart).\\n\\n" +
          "Prefer Local Whisper to keep STT tokens at zero.\\n\\n" +
          "Enable Cartesia Ink anyway?"
        );
        if (!ok) {
          sttHint.textContent = "Cancelled — stayed on current STT backend.";
          await loadSttSettings();
          return;
        }
        confirmCost = true;
      }
      sttHint.textContent = "Applying STT…";
      try {
        const res = await fetch("/api/stt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ backend: chosen, confirm_cost: confirmCost }),
        });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          sttHint.textContent = data.error || "Could not apply STT";
          await loadSttSettings();
          return;
        }
        if (data.backends && data.backends.length) {
          fillSelectChoices(sttBackend, data.backends, data.backend, "id", "label");
        } else if (data.backend) {
          sttBackend.value = data.backend;
        }
        sttHint.textContent = "Now using " + (data.active_label || data.backend) + ". " + sttHintFrom(data);
      } catch (err) {
        sttHint.textContent = String(err);
      }
    });

    async function loadLlmSettings() {
      try {
        const data = await fetch("/api/llm").then((r) => readJson(r));
        if (data.backends && data.backends.length) {
          fillSelectChoices(llmBackend, data.backends, data.backend, "id", "label");
        } else if (data.backend) {
          llmBackend.value = data.backend;
        }
        if (data.local_url) localLlmUrl.value = data.local_url;
        fillLocalModels(data.local_models || (data.local_health && data.local_health.models) || [], data.local_model);
        llmHint.textContent = llmHintFrom(data);
      } catch (err) {
        llmHint.textContent = "Could not load reply model settings: " + err;
      }
    }

    applyLlmBtn.addEventListener("click", async () => {
      llmHint.textContent = "Applying reply model…";
      try {
        const res = await fetch("/api/llm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            backend: llmBackend.value,
            local_url: localLlmUrl.value,
            local_model: localLlmModel.value,
          }),
        });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          llmHint.textContent = data.error || "Could not apply reply model";
          await loadLlmSettings();
          return;
        }
        if (data.backends && data.backends.length) {
          fillSelectChoices(llmBackend, data.backends, data.backend, "id", "label");
        } else if (data.backend) {
          llmBackend.value = data.backend;
        }
        if (data.local_url) localLlmUrl.value = data.local_url;
        fillLocalModels(data.local_models || (data.local_health && data.local_health.models) || [], data.local_model);
        llmHint.textContent = llmHintFrom(data);
      } catch (err) {
        llmHint.textContent = String(err);
      }
    });

    function ttsHintFrom(data) {
      const h = data.local_health || {};
      const voiceLabel = (cartesiaVoice.selectedOptions[0] && cartesiaVoice.selectedOptions[0].textContent) || data.voice_id || "";
      const spd = data.speed != null ? Number(data.speed) : 1;
      const tone = data.tonality || "neutral";
      const delivery = (Math.round(spd * 100) / 100) + "× · " + tone;
      const styleNote = data.hint || "Cartesia uses Sonic speed/emotion; local uses temperature + time-stretch.";
      if (data.backend === "local") {
        return h.ok
          ? ("Local GPU ready — " + (h.device || "?") + " · " + (h.model || "?") + " @ " + (data.chatterbox_url || "") + ". Delivery: " + delivery + ". " + styleNote)
          : ("Local TTS not ready: " + (h.error || "unreachable") + " — start chatterbox GPU container");
      }
      const base = h.reachable
        ? "Using Cartesia cloud"
        : ("Using Cartesia cloud. Local URL checked: " + (h.error || "offline"));
      return base + " — voice: " + voiceLabel + " · delivery: " + delivery;
    }

    function applyTtsPayload(data) {
      if (data.backends && data.backends.length) {
        fillSelectChoices(ttsBackend, data.backends, data.backend, "id", "label");
      } else if (data.backend) {
        ttsBackend.value = data.backend;
      }
      if (data.chatterbox_url) chatterboxUrl.value = data.chatterbox_url;
      fillCartesiaVoices(data.voices || [], data.voice_id);
      if (data.speeds && data.speeds.length) {
        fillSelectChoices(ttsSpeed, data.speeds, data.speed, "value", "label");
      } else if (data.speed != null) {
        ttsSpeed.value = String(data.speed);
      }
      if (data.tonalities && data.tonalities.length) {
        fillSelectChoices(ttsTonality, data.tonalities, data.tonality, "id", "label");
      } else if (data.tonality) {
        ttsTonality.value = data.tonality;
      }
    }

    async function loadTtsSettings(opts) {
      const force = !!(opts && opts.refreshVoices);
      try {
        const url = force ? "/api/tts?refresh_voices=1" : "/api/tts";
        if (force) ttsHint.textContent = "Refreshing Cartesia voices from API…";
        const data = await fetch(url).then((r) => readJson(r));
        applyTtsPayload(data);
        const count = data.voices_count != null ? data.voices_count : ((data.voices && data.voices.length) || 0);
        if (force) {
          ttsHint.textContent = "Loaded " + count + " Cartesia voices from API. " + ttsHintFrom(data);
        } else {
          ttsHint.textContent = ttsHintFrom(data) + " (" + count + " voices)";
        }
      } catch (err) {
        ttsHint.textContent = "Could not load voice engine settings: " + err;
      }
    }

    applyTtsBtn.addEventListener("click", async () => {
      ttsHint.textContent = "Applying voice settings…";
      try {
        const res = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            backend: ttsBackend.value,
            chatterbox_url: chatterboxUrl.value,
            voice_id: cartesiaVoice.value,
            speed: Number(ttsSpeed.value),
            tonality: ttsTonality.value,
          }),
        });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          ttsHint.textContent = data.error || "Could not apply voice settings";
          await loadTtsSettings();
          return;
        }
        applyTtsPayload(data);
        ttsHint.textContent = data.backend === "local"
          ? ttsHintFrom(data)
          : ("Now using Cartesia — " + ((cartesiaVoice.selectedOptions[0] && cartesiaVoice.selectedOptions[0].textContent) || data.voice_id) + " · " + (data.speed != null ? Number(data.speed) : 1) + "× · " + (data.tonality || "neutral"));
      } catch (err) {
        ttsHint.textContent = String(err);
      }
    });

    refreshVoicesBtn.addEventListener("click", async () => {
      await loadTtsSettings({ refreshVoices: true });
    });

    async function loadDevices() {
      loadingDevices = true;
      try {
        const data = await fetch("/api/audio/devices").then((r) => readJson(r));
        fillSelect(inputDevice, data.inputs || [], data.current_input);
        fillSelect(outputDevice, data.outputs || [], data.current_output);
        deviceHint.textContent = data.hint
          || (data.has_loopback
            ? "TikTok / livestream → Stereo Mix (loopback / system audio). Room mic → Microphone Array."
            : "No loopback device — enable Stereo Mix (Recording → Show Disabled Devices) or use VB-Audio Cable.");
      } catch (err) {
        deviceHint.textContent = "Could not load audio devices: " + err;
      } finally {
        loadingDevices = false;
      }
    }

    refreshDevicesBtn.addEventListener("click", async () => {
      deviceHint.textContent = "Refreshing audio devices…";
      await loadDevices();
    });

    async function setDevice(kind, index, name) {
      deviceHint.textContent = "Switching " + kind + "…";
      try {
        const res = await fetch("/api/audio/" + kind, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device: Number(index), name: name || "" }),
        });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          deviceHint.textContent = data.error || ("Failed to set " + kind);
          await loadDevices();
          return;
        }
        if (kind === "input") {
          const opt = inputDevice.selectedOptions[0];
          const label = (opt && opt.textContent) || name || ("device " + index);
          const confirm = label.toLowerCase().includes("loopback")
            ? "Capturing system / livestream audio via " + label + ". Joe mutes while speaking."
            : "Microphone updated -> " + label + ". For TikTok audio, pick Stereo Mix (loopback).";
          deviceHint.textContent = confirm;
        } else {
          deviceHint.textContent = "Speakers updated.";
        }
      } catch (err) {
        deviceHint.textContent = String(err);
      }
    }

    inputDevice.addEventListener("change", () => {
      if (loadingDevices) return;
      const opt = inputDevice.selectedOptions[0];
      setDevice("input", inputDevice.value, opt && opt.dataset.rawName);
    });
    outputDevice.addEventListener("change", () => {
      if (loadingDevices) return;
      const opt = outputDevice.selectedOptions[0];
      setDevice("output", outputDevice.value, opt && opt.dataset.rawName);
    });

    function autoHint() {
      if (!powered) return "off";
      if (talking) return "speaking";
      return autoReply
        ? "auto on — pause starts reply"
        : "auto off — enable Auto-reply";
    }

    function setAutoReplyUI(on) {
      autoReply = !!on;
      if (autoReplyBtn) {
        autoReplyBtn.textContent = autoReply ? "Auto-reply: on" : "Auto-reply: off";
        autoReplyBtn.classList.toggle("on", autoReply);
        autoReplyBtn.disabled = !powered;
      }
      if (!talking) btnHint.textContent = autoHint();
    }

    function setTalkingUI(on) {
      talking = on;
      respondBtn.disabled = !powered;
      stopBtn.disabled = !powered;
      if (autoReplyBtn) autoReplyBtn.disabled = !powered;
      btnHint.textContent = autoHint();
    }

    function setPowerUI(on) {
      powered = on;
      if (on) {
        quitBtn.classList.remove("off");
        quitBtn.textContent = "Turn off";
        statusEl.textContent = "live";
        statusEl.className = "ok";
        btnHint.textContent = autoHint();
      } else {
        quitBtn.classList.add("off");
        quitBtn.textContent = "Turn on";
        statusEl.textContent = "off";
        statusEl.className = "";
        talking = false;
        btnHint.textContent = "off";
      }
      respondBtn.disabled = !on;
      stopBtn.disabled = !on;
      if (autoReplyBtn) autoReplyBtn.disabled = !on;
    }

    async function doStart() {
      if (!powered) {
        btnHint.textContent = "App is off — click Turn on first";
        return;
      }
      btnHint.textContent = "Starting reply…";
      try {
        const res = await fetch("/api/talk/start", { method: "POST" });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not start";
          setTalkingUI(false);
          return;
        }
        if (typeof data.auto_reply === "boolean") setAutoReplyUI(data.auto_reply);
        else setAutoReplyUI(true);
        setTalkingUI(true);
      } catch (err) {
        btnHint.textContent = String(err);
        setTalkingUI(false);
      }
    }

    async function doStop() {
      btnHint.textContent = "Stopping…";
      try {
        const res = await fetch("/api/talk/stop", { method: "POST" });
        const data = await readJson(res);
        if (typeof data.auto_reply === "boolean") setAutoReplyUI(data.auto_reply);
        else setAutoReplyUI(false);
      } catch (err) {
        btnHint.textContent = String(err);
        setAutoReplyUI(false);
      }
      setTalkingUI(false);
    }

    async function doToggleAuto() {
      if (!powered) {
        btnHint.textContent = "App is off — click Turn on first";
        return;
      }
      const next = !autoReply;
      btnHint.textContent = next ? "Enabling auto…" : "Disabling auto…";
      try {
        const res = await fetch("/api/auto-reply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: next }),
        });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not set auto-reply";
          return;
        }
        if (typeof data.auto_reply === "boolean") setAutoReplyUI(data.auto_reply);
        else setAutoReplyUI(next);
      } catch (err) {
        btnHint.textContent = String(err);
      }
    }

    async function doPowerOff() {
      btnHint.textContent = "Turning off…";
      try {
        if (talking) await doStop();
        const res = await fetch("/api/power/off", { method: "POST" });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not turn off";
          return;
        }
        if (typeof data.auto_reply === "boolean") setAutoReplyUI(data.auto_reply);
        else setAutoReplyUI(false);
        setPowerUI(false);
      } catch (err) {
        btnHint.textContent = String(err);
      }
    }

    async function doPowerOn() {
      btnHint.textContent = "Turning on…";
      try {
        const res = await fetch("/api/power/on", { method: "POST" });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not turn on";
          return;
        }
        if (typeof data.auto_reply === "boolean") setAutoReplyUI(data.auto_reply);
        setPowerUI(true);
      } catch (err) {
        btnHint.textContent = String(err);
      }
    }

    respondBtn.addEventListener("click", async () => {
      if (!powered) {
        btnHint.textContent = "App is off — click Turn on first";
        return;
      }
      await doStart();
    });

    if (autoReplyBtn) {
      autoReplyBtn.addEventListener("click", () => { doToggleAuto(); });
    }

    stopBtn.addEventListener("click", async () => {
      if (!powered) {
        btnHint.textContent = "App is off — click Turn on first";
        return;
      }
      await doStop();
    });

    quitBtn.addEventListener("click", async () => {
      if (powered) {
        await doPowerOff();
      } else {
        await doPowerOn();
      }
    });

    clearBtn.addEventListener("click", async () => {
      btnHint.textContent = "Refreshing chat…";
      try {
        if (talking) await doStop();
        const res = await fetch("/api/clear", { method: "POST" });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not refresh chat";
          return;
        }
        log.innerHTML = "";
        lastPartialYou = null;
        lastPartialAgent = null;
        setLive("");
        if (typeof data.pinned === "string") {
          pinnedContext.value = data.pinned;
        }
        const kept = data.pinned_chars || 0;
        btnHint.textContent = kept
          ? "Chat refreshed — saved context kept (" + kept + " chars)"
          : "Chat refreshed — no saved context on file";
        contextHint.textContent = kept
          ? "Saved context still on disk (" + kept + " chars)."
          : "No saved context yet — write some and click Save context.";
      } catch (err) {
        btnHint.textContent = String(err);
      }
    });

    saveCtxBtn.addEventListener("click", async () => {
      contextHint.textContent = "Saving…";
      try {
        const res = await fetch("/api/context", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: pinnedContext.value }),
        });
        const data = await readJson(res);
        if (!res.ok || data.ok === false) {
          contextHint.textContent = data.error || "Could not save context";
          return;
        }
        pinnedContext.value = data.text || "";
        const n = data.chars || 0;
        contextHint.textContent = n
          ? "Saved (" + n + " chars). Refresh chat keeps this."
          : "Saved context cleared on disk.";
      } catch (err) {
        contextHint.textContent = String(err);
      }
    });

    async function loadPinnedContext() {
      try {
        const data = await fetch("/api/context").then((r) => readJson(r));
        if (data && typeof data.text === "string") {
          pinnedContext.value = data.text;
          const n = data.chars || 0;
          contextHint.textContent = n
            ? "Loaded saved context (" + n + " chars)."
            : "No saved context yet — write some and click Save context.";
        }
      } catch (err) {
        contextHint.textContent = "Could not load saved context: " + err;
      }
    }

    function statusLabel(text) {
      const raw = text || "";
      const parts = raw.split("—");
      if (parts.length > 1) return parts.slice(1).join("—").trim() || raw;
      return raw;
    }

    function attachStatusToAgent(label) {
      const target = lastPartialAgent || log.querySelector(".msg.agent:last-of-type");
      if (!target) return false;
      let note = target.querySelector(".status-note");
      if (!note) {
        note = document.createElement("span");
        note.className = "status-note";
        target.appendChild(note);
      }
      note.textContent = label;
      return true;
    }

    function addOrUpdate(ev) {
      const text = ev.text || "";
      if (ev.role === "status" && text.includes("TALKING_ON")) setTalkingUI(true);
      if (ev.role === "status" && text.includes("TALKING_OFF")) setTalkingUI(false);
      if (ev.role === "status" && text.includes("APP_OFF")) setPowerUI(false);
      if (ev.role === "status" && text.includes("APP_ON")) setPowerUI(true);
      if (
        ev.role === "status" &&
        (text.includes("CONTEXT_CLEARED") || text.includes("CONVERSATION_REFRESHED"))
      ) {
        log.innerHTML = "";
        lastPartialYou = null;
        lastPartialAgent = null;
        setLive("");
      }
      if (ev.role === "status") {
        const label = statusLabel(text);
        if (!label) return;
        // Prefer nesting under Joe's current/last line; else compact status row.
        if (
          text.includes("TALKING_ON") ||
          text.includes("TALKING_OFF") ||
          text.includes("speaking")
        ) {
          if (attachStatusToAgent(label)) {
            scroll();
            return;
          }
        }
        const row = el("status", label, false);
        log.appendChild(row);
        scroll();
        return;
      }

      if (ev.role === "you" && ev.partial) {
        setLive(ev.text, "you");
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
        setLive(ev.text, "agent");
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
        setLive("");
        if (lastPartialAgent) {
          lastPartialAgent.classList.remove("partial");
          lastPartialAgent.querySelector(".body").textContent = ev.text;
          lastPartialAgent = null;
        } else {
          // Prefer updating the last Joe bubble if a prior partial was missed.
          const lastJoe = log.querySelector(".msg.agent:last-of-type");
          if (lastJoe && lastJoe.classList.contains("partial")) {
            lastJoe.classList.remove("partial");
            lastJoe.querySelector(".body").textContent = ev.text;
          } else {
            log.appendChild(el("agent", ev.text, false));
          }
        }
        scroll();
        return;
      }
      if (ev.role === "error") {
        log.appendChild(el("error", text, false));
        scroll();
        return;
      }
      log.appendChild(el(ev.role, text, false));
      scroll();
    }

    function el(role, text, partial) {
      const div = document.createElement("div");
      div.className = "msg " + role + (partial ? " partial" : "");
      const who = document.createElement("span");
      who.className = "who";
      who.textContent =
        role === "you" ? "Heard" :
        role === "agent" ? "Joe" :
        role;
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

    async function reloadTranscript() {
      histReady = false;
      pendingEvents = [];
      log.innerHTML = "";
      lastPartialYou = null;
      lastPartialAgent = null;
      setLive("");
      try {
        const hist = await fetch("/api/history").then((r) => readJson(r));
        if (Array.isArray(hist)) {
          for (const ev of hist) addOrUpdate(ev);
        }
      } catch (err) {
        console.warn("Could not load history:", err);
        btnHint.textContent = String(err);
      }
      histReady = true;
      const queued = pendingEvents.slice();
      pendingEvents = [];
      for (const ev of queued) addOrUpdate(ev);
    }

    async function loadSessionState() {
      try {
        const sess = await fetch("/api/session").then((r) => readJson(r));
        if (sess && typeof sess.auto_reply === "boolean") {
          setAutoReplyUI(sess.auto_reply);
        }
        if (sess && typeof sess.powered === "boolean") {
          setPowerUI(sess.powered);
        }
        if (sess && typeof sess.talking === "boolean") {
          setTalkingUI(sess.talking);
        }
      } catch (err) {
        console.warn("Could not load session:", err);
        setAutoReplyUI(false);
      }
    }

    function connectWs() {
      if (
        ws &&
        (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(proto + "://" + location.host + "/ws");
      ws.onopen = () => {
        statusEl.textContent = powered ? "live" : "off";
        statusEl.className = powered ? "ok" : "";
      };
      ws.onclose = () => {
        statusEl.textContent = "disconnected — reconnecting…";
        statusEl.className = "";
        setTimeout(async () => {
          await loadSessionState();
          await reloadTranscript();
          connectWs();
        }, 1200);
      };
      ws.onmessage = (m) => {
        const ev = parseWsJson(m.data);
        if (!ev) return;
        if (!histReady) {
          pendingEvents.push(ev);
          return;
        }
        addOrUpdate(ev);
      };
    }

    async function boot() {
      if (!bootStarted) {
        bootStarted = true;
        await loadDevices();
        await loadSttSettings();
        await loadLlmSettings();
        await loadTtsSettings();
        await loadPinnedContext();
      }
      await loadSessionState();
      await reloadTranscript();
      connectWs();
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
    on_clear: Optional[RespondHandler] = None,
    on_power_on: Optional[RespondHandler] = None,
    on_power_off: Optional[RespondHandler] = None,
    on_respond: Optional[RespondHandler] = None,
    set_auto_reply: Optional[AutoReplySetter] = None,
    get_session: Optional[SessionGetter] = None,
    get_context: Optional[ContextGetter] = None,
    set_context: Optional[ContextSetter] = None,
    get_tts: Optional[TtsGetter] = None,
    set_tts: Optional[TtsSetter] = None,
    get_llm: Optional[LlmGetter] = None,
    set_llm: Optional[LlmSetter] = None,
    get_stt: Optional[SttGetter] = None,
    set_stt: Optional[SttSetter] = None,
    get_devices: Optional[Callable[[], dict[str, Any]]] = None,
    set_input: Optional[AudioDeviceSetter] = None,
    set_output: Optional[AudioDeviceSetter] = None,
) -> web.Application:
    h = transcript_hub or hub
    start_handler = on_start or on_respond
    app = web.Application()

    async def index(_: web.Request) -> web.Response:
        return web.Response(text=HTML, content_type="text/html")

    async def history(_: web.Request) -> web.Response:
        try:
            return _json_response(h.snapshot())
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def session_state(_: web.Request) -> web.Response:
        if get_session is None:
            return _json_response(
                {
                    "ok": True,
                    "powered": True,
                    "talking": False,
                    "auto_reply": False,
                }
            )
        try:
            return _json_response(await get_session())
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def talk_start(_: web.Request) -> web.Response:
        if start_handler is None:
            return _json_response(
                {"ok": False, "error": "Start handler not ready"}, status=503
            )
        try:
            result = await start_handler()
            if not isinstance(result, dict):
                return _json_response(
                    {"ok": False, "error": f"Bad start result type: {type(result).__name__}"},
                    status=500,
                )
            ok = result.get("ok") is True or result.get("ok") == "true"
            return _json_response(result, status=200 if ok else 409)
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def talk_stop(_: web.Request) -> web.Response:
        if on_stop is None:
            return _json_response(
                {"ok": False, "error": "Stop handler not ready"}, status=503
            )
        try:
            result = await on_stop()
            return _json_response(result if isinstance(result, dict) else {"ok": True})
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def auto_reply_set(request: web.Request) -> web.Response:
        if set_auto_reply is None:
            return _json_response(
                {"ok": False, "error": "Auto-reply handler not ready"}, status=503
            )
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        raw = body.get("enabled", body.get("auto_reply", True))
        if isinstance(raw, str):
            enabled = raw.strip().lower() in ("1", "true", "on", "yes")
        else:
            enabled = bool(raw)
        try:
            result = await set_auto_reply(enabled)
            return _json_response(
                result if isinstance(result, dict) else {"ok": True, "auto_reply": enabled}
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def clear_context(_: web.Request) -> web.Response:
        if on_clear is None:
            return _json_response(
                {"ok": False, "error": "Clear handler not ready"}, status=503
            )
        try:
            result = await on_clear()
            if not isinstance(result, dict):
                return _json_response(
                    {"ok": False, "error": "Bad clear result"}, status=500
                )
            ok = result.get("ok") is True
            return _json_response(result, status=200 if ok else 500)
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def power_off(_: web.Request) -> web.Response:
        if on_power_off is None:
            return _json_response(
                {"ok": False, "error": "Power off not ready"}, status=503
            )
        try:
            result = await on_power_off()
            return _json_response(result if isinstance(result, dict) else {"ok": True})
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def power_on(_: web.Request) -> web.Response:
        if on_power_on is None:
            return _json_response(
                {"ok": False, "error": "Power on not ready"}, status=503
            )
        try:
            result = await on_power_on()
            return _json_response(result if isinstance(result, dict) else {"ok": True})
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def get_pinned(_: web.Request) -> web.Response:
        if get_context is None:
            return _json_response(
                {"ok": False, "error": "Context not ready"}, status=503
            )
        try:
            return _json_response(await get_context())
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def set_pinned(request: web.Request) -> web.Response:
        if set_context is None:
            return _json_response(
                {"ok": False, "error": "Context not ready"}, status=503
            )
        try:
            body = await request.json()
            result = await set_context(str(body.get("text") or ""))
            ok = isinstance(result, dict) and result.get("ok") is True
            return _json_response(
                result if isinstance(result, dict) else {"ok": False},
                status=200 if ok else 500,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def get_tts_settings(request: web.Request) -> web.Response:
        if get_tts is None:
            return _json_response(
                {"ok": False, "error": "TTS settings not ready"}, status=503
            )
        try:
            q = (request.rel_url.query.get("refresh_voices") or "").strip().lower()
            force = q in ("1", "true", "yes")
            payload = await get_tts(force_refresh_voices=force)
            if not isinstance(payload, dict):
                return _json_response(
                    {"ok": False, "error": f"Bad TTS payload type: {type(payload).__name__}"},
                    status=500,
                )
            return _json_response(payload)
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def set_tts_settings(request: web.Request) -> web.Response:
        if set_tts is None:
            return _json_response(
                {"ok": False, "error": "TTS settings not ready"}, status=503
            )
        try:
            body = await request.json()
            result = await set_tts(
                backend=body.get("backend"),
                chatterbox_url=body.get("chatterbox_url"),
                voice_id=body.get("voice_id"),
                speed=body.get("speed"),
                tonality=body.get("tonality"),
            )
            ok = isinstance(result, dict) and result.get("ok") is True
            return _json_response(
                result if isinstance(result, dict) else {"ok": False},
                status=200 if ok else 400,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def get_llm_settings(_: web.Request) -> web.Response:
        if get_llm is None:
            return _json_response(
                {"ok": False, "error": "LLM settings not ready"}, status=503
            )
        try:
            return _json_response(await get_llm())
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def set_llm_settings(request: web.Request) -> web.Response:
        if set_llm is None:
            return _json_response(
                {"ok": False, "error": "LLM settings not ready"}, status=503
            )
        try:
            body = await request.json()
            result = await set_llm(
                backend=body.get("backend"),
                local_url=body.get("local_url"),
                local_model=body.get("local_model"),
            )
            ok = isinstance(result, dict) and result.get("ok") is True
            return _json_response(
                result if isinstance(result, dict) else {"ok": False},
                status=200 if ok else 400,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def get_stt_settings(_: web.Request) -> web.Response:
        if get_stt is None:
            return _json_response(
                {"ok": False, "error": "STT settings not ready"}, status=503
            )
        try:
            return _json_response(await get_stt())
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def set_stt_settings(request: web.Request) -> web.Response:
        if set_stt is None:
            return _json_response(
                {"ok": False, "error": "STT settings not ready"}, status=503
            )
        try:
            body = await request.json()
            confirm = body.get("confirm_cost")
            confirm_cost = confirm is True or str(confirm).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            result = await set_stt(
                backend=body.get("backend"),
                confirm_cost=confirm_cost,
            )
            ok = isinstance(result, dict) and result.get("ok") is True
            return _json_response(
                result if isinstance(result, dict) else {"ok": False},
                status=200 if ok else 400,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def respond(request: web.Request) -> web.Response:
        return await talk_start(request)

    async def audio_devices(_: web.Request) -> web.Response:
        if get_devices is None:
            return _json_response(
                {"ok": False, "error": "Audio devices not ready"}, status=503
            )
        try:
            payload = get_devices()
            if not isinstance(payload, dict):
                return _json_response(
                    {"ok": False, "error": f"Bad devices payload type: {type(payload).__name__}"},
                    status=500,
                )
            return _json_response(payload)
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def audio_input(request: web.Request) -> web.Response:
        if set_input is None:
            return _json_response(
                {"ok": False, "error": "Input switch not ready"}, status=503
            )
        try:
            body = await request.json()
            raw_dev = body.get("device")
            index = None if raw_dev is None or raw_dev == "" else int(raw_dev)
            name = body.get("name")
            result = await set_input(index, name=str(name) if name else None)
            ok = isinstance(result, dict) and result.get("ok") is True
            return _json_response(
                result if isinstance(result, dict) else {"ok": False},
                status=200 if ok else 400,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

    async def audio_output(request: web.Request) -> web.Response:
        if set_output is None:
            return _json_response(
                {"ok": False, "error": "Output switch not ready"}, status=503
            )
        try:
            body = await request.json()
            raw_dev = body.get("device")
            index = None if raw_dev is None or raw_dev == "" else int(raw_dev)
            name = body.get("name")
            result = await set_output(index, name=str(name) if name else None)
            ok = isinstance(result, dict) and result.get("ok") is True
            return _json_response(
                result if isinstance(result, dict) else {"ok": False},
                status=200 if ok else 400,
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"ok": False, "error": str(exc)}, status=500)

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
    app.router.add_get("/api/session", session_state)
    app.router.add_post("/api/talk/start", talk_start)
    app.router.add_post("/api/talk/stop", talk_stop)
    app.router.add_post("/api/auto-reply", auto_reply_set)
    app.router.add_post("/api/clear", clear_context)
    app.router.add_post("/api/power/off", power_off)
    app.router.add_post("/api/power/on", power_on)
    app.router.add_get("/api/context", get_pinned)
    app.router.add_post("/api/context", set_pinned)
    app.router.add_get("/api/tts", get_tts_settings)
    app.router.add_post("/api/tts", set_tts_settings)
    app.router.add_get("/api/llm", get_llm_settings)
    app.router.add_post("/api/llm", set_llm_settings)
    app.router.add_get("/api/stt", get_stt_settings)
    app.router.add_post("/api/stt", set_stt_settings)
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
    on_clear: Optional[RespondHandler] = None,
    on_power_on: Optional[RespondHandler] = None,
    on_power_off: Optional[RespondHandler] = None,
    on_respond: Optional[RespondHandler] = None,
    set_auto_reply: Optional[AutoReplySetter] = None,
    get_session: Optional[SessionGetter] = None,
    get_context: Optional[ContextGetter] = None,
    set_context: Optional[ContextSetter] = None,
    get_tts: Optional[TtsGetter] = None,
    set_tts: Optional[TtsSetter] = None,
    get_llm: Optional[LlmGetter] = None,
    set_llm: Optional[LlmSetter] = None,
    get_stt: Optional[SttGetter] = None,
    set_stt: Optional[SttSetter] = None,
    get_devices: Optional[Callable[[], dict[str, Any]]] = None,
    set_input: Optional[AudioDeviceSetter] = None,
    set_output: Optional[AudioDeviceSetter] = None,
) -> web.AppRunner:
    runner = web.AppRunner(
        create_app(
            on_start=on_start,
            on_stop=on_stop,
            on_clear=on_clear,
            on_power_on=on_power_on,
            on_power_off=on_power_off,
            on_respond=on_respond,
            set_auto_reply=set_auto_reply,
            get_session=get_session,
            get_context=get_context,
            set_context=set_context,
            get_tts=get_tts,
            set_tts=set_tts,
            get_llm=get_llm,
            set_llm=set_llm,
            get_stt=get_stt,
            set_stt=set_stt,
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
