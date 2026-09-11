"""Live transcript UI at http://localhost:7860/"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional

from aiohttp import web

from agent.transcript_hub import TranscriptHub, hub

RespondHandler = Callable[[], Awaitable[dict[str, Any]]]
ContextGetter = Callable[[], Awaitable[dict[str, Any]]]
ContextSetter = Callable[[str], Awaitable[dict[str, Any]]]
TtsGetter = Callable[[], Awaitable[dict[str, Any]]]
TtsSetter = Callable[..., Awaitable[dict[str, Any]]]
LlmGetter = Callable[[], Awaitable[dict[str, Any]]]
LlmSetter = Callable[..., Awaitable[dict[str, Any]]]

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
    #ttsHint {
      font-size: 0.75rem;
      color: var(--muted);
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
        <p>Auto turn-taking · <strong>Respond</strong> / <strong>Stop</strong></p>
      </div>
      <div>
        <button id="respond" type="button">Respond as Joe</button>
        <button id="stopTalk" type="button">Stop</button>
        <button id="clearCtx" type="button" class="secondary">Refresh chat</button>
        <button id="quitApp" type="button">Turn off</button>
        <button id="toggleSetup" type="button">Hide setup</button>
        <div id="btnHint">auto on</div>
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
          <label for="inputDevice">Microphone input</label>
          <select id="inputDevice"></select>
        </div>
        <div>
          <label for="outputDevice">Speaker output</label>
          <select id="outputDevice"></select>
        </div>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="llmBackend">Reply model</label>
          <select id="llmBackend">
            <option value="cursor">Cursor local (composer-2.5)</option>
            <option value="local">Local (OpenAI-compatible)</option>
          </select>
        </div>
        <div>
          <label for="localLlmModel">Local model name</label>
          <input id="localLlmModel" type="text" value="llama3.2" />
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
        <span id="llmHint">Cursor uses the Cursor SDK local runtime (this machine). Local uses an OpenAI-compatible chat API (Ollama / LM Studio).</span>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="ttsBackend">Voice engine</label>
          <select id="ttsBackend">
            <option value="cartesia">Cartesia Sonic (cloud)</option>
            <option value="local">Local GPU Chatterbox (Docker)</option>
          </select>
        </div>
        <div>
          <label for="cartesiaVoice">Cartesia voice</label>
          <select id="cartesiaVoice"></select>
        </div>
      </div>
      <div class="devices" style="margin-top:0.65rem">
        <div>
          <label for="ttsSpeed">Speaking speed</label>
          <select id="ttsSpeed">
            <option value="0.8">Slow (0.8×)</option>
            <option value="0.9">Slightly slow (0.9×)</option>
            <option value="1.0" selected>Normal (1.0×)</option>
            <option value="1.15">Slightly fast (1.15×)</option>
            <option value="1.3">Fast (1.3×)</option>
            <option value="1.45">Very fast (1.45×)</option>
          </select>
        </div>
        <div>
          <label for="ttsTonality">Tonality / delivery</label>
          <select id="ttsTonality">
            <option value="neutral" selected>Neutral</option>
            <option value="calm">Calm</option>
            <option value="warm">Warm</option>
            <option value="energetic">Energetic</option>
            <option value="serious">Serious</option>
            <option value="cheerful">Cheerful</option>
          </select>
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
    const ttsHint = document.getElementById("ttsHint");
    const llmBackend = document.getElementById("llmBackend");
    const localLlmUrl = document.getElementById("localLlmUrl");
    const localLlmModel = document.getElementById("localLlmModel");
    const applyLlmBtn = document.getElementById("applyLlm");
    const llmHint = document.getElementById("llmHint");
    const setupPanel = document.getElementById("setupPanel");
    const toggleSetupBtn = document.getElementById("toggleSetup");
    let lastPartialYou = null;
    let lastPartialAgent = null;
    let talking = false;
    let loadingDevices = false;
    let powered = true;
    let setupCollapsed = false;

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
        opt.textContent = d.index + ": " + d.name;
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
        opt.value = String(item[valueKey]);
        opt.textContent = item[labelKey] || String(item[valueKey]);
        if (String(item[valueKey]) === cur) opt.selected = true;
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

    function llmHintFrom(data) {
      const h = data.local_health || {};
      const label = data.active_label || data.backend || "?";
      const note = data.hint || "Pinned scenario context works on both backends.";
      if (data.backend === "local") {
        return h.ok
          ? ("Local LLM ready — " + (data.local_model || "?") + " @ " + (data.local_url || "") + ". Active: " + label + ". " + note)
          : ("Local LLM not ready: " + (h.error || "unreachable") + " — start Ollama/LM Studio at " + (data.local_url || ""));
      }
      const runtime = data.cursor_runtime || "local";
      const base = h.reachable
        ? ("Using Cursor (" + runtime + ") — " + (data.cursor_model || "composer-2.5"))
        : ("Using Cursor (" + runtime + ") — " + (data.cursor_model || "composer-2.5") + ". Ollama probe: " + (h.error || "offline"));
      return base + ". " + note;
    }

    async function loadLlmSettings() {
      try {
        const data = await fetch("/api/llm").then((r) => r.json());
        if (data.backends && data.backends.length) {
          fillSelectChoices(llmBackend, data.backends, data.backend, "id", "label");
        } else if (data.backend) {
          llmBackend.value = data.backend;
        }
        if (data.local_url) localLlmUrl.value = data.local_url;
        if (data.local_model) localLlmModel.value = data.local_model;
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
        const data = await res.json();
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
        if (data.local_model) localLlmModel.value = data.local_model;
        llmHint.textContent = "Now using " + (data.active_label || data.backend) + ". " + llmHintFrom(data);
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

    async function loadTtsSettings() {
      try {
        const data = await fetch("/api/tts").then((r) => r.json());
        if (data.backend) ttsBackend.value = data.backend;
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
        ttsHint.textContent = ttsHintFrom(data);
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
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          ttsHint.textContent = data.error || "Could not apply voice settings";
          await loadTtsSettings();
          return;
        }
        if (data.backend) ttsBackend.value = data.backend;
        if (data.chatterbox_url) chatterboxUrl.value = data.chatterbox_url;
        fillCartesiaVoices(data.voices || [], data.voice_id);
        if (data.speeds && data.speeds.length) {
          fillSelectChoices(ttsSpeed, data.speeds, data.speed, "value", "label");
        }
        if (data.tonalities && data.tonalities.length) {
          fillSelectChoices(ttsTonality, data.tonalities, data.tonality, "id", "label");
        }
        ttsHint.textContent = data.backend === "local"
          ? ttsHintFrom(data)
          : ("Now using Cartesia — " + ((cartesiaVoice.selectedOptions[0] && cartesiaVoice.selectedOptions[0].textContent) || data.voice_id) + " · " + (data.speed != null ? Number(data.speed) : 1) + "× · " + (data.tonality || "neutral"));
      } catch (err) {
        ttsHint.textContent = String(err);
      }
    });

    async function loadDevices() {
      loadingDevices = true;
      try {
        const data = await fetch("/api/audio/devices").then((r) => r.json());
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
        if (kind === "input") {
          const opt = inputDevice.selectedOptions[0];
          const label = (opt && opt.textContent) || ("device " + index);
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
      setDevice("input", inputDevice.value);
    });
    outputDevice.addEventListener("change", () => {
      if (loadingDevices) return;
      setDevice("output", outputDevice.value);
    });

    function setTalkingUI(on) {
      talking = on;
      respondBtn.disabled = !powered;
      stopBtn.disabled = !powered;
      if (on) {
        btnHint.textContent = "speaking";
      } else {
        btnHint.textContent = powered ? "auto on" : "off";
      }
    }

    function setPowerUI(on) {
      powered = on;
      if (on) {
        quitBtn.classList.remove("off");
        quitBtn.textContent = "Turn off";
        statusEl.textContent = "live";
        statusEl.className = "ok";
        btnHint.textContent = talking ? "speaking" : "auto on";
      } else {
        quitBtn.classList.add("off");
        quitBtn.textContent = "Turn on";
        statusEl.textContent = "off";
        statusEl.className = "";
        setTalkingUI(false);
        btnHint.textContent = "off";
      }
      respondBtn.disabled = !on;
      stopBtn.disabled = !on;
    }

    async function doStart() {
      if (!powered) {
        btnHint.textContent = "App is off — click Turn on first";
        return;
      }
      btnHint.textContent = "Starting reply…";
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
      btnHint.textContent = "auto paused";
    }

    async function doPowerOff() {
      btnHint.textContent = "Turning off…";
      try {
        if (talking) await doStop();
        const res = await fetch("/api/power/off", { method: "POST" });
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not turn off";
          return;
        }
        setPowerUI(false);
      } catch (err) {
        btnHint.textContent = String(err);
      }
    }

    async function doPowerOn() {
      btnHint.textContent = "Turning on…";
      try {
        const res = await fetch("/api/power/on", { method: "POST" });
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          btnHint.textContent = data.error || "Could not turn on";
          return;
        }
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
        const data = await res.json();
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
        const data = await res.json();
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
        const data = await fetch("/api/context").then((r) => r.json());
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

    async function boot() {
      await loadDevices();
      await loadLlmSettings();
      await loadTtsSettings();
      await loadPinnedContext();
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
    on_clear: Optional[RespondHandler] = None,
    on_power_on: Optional[RespondHandler] = None,
    on_power_off: Optional[RespondHandler] = None,
    on_respond: Optional[RespondHandler] = None,
    get_context: Optional[ContextGetter] = None,
    set_context: Optional[ContextSetter] = None,
    get_tts: Optional[TtsGetter] = None,
    set_tts: Optional[TtsSetter] = None,
    get_llm: Optional[LlmGetter] = None,
    set_llm: Optional[LlmSetter] = None,
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

    async def clear_context(_: web.Request) -> web.Response:
        if on_clear is None:
            return web.json_response(
                {"ok": False, "error": "Clear handler not ready"}, status=503
            )
        result = await on_clear()
        ok = result.get("ok") is True
        return web.json_response(result, status=200 if ok else 500)

    async def power_off(_: web.Request) -> web.Response:
        if on_power_off is None:
            return web.json_response(
                {"ok": False, "error": "Power off not ready"}, status=503
            )
        result = await on_power_off()
        return web.json_response(result)

    async def power_on(_: web.Request) -> web.Response:
        if on_power_on is None:
            return web.json_response(
                {"ok": False, "error": "Power on not ready"}, status=503
            )
        result = await on_power_on()
        return web.json_response(result)

    async def get_pinned(_: web.Request) -> web.Response:
        if get_context is None:
            return web.json_response(
                {"ok": False, "error": "Context not ready"}, status=503
            )
        return web.json_response(await get_context())

    async def set_pinned(request: web.Request) -> web.Response:
        if set_context is None:
            return web.json_response(
                {"ok": False, "error": "Context not ready"}, status=503
            )
        body = await request.json()
        result = await set_context(str(body.get("text") or ""))
        ok = result.get("ok") is True
        return web.json_response(result, status=200 if ok else 500)

    async def get_tts_settings(_: web.Request) -> web.Response:
        if get_tts is None:
            return web.json_response(
                {"ok": False, "error": "TTS settings not ready"}, status=503
            )
        return web.json_response(await get_tts())

    async def set_tts_settings(request: web.Request) -> web.Response:
        if set_tts is None:
            return web.json_response(
                {"ok": False, "error": "TTS settings not ready"}, status=503
            )
        body = await request.json()
        result = await set_tts(
            backend=body.get("backend"),
            chatterbox_url=body.get("chatterbox_url"),
            voice_id=body.get("voice_id"),
            speed=body.get("speed"),
            tonality=body.get("tonality"),
        )
        ok = result.get("ok") is True
        return web.json_response(result, status=200 if ok else 400)

    async def get_llm_settings(_: web.Request) -> web.Response:
        if get_llm is None:
            return web.json_response(
                {"ok": False, "error": "LLM settings not ready"}, status=503
            )
        return web.json_response(await get_llm())

    async def set_llm_settings(request: web.Request) -> web.Response:
        if set_llm is None:
            return web.json_response(
                {"ok": False, "error": "LLM settings not ready"}, status=503
            )
        body = await request.json()
        result = await set_llm(
            backend=body.get("backend"),
            local_url=body.get("local_url"),
            local_model=body.get("local_model"),
        )
        ok = result.get("ok") is True
        return web.json_response(result, status=200 if ok else 400)

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
    app.router.add_post("/api/clear", clear_context)
    app.router.add_post("/api/power/off", power_off)
    app.router.add_post("/api/power/on", power_on)
    app.router.add_get("/api/context", get_pinned)
    app.router.add_post("/api/context", set_pinned)
    app.router.add_get("/api/tts", get_tts_settings)
    app.router.add_post("/api/tts", set_tts_settings)
    app.router.add_get("/api/llm", get_llm_settings)
    app.router.add_post("/api/llm", set_llm_settings)
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
    get_context: Optional[ContextGetter] = None,
    set_context: Optional[ContextSetter] = None,
    get_tts: Optional[TtsGetter] = None,
    set_tts: Optional[TtsSetter] = None,
    get_llm: Optional[LlmGetter] = None,
    set_llm: Optional[LlmSetter] = None,
    get_devices: Optional[Callable[[], dict[str, Any]]] = None,
    set_input: Optional[Callable[[int], Awaitable[dict[str, Any]]]] = None,
    set_output: Optional[Callable[[int], Awaitable[dict[str, Any]]]] = None,
) -> web.AppRunner:
    runner = web.AppRunner(
        create_app(
            on_start=on_start,
            on_stop=on_stop,
            on_clear=on_clear,
            on_power_on=on_power_on,
            on_power_off=on_power_off,
            on_respond=on_respond,
            get_context=get_context,
            set_context=set_context,
            get_tts=get_tts,
            set_tts=set_tts,
            get_llm=get_llm,
            set_llm=set_llm,
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
