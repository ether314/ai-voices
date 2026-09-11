import io, os, struct, wave, urllib.request
from pathlib import Path

# ~45s of speech if spoken naturally; split into chunks for stable Nano CPU gens
CHUNKS = [
    "Alright, this is a longer sample of my local voice clone. I'm speaking through Chatterbox Nano, running on my own machine, with no cloud TTS in the loop.",
    "The reference clip was taken from my recording, trimmed to about ten seconds of clearer speech, then locked in as the default voice for the API.",
    "If this sounds like me, the clone is working. If it drifts, I can swap the reference wav and regenerate without any training run.",
    "For day to day use, short sentences land cleanest. Longer paragraphs still work if you chunk the text and stitch the audio back together, like this script is doing right now.",
    "That wraps the demo. Thanks for listening to this test of my personal voice model.",
]

OUT = Path("voices/long_sample.wav")
API = "http://localhost:8090/tts"

def post_tts(text: str) -> bytes:
    boundary = "----CursorBoundary7f3a"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="text"\r\n\r\n'
        f"{text}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        if resp.status != 200:
            raise SystemExit(f"HTTP {resp.status}")
        return resp.read()

def read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())

parts = []
sr = sw = ch = None
for i, text in enumerate(CHUNKS, 1):
    print(f"[{i}/{len(CHUNKS)}] generating ({len(text)} chars)...", flush=True)
    raw = post_tts(text)
    c, s, r, frames = read_wav(raw)
    if sr is None:
        ch, sw, sr = c, s, r
    elif (c, s, r) != (ch, sw, sr):
        raise SystemExit(f"format mismatch chunk {i}: {(c,s,r)} vs {(ch,sw,sr)}")
    parts.append(frames)
    print(f"  got {len(frames)} bytes frames", flush=True)

# 250ms silence between chunks
gap = b"\x00" * (ch * sw * int(sr * 0.25))
joined = gap.join(parts)
with wave.open(str(OUT), "wb") as w:
    w.setnchannels(ch)
    w.setsampwidth(sw)
    w.setframerate(sr)
    w.writeframes(joined)
dur = len(joined) / (ch * sw * sr)
print(f"wrote {OUT} duration_s={dur:.2f} bytes={OUT.stat().st_size}", flush=True)
