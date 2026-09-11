import io, wave, urllib.request
from pathlib import Path

# Script sized for ~60s spoken audio when chunked
CHUNKS = [
    "This is a sixty second sample of my local voice clone running on the GPU.",
    "Chatterbox Nano is generating each sentence from my reference clip, then stitching the pieces into one longer take.",
    "The goal is a continuous listening test, so you can hear tone, pacing, and consistency across a full minute.",
    "If something drifts, it is usually the reference audio, not a training step. Better clips make a cleaner clone.",
    "Short turns still feel more natural for an agent, but longer scripts like this are useful for judging quality.",
    "I am speaking clearly, with ordinary conversational rhythm, and without background music or effects.",
    "That covers the demo. Thanks for listening to this sixty second GPU voice sample.",
]

OUT = Path("voices/sample_60s.wav")
API = "http://localhost:8090/tts"

def post_tts(text: str) -> bytes:
    boundary = "----CursorBoundary60s"
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
            raise SystemExit(f"HTTP {resp.status}: {resp.read()[:200]!r}")
        return resp.read()

def read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())

parts = []
sr = sw = ch = None
for i, text in enumerate(CHUNKS, 1):
    print(f"[{i}/{len(CHUNKS)}] {len(text)} chars...", flush=True)
    raw = post_tts(text)
    c, s, r, frames = read_wav(raw)
    if sr is None:
        ch, sw, sr = c, s, r
    elif (c, s, r) != (ch, sw, sr):
        raise SystemExit(f"format mismatch chunk {i}")
    parts.append(frames)
    dur = len(frames) / (ch * sw * sr)
    print(f"  chunk_s={dur:.2f}", flush=True)

gap = b"\x00" * (ch * sw * int(sr * 0.2))
joined = gap.join(parts)
total = len(joined) / (ch * sw * sr)
print(f"stitched_s={total:.2f}", flush=True)

# If short of 60s, append extra filler chunks until we hit ~60
extra = [
    "Here is a little more speech so the sample reaches a full minute of audio.",
    "One more sentence keeps the pacing steady and fills out the remaining time.",
    "And a final line to round out the sixty second listening check.",
    "Almost there. This extra passage helps stretch the sample without changing the voice.",
    "Last bit of filler text for length, spoken in the same cloned voice.",
]
ei = 0
while total < 58.0 and ei < len(extra) * 3:
    text = extra[ei % len(extra)]
    ei += 1
    print(f"[extra {ei}] {len(text)} chars...", flush=True)
    raw = post_tts(text)
    c, s, r, frames = read_wav(raw)
    if (c, s, r) != (ch, sw, sr):
        raise SystemExit("format mismatch extra")
    joined = joined + gap + frames
    total = len(joined) / (ch * sw * sr)
    print(f"  now_s={total:.2f}", flush=True)

# Trim to ~60s if over
max_bytes = int(60 * ch * sw * sr)
if len(joined) > max_bytes:
    # keep whole frames
    frame = ch * sw
    joined = joined[: (max_bytes // frame) * frame]
    total = len(joined) / (ch * sw * sr)

with wave.open(str(OUT), "wb") as w:
    w.setnchannels(ch)
    w.setsampwidth(sw)
    w.setframerate(sr)
    w.writeframes(joined)
print(f"wrote {OUT} duration_s={total:.2f} bytes={OUT.stat().st_size}", flush=True)
