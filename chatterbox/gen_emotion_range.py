import io, wave, urllib.request
from pathlib import Path

# Nano/Turbo: emotion via wording + paralinguistic tags (exaggeration ignored)
SECTIONS = [
    ("Warm and friendly",
     "Hey, it's really good to hear from you. I hope your day's going well."),
    ("Calm and steady",
     "Take a breath. We can go through this carefully, one step at a time."),
    ("Excited and upbeat",
     "No way — that actually worked! This is going to be so much fun."),
    ("Serious and firm",
     "I need you to listen carefully. This part matters, and we cannot skip it."),
    ("Soft and reassuring",
     "It's okay. You're not alone in this, and we will figure it out together."),
    ("Urgent",
     "We need to move now. There's no time to wait — follow my lead."),
    ("Tired and resigned",
     "Yeah... it's been a long day. I'm ready for this to be over."),
    ("Curious",
     "Hmm. That's interesting. Why would it behave like that?"),
    ("Laugh",
     "Oh come on, that is ridiculous. [laugh] Okay, okay, I'm done."),
    ("Chuckle",
     "Well, that was unexpected. [chuckle] Fair enough, I guess."),
    ("Sigh",
     "[sigh] Alright. Let's try that again, a little more carefully this time."),
    ("Gasp",
     "[gasp] Wait — did you see that? I did not see that coming."),
    ("Cough and clear throat",
     "[clear throat] Excuse me. [cough] Sorry — where was I?"),
    ("Groan",
     "[groan] Ugh, not this again. Fine, I'll deal with it."),
    ("Shush and sniff",
     "[shush] Keep it down for a second. [sniff] Okay... something smells off."),
]

OUT = Path("voices/emotion_range.wav")
API = "http://localhost:8090/tts"

def post_tts(text: str) -> bytes:
    boundary = "----EmotionRangeDemo"
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
            raise SystemExit(f"HTTP {resp.status}: {resp.read()[:300]!r}")
        return resp.read()

def read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())

parts = []
sr = sw = ch = None
for i, (label, line) in enumerate(SECTIONS, 1):
    # Speak label then line so listener can identify each mode
    text = f"{label}. {line}"
    print(f"[{i}/{len(SECTIONS)}] {label}...", flush=True)
    raw = post_tts(text)
    c, s, r, frames = read_wav(raw)
    if sr is None:
        ch, sw, sr = c, s, r
    elif (c, s, r) != (ch, sw, sr):
        raise SystemExit(f"format mismatch at {label}")
    parts.append(frames)
    print(f"  {len(frames)/(ch*sw*sr):.2f}s", flush=True)

gap = b"\x00" * (ch * sw * int(sr * 0.35))
joined = gap.join(parts)
total = len(joined) / (ch * sw * sr)
with wave.open(str(OUT), "wb") as w:
    w.setnchannels(ch)
    w.setsampwidth(sw)
    w.setframerate(sr)
    w.writeframes(joined)
print(f"wrote {OUT} duration_s={total:.2f}", flush=True)
