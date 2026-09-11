"""Paralinguistic tags + delivery tone parsing for spoken replies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

try:
    import wordninja
except ImportError:  # pragma: no cover
    wordninja = None  # type: ignore[assignment]

# Chatterbox Turbo added_tokens (ResembleAI/chatterbox-turbo).
CHATTERBOX_TAGS: tuple[str, ...] = (
    "angry",
    "fear",
    "surprised",
    "whispering",
    "advertisement",
    "dramatic",
    "narration",
    "crying",
    "happy",
    "sarcastic",
    "clear throat",
    "sigh",
    "shush",
    "cough",
    "groan",
    "sniff",
    "gasp",
    "chuckle",
    "laugh",
)

_TAG_ALT = "|".join(re.escape(t) for t in CHATTERBOX_TAGS)
TAG_RE = re.compile(rf"\[({_TAG_ALT})\]", re.IGNORECASE)
DELIVERY_LINE_RE = re.compile(
    r"^\s*DELIVERY\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
TAG_LIST_FOR_PROMPT = ", ".join(f"[{t}]" for t in CHATTERBOX_TAGS)

# Alphabetic runs that may be missing spaces (wordninja is safe on real words).
_GLUED_RUN_RE = re.compile(r"[A-Za-z]{5,}")
# Straight + curly apostrophes.
_APOS = "'’‘"
# Split tokens while leaving punctuation / tags alone.
_SPEECH_TOKEN_RE = re.compile(
    rf"\[[^\]]*\]|[A-Za-z]+(?:[{_APOS}][A-Za-z]+)?|[^A-Za-z\[]+"
)
# Two words that may be a single word falsely split by the streamer ("sharp ens").
_FALSE_SPLIT_RE = re.compile(r"\b([A-Za-z]{2,})\s+([a-zA-Z]{2,6})\b")
# Possessive / contraction glued to the next word: Apple'smagnetic, don'tknow.
_APOS_GLUE_RE = re.compile(
    rf"^([A-Za-z]+)([{_APOS}])(s|S|t|T|d|D|ll|LL|ve|VE|re|RE|m|M)([A-Za-z].+)$"
)

# Morphological / streamer suffix shards — safe to reattach when join is one word.
_MORPH_SUFFIXES: frozenset[str] = frozenset(
    {
        "ing",
        "ings",
        "ed",
        "en",
        "ens",
        "er",
        "ers",
        "est",
        "ly",
        "ally",
        "edly",
        "tion",
        "sion",
        "ment",
        "ments",
        "ness",
        "able",
        "ible",
        "ful",
        "less",
        "ous",
        "ious",
        "ive",
        "ize",
        "ise",
        "ity",
        "ities",
        "ary",
        "ory",
        "ance",
        "ence",
        "ant",
        "ent",
        "al",
        "ial",
        "ic",
        "ical",
        "ure",
        "ures",
        "ated",
        "ating",
        "ened",
        "ening",
        "ized",
        "izing",
        "ship",
        "hood",
        "dom",
        "or",
        "ors",
        "es",
        "ies",
        "ied",
        "ues",
        "ne",
        "gi",
        "magi",
    }
)

# Brands wordninja often splits or streams split — keep canonical spoken forms.
_BRAND_COMPOUNDS: dict[str, str] = {
    "magsafe": "MagSafe",
    "iphone": "iPhone",
    "iphones": "iPhones",
    "ipad": "iPad",
    "ipads": "iPads",
    "ipod": "iPod",
    "imac": "iMac",
    "ios": "iOS",
    "macos": "macOS",
    "macbook": "MacBook",
    "macbookpro": "MacBook Pro",
    "airpods": "AirPods",
    "airpodspro": "AirPods Pro",
    "applewatch": "Apple Watch",
    "appletv": "Apple TV",
    "homepod": "HomePod",
    "visionpro": "Vision Pro",
    "imessage": "iMessage",
    "icloud": "iCloud",
    "facetime": "FaceTime",
    "airplay": "AirPlay",
    "appstore": "App Store",
    "wifi": "Wi-Fi",
    "wi-fi": "Wi-Fi",
    "usb": "USB",
    "usbc": "USB-C",
    "chatgpt": "ChatGPT",
    "openai": "OpenAI",
    "github": "GitHub",
    "whatsapp": "WhatsApp",
    "linkedin": "LinkedIn",
    "youtube": "YouTube",
}

_BRAND_SPACE_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bMag\s+Safe\b", re.IGNORECASE), "MagSafe"),
    (re.compile(r"\bi\s+Phones\b", re.IGNORECASE), "iPhones"),
    (re.compile(r"\bi\s+Phone\b", re.IGNORECASE), "iPhone"),
    (re.compile(r"\bi\s+Pads\b", re.IGNORECASE), "iPads"),
    (re.compile(r"\bi\s+Pad\b", re.IGNORECASE), "iPad"),
    (re.compile(r"\bi\s+Mac\b", re.IGNORECASE), "iMac"),
    (re.compile(r"\bi\s+Cloud\b", re.IGNORECASE), "iCloud"),
    (re.compile(r"\bi\s+Message\b", re.IGNORECASE), "iMessage"),
    (re.compile(r"\bMac\s+Book\s+Pro\b", re.IGNORECASE), "MacBook Pro"),
    (re.compile(r"\bMac\s+Book\b", re.IGNORECASE), "MacBook"),
    (re.compile(r"\bAir\s+Pods\s+Pro\b", re.IGNORECASE), "AirPods Pro"),
    (re.compile(r"\bAir\s+Pods\b", re.IGNORECASE), "AirPods"),
    (re.compile(r"\bApple\s+Watch\b", re.IGNORECASE), "Apple Watch"),
    (re.compile(r"\bApple\s+TV\b", re.IGNORECASE), "Apple TV"),
    (re.compile(r"\bHome\s+Pod\b", re.IGNORECASE), "HomePod"),
    (re.compile(r"\bVision\s+Pro\b", re.IGNORECASE), "Vision Pro"),
    (re.compile(r"\bFace\s+Time\b", re.IGNORECASE), "FaceTime"),
    (re.compile(r"\bAir\s+Play\b", re.IGNORECASE), "AirPlay"),
    (re.compile(r"\bApp\s+Store\b", re.IGNORECASE), "App Store"),
    (re.compile(r"\bWi\s*-\s*Fi\b", re.IGNORECASE), "Wi-Fi"),
    (re.compile(r"\bWi\s+Fi\b", re.IGNORECASE), "Wi-Fi"),
    (re.compile(r"\bUSB\s*-\s*C\b", re.IGNORECASE), "USB-C"),
    (re.compile(r"\bChat\s*GPT\b", re.IGNORECASE), "ChatGPT"),
    (re.compile(r"\bOpen\s*AI\b", re.IGNORECASE), "OpenAI"),
    (re.compile(r"\bGit\s*Hub\b", re.IGNORECASE), "GitHub"),
    (re.compile(r"\bWhats\s*App\b", re.IGNORECASE), "WhatsApp"),
    (re.compile(r"\bLinked\s*In\b", re.IGNORECASE), "LinkedIn"),
    (re.compile(r"\bYou\s*Tube\b", re.IGNORECASE), "YouTube"),
)


def should_flush_phrase(text: str, *, first: bool) -> bool:
    """Flush early on the first phrase so audio can start before the LLM finishes."""
    s = text.rstrip()
    if not s:
        return False
    min_sent = 16 if first else 24
    hard_cap = 52 if first else 110
    if len(s) < min_sent:
        return False
    if s[-1] in ".!?…" and len(s) >= min_sent:
        return True
    if first and s[-1] in ",;:" and len(s) >= 28:
        return True
    if len(s) >= hard_cap:
        return True
    return False


def smart_append(buf: str, chunk: str) -> str:
    """Join streamed LLM pieces with almost no invented spaces.

    Heuristic joiners kept inserting mid-word gaps (sharp+ens) then gluing the
    next token (ens+your). We only:
      - preserve explicit whitespace from the model,
      - space after punctuation,
      - never touch text inside an open [tag].
    All other spacing is repaired in normalize_speech_for_tts().
    """
    if not chunk:
        return buf
    if not buf:
        return chunk

    # Preserve bracket tags: "[ch" + "uckle]" must not become "[ch uckle]"
    if buf.rfind("[") > buf.rfind("]"):
        return buf + chunk

    # Model already sent a leading space / newline — trust it.
    if chunk[0].isspace():
        return buf + chunk

    left = buf[-1]

    # Contraction / possessive fragments stay attached: "I" + "'m", "'" + "s"
    if left.isalnum() and chunk[0] in "'’":
        return buf + chunk
    if left in "'’" and chunk[0].isalnum():
        return buf + chunk

    # Gap after clause/sentence punctuation before a word.
    if left in ",.;:!?)]\"" and (
        chunk[0].isalnum() or chunk[0] in "\"'“‘([{"
    ):
        return buf + " " + chunk

    # camelCase / new capital after lowercase (rare without punct).
    if left.islower() and chunk[0].isupper() and chunk[0].isalpha():
        return buf + " " + chunk

    return buf + chunk


def join_speech_chunks(chunks: list[str] | tuple[str, ...]) -> str:
    """Fold streamed speech pieces with smart_append, then normalize."""
    out = ""
    for piece in chunks:
        if piece:
            out = smart_append(out, piece)
    return normalize_speech_for_tts(out)


def _canonical_brand(compact: str) -> str | None:
    key = compact.lower().replace("-", "").replace(" ", "")
    return _BRAND_COMPOUNDS.get(key) or _BRAND_COMPOUNDS.get(compact.lower())


def _peel_leading_brand(run: str) -> tuple[str, str] | None:
    """If run starts with a known brand, return (canonical, remainder)."""
    if len(run) < 4:
        return None
    compact = run.lower().replace("-", "")
    # Longest brand key first so macbookpro wins over macbook / iphones over iphone.
    keys = sorted(
        {k.replace("-", "").replace(" ", "") for k in _BRAND_COMPOUNDS},
        key=len,
        reverse=True,
    )
    for key in keys:
        if len(key) < 4 or len(compact) <= len(key):
            continue
        if compact.startswith(key):
            taken = 0
            letters = 0
            for ch in run:
                taken += 1
                if ch not in "- ":
                    letters += 1
                if letters >= len(key):
                    break
            rest = run[taken:]
            if not rest or not rest[0].isalpha():
                continue
            # Don't peel "iPhone" out of "...iPhones" style leftovers.
            if rest.lower() in {"s", "es"}:
                continue
            brand = _BRAND_COMPOUNDS.get(key) or _canonical_brand(run[:taken])
            if brand:
                return brand, rest
    return None


def _apply_brand_space_fixes(text: str) -> str:
    for pattern, repl in _BRAND_SPACE_FIXES:
        text = pattern.sub(repl, text)
    # Brands glued to the next word: "Chat GPTcan" / "i Phoneover".
    # Important: never peel a singular brand out of its plural (iPhone≠iPhones).
    glued_brand_res = (
        (re.compile(r"\bChat\s*GPT(?=[A-Za-z])", re.IGNORECASE), "ChatGPT "),
        (re.compile(r"\bOpen\s*AI(?=[A-Za-z])", re.IGNORECASE), "OpenAI "),
        (re.compile(r"\bGit\s*Hub(?=[A-Za-z])", re.IGNORECASE), "GitHub "),
        (re.compile(r"\bWhats\s*App(?=[A-Za-z])", re.IGNORECASE), "WhatsApp "),
        (re.compile(r"\bLinked\s*In(?=[A-Za-z])", re.IGNORECASE), "LinkedIn "),
        (re.compile(r"\bYou\s*Tube(?=[A-Za-z])", re.IGNORECASE), "YouTube "),
        (re.compile(r"\bMag\s*Safe(?=[A-Za-z])", re.IGNORECASE), "MagSafe "),
        (re.compile(r"\bi\s*Phones(?=[A-Za-z])", re.IGNORECASE), "iPhones "),
        (re.compile(r"\bi\s*Phone(?![sS])(?=[A-Za-z])", re.IGNORECASE), "iPhone "),
        (re.compile(r"\bi\s*Pads(?=[A-Za-z])", re.IGNORECASE), "iPads "),
        (re.compile(r"\bi\s*Pad(?![sS])(?=[A-Za-z])", re.IGNORECASE), "iPad "),
        (re.compile(r"\bAir\s*Pods(?=[A-Za-z])", re.IGNORECASE), "AirPods "),
        (re.compile(r"\bMac\s*Book(?=[A-Za-z])", re.IGNORECASE), "MacBook "),
        (re.compile(r"\bFace\s*Time(?=[A-Za-z])", re.IGNORECASE), "FaceTime "),
        (re.compile(r"\bWi\s*-?\s*Fi(?=[A-Za-z])", re.IGNORECASE), "Wi-Fi "),
        (re.compile(r"\bApp\s*Store(?=[A-Za-z])", re.IGNORECASE), "App Store "),
    )
    for pattern, repl in glued_brand_res:
        text = pattern.sub(repl, text)
    return text


def _word_cost(word: str) -> float:
    if wordninja is None:
        return 0.0
    return wordninja.DEFAULT_LANGUAGE_MODEL._wordcost.get(  # type: ignore[attr-defined]
        word.lower(), float("inf")
    )


def _is_known_word(word: str) -> bool:
    return _word_cost(word) < float("inf")


def _format_merged(left: str, joined: str) -> str:
    if left[0].isupper():
        return joined[0].upper() + joined[1:]
    return joined


def _should_merge_false_split(left: str, right: str, joined: str) -> bool:
    """Decide whether a spaced pair is a streamer mid-word break.

    Prefer keeping real two-word phrases (ice cream, come back) over collapsing
    them into compounds that happen to also be dictionary entries.
    """
    brand = _canonical_brand(joined)
    if brand:
        return True
    if wordninja is None:
        return False
    parts = wordninja.split(joined)
    brand = _canonical_brand("".join(parts))
    if brand:
        return True

    single = len(parts) == 1 and parts[0].lower() == joined.lower()
    rl = right.lower()

    # Morphological reassembly: sharp ens / charg ing / import ant.
    if single and rl in _MORPH_SUFFIXES:
        return True
    # Incomplete stem + common ending (charg+ing already covered; also rare stems).
    if single and not _is_known_word(left) and _is_known_word(joined):
        return True
    # Short left stem (im+port → import) completing a known word.
    if single and len(left) <= 2 and _is_known_word(joined):
        return True
    # Very short right shard that completes a known word (en, or, al…).
    if single and len(right) <= 2 and _is_known_word(joined):
        return True
    # Two solid content words → keep the space (ice cream, come back, high school).
    if (
        len(left) >= 3
        and len(right) >= 3
        and _is_known_word(left)
        and _is_known_word(right)
        and rl not in _MORPH_SUFFIXES
    ):
        return False
    # Remaining single-word joins with a short right fragment.
    if single and len(right) <= 4 and len(left) >= 3:
        return True
    return False


def _unglue_run(run: str) -> str:
    """Split a glued alphabetic run; leave real dictionary words / brands intact."""
    if len(run) < 4:
        brand = _canonical_brand(run)
        return brand if brand else run
    brand = _canonical_brand(run)
    if brand:
        return brand
    peeled = _peel_leading_brand(run)
    if peeled:
        brand_form, rest = peeled
        rest_fixed = _unglue_run(rest) if len(rest) >= 3 else rest
        return f"{brand_form} {rest_fixed}".rstrip()
    if wordninja is None:
        return run
    parts = wordninja.split(run)
    compact = "".join(parts)
    brand = _canonical_brand(compact)
    if brand:
        return brand
    # "Ifixed" / "Iwillspeak" — wordninja already isolates leading I.
    # Do NOT strip I from real words (Imagine, Indeed, Important).
    if (
        run.startswith("I")
        and len(run) > 1
        and run[1].islower()
        and parts
        and parts[0] == "I"
        and len(parts) > 1
    ):
        return " ".join(parts)
    if len(parts) <= 1:
        return run
    # Preserve leading capital: "Redflag" -> "Red flag"
    if run[0].isupper() and parts[0] and parts[0][0].islower():
        parts[0] = parts[0][:1].upper() + parts[0][1:]
    return " ".join(parts)


def _repair_broken_I_words(text: str) -> str:
    """Rejoin 'I magi ne' → 'Imagine' when a capital-I word was falsely split."""
    if not text or wordninja is None:
        return text

    # Greedy: I + lowercase shards. Replacer keeps real "I fixed" / "I will".
    pattern = re.compile(r"\bI(?:\s+[a-z]{1,8}){1,4}\b")

    def _repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        letters = re.sub(r"\s+", "", raw)
        if len(letters) < 5:
            return raw
        # Prefer the longest head that is a single dictionary word (Imagine, not I).
        for end in range(len(letters), 4, -1):
            head = letters[:end]
            brand = _canonical_brand(head)
            if brand:
                tail = letters[end:]
                if not tail:
                    return brand
                return f"{brand} {_unglue_run(tail)}" if len(tail) >= 3 else f"{brand} {tail}"
            parts = wordninja.split(head)
            if len(parts) == 1 and parts[0].lower() == head.lower() and len(head) >= 5:
                word = parts[0]
                if word[0].islower():
                    word = word[0].upper() + word[1:]
                # Must be a real capital-I word, not pronoun + glue artifact.
                if not word.lower().startswith("i") or len(word) < 5:
                    continue
                if word.lower() == "i":
                    continue
                tail = letters[end:]
                if not tail:
                    return word
                rest = _unglue_run(tail) if len(tail) >= 3 else tail
                return f"{word} {rest}"
        return raw

    return pattern.sub(_repl, text)


def _merge_false_splits(text: str) -> str:
    """Rejoin streamer splits like 'sharp ens' -> 'sharpens' when that is one word."""
    if not text:
        return text

    def _repl(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        joined = left + right
        brand = _canonical_brand(joined)
        if brand:
            return brand
        if not _should_merge_false_split(left, right, joined):
            return match.group(0)
        brand = _canonical_brand(joined)
        if brand:
            return brand
        return _format_merged(left, joined)

    prev = None
    cur = text
    for _ in range(6):
        if cur == prev:
            break
        prev = cur
        cur = _FALSE_SPLIT_RE.sub(_repl, cur)
        cur = _apply_brand_space_fixes(cur)
        cur = _repair_broken_I_words(cur)
    return cur


def repair_tag_spacing(text: str) -> str:
    """Fix tags broken by stream joining, e.g. '[ch uckle]' -> '[chuckle]'."""
    if not text or "[" not in text:
        return text

    def _fix(match: re.Match[str]) -> str:
        inner = match.group(1)
        compact = re.sub(r"\s+", "", inner).lower()
        spaced = re.sub(r"\s+", " ", inner.strip()).lower()
        for tag in CHATTERBOX_TAGS:
            tag_compact = tag.replace(" ", "").lower()
            if compact == tag_compact or spaced == tag.lower():
                return f"[{tag}]"
        return match.group(0)

    return re.sub(r"\[([^\]]+)\]", _fix, text)


def ensure_speech_spacing(text: str) -> str:
    """Light structural spacing fixes (punct, camelCase)."""
    if not text:
        return text
    text = repair_tag_spacing(text)
    text = re.sub(r"([,.;:!?])([A-Za-z0-9\"'“‘])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Single-letter BPE breaks: "L osing" -> "Losing" (keep "I fixed" / "A day").
    # Leading-I glued words (Ifixed / Imagine) are handled in _unglue_run — do not
    # blindly insert a space after capital I (that turns Imagine into I magine).
    text = re.sub(r"\b([B-HJ-Z]) ([a-z]{2,})\b", r"\1\2", text)
    text = re.sub(r"\b([b-hj-z]) ([a-z]{2,})\b", r"\1\2", text)
    text = re.sub(r" {2,}", " ", text)
    return text


def _unglue_token(tok: str) -> str:
    """Unglue one alphabetic / possessive / contraction token."""
    apos = _APOS_GLUE_RE.match(tok)
    if apos:
        stem, mark, contr, rest = (
            apos.group(1),
            apos.group(2),
            apos.group(3),
            apos.group(4),
        )
        return f"{stem}{mark}{contr} {_unglue_run(rest)}"
    for mark in _APOS:
        if mark in tok:
            head, _, tail = tok.partition(mark)
            fixed = _unglue_run(head)
            return f"{fixed}{mark}{tail}"
    return _unglue_run(tok)


def normalize_speech_for_tts(text: str) -> str:
    """Hard gate before audio: merge false splits, unglue compounds, tidy spaces.

    This is the authoritative spacing pass. Call it on every phrase before synth.
    """
    if not text or not text.strip():
        return text
    original = text

    text = ensure_speech_spacing(text)
    text = repair_tag_spacing(text)
    text = _apply_brand_space_fixes(text)

    # 1) Unglue each alphabetic token ("ownvoice", "ensyour", "redflag").
    pieces: list[str] = []
    for tok in _SPEECH_TOKEN_RE.findall(text):
        if tok.startswith("["):
            pieces.append(tok)
        elif tok.isalpha() or (
            len(tok) > 2 and re.sub(rf"[{_APOS}]", "", tok).isalpha()
        ):
            pieces.append(_unglue_token(tok))
        else:
            pieces.append(tok)
    text = "".join(pieces)

    # 2) Rejoin false mid-word spaces ("sharp ens" -> "sharpens").
    text = _repair_broken_I_words(text)
    text = _merge_false_splits(text)
    text = _apply_brand_space_fixes(text)

    # 3) Another unglue + merge pass for leftovers.
    text = _GLUED_RUN_RE.sub(lambda m: _unglue_run(m.group(0)), text)
    text = _repair_broken_I_words(text)
    text = _merge_false_splits(text)
    text = _apply_brand_space_fixes(text)

    text = repair_tag_spacing(text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.strip()
    if text != original.strip() and original.strip():
        print(f"[tts-normalize] {original.strip()!r} -> {text!r}", flush=True)
    return text


def sanitize_delivery_header(text: str) -> str:
    """Fix 'DEL IVERY' corruption from bad spacing inserts."""
    return re.sub(r"\bD\s*E\s*L\s*I\s*V\s*E\s*R\s*Y\s*:", "DELIVERY:", text, flags=re.I)


@dataclass
class ExpressionState:
    delivery: str = ""
    tags: list[str] = field(default_factory=list)

    def to_event_text(self) -> str:
        return json.dumps(
            {"delivery": self.delivery, "tags": list(self.tags)},
            ensure_ascii=False,
        )


def strip_tags(text: str) -> str:
    """Remove tags / DELIVERY headers; normalize spacing for captions + TTS."""
    cleaned = text or ""
    cleaned = re.sub(
        r"^\s*DELIVERY\s*:.*(?:\n|$)", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    )
    cleaned = repair_tag_spacing(cleaned)
    cleaned = TAG_RE.sub("", cleaned)
    # Drop any leftover bracket junk (legacy emotion tags).
    cleaned = re.sub(r"\[[^\]]{0,40}\]", "", cleaned)
    cleaned = normalize_speech_for_tts(cleaned)
    return re.sub(r" {2,}", " ", cleaned).strip()


def extract_tags(text: str) -> list[str]:
    found: list[str] = []
    repaired = repair_tag_spacing(text or "")
    for match in TAG_RE.finditer(repaired):
        tag = match.group(1).lower()
        if tag not in found:
            found.append(tag)
    return found


def speech_for_tts(text: str, *, backend: str = "") -> str:
    """Normalize speech for any TTS backend (no emotion tags)."""
    del backend  # kept for call-site compatibility
    return strip_tags(text)


@dataclass
class _Emit:
    kind: str  # "delivery" | "speech" | "expression"
    text: str = ""
    state: ExpressionState | None = None


class ExpressionStreamParser:
    """Pull DELIVERY: line + tags out of a streamed LLM reply."""

    def __init__(self) -> None:
        self._header_buf = ""
        self._speech = ""
        self._emitted = 0
        self._header_done = False
        self.state = ExpressionState()
        self._seen_tags: set[str] = set()

    def feed(self, chunk: str) -> list[_Emit]:
        if not chunk:
            return []
        out: list[_Emit] = []

        if not self._header_done:
            self._header_buf += chunk
            self._header_buf = sanitize_delivery_header(self._header_buf)

            if "\n" not in self._header_buf:
                upper = re.sub(r"\s+", "", self._header_buf.upper())
                if len(self._header_buf) > 24 and not upper.startswith("DELIVERY"):
                    # Model skipped DELIVERY — treat everything as speech.
                    self._header_done = True
                    self._speech = smart_append("", self._header_buf)
                    self._header_buf = ""
                    out.extend(self._emit_new_speech())
                return out

            first, rest = self._header_buf.split("\n", 1)
            first = sanitize_delivery_header(first)
            match = re.match(r"^\s*DELIVERY\s*:\s*(.+?)\s*$", first, re.IGNORECASE)
            self._header_done = True
            self._header_buf = ""
            if match:
                self.state.delivery = match.group(1).strip()
                out.append(_Emit(kind="delivery", text=self.state.delivery))
                out.append(
                    _Emit(
                        kind="expression",
                        state=ExpressionState(
                            delivery=self.state.delivery, tags=list(self.state.tags)
                        ),
                    )
                )
                if rest:
                    self._speech = smart_append(self._speech, rest)
                    out.extend(self._emit_new_speech())
            else:
                # No delivery line — whole buffer is speech (keep the newline).
                self._speech = smart_append(self._speech, first + "\n" + rest)
                out.extend(self._emit_new_speech())
            return out

        self._speech = smart_append(self._speech, chunk)
        out.extend(self._emit_new_speech())
        return out

    def _emit_new_speech(self) -> list[_Emit]:
        if len(self._speech) <= self._emitted:
            return []
        delta = self._speech[self._emitted :]
        self._emitted = len(self._speech)
        out = self._speech_emits(delta)
        # Tags may complete across chunk boundaries — scan the full speech.
        for tag in extract_tags(self._speech):
            if tag not in self._seen_tags:
                self._seen_tags.add(tag)
                self.state.tags.append(tag)
                out.append(
                    _Emit(
                        kind="expression",
                        state=ExpressionState(
                            delivery=self.state.delivery, tags=list(self.state.tags)
                        ),
                    )
                )
        return out

    def flush(self) -> list[_Emit]:
        if not self._header_done and self._header_buf:
            speech = self._header_buf
            self._header_buf = ""
            self._header_done = True
            if re.match(r"^\s*DELIVERY\s*:", speech, re.IGNORECASE) and "\n" not in speech:
                match = re.match(r"^\s*DELIVERY\s*:\s*(.+?)\s*$", speech, re.IGNORECASE)
                if match:
                    self.state.delivery = match.group(1).strip()
                    return [
                        _Emit(kind="delivery", text=self.state.delivery),
                        _Emit(
                            kind="expression",
                            state=ExpressionState(
                                delivery=self.state.delivery, tags=list(self.state.tags)
                            ),
                        ),
                    ]
                return []
            self._speech = smart_append(self._speech, speech)
        return self._emit_new_speech()

    def _speech_emits(self, speech: str) -> list[_Emit]:
        out: list[_Emit] = []
        if not speech:
            return out
        # Strip a late DELIVERY line if the model put it mid-stream.
        speech = sanitize_delivery_header(speech)
        speech = DELIVERY_LINE_RE.sub("", speech)
        # Drop a corrupted spoken "DEL IVERY: ..." prefix if header parse missed it.
        speech = re.sub(
            r"^\s*DEL\s*IVERY\s*:\s*[^\n]*\n?",
            "",
            speech,
            count=1,
            flags=re.IGNORECASE,
        )
        speech = ensure_speech_spacing(speech)
        new_tags = extract_tags(speech)
        added = False
        for tag in new_tags:
            if tag not in self._seen_tags:
                self._seen_tags.add(tag)
                self.state.tags.append(tag)
                added = True
        out.append(_Emit(kind="speech", text=speech))
        if added:
            out.append(
                _Emit(
                    kind="expression",
                    state=ExpressionState(
                        delivery=self.state.delivery, tags=list(self.state.tags)
                    ),
                )
            )
        return out


def expression_prompt_block() -> str:
    return (
        "Emotional delivery (required format):\n"
        "1) First line MUST be exactly: DELIVERY: <short tone phrase>\n"
        "   Pick a tone that fits the call context — e.g. warm coaching, calm urgency, "
        "curious probe, light humor, firm redirect, empathetic support.\n"
        "2) Then speak 2–4 short sentences. Start with one short sentence.\n"
        "3) Optionally embed at most 1–2 paralinguistic tags from this allowlist when "
        "they feel natural (do not overuse):\n"
        f"   {TAG_LIST_FOR_PROMPT}\n"
        "   Prefer emotion tags like [happy]/[sigh]/[chuckle] that match the DELIVERY tone "
        "and the transcript. Never invent tags outside the allowlist.\n"
        "4) Do not explain the tags. Do not use markdown. Spoken words only after DELIVERY.\n"
        "5) Spacing matters: put a normal space between every word. "
        "Never concatenate words.\n"
    )
