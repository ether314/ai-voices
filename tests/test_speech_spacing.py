"""Smoke / regression tests for TTS speech spacing, unglue, and brand repairs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.expression import (  # noqa: E402
    join_speech_chunks,
    normalize_speech_for_tts as norm,
    repair_tag_spacing,
    smart_append,
    strip_tags,
)


def _char_stream(text: str) -> str:
    out = ""
    for ch in text:
        out = smart_append(out, ch)
    return norm(out)


def test_magsafe_sentence() -> None:
    raw = (
        "Mag Safeis Apple'smagneticchargingconnector, "
        "buttheyremoveditfromi Phonesyearsago."
    )
    got = norm(raw)
    assert got == (
        "MagSafe is Apple's magnetic charging connector, "
        "but they removed it from iPhones years ago."
    )


def test_magsafe_curly_apostrophe() -> None:
    raw = (
        "Mag Safeis Apple\u2019smagneticchargingconnector, "
        "buttheyremoveditfromi Phonesyearsago."
    )
    got = norm(raw)
    assert "MagSafe is" in got
    assert "magnetic charging connector" in got
    assert "iPhones years ago" in got
    assert "Apple" in got
    # Curly or straight possessive is fine; must be unglued from following word.
    assert "smagnetic" not in got.lower().replace("\u2019", "'")


def test_sharp_ensyour() -> None:
    assert norm("sharp ensyour") == "sharpens your"
    assert norm("sharp ens") == "sharpens"
    assert norm("ownvoice") == "own voice"
    assert norm("charg ing") == "charging"
    assert norm("import ant") == "important"
    assert norm("im port ant") == "important"
    assert norm("connect or") == "connector"


def test_common_glues() -> None:
    assert norm("redflag") == "red flag"
    assert "but it" in norm("butit")
    assert "the wallet" in norm("thewallet")
    assert norm("buttheyremoveditfrom") == "but they removed it from"
    assert norm("gotit") == "got it"
    assert norm("inthe") == "in the"
    assert norm("ofthe") == "of the"
    assert norm("andthe") == "and the"
    assert norm("thisis") == "this is"
    assert norm("lookslike") == "looks like"
    assert norm("makessense") == "makes sense"
    assert norm("soundsgood") == "sounds good"
    assert "they cannot see the issue" == norm("theycannotseetheissue")
    assert norm("lookslikeagoodplan") == "looks like a good plan"
    assert norm("Hey,howareyoudoing?") == "Hey, how are you doing?"


def test_punct_gap() -> None:
    assert norm("Yes,I fixed that.") == "Yes, I fixed that."
    assert norm("Wait;I need a second.") == "Wait; I need a second."
    assert norm("Really?I doubt it.") == "Really? I doubt it."
    assert norm("Stop:I heard you.") == "Stop: I heard you."


def test_char_stream_join() -> None:
    assert _char_stream("Yes, I fixed that.") == "Yes, I fixed that."
    assert "MagSafe" in _char_stream("MagSafe is ready.")
    assert _char_stream("Imagine a drone.") == "Imagine a drone."


def test_join_speech_chunks() -> None:
    pieces = ["Mag", " Safeis", " Apple'smagnetic", "chargingconnector."]
    got = join_speech_chunks(pieces)
    assert "MagSafe is" in got
    assert "magnetic charging connector" in got

    glued = join_speech_chunks(
        ["but", "they", "removed", "it", "from", "i", " Phones", "years", "ago."]
    )
    assert "iPhones" in glued or "i Phones" not in glued


def test_chunk_stream_adversarial() -> None:
    # Mid-word BPE-like cuts that smart_append must not "fix" with spaces.
    pieces = ["shar", "p", " ens", "your idea."]
    out = ""
    for p in pieces:
        out = smart_append(out, p)
    assert "sharp ens" in out or "sharpens" in norm(out)
    assert norm(out) == "sharpens your idea."

    pieces2 = ["I", "magi", "ne", "a", "drone."]
    # Without spaces between pieces that should glue into Imagine.
    out2 = ""
    for p in ["I", "magi", "ne a drone."]:
        out2 = smart_append(out2, p)
    # Conservative append may glue Imaginea...; normalize must repair.
    assert "Imagine" in norm(out2)


def test_strip_tags_keeps_speech() -> None:
    assert strip_tags("Hello [laugh] there.") == "Hello there."
    assert strip_tags("[chuckle] Okay then.") == "Okay then."
    assert "[laugh]" not in strip_tags("Hi [laugh] there [sigh] friend.")


def test_tag_spacing_repair() -> None:
    assert repair_tag_spacing("[ch uckle]") == "[chuckle]"
    assert repair_tag_spacing("[clear  throat]") == "[clear throat]"
    assert "[chuckle]" in norm("[ch uckle] Hello")


def test_iphone_not_split_as_i_word() -> None:
    assert norm("iPhone") == "iPhone"
    assert norm("iPhones") == "iPhones"
    assert norm("MagSafe") == "MagSafe"
    assert norm("iPad") == "iPad"
    assert norm("iMessage") == "iMessage"
    assert norm("iCloud") == "iCloud"


def test_brand_space_splits() -> None:
    assert norm("Mag Safe") == "MagSafe"
    assert norm("i Phones") == "iPhones"
    assert norm("i Phone") == "iPhone"
    assert norm("Air Pods") == "AirPods"
    assert norm("Air Pods Pro") == "AirPods Pro"
    assert norm("Mac Book") == "MacBook"
    assert norm("Mac Book Pro") == "MacBook Pro"
    assert norm("Face Time") == "FaceTime"
    assert norm("Air Play") == "AirPlay"
    assert norm("Home Pod") == "HomePod"
    assert norm("Vision Pro") == "Vision Pro"
    assert norm("Apple Watch") == "Apple Watch"
    assert norm("Apple TV") == "Apple TV"
    assert norm("App Store") == "App Store"
    assert norm("Wi Fi") == "Wi-Fi"
    assert norm("USB - C") == "USB-C"
    assert norm("Chat GPT") == "ChatGPT"
    assert norm("Open AI") == "OpenAI"
    assert norm("Git Hub") == "GitHub"
    assert norm("Whats App") == "WhatsApp"
    assert norm("Linked In") == "LinkedIn"
    assert norm("You Tube") == "YouTube"


def test_brand_compounds_glued() -> None:
    assert norm("wifi") == "Wi-Fi"
    assert norm("imessage") == "iMessage"
    assert norm("icloud") == "iCloud"
    assert norm("chatgpt") == "ChatGPT"
    assert norm("openai") == "OpenAI"
    assert norm("github") == "GitHub"
    assert norm("whatsapp") == "WhatsApp"
    assert norm("linkedin") == "LinkedIn"
    assert norm("facetime") == "FaceTime"
    assert norm("airplay") == "AirPlay"
    assert norm("magsafe") == "MagSafe"
    assert norm("iphones") == "iPhones"


def test_imagine_drone_sentence() -> None:
    raw = (
        "Sure. I magi neadronespottinganobjectanddecidingwhethertofollowit. "
        "Theimportantquestioniswhetherahumanapprovesthatactionfirst."
    )
    got = norm(raw)
    assert got == (
        "Sure. Imagine a drone spotting an object and deciding whether to follow it. "
        "The important question is whether a human approves that action first."
    )


def test_imagine_not_split_to_i() -> None:
    assert norm("Imagine a drone.") == "Imagine a drone."
    assert norm("Indeed it works.") == "Indeed it works."
    assert norm("Important news today.") == "Important news today."
    assert norm("Immediate reply please.") == "Immediate reply please."
    assert norm("Instead of waiting.") == "Instead of waiting."
    assert "I fixed" in norm("Ifixed that.")
    assert norm("Iwillspeak") == "I will speak"
    assert norm("Iwant") == "I want"
    assert norm("Ineed") == "I need"
    assert norm("Ihave") == "I have"
    assert norm("I think so") == "I think so"
    assert norm("I know") == "I know"


def test_broken_I_word_repairs() -> None:
    assert norm("I magi ne a drone") == "Imagine a drone"
    assert norm("I ndeed it works") == "Indeed it works"
    assert norm("I mportant news") == "Important news"
    # True pronoun phrases must survive.
    assert norm("I fixed that") == "I fixed that"
    assert norm("I will speak") == "I will speak"


def test_possessives_and_contractions() -> None:
    assert norm("Apple'smagnetic") == "Apple's magnetic"
    assert "magnetic" in norm("Apple\u2019smagnetic")
    assert norm("don'tknow") == "don't know"
    assert norm("can'tsee") == "can't see"
    assert norm("I'mgoing") == "I'm going"
    assert norm("it'sworking") == "it's working"
    assert norm("that'sright") == "that's right"
    assert norm("they'rehere") == "they're here"
    assert norm("we'vebeen") == "we've been"
    assert norm("I'drather") == "I'd rather"


def test_avoid_over_merge_real_phrases() -> None:
    assert norm("ice cream") == "ice cream"
    assert norm("come back") == "come back"
    assert norm("high school") == "high school"
    assert norm("cell phone") == "cell phone"
    assert norm("New York") == "New York"
    assert norm("go ahead") == "go ahead"
    assert norm("all right") == "all right"
    assert norm("right here") == "right here"
    assert norm("hang on") == "hang on"
    assert norm("hold on") == "hold on"
    assert norm("real time") == "real time"
    assert norm("open source") == "open source"
    assert norm("pass word") == "pass word"
    assert norm("web site") == "web site"
    assert norm("note book") == "note book"
    assert norm("home page") == "home page"
    assert norm("text book") == "text book"
    assert norm("voice mail") == "voice mail"
    assert norm("full screen") == "full screen"
    assert norm("dark mode") == "dark mode"
    assert norm("machine learning") == "machine learning"
    assert norm("soft ware") == "soft ware"
    assert norm("data base") == "data base"
    # Already-correct compounds stay intact.
    assert norm("software") == "software"
    assert norm("database") == "database"
    assert norm("sharpens your") == "sharpens your"


def test_round_trip_stable() -> None:
    samples = [
        "ice cream",
        "Imagine a drone.",
        "MagSafe is great.",
        "Yes, I fixed that.",
        "sharpens your",
        "Indeed it works.",
        "AirPods Pro",
        "but they removed it from iPhones years ago.",
        "looks like a good plan",
        "I'm going",
        "don't know",
        "Wi-Fi",
        "ChatGPT",
        "come back",
        "New York",
    ]
    for s in samples:
        once = norm(s)
        twice = norm(once)
        assert once == twice, (s, once, twice)


def test_smart_append_conservative() -> None:
    # Must not invent a mid-word space for lowercase continuations.
    assert smart_append("sharp", "ens") == "sharpens"
    # camelCase / Cap after lower still gets a space (brand fix later).
    assert smart_append("Mag", "Safe") == "Mag Safe"
    # Trust explicit leading space from the model.
    assert smart_append("sharp", " ens") == "sharp ens"
    # Punctuation gap.
    assert smart_append("Yes,", "I") == "Yes, I"
    # Open tag must not gain spaces.
    assert smart_append("[ch", "uckle]") == "[chuckle]"
    # Apostrophe contraction fragment.
    assert smart_append("I", "'m") == "I'm"
    assert smart_append("I", "\u2019m") == "I\u2019m"
    assert smart_append("Apple'", "s") == "Apple's"
    assert smart_append("Apple\u2019", "s") == "Apple\u2019s"


def test_long_glued_conversational() -> None:
    raw = (
        "Okay,soifyouconnectyourAir Podstothei PhoneoverWi Fi,"
        "Chat GPTcanhelpyoudebugtheissueintheApp Storebuild."
    )
    got = norm(raw)
    assert "AirPods" in got
    assert "iPhone" in got
    assert "Wi-Fi" in got
    assert "ChatGPT" in got
    assert "App Store" in got
    assert "so if you connect" in got or "if you connect" in got


def test_idempotent_on_good_speech() -> None:
    good = (
        "Sure. Imagine a drone spotting an object and deciding whether to follow it. "
        "The important question is whether a human approves that action first."
    )
    assert norm(good) == good
    assert norm(norm(good)) == good


def test_magsafe_char_stream() -> None:
    raw = (
        "Mag Safeis Apple'smagneticchargingconnector, "
        "buttheyremoveditfromi Phonesyearsago."
    )
    assert _char_stream(raw) == (
        "MagSafe is Apple's magnetic charging connector, "
        "but they removed it from iPhones years ago."
    )


def test_plural_brands_stable() -> None:
    assert norm("iPhones") == "iPhones"
    assert norm("iPads") == "iPads"
    assert norm("AirPods") == "AirPods"
    assert "iPhone s" not in norm("from iPhones years ago.")
    assert norm("from iPhones years ago.") == "from iPhones years ago."


def test_delivery_strip_normalize() -> None:
    assert strip_tags("DELIVERY: warm\nHello there.") == "Hello there."
    assert "MagSafe" in strip_tags("DELIVERY: calm\nMag Safe is ready.")


if __name__ == "__main__":
    tests = [
        test_magsafe_sentence,
        test_magsafe_curly_apostrophe,
        test_sharp_ensyour,
        test_common_glues,
        test_punct_gap,
        test_char_stream_join,
        test_join_speech_chunks,
        test_chunk_stream_adversarial,
        test_strip_tags_keeps_speech,
        test_tag_spacing_repair,
        test_iphone_not_split_as_i_word,
        test_brand_space_splits,
        test_brand_compounds_glued,
        test_imagine_drone_sentence,
        test_imagine_not_split_to_i,
        test_broken_I_word_repairs,
        test_possessives_and_contractions,
        test_avoid_over_merge_real_phrases,
        test_round_trip_stable,
        test_smart_append_conservative,
        test_long_glued_conversational,
        test_idempotent_on_good_speech,
        test_magsafe_char_stream,
        test_plural_brands_stable,
        test_delivery_strip_normalize,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"ERROR {fn.__name__}: {exc!r}")
    if failed:
        raise SystemExit(1)
    print(f"OK {len(tests)} tests")
