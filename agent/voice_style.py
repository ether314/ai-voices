"""Shared TTS delivery settings: speed + tonality for Cartesia and local Chatterbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SPEED_PATH = _DATA_DIR / "tts_speed.txt"
_TONALITY_PATH = _DATA_DIR / "tts_tonality.txt"

DEFAULT_SPEED = 1.0
DEFAULT_TONALITY = "neutral"

# Discrete UI options (Cartesia generation_config.speed is 0.6–1.5).
SPEED_CHOICES: tuple[dict[str, Any], ...] = (
    {"id": "0.8", "value": 0.8, "label": "Slow (0.8×)"},
    {"id": "0.9", "value": 0.9, "label": "Slightly slow (0.9×)"},
    {"id": "1.0", "value": 1.0, "label": "Normal (1.0×)"},
    {"id": "1.15", "value": 1.15, "label": "Slightly fast (1.15×)"},
    {"id": "1.3", "value": 1.3, "label": "Fast (1.3×)"},
    {"id": "1.45", "value": 1.45, "label": "Very fast (1.45×)"},
)

# Shared labels; each backend maps these to native knobs.
TONALITY_CHOICES: tuple[dict[str, str], ...] = (
    {"id": "neutral", "label": "Neutral"},
    {"id": "calm", "label": "Calm"},
    {"id": "warm", "label": "Warm"},
    {"id": "energetic", "label": "Energetic"},
    {"id": "serious", "label": "Serious"},
    {"id": "cheerful", "label": "Cheerful"},
)

# Cartesia Sonic generation_config.emotion (primary / well-supported values preferred).
_CARTESIA_EMOTION: dict[str, str] = {
    "neutral": "neutral",
    "calm": "calm",
    "warm": "affectionate",
    "energetic": "excited",
    "serious": "contemplative",
    "cheerful": "happy",
}

# Local Nano/Turbo: exaggeration is ignored; temperature + post speed are best-effort.
_LOCAL_TEMPERATURE: dict[str, float] = {
    "neutral": 0.8,
    "calm": 0.55,
    "warm": 0.7,
    "energetic": 1.05,
    "serious": 0.5,
    "cheerful": 0.95,
}

_LOCAL_EXAGGERATION: dict[str, float] = {
    "neutral": 0.0,
    "calm": 0.15,
    "warm": 0.35,
    "energetic": 0.7,
    "serious": 0.2,
    "cheerful": 0.55,
}


def clamp_speed(speed: float) -> float:
    return max(0.6, min(1.5, float(speed)))


def normalize_tonality(tonality: str) -> str:
    t = (tonality or "").strip().lower()
    known = {c["id"] for c in TONALITY_CHOICES}
    return t if t in known else DEFAULT_TONALITY


def _load_float(path: Path, default: float) -> float:
    try:
        if path.is_file():
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                return clamp_speed(float(raw))
    except (OSError, ValueError):
        pass
    return default


def _load_text(path: Path, default: str) -> str:
    try:
        if path.is_file():
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                return normalize_tonality(raw)
    except OSError:
        pass
    return default


def load_speed(default: float = DEFAULT_SPEED) -> float:
    return _load_float(_SPEED_PATH, default)


def load_tonality(default: str = DEFAULT_TONALITY) -> str:
    return _load_text(_TONALITY_PATH, default)


def save_speed(speed: float) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SPEED_PATH.write_text(f"{clamp_speed(speed):g}\n", encoding="utf-8")


def save_tonality(tonality: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _TONALITY_PATH.write_text(normalize_tonality(tonality) + "\n", encoding="utf-8")


def cartesia_generation_config(
    *,
    speed: float = DEFAULT_SPEED,
    tonality: str = DEFAULT_TONALITY,
) -> dict[str, Any]:
    """Sonic-3+ generation_config (speed + emotion)."""
    spd = clamp_speed(speed)
    emotion = _CARTESIA_EMOTION.get(normalize_tonality(tonality), "neutral")
    cfg: dict[str, Any] = {"speed": spd, "emotion": emotion}
    # Slight volume lift for energetic delivery.
    if normalize_tonality(tonality) == "energetic":
        cfg["volume"] = 1.15
    return cfg


def local_tts_params(
    *,
    speed: float = DEFAULT_SPEED,
    tonality: str = DEFAULT_TONALITY,
) -> dict[str, float]:
    """Form fields for Chatterbox /tts (temperature + optional exaggeration + speed)."""
    tone = normalize_tonality(tonality)
    return {
        "speed": clamp_speed(speed),
        "temperature": _LOCAL_TEMPERATURE.get(tone, 0.8),
        "exaggeration": _LOCAL_EXAGGERATION.get(tone, 0.0),
    }


def style_cache_tag(*, speed: float, tonality: str) -> str:
    """Short tag for filler PCM cache filenames."""
    return f"s{clamp_speed(speed):.2f}_{normalize_tonality(tonality)}"


def choices_payload() -> dict[str, Any]:
    return {
        "speeds": [dict(c) for c in SPEED_CHOICES],
        "tonalities": [dict(c) for c in TONALITY_CHOICES],
        "hint": (
            "Cartesia: Sonic generation_config speed + emotion (UI baseline), "
            "plus inline SSML / [laughter] from the LLM when present. "
            "Local: temperature (+ exaggeration if supported) and time-stretch speed; "
            "Cartesia SSML is adapted or stripped."
        ),
    }
