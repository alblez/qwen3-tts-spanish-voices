"""Configuration management for spanish-tts."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("spanish-tts.config")

VOICES_FILENAME = "voices.yaml"
DEFAULT_CONFIG_DIR = Path.home() / ".spanish-tts"
DEFAULT_VOICES_FILE = Path(__file__).parent.parent.parent / "presets" / VOICES_FILENAME


def get_config_dir() -> Path:
    """Get or create config directory."""
    config_dir = Path(os.environ.get("SPANISH_TTS_CONFIG", DEFAULT_CONFIG_DIR))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_references_dir() -> Path:
    """Get or create references directory."""
    ref_dir = get_config_dir() / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir


def load_voices(voices_file: Path | None = None) -> dict[str, Any]:
    """Load voice registry from YAML."""
    if voices_file is None:
        # Check user config first, fall back to bundled presets
        user_voices = get_config_dir() / VOICES_FILENAME
        if user_voices.exists():
            voices_file = user_voices
        else:
            voices_file = DEFAULT_VOICES_FILE

    with open(voices_file) as f:
        data = yaml.safe_load(f)

    return data


def save_voices(data: dict[str, Any], voices_file: Path | None = None):
    """Save voice registry to YAML."""
    if voices_file is None:
        voices_file = get_config_dir() / VOICES_FILENAME

    voices_file.parent.mkdir(parents=True, exist_ok=True)
    with open(voices_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_voice(name: str, voices_file: Path | None = None) -> dict[str, Any] | None:
    """Get a specific voice by name."""
    data = load_voices(voices_file)
    voices = data.get("voices", {})
    return voices.get(name)


def add_voice(
    name: str,
    voice_data: dict[str, Any],
    voices_file: Path | None = None,
    allow_overwrite: bool = True,
):
    """Add or update a voice in the registry.

    When a voice with ``name`` already exists, log a warning before
    overwriting. If the existing entry has a different ``type`` than the
    new one (typically ``design`` being shadowed by ``clone``, which is
    the real collision `scripts/curate.py` produces against bundled
    presets like ``neutral_female`` and ``warm_female``), log a more
    prominent warning message so the caller cannot miss the shadowing.
    Both cases emit at the same WARNING level.

    Args:
        name: Voice key.
        voice_data: Voice payload (must include ``type``).
        voices_file: Registry path; defaults to user config.
        allow_overwrite: If False, raise ``ValueError`` instead of
            overwriting an existing entry. Default True to preserve
            existing callers (`scripts/curate.py` intentionally
            replaces entries when re-exporting a clone).
    """
    data = load_voices(voices_file)
    if "voices" not in data:
        data["voices"] = {}

    existing = data["voices"].get(name)
    if existing is not None:
        if not allow_overwrite:
            raise ValueError(
                f"Voice {name!r} already exists. "
                f"Pass allow_overwrite=True to overwrite it, or choose a different name."
            )
        existing_type = existing.get("type", "unknown")
        new_type = voice_data.get("type", "unknown")
        if existing_type != new_type:
            logger.warning(
                "Voice %r is being OVERWRITTEN and its type CHANGES "
                "(%s -> %s). This shadows the previous registration. "
                "If %r is a bundled preset, consider registering your "
                "custom voice under a distinct name with a regional suffix "
                "(e.g. %r) to avoid colliding with it.",
                name,
                existing_type,
                new_type,
                name,
                f"{name}_es",
            )
        else:
            logger.warning(
                "Voice %r already exists and is being overwritten (same type: %s). "
                "To keep both, register your voice under a distinct name.",
                name,
                existing_type,
            )

    data["voices"][name] = voice_data
    save_voices(data, voices_file)


def list_voices(voices_file: Path | None = None) -> dict[str, dict[str, Any]]:
    """List all registered voices."""
    data = load_voices(voices_file)
    return data.get("voices", {})


def get_defaults(voices_file: Path | None = None) -> dict[str, Any]:
    """Get default settings."""
    data = load_voices(voices_file)
    return data.get(
        "defaults", {"language": "Spanish", "speed": 1.0, "output_dir": "~/tts-output/spanish"}
    )
