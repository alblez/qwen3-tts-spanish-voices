"""Configuration management for spanish-tts."""

import os
from pathlib import Path
from typing import Any

import yaml

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


def add_voice(name: str, voice_data: dict[str, Any], voices_file: Path | None = None):
    """Add or update a voice in the registry."""
    data = load_voices(voices_file)
    if "voices" not in data:
        data["voices"] = {}
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
