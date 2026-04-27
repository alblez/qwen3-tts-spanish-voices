"""Configuration management for spanish-tts."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("spanish_tts.config")

VOICES_FILENAME = "voices.yaml"
DEFAULT_CONFIG_DIR = Path.home() / ".spanish-tts"
DEFAULT_VOICES_FILE = Path(__file__).parent.parent.parent / "presets" / VOICES_FILENAME

# Valid voice types for schema validation.
_VALID_VOICE_TYPES = {"clone", "design"}


def get_config_dir() -> Path:
    """Get or create config directory.

    Validates that the resolved path is under the user's home directory (or a
    known-safe temporary directory) when the SPANISH_TTS_CONFIG environment
    variable is set, to prevent accidental writes to system directories.

    NOTE: The path is resolved with `.expanduser().resolve()` before the guard
    check. Relative paths in SPANISH_TTS_CONFIG are therefore resolved against
    CWD. If CWD is outside $HOME (e.g. `/srv/app`), a relative path that would
    have worked in prior versions now raises ValueError. This is an intentional
    behavior change: use an absolute path to avoid ambiguity.

    Raises:
        ValueError: If SPANISH_TTS_CONFIG resolves outside $HOME and outside
            the system temp directory.
    """
    raw = os.environ.get("SPANISH_TTS_CONFIG")
    if raw is not None:
        config_dir = Path(raw).expanduser().resolve()
        home = Path.home().resolve()
        # Also allow the system temp dir (e.g. pytest tmp_path on macOS resolves
        # to /private/var/folders/… which is outside $HOME).
        tmpdir = Path(tempfile.gettempdir()).resolve()

        def _under(parent: Path) -> bool:
            try:
                config_dir.relative_to(parent)
                return True
            except ValueError:
                return False

        if not (_under(home) or _under(tmpdir)):
            raise ValueError(
                f"SPANISH_TTS_CONFIG must resolve under $HOME ({home}), got: {config_dir}"
            )
    else:
        config_dir = DEFAULT_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_references_dir() -> Path:
    """Get or create references directory."""
    ref_dir = get_config_dir() / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir


def _validate_voices_schema(data: dict[str, Any], source: Path) -> None:
    """Raise ValueError if *data* does not satisfy the minimal voices schema.

    Checks:
    - ``data["voices"]`` is a mapping.
    - Each entry has ``type`` in ``{"clone", "design"}``.
    - Clone entries have ``ref_audio`` as a non-empty string.

    Optional keys ``source_license`` (SPDX string) and ``source_url`` (str)
    are allowed but not validated beyond being strings when present.
    """
    voices = data.get("voices", {})
    if not isinstance(voices, dict):
        raise ValueError(
            f"voices.yaml 'voices' key must be a mapping, got {type(voices).__name__} in {source}"
        )
    for name, entry in voices.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Voice entry {name!r} must be a mapping, got {type(entry).__name__} in {source}"
            )
        vtype = entry.get("type")
        if vtype not in _VALID_VOICE_TYPES:
            raise ValueError(
                f"Voice {name!r} has invalid type {vtype!r}; "
                f"must be one of {sorted(_VALID_VOICE_TYPES)} in {source}"
            )
        if vtype == "clone":
            ref = entry.get("ref_audio")
            if not isinstance(ref, str) or not ref:
                raise ValueError(
                    f"Clone voice {name!r} must have a non-empty string 'ref_audio' in {source}"
                )
        for opt_key in ("source_license", "source_url"):
            val = entry.get(opt_key)
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"Voice {name!r} key {opt_key!r} must be a string when present in {source}"
                )


def load_voices(voices_file: Path | None = None) -> dict[str, Any]:
    """Load voice registry from YAML.

    Falls back to the bundled presets if the user voices file is corrupt or
    schema-invalid (logs the error; does **not** overwrite the corrupt file —
    the user must recover it manually or delete it to reset to presets).

    Returns:
        Parsed voice registry dict.  Always a non-None mapping.  When the user
        file is schema-invalid, the bundled presets are returned instead of the
        user data; the return value does not distinguish this substitution — check
        logs if you need to detect the fallback.

    Raises:
        ValueError: If the loaded data is not a YAML mapping (dict).
        ValueError: If the bundled presets themselves fail schema validation
            (indicates a corrupt install; no fallback is possible).
    """
    if voices_file is None:
        user_voices = get_config_dir() / VOICES_FILENAME
        if user_voices.exists():
            voices_file = user_voices
        else:
            voices_file = DEFAULT_VOICES_FILE

    try:
        raw = voices_file.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.error(
            "Corrupt voices.yaml at %s: %s; falling back to bundled presets. "
            "Fix or delete the file to restore normal operation.",
            voices_file,
            exc,
        )
        data = yaml.safe_load(DEFAULT_VOICES_FILE.read_text(encoding="utf-8"))

    # safe_load returns None for an empty file; normalise to empty dict.
    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(
            f"voices.yaml must be a YAML mapping (dict), got {type(data).__name__} in {voices_file}"
        )

    try:
        _validate_voices_schema(data, voices_file)
    except ValueError as exc:
        logger.error(
            "Schema-invalid voices.yaml at %s: %s; falling back to bundled "
            "presets. Fix or delete the file to restore normal operation.",
            voices_file,
            exc,
        )
        data = yaml.safe_load(DEFAULT_VOICES_FILE.read_text(encoding="utf-8"))
        # Re-validate presets; if THOSE are broken the install is corrupt
        # and we should raise.
        _validate_voices_schema(data, DEFAULT_VOICES_FILE)

    return data


def save_voices(data: dict[str, Any], voices_file: Path | None = None) -> None:
    """Save voice registry to YAML atomically.

    Writes to a ``.yaml.tmp`` sibling first, then renames atomically via
    ``os.replace``.  On POSIX this is guaranteed atomic within the same
    filesystem; a crash between the write and the rename leaves the tmp file
    behind but the original intact.

    Args:
        data: Voice registry dict to serialise.  Must pass
            :func:`_validate_voices_schema` — invalid entries raise
            ``ValueError`` before any file I/O occurs.
        voices_file: Destination path; defaults to user config.

    Raises:
        ValueError: If *data* contains schema-invalid voice entries.
    """
    if voices_file is None:
        voices_file = get_config_dir() / VOICES_FILENAME

    voices_file.parent.mkdir(parents=True, exist_ok=True)
    _validate_voices_schema(data, voices_file)
    tmp = voices_file.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, voices_file)


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
        voice_data: Voice payload (must include a valid ``type`` and satisfy
            :func:`_validate_voices_schema`).  Invalid entries raise
            ``ValueError`` via :func:`save_voices` before any write occurs.
        voices_file: Registry path; defaults to user config.
        allow_overwrite: If False, raise ``ValueError`` instead of
            overwriting an existing entry. Default True to preserve
            existing callers (`scripts/curate.py` intentionally
            replaces entries when re-exporting a clone).

    Raises:
        ValueError: If *name* already exists and *allow_overwrite* is False.
        ValueError: If *voice_data* contains schema-invalid entries.
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
