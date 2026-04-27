# SPDX-License-Identifier: MIT
"""TTS engine wrapping Qwen3-TTS MLX for clone and design modes."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("spanish_tts.engine")


@dataclass(frozen=True)
class TtsResult:
    """Return value from all ``generate_*`` engine entry points.

    Attributes:
        path: Absolute path to the written ``.wav`` file.
        duration_seconds: Exact duration computed from the raw audio array
            (``len(audio) / sample_rate``).  Always a finite float — never
            re-read from the file via ``sf.info``.

    The ``__str__`` shim returns ``self.path`` so existing callers that
    treat the result as a string via coercion (``click.echo(result)``,
    f-strings, ``str(result)``, ``print(result)``) continue working
    without modification.  Equality comparisons against plain path strings
    (``result == "/path/foo.wav"``) will **not** match — use
    ``result.path == "/path/foo.wav"`` or ``str(result) == "/path/foo.wav"``.
    """

    path: str
    duration_seconds: float

    def __str__(self) -> str:  # backward-compat shim
        return self.path


# Speed range constants
SPEED_MIN, SPEED_MAX = 0.5, 2.0

# Language name → Qwen3-TTS lang_code mapping (used by generate_design).
LANG_MAP: dict[str, str] = {
    "Spanish": "spanish",
    "English": "english",
    "Chinese": "chinese",
    "French": "french",
    "German": "german",
    "Italian": "italian",
    "Portuguese": "portuguese",
    "Japanese": "japanese",
    "Korean": "korean",
    "Russian": "russian",
}

_MAX_TEXT_LEN = 10_000


def _validate_text(text: str, max_len: int | None = _MAX_TEXT_LEN) -> str:
    """Validate text before synthesis.  Returns *text* unchanged on success.

    Args:
        text: Text to validate.
        max_len: Maximum allowed character count (default 10 000).  Pass
            ``None`` to skip the length check.

    Returns:
        The *text* argument unchanged (convenience for inline use).

    Raises:
        ValueError: If the text is empty/whitespace-only, contains a NUL
            byte, or exceeds *max_len* characters.
    """
    if text is None or not text.strip():
        raise ValueError("text is empty")
    if "\x00" in text:
        raise ValueError("text contains NUL byte")
    if max_len is not None and len(text) > max_len:
        raise ValueError(f"text too long ({len(text)} chars, max {max_len})")
    return text


# Default MLX models
MODELS = {
    "clone": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
}

# Known-good Hugging Face commit SHAs for each model.
#
# These are verified-good revision hashes. Passing revision= to
# mlx_audio.tts.load() pins the download to this exact commit and
# prevents silent breakage when the repo is updated.
#
# Bump process (see CONTRIBUTING.md § "Updating model revisions"):
#   1. Verify the new commit on the HF model card (check README, config).
#   2. Update the SHA below.
#   3. Test locally: `conda run -n qwen3-tts pytest -m requires_mlx -q`.
#   4. Add a CHANGELOG entry under [Unreleased] → ### Changed.
MODEL_REVISIONS: dict[str, str] = {
    "clone": "e7dd0585652209fa0d7783659aad4e8a324de11c",
    "design": "f90d617701d9f7f4ca499291e0b57f2b3c2fd2ee",
}

# Thread-safety: _model_cache is shared across threads.
# _cache_lock guards the cache dict — prevents duplicate loads when two threads
# call _get_model simultaneously for the same model_id.
# _generate_lock serializes the entire generation pipeline (model.generate →
# _collect_audio → _apply_speed → sf.write).  The MLX/MPS execution environment
# is not thread-safe; and stdio-MCP is pragmatically single-request-at-a-time,
# so this lock imposes no real-world throughput cost.
_model_cache: dict = {}
_cache_lock = threading.Lock()
_generate_lock = threading.Lock()


def _clear_cache() -> None:
    """Evict all cached models.  Used in tests to reset between runs."""
    with _cache_lock:
        _model_cache.clear()


def _apply_speed(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """Post-synthesis pitch-preserving time-stretch. No-op at speed=1.0.

    Contract: mono float audio only. Multi-channel input is rejected
    rather than silently averaged. Empty arrays and NaN/inf are rejected
    rather than passed to librosa where they produce confusing errors.

    Args:
        audio: Mono audio array (1-D). dtype is cast to float32 internally.
        speed: Speed factor (0.5-2.0; rate > 1.0 speeds up).
        sample_rate: Sample rate in Hz.

    Returns:
        Time-stretched audio array (float32, 1-D). Falls back to unmodified
        audio if librosa is not installed.

    Raises:
        ValueError: If speed is outside 0.5-2.0, audio is not a numpy array,
            audio is not 1-D (mono), audio is empty, audio contains NaN/inf,
            or audio length is below the librosa STFT window (n_fft=2048).
            NOTE: validation is skipped on the speed==1.0 no-op fast path
            for backward compatibility — callers should not rely on that.
    """
    if abs(speed - 1.0) < 1e-6:
        return audio
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        raise ValueError(f"speed out of range {SPEED_MIN}-{SPEED_MAX}: {speed}")

    # Input contract: mono 1-D numpy array, non-empty, finite samples.
    if not isinstance(audio, np.ndarray):
        raise ValueError(f"_apply_speed expects numpy.ndarray, got {type(audio).__name__}")
    if audio.ndim != 1:
        raise ValueError(
            f"_apply_speed expects mono 1-D audio, got shape {audio.shape}. "
            "Downmix to mono before calling (e.g. audio.mean(axis=0) for (channels, N))."
        )
    if audio.size == 0:
        raise ValueError("_apply_speed got empty audio array")

    # librosa.effects.time_stretch uses STFT with default n_fft=2048.
    # Signals shorter than that either raise inside the STFT (very short
    # inputs, below reflect-padding threshold) or produce degenerate
    # phase-vocoder output. Gate explicitly with a clear message.
    _MIN_SAMPLES_FOR_STFT = 2048
    if audio.size < _MIN_SAMPLES_FOR_STFT:
        raise ValueError(
            f"_apply_speed got audio of length {audio.size}; need at least "
            f"{_MIN_SAMPLES_FOR_STFT} samples for librosa time_stretch "
            f"(~{_MIN_SAMPLES_FOR_STFT / sample_rate * 1000:.0f}ms at "
            f"{sample_rate}Hz)."
        )

    audio_f = audio.astype(np.float32, copy=False)
    if not np.all(np.isfinite(audio_f)):
        raise ValueError("_apply_speed got non-finite samples (NaN or inf)")

    try:
        import librosa
    except ImportError:
        logger.warning(
            "librosa not installed; speed parameter ignored. Install with: pip install 'spanish-tts[speed]'"
        )
        return audio

    stretched = librosa.effects.time_stretch(y=audio_f, rate=speed)
    return stretched


def _get_model(model_id: str):
    """Load MLX model with thread-safe caching.

    Uses a module-level lock to ensure at most one thread loads a given
    model_id; subsequent callers receive the cached instance immediately
    without touching the MLX runtime.

    The model is loaded at the pinned revision in :data:`MODEL_REVISIONS`
    (if a matching entry exists), otherwise at the repo's default branch.
    """
    with _cache_lock:
        if model_id not in _model_cache:
            from mlx_audio.tts import load

            revision = _revision_for(model_id)
            logger.info("Loading model: %s (revision=%s)", model_id, revision or "default")
            _model_cache[model_id] = load(model_id, revision=revision)
        return _model_cache[model_id]


def _revision_for(model_id: str) -> str | None:
    """Return the pinned revision SHA for *model_id*, or None.

    Looks up ``model_id`` in :data:`MODEL_REVISIONS` by both the full ID
    and by the short key (``"clone"``/``"design"``).
    """
    # Direct match first (future custom model IDs).
    if model_id in MODEL_REVISIONS:
        return MODEL_REVISIONS[model_id]
    # Fall back to short-key lookup via MODELS inverse.
    for key, mid in MODELS.items():
        if mid == model_id:
            return MODEL_REVISIONS.get(key)
    return None


def _resolve_output(output: str | None, prefix: str, output_dir: str | None = None) -> str:
    """Resolve output file path."""
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    out_dir = Path(output_dir or "~/tts-output/spanish").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"{prefix}_{ts}.wav")


def _collect_audio(
    results, stream: bool, on_chunk: Callable[[int, int, float], None] | None = None
):
    """Collect audio from a generator of GenerationResults.

    When stream=False, takes the first result (non-streaming behavior).
    When stream=True, iterates all chunks and concatenates audio.

    Args:
        results: Generator from model.generate() or similar.
        stream: Whether results are streaming chunks.
        on_chunk: Optional callback(chunk_index, total_samples, est_duration).

    Returns:
        Concatenated numpy audio array.
    """
    if not stream:
        first_result = next(iter(results))
        return np.array(first_result.audio)

    chunks = []
    total_samples = 0
    for i, chunk in enumerate(results):
        audio = np.array(chunk.audio)
        chunks.append(audio)
        total_samples += len(audio)
        if on_chunk:
            on_chunk(i, total_samples, total_samples / 24000.0)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def generate_clone(
    text: str,
    ref_audio: str,
    ref_text: str,
    speed: float = 1.0,
    output: str | None = None,
    output_dir: str | None = None,
    model_id: str | None = None,
    stream: bool = False,
    streaming_interval: float = 2.0,
    on_chunk: Callable[[int, int, float], None] | None = None,
) -> TtsResult:
    """Generate speech by cloning a reference voice.

    Args:
        text: Text to synthesize.
        ref_audio: Path to reference audio file (5-10s, clean).
        ref_text: Transcript of the reference audio.
        speed: Speed factor (0.5-2.0).
        output: Output file path (auto-generated if None).
        output_dir: Base output directory.
        model_id: Override model ID.
        stream: Use streaming decode (lower memory, progress callbacks).
        streaming_interval: Seconds of audio per streaming chunk (default 2.0).
        on_chunk: Optional callback(chunk_index, total_samples, est_duration).

    Returns:
        :class:`TtsResult` with the path to the generated ``.wav`` and its
        exact duration.  ``str(result)`` returns ``result.path`` for
        backward-compatible callers.
    """
    _validate_text(text)
    ref_path = Path(ref_audio).expanduser()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    with _generate_lock:
        model = _get_model(model_id or MODELS["clone"])
        logger.info("Cloning voice from: %s%s", ref_path, " [streaming]" if stream else "")
        results = model.generate(
            text=text,
            lang_code="auto",
            ref_audio=str(ref_path),
            ref_text=ref_text,
            speed=speed,  # upstream no-op; real stretch is post-process
            stream=stream,
            streaming_interval=streaming_interval,
        )

        output_path = _resolve_output(output, "clone", output_dir)
        audio_np = _collect_audio(results, stream=stream, on_chunk=on_chunk)

        sample_rate = model.sample_rate
        audio_np = _apply_speed(audio_np, speed, sample_rate)
        # NOTE: _apply_speed (librosa time_stretch) is CPU-bound and may take
        # several seconds at non-trivial speeds.  Move outside _generate_lock
        # if the transport ever moves beyond single-request stdio-MCP.
        sf.write(output_path, audio_np, sample_rate)

    duration = len(audio_np) / sample_rate
    logger.info("Saved: %s (%.1fs)", output_path, duration)
    return TtsResult(path=output_path, duration_seconds=duration)


def generate_design(
    text: str,
    instruct: str,
    language: str = "Spanish",
    speed: float = 1.0,
    output: str | None = None,
    output_dir: str | None = None,
    model_id: str | None = None,
    stream: bool = False,
    streaming_interval: float = 2.0,
    on_chunk: Callable[[int, int, float], None] | None = None,
) -> TtsResult:
    """Generate speech from a voice description.

    Args:
        text: Text to synthesize.
        instruct: Natural language description of the desired voice.
        language: Target language.
        speed: Speed factor (0.5-2.0).
        output: Output file path (auto-generated if None).
        output_dir: Base output directory.
        model_id: Override model ID.
        stream: Use streaming decode (lower memory, progress callbacks).
        streaming_interval: Seconds of audio per streaming chunk (default 2.0).
        on_chunk: Optional callback(chunk_index, total_samples, est_duration).

    Returns:
        :class:`TtsResult` with the path to the generated ``.wav`` and its
        exact duration.  ``str(result)`` returns ``result.path`` for
        backward-compatible callers.
    """
    _validate_text(text)
    lang_code = LANG_MAP.get(language, "spanish")

    with _generate_lock:
        model = _get_model(model_id or MODELS["design"])
        logger.debug(
            "Designing voice: instruct=%r (lang=%s)%s",
            instruct,
            lang_code,
            " [streaming]" if stream else "",
        )
        results = model.generate(
            text=text,
            lang_code=lang_code,
            instruct=instruct,
            speed=speed,  # upstream no-op; real stretch is post-process
            stream=stream,
            streaming_interval=streaming_interval,
        )

        output_path = _resolve_output(output, "design", output_dir)
        audio_np = _collect_audio(results, stream=stream, on_chunk=on_chunk)

        sample_rate = model.sample_rate
        audio_np = _apply_speed(audio_np, speed, sample_rate)
        # NOTE: _apply_speed (librosa time_stretch) is CPU-bound and may take
        # several seconds at non-trivial speeds.  Move outside _generate_lock
        # if the transport ever moves beyond single-request stdio-MCP.
        sf.write(output_path, audio_np, sample_rate)

    duration = len(audio_np) / sample_rate
    logger.info("Saved: %s (%.1fs)", output_path, duration)
    return TtsResult(path=output_path, duration_seconds=duration)


def generate(
    text: str,
    voice_config: dict,
    speed: float = 1.0,
    output: str | None = None,
    output_dir: str | None = None,
    stream: bool = False,
    streaming_interval: float = 2.0,
    on_chunk: Callable[[int, int, float], None] | None = None,
) -> TtsResult:
    """Generate speech using a voice config from the registry.

    Args:
        text: Text to synthesize.
        voice_config: Voice dict from voices.yaml (must have 'type' key).
        speed: Speed factor override.
        output: Output path override.
        output_dir: Output directory override.
        stream: Use streaming decode (lower memory, progress callbacks).
        streaming_interval: Seconds of audio per streaming chunk (default 2.0).
        on_chunk: Optional callback(chunk_index, total_samples, est_duration).

    Returns:
        :class:`TtsResult` with the path to the generated ``.wav`` and its
        exact duration.  ``str(result)`` returns ``result.path`` for
        backward-compatible callers.
    """
    voice_type = voice_config.get("type", "design")
    language = voice_config.get("language", "Spanish")

    if voice_type == "clone":
        ref_audio = voice_config.get("ref_audio")
        ref_text = voice_config.get("ref_text", "")
        if not ref_audio:
            raise ValueError("Clone voice requires 'ref_audio' in config")
        return generate_clone(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=speed,
            output=output,
            output_dir=output_dir,
            stream=stream,
            streaming_interval=streaming_interval,
            on_chunk=on_chunk,
        )
    elif voice_type == "design":
        instruct = voice_config.get("instruct")
        if not instruct:
            raise ValueError("Design voice requires 'instruct' in config")
        return generate_design(
            text=text,
            instruct=instruct,
            language=language,
            speed=speed,
            output=output,
            output_dir=output_dir,
            stream=stream,
            streaming_interval=streaming_interval,
            on_chunk=on_chunk,
        )
    else:
        raise ValueError(f"Unknown voice type: {voice_type}")
