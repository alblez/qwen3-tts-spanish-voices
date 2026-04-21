"""TTS engine wrapping Qwen3-TTS MLX for clone and design modes."""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


# Default MLX models
MODELS = {
    "clone": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "custom": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
}

_model_cache = {}


def _get_model(model_id: str):
    """Load MLX model with caching."""
    if model_id not in _model_cache:
        from mlx_audio.tts import load_model
        print(f"Loading model: {model_id}", file=sys.stderr)
        _model_cache[model_id] = load_model(model_id)
    return _model_cache[model_id]


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


def generate_clone(
    text: str,
    ref_audio: str,
    ref_text: str,
    speed: float = 1.0,
    output: str | None = None,
    output_dir: str | None = None,
    model_id: str | None = None,
) -> str:
    """Generate speech by cloning a reference voice.

    Args:
        text: Text to synthesize.
        ref_audio: Path to reference audio file (5-10s, clean).
        ref_text: Transcript of the reference audio.
        speed: Speed factor (0.8-1.3).
        output: Output file path (auto-generated if None).
        output_dir: Base output directory.
        model_id: Override model ID.

    Returns:
        Path to generated .wav file.
    """
    model = _get_model(model_id or MODELS["clone"])

    ref_path = Path(ref_audio).expanduser()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    print(f"Cloning voice from: {ref_path}", file=sys.stderr)
    results = model.generate(
        text=text,
        lang_code="auto",
        ref_audio=str(ref_path),
        ref_text=ref_text,
        speed=speed,
    )

    output_path = _resolve_output(output, "clone", output_dir)
    first_result = next(iter(results))
    audio_np = np.array(first_result.audio)

    sample_rate = model.sample_rate
    sf.write(output_path, audio_np, sample_rate)

    duration = len(audio_np) / sample_rate
    print(f"Saved: {output_path} ({duration:.1f}s)", file=sys.stderr)
    return output_path


def generate_design(
    text: str,
    instruct: str,
    language: str = "Spanish",
    speed: float = 1.0,
    output: str | None = None,
    output_dir: str | None = None,
    model_id: str | None = None,
) -> str:
    """Generate speech from a voice description.

    Args:
        text: Text to synthesize.
        instruct: Natural language description of the desired voice.
        language: Target language.
        speed: Speed factor (0.8-1.3).
        output: Output file path (auto-generated if None).
        output_dir: Base output directory.
        model_id: Override model ID.

    Returns:
        Path to generated .wav file.
    """
    model = _get_model(model_id or MODELS["design"])

    # Map language name to lang_code for the codec language token.
    # "auto" skips the language token entirely, causing the model to
    # default to English prosody when the instruct is in English.
    lang_map = {
        "Spanish": "spanish", "English": "english", "Chinese": "chinese",
        "French": "french", "German": "german", "Italian": "italian",
        "Portuguese": "portuguese", "Japanese": "japanese", "Korean": "korean",
        "Russian": "russian",
    }
    lang_code = lang_map.get(language, "spanish")

    print(f"Designing voice: '{instruct[:60]}...' (lang={lang_code})", file=sys.stderr)
    results = model.generate(
        text=text,
        lang_code=lang_code,
        instruct=instruct,
        speed=speed,
    )

    output_path = _resolve_output(output, "design", output_dir)
    first_result = next(iter(results))
    audio_np = np.array(first_result.audio)

    sample_rate = model.sample_rate
    sf.write(output_path, audio_np, sample_rate)

    duration = len(audio_np) / sample_rate
    print(f"Saved: {output_path} ({duration:.1f}s)", file=sys.stderr)
    return output_path


def generate(
    text: str,
    voice_config: dict,
    speed: float = 1.0,
    output: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Generate speech using a voice config from the registry.

    Args:
        text: Text to synthesize.
        voice_config: Voice dict from voices.yaml (must have 'type' key).
        speed: Speed factor override.
        output: Output path override.
        output_dir: Output directory override.

    Returns:
        Path to generated .wav file.
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
        )
    else:
        raise ValueError(f"Unknown voice type: {voice_type}")
