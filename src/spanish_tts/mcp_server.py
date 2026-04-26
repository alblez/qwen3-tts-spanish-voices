"""MCP server for spanish-tts.

Exposes Qwen3-TTS Spanish voice generation as MCP tools so Hermes Agent
can call them without subprocess overhead. The MLX model stays loaded
across calls via engine._model_cache.

Usage:
    python -m spanish_tts.mcp_server

Hermes config.yaml:
    mcp_servers:
      spanish-tts:
        command: ["conda", "run", "-n", "qwen3-tts", "python", "-m", "spanish_tts.mcp_server"]
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

from spanish_tts.config import get_defaults, get_voice, list_voices
from spanish_tts.engine import generate, generate_clone, generate_design

logger = logging.getLogger("spanish-tts-mcp")

mcp = FastMCP("spanish-tts")


@mcp.tool()
def say(
    text: str,
    voice: str = "neutral_male",
    speed: float | None = None,
    output: str | None = None,
    stream: bool = False,
) -> dict:
    """Generate Spanish speech from text using a registered voice.

    Args:
        text: Text to synthesize (Spanish).
        voice: Voice name from registry (default: neutral_male).
        speed: Speed factor 0.5-2.0. If omitted, falls back to
            voices.yaml `defaults.speed` (or 1.0). Matches CLI behaviour.
        output: Output .wav path (auto-generated if omitted).
        stream: Use streaming decode for lower memory on long texts (default: false).

    Returns:
        Dict with 'path' and 'duration_seconds', or 'error'.
    """
    if not text or not text.strip():
        return {"error": "text is empty"}
    if len(text) > 10000:
        return {"error": f"text too long ({len(text)} chars, max 10000)"}

    voice_config = get_voice(voice)
    if voice_config is None:
        available = list(list_voices().keys())
        return {"error": f"Voice '{voice}' not found. Available: {', '.join(available)}"}

    defaults = get_defaults()
    effective_speed = speed if speed is not None else defaults.get("speed", 1.0)
    if not (0.5 <= effective_speed <= 2.0):
        return {"error": f"speed out of range 0.5-2.0: {effective_speed}"}
    output_dir = defaults.get("output_dir", "~/tts-output/spanish")

    def _on_chunk(idx: int, total_samples: int, est_duration: float) -> None:
        logger.info("Chunk %d: %.1fs generated so far", idx, est_duration)

    try:
        path = generate(
            text=text,
            voice_config=voice_config,
            speed=effective_speed,
            output=output,
            output_dir=output_dir,
            stream=stream,
            on_chunk=_on_chunk if stream else None,
        )
    except Exception as e:
        logger.error("generate() failed: %s", e, exc_info=True)
        return {"error": f"Generation failed: {e}"}

    try:
        import soundfile as sf

        info = sf.info(path)
        duration = round(info.duration, 2)
    except Exception:
        duration = None

    return {"path": path, "duration_seconds": duration}


@mcp.tool()
def list_all_voices() -> dict:
    """List all registered Spanish TTS voices with their type and metadata.

    Returns:
        Dict with 'voices' mapping name -> config.
    """
    try:
        voices = list_voices()
        # Simplify for the agent: only include useful fields
        summary = {}
        for name, config in voices.items():
            summary[name] = {
                "type": config.get("type", "unknown"),
                "gender": config.get("gender", "unknown"),
                "language": config.get("language", "Spanish"),
            }
            if config.get("type") == "clone":
                summary[name]["accent"] = config.get("accent", "")
            elif config.get("type") == "design":
                instruct = config.get("instruct", "")
                summary[name]["description"] = instruct[:80]
        return {"voices": summary}
    except Exception as e:
        logger.error("list_voices() failed: %s", e, exc_info=True)
        return {"error": f"Failed to list voices: {e}"}


@mcp.tool()
def demo(text: str, output_dir: str = "/tmp/spanish-tts-demo", speed: float = 1.0) -> dict:
    """Generate the same text with ALL registered voices for comparison.

    Args:
        text: Text to synthesize (Spanish).
        output_dir: Directory for output files (default: /tmp/spanish-tts-demo).
        speed: Speed factor 0.5-2.0 (default: 1.0).

    Returns:
        Dict with 'results' list of per-voice outcomes.
    """
    if not text or not text.strip():
        return {"error": "text is empty"}
    if not (0.5 <= speed <= 2.0):
        return {"error": f"speed out of range 0.5-2.0: {speed}"}

    from pathlib import Path

    voices = list_voices()
    if not voices:
        return {"error": "No voices registered"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, config in voices.items():
        try:
            path = generate(
                text=text,
                voice_config=config,
                speed=speed,
                output=str(out / f"{name}.wav"),
            )
            results.append({"voice": name, "path": path, "status": "ok"})
        except Exception as e:
            logger.error("demo generate for %s failed: %s", name, e)
            results.append({"voice": name, "error": str(e), "status": "failed"})

    return {"results": results}


def _preload_models():
    """Eagerly load common models + warm librosa so first tool calls are fast."""
    from spanish_tts.engine import MODELS, _apply_speed, _get_model

    for mode in ("clone", "design"):
        model_id = MODELS[mode]
        logger.info("Pre-loading model: %s", model_id)
        try:
            _get_model(model_id)
            logger.info("Model '%s' loaded successfully", mode)
        except Exception as e:
            # Non-fatal: model will load on first tool call instead
            logger.warning("Pre-load of '%s' failed (will retry on first call): %s", mode, e)

    # Warm librosa JIT so first `say` with speed != 1.0 doesn't pay the
    # one-off ~18s time_stretch init cost. Graceful no-op if librosa
    # isn't installed (the [speed] extra is optional).
    logger.info("Warming librosa time-stretch...")
    try:
        import numpy as np

        _apply_speed(np.zeros(24000, dtype=np.float32), 1.5, 24000)
        logger.info("librosa warmup complete")
    except Exception as e:
        logger.warning("librosa warmup skipped: %s", e)


def main():
    """Entry point for python -m spanish_tts.mcp_server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting spanish-tts MCP server")
    _preload_models()
    mcp.run()


if __name__ == "__main__":
    main()
