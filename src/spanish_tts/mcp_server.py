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
import math
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from spanish_tts.config import get_defaults, get_voice, list_voices
from spanish_tts.engine import generate

logger = logging.getLogger("spanish_tts.mcp_server")

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
        output: Output .wav path (auto-generated if omitted). MUST
            resolve inside the configured output_dir (voices.yaml
            defaults.output_dir). Absolute paths outside that root and
            relative ``..`` escapes are rejected — the tool returns an
            error dict instead of writing. Absolute paths that already
            resolve inside output_dir are accepted. Hardens the MCP
            surface for cases where the server is exposed over a
            transport other than a trusted local stdio parent. CLI
            users are not affected; they call engine.generate directly
            and get the untrusted-but-local semantics they always had.
        stream: Use streaming decode for lower memory on long texts (default: false).

    Returns:
        Dict with 'path' and 'duration_seconds', or {'error': ..., 'code': ...}.
        The 'code' key is a stable enum value (see CONTRACT.md).
    """
    if not text or not text.strip():
        return {"error": "text is empty", "code": "text_empty"}
    if "\x00" in text:
        return {"error": "text contains NUL byte", "code": "text_nul"}
    if len(text) > 10000:
        return {
            "error": f"text too long ({len(text)} chars, max 10000)",
            "code": "text_too_long",
        }

    voice_config = get_voice(voice)
    if voice_config is None:
        available = list(list_voices().keys())
        return {
            "error": f"Voice '{voice}' not found. Available: {', '.join(available)}",
            "code": "voice_not_found",
        }

    defaults = get_defaults()
    effective_speed = speed if speed is not None else defaults.get("speed", 1.0)
    if not math.isfinite(effective_speed):
        return {
            "error": f"speed must be a finite number, got {effective_speed}",
            "code": "speed_not_finite",
        }
    if not (0.5 <= effective_speed <= 2.0):
        return {
            "error": f"speed out of range 0.5-2.0: {effective_speed}",
            "code": "speed_out_of_range",
        }
    output_dir = defaults.get("output_dir", "~/tts-output/spanish")

    # MCP-1 path-traversal hardening. If the caller supplied `output`,
    # it MUST land inside output_dir once resolved. Rejects absolute
    # paths and ../../escape attempts. Asymmetric with the CLI path
    # on purpose: `engine.generate` still accepts arbitrary paths for
    # local-user invocations. Only the MCP tool is sandboxed.
    if output is not None:
        # Quick structural rejections before touching the filesystem:
        # empty string, NUL byte (pathlib would raise mid-resolve with
        # a ValueError that escapes our error-dict contract).
        if output == "" or "\x00" in output:
            return {
                "error": f"output path {output!r} is not a valid filename",
                "code": "path_invalid",
            }

        safe_root = Path(output_dir).expanduser().resolve()
        # Join then resolve so both relative and absolute `output`
        # values get normalised against the same safe_root. resolve()
        # follows symlinks, so a symlink inside safe_root pointing out
        # is caught by the relative_to check below.
        try:
            candidate = (
                (safe_root / output).resolve()
                if not Path(output).is_absolute()
                else Path(output).resolve()
            )
        except (OSError, ValueError) as e:
            return {
                "error": f"output path {output!r} cannot be resolved: {e}",
                "code": "path_invalid",
            }

        # Reject writing to safe_root itself — it is a directory and
        # engine.generate would fail downstream with a less clear error.
        if candidate == safe_root:
            return {
                "error": (
                    f"output path {output!r} resolves to output_dir "
                    "itself; supply a filename under it or omit `output`."
                ),
                "code": "path_is_dir",
            }

        try:
            candidate.relative_to(safe_root)
        except ValueError:
            return {
                "error": (
                    f"output path {output!r} escapes output_dir "
                    f"{str(safe_root)!r}. MCP refuses path traversal; "
                    "use a path relative to output_dir or omit `output`."
                ),
                "code": "path_escape",
            }
        output = str(candidate)

    def _on_chunk(idx: int, total_samples: int, est_duration: float) -> None:
        logger.info("Chunk %d: %.1fs generated so far", idx, est_duration)

    try:
        result = generate(
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
        return {"error": f"Generation failed: {e}", "code": "generation_failed"}

    return {"path": result.path, "duration_seconds": round(result.duration_seconds, 2)}


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
                # Return the full instruct prompt. Truncating silently
                # destroys the persona description — presets ship with
                # 99-121 char instructs and clipping at 80 drops the
                # trailing clause (tone/pacing/style) that defines the
                # voice. Token cost is negligible; 4 presets total.
                summary[name]["description"] = config.get("instruct", "")
        return {"voices": summary}
    except Exception as e:
        logger.error("list_voices() failed: %s", e, exc_info=True)
        return {"error": f"Failed to list voices: {e}", "code": "internal_error"}


@mcp.tool()
def demo(text: str, output_dir: str = "/tmp/spanish-tts-demo", speed: float = 1.0) -> dict:
    """Generate the same text with ALL registered voices for comparison.

    Args:
        text: Text to synthesize (Spanish).
        output_dir: Directory for output files (default: /tmp/spanish-tts-demo).
        speed: Speed factor 0.5-2.0 (default: 1.0).

    Returns:
        Dict with 'results' list of per-voice outcomes, or {'error': ..., 'code': ...}.
    """
    if not text or not text.strip():
        return {"error": "text is empty", "code": "text_empty"}
    if not math.isfinite(speed):
        return {"error": f"speed must be a finite number, got {speed}", "code": "speed_not_finite"}
    if not (0.5 <= speed <= 2.0):
        return {"error": f"speed out of range 0.5-2.0: {speed}", "code": "speed_out_of_range"}

    from pathlib import Path

    voices = list_voices()
    if not voices:
        return {"error": "No voices registered", "code": "voices_empty"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, config in voices.items():
        try:
            result = generate(
                text=text,
                voice_config=config,
                speed=speed,
                output=str(out / f"{name}.wav"),
            )
            results.append({"voice": name, "path": result.path, "status": "ok"})
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
