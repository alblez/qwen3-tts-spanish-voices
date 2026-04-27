"""spanish-tts: Curated Spanish TTS voices using Qwen3-TTS."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__: str = version("qwen3-tts-spanish-voices")
except PackageNotFoundError:  # editable install not yet built
    __version__ = "0.0.0+dev"

from spanish_tts.engine import TtsResult

__all__ = ["TtsResult", "__version__"]
