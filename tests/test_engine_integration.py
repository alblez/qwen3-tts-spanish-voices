"""Engine slow/MLX integration tests (U3-6, Subagent B).

Covers engine.py lines uncovered by the fast test suite:
  - _collect_audio streaming path (stream=True with chunks)
  - generate_clone missing ref_audio (FileNotFoundError)
  - _get_model cache hit via direct cache injection
  - generate_design language mapping for all 10 lang_map entries

All tests carry @pytest.mark.slow and @pytest.mark.requires_mlx.
No real MLX model is loaded — all heavy callouts are monkeypatched.
"""

import numpy as np
import pytest

import spanish_tts.engine as eng
from spanish_tts.engine import (
    MODELS,
    _cache_lock,
    _clear_cache,
    _collect_audio,
    _get_model,
    _model_cache,
    generate_clone,
    generate_design,
)

# ---------------------------------------------------------------------------
# Helper: minimal stand-in for mlx_audio GenerationResult
# ---------------------------------------------------------------------------


class _R:
    """Minimal stand-in for mlx_audio GenerationResult."""

    def __init__(self, n_samples: int):
        self.audio = [0.0] * n_samples


# ---------------------------------------------------------------------------
# A. _collect_audio streaming path (lines 159-167)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_mlx
class TestCollectAudioStreaming:
    def test_three_chunks_concatenated(self):
        """stream=True: 3 × 1000-sample chunks → 3000-sample output."""
        results = iter([_R(1000), _R(1000), _R(1000)])
        out = _collect_audio(results, stream=True)
        assert len(out) == 3000

    def test_on_chunk_called_per_chunk(self):
        """on_chunk invoked once per chunk with correct chunk_index and cumulative totals."""
        calls = []

        def cb(chunk_index, total_samples, est_duration):
            calls.append((chunk_index, total_samples))

        results = iter([_R(1000), _R(1000), _R(1000)])
        _collect_audio(results, stream=True, on_chunk=cb)

        assert len(calls) == 3
        assert calls[0] == (0, 1000)
        assert calls[1] == (1, 2000)
        assert calls[2] == (2, 3000)

    def test_on_chunk_none_does_not_crash(self):
        """stream=True with on_chunk=None runs without error."""
        results = iter([_R(500), _R(500)])
        out = _collect_audio(results, stream=True, on_chunk=None)
        assert len(out) == 1000

    def test_empty_stream_returns_empty_float32(self):
        """stream=True with no chunks → empty float32 array."""
        out = _collect_audio(iter([]), stream=True)
        assert isinstance(out, np.ndarray)
        assert len(out) == 0
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# B. generate_clone missing ref_audio (line 201)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_mlx
class TestGenerateCloneMissingRefAudio:
    def test_nonexistent_ref_audio_raises_fnf(self):
        """Passing a non-existent path raises FileNotFoundError before model load."""
        with pytest.raises(FileNotFoundError, match="Reference audio not found"):
            generate_clone(
                text="hola",
                ref_audio="/nonexistent/path/ref.wav",
                ref_text="hola",
            )

    def test_error_message_includes_path(self):
        """FileNotFoundError message includes the bad path for debuggability."""
        bad = "/absolutely/not/there.wav"
        with pytest.raises(FileNotFoundError) as exc_info:
            generate_clone(text="hola", ref_audio=bad, ref_text="")
        assert bad in str(exc_info.value)


# ---------------------------------------------------------------------------
# C. _get_model cache hit
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_mlx
class TestGetModelCacheHit:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        _clear_cache()
        yield
        _clear_cache()

    def test_same_object_on_repeated_call(self, monkeypatch):
        """Pre-seed _model_cache; two _get_model calls return the same instance."""

        class FakeModel:
            sample_rate = 24000

        fake = FakeModel()
        model_id = MODELS["design"]
        with _cache_lock:
            _model_cache[model_id] = fake

        assert _get_model(model_id) is fake
        assert _get_model(model_id) is fake


# ---------------------------------------------------------------------------
# D. generate_design language mapping — all 10 lang_map entries
# ---------------------------------------------------------------------------

_LANG_MAP = {
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


@pytest.mark.slow
@pytest.mark.requires_mlx
@pytest.mark.parametrize("language,expected_code", list(_LANG_MAP.items()))
def test_generate_design_language_mapping(monkeypatch, tmp_path, language, expected_code):
    """generate_design passes the correct lang_code to model.generate() for each language."""
    captured = {}

    class FakeModel:
        sample_rate = 24000

        def generate(self, **kwargs):
            captured["lang_code"] = kwargs.get("lang_code")
            yield _R(100)

    # Pre-seed the cache so _get_model returns FakeModel without touching mlx_audio.
    monkeypatch.setattr(eng, "_model_cache", {MODELS["design"]: FakeModel()})

    try:
        generate_design(
            text="prueba",
            instruct="a calm voice",
            language=language,
            output=str(tmp_path / "out.wav"),
        )
    finally:
        _clear_cache()

    assert captured.get("lang_code") == expected_code, (
        f"language={language!r}: expected={expected_code!r}, got={captured.get('lang_code')!r}"
    )
