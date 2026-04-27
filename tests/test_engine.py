"""Tests for spanish_tts.engine module (no model loading)."""

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from spanish_tts.engine import (
    MODELS,
    _cache_lock,
    _clear_cache,
    _generate_lock,
    _resolve_output,
    generate,
    generate_clone,
    generate_design,
)


class TestModels:
    def test_model_keys(self):
        assert "clone" in MODELS
        assert "design" in MODELS
        assert "custom" not in MODELS  # dropped in M7: dead code

    def test_model_ids_are_mlx(self):
        for key, model_id in MODELS.items():
            assert "mlx-community" in model_id, f"{key} model should be mlx-community"


class TestGenerateCloneSignature:
    def test_no_language_param(self):
        sig = inspect.signature(generate_clone)
        assert "language" not in sig.parameters

    def test_required_params(self):
        sig = inspect.signature(generate_clone)
        params = sig.parameters
        # text, ref_audio, ref_text are required (no default)
        for name in ("text", "ref_audio", "ref_text"):
            assert params[name].default is inspect.Parameter.empty

    def test_optional_params(self):
        sig = inspect.signature(generate_clone)
        params = sig.parameters
        assert params["speed"].default == 1.0
        assert params["output"].default is None
        assert params["output_dir"].default is None
        assert params["model_id"].default is None


class TestGenerateDesignSignature:
    def test_has_language_param(self):
        sig = inspect.signature(generate_design)
        assert "language" in sig.parameters
        assert sig.parameters["language"].default == "Spanish"

    def test_required_params(self):
        sig = inspect.signature(generate_design)
        params = sig.parameters
        for name in ("text", "instruct"):
            assert params[name].default is inspect.Parameter.empty


class TestGenerateSignature:
    def test_required_params(self):
        sig = inspect.signature(generate)
        params = sig.parameters
        for name in ("text", "voice_config"):
            assert params[name].default is inspect.Parameter.empty


class TestResolveOutput:
    def test_explicit_output(self, tmp_path):
        path = str(tmp_path / "test.wav")
        result = _resolve_output(path, "clone")
        assert result == path

    def test_auto_generated_output(self, tmp_path):
        result = _resolve_output(None, "clone", str(tmp_path))
        assert result.startswith(str(tmp_path))
        assert "clone_" in result
        assert result.endswith(".wav")

    def test_auto_creates_parent(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "test.wav")
        result = _resolve_output(deep, "clone")
        assert result == deep
        assert (tmp_path / "a" / "b").exists()


class TestGenerateValidation:
    """Test generate() input validation (no model needed)."""

    def test_clone_missing_ref_audio(self):
        config = {"type": "clone", "ref_text": "hello"}
        with pytest.raises(ValueError, match="ref_audio"):
            generate("test", config)

    def test_design_missing_instruct(self):
        config = {"type": "design"}
        with pytest.raises(ValueError, match="instruct"):
            generate("test", config)

    def test_unknown_type(self):
        config = {"type": "unknown"}
        with pytest.raises(ValueError, match="Unknown voice type"):
            generate("test", config)


# ---------------------------------------------------------------------------
# U3-4: _model_cache concurrency + serialization lock
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_engine_cache():
    """Reset the engine cache before and after each test in this module."""
    _clear_cache()
    yield
    _clear_cache()


class TestCacheLock:
    def test_get_model_called_once_under_concurrency(self, monkeypatch):
        """Concurrent _get_model calls for the same model_id load the model
        exactly once, even under a ThreadPoolExecutor.
        """
        import spanish_tts.engine as eng

        load_count = {"n": 0}

        class FakeModel:
            sample_rate = 24000

        def fake_load_model(model_id):
            load_count["n"] += 1
            # Simulate some work so threads have time to race.
            import time

            time.sleep(0.02)
            return FakeModel()

        monkeypatch.setattr(eng, "_model_cache", {})
        # Patch at the source where _get_model imports it.
        import sys

        if "mlx_audio.tts" in sys.modules:
            monkeypatch.setattr(sys.modules["mlx_audio.tts"], "load_model", fake_load_model)
        else:
            # mlx_audio not installed (CI) — patch via importlib side-channel.
            import types

            fake_mlx = types.ModuleType("mlx_audio")
            fake_tts = types.ModuleType("mlx_audio.tts")
            fake_tts.load_model = fake_load_model
            fake_mlx.tts = fake_tts
            monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx)
            monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)

        model_id = MODELS["design"]
        results = []

        def call_get_model():
            results.append(eng._get_model(model_id))

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(call_get_model) for _ in range(5)]
            for f in futures:
                f.result()

        assert load_count["n"] == 1, f"Expected 1 load, got {load_count['n']}"
        assert len(results) == 5
        # All threads got the same cached object.
        assert all(r is results[0] for r in results)

    def test_clear_cache_evicts_all(self, monkeypatch):
        """_clear_cache() empties the module-level _model_cache dict."""
        import spanish_tts.engine as eng

        with _cache_lock:
            eng._model_cache["fake_id"] = object()
        assert "fake_id" in eng._model_cache
        _clear_cache()
        assert eng._model_cache == {}


class TestGenerateLock:
    def test_generate_lock_is_threading_lock(self):
        """_generate_lock must be a real threading.Lock (not a no-op)."""
        assert isinstance(_generate_lock, type(threading.Lock()))

    def test_generate_serialized_under_concurrency(self, monkeypatch, tmp_path):
        """Concurrent generate_design calls must not overlap — _generate_lock
        serialises the critical section.  We detect overlap by recording
        entry/exit timestamps and asserting no two intervals overlap.
        """
        import time

        import spanish_tts.engine as eng

        intervals = []
        lock = threading.Lock()

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                # Tiny sleep simulates generation work.
                start = time.monotonic()
                time.sleep(0.05)
                end = time.monotonic()
                with lock:
                    intervals.append((start, end))

                class R:
                    audio = [0.0] * 100

                yield R()

        monkeypatch.setattr(eng, "_model_cache", {MODELS["design"]: FakeModel()})

        errors = []

        def run_generate(i):
            out = str(tmp_path / f"out_{i}.wav")
            try:
                eng.generate_design(
                    text="hola", instruct="calm voice", output=out, output_dir=str(tmp_path)
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_generate, i) for i in range(3)]
            for f in futures:
                f.result()

        assert not errors, f"generate_design raised: {errors}"
        # Verify no two intervals overlap (serialized execution).
        sorted_ivs = sorted(intervals)
        for i in range(1, len(sorted_ivs)):
            prev_end = sorted_ivs[i - 1][1]
            cur_start = sorted_ivs[i][0]
            assert cur_start >= prev_end - 1e-3, (
                f"Overlap detected: interval {i - 1} ends {prev_end:.4f}, "
                f"interval {i} starts {cur_start:.4f}"
            )
