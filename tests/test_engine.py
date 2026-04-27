"""Tests for spanish_tts.engine module (no model loading)."""

import inspect
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from spanish_tts.engine import (
    MODELS,
    TtsResult,
    _cache_lock,
    _clear_cache,
    _generate_lock,
    _resolve_output,
    _validate_text,
    generate,
    generate_clone,
    generate_design,
)

# ---------------------------------------------------------------------------
# U3-18: Logger hygiene — caplog tests for instruct body visibility
# ---------------------------------------------------------------------------


class TestModels:
    def test_model_keys(self):
        assert "clone" in MODELS
        assert "design" in MODELS
        assert "custom" not in MODELS  # dropped in M7: dead code

    def test_model_ids_are_mlx(self):
        for key, model_id in MODELS.items():
            assert "mlx-community" in model_id, f"{key} model should be mlx-community"


class TestLoggerHygiene:
    """U3-18: instruct body (voice-persona PII) demoted to DEBUG."""

    def _setup_fake_design(self, monkeypatch, tmp_path):
        """Patch mlx_audio so generate_design runs without a real model."""
        import types

        import spanish_tts.engine as eng

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                class R:
                    audio = [0.0] * 4096

                yield R()

        fake_mlx = types.ModuleType("mlx_audio")
        fake_tts = types.ModuleType("mlx_audio.tts")
        fake_tts.load_model = lambda model_id: FakeModel()
        fake_mlx.tts = fake_tts
        monkeypatch.setitem(__import__("sys").modules, "mlx_audio", fake_mlx)
        monkeypatch.setitem(__import__("sys").modules, "mlx_audio.tts", fake_tts)
        eng._model_cache.clear()
        return tmp_path

    def test_instruct_not_logged_at_info(self, monkeypatch, tmp_path, caplog):
        """At INFO level, instruct body must NOT appear in logs (voice-persona PII)."""
        self._setup_fake_design(monkeypatch, tmp_path)
        secret_instruct = "SECRET_VOICE_PERSONA_DO_NOT_LOG"
        with caplog.at_level(logging.INFO, logger="spanish_tts.engine"):
            generate_design(
                text="hola",
                instruct=secret_instruct,
                output=str(tmp_path / "out.wav"),
            )
        log_text = " ".join(r.message for r in caplog.records)
        assert secret_instruct not in log_text, (
            "Instruct body leaked at INFO level — must be demoted to DEBUG"
        )

    def test_instruct_logged_at_debug(self, monkeypatch, tmp_path, caplog):
        """At DEBUG level, instruct body IS captured (developers need it for diagnostics)."""
        self._setup_fake_design(monkeypatch, tmp_path)
        secret_instruct = "SECRET_VOICE_PERSONA_FOR_DEBUG"
        with caplog.at_level(logging.DEBUG, logger="spanish_tts.engine"):
            generate_design(
                text="hola",
                instruct=secret_instruct,
                output=str(tmp_path / "out.wav"),
            )
        log_text = " ".join(r.message for r in caplog.records)
        assert secret_instruct in log_text, "Instruct body should be visible at DEBUG level"


class TestTtsResult:
    def test_is_dataclass(self):
        r = TtsResult(path="/tmp/foo.wav", duration_seconds=1.5)
        assert isinstance(r, TtsResult)

    def test_str_returns_path(self):
        r = TtsResult(path="/tmp/foo.wav", duration_seconds=1.5)
        assert str(r) == "/tmp/foo.wav"
        assert str(r) == r.path

    def test_duration_seconds_stored(self):
        r = TtsResult(path="/tmp/foo.wav", duration_seconds=2.345)
        assert r.duration_seconds == pytest.approx(2.345)

    def test_fstring_interpolation(self):
        r = TtsResult(path="/tmp/foo.wav", duration_seconds=1.0)
        assert f"{r}" == "/tmp/foo.wav"

    def test_frozen(self):
        r = TtsResult(path="/tmp/foo.wav", duration_seconds=1.0)
        with pytest.raises(AttributeError):  # frozen dataclass raises FrozenInstanceError
            r.path = "/other"  # type: ignore[misc]


class TestValidateText:
    """Unit tests for _validate_text helper (U3-17)."""

    def test_valid_ascii(self):
        assert _validate_text("hello world") == "hello world"

    def test_valid_spanish_unicode(self):
        assert _validate_text("¡Hola, cómo estás!") == "¡Hola, cómo estás!"

    def test_valid_nfc_accent(self):
        # NFC: é (U+00E9)
        assert _validate_text("caf\u00e9") == "caf\u00e9"

    def test_valid_nfd_accent(self):
        # NFD: e + combining accent (U+0301)
        assert _validate_text("cafe\u0301") == "cafe\u0301"

    def test_valid_mixed_scripts(self):
        text = "Spanish: ñ, Arabic: نعم, CJK: 你好"
        assert _validate_text(text) == text

    def test_valid_long_accent(self):
        long_text = "a" * 9999 + "ñ"
        assert _validate_text(long_text) == long_text

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="text is empty"):
            _validate_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="text is empty"):
            _validate_text("   \t\n")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="text is empty"):
            _validate_text(None)  # type: ignore[arg-type]

    def test_nul_byte_raises(self):
        with pytest.raises(ValueError, match="NUL"):
            _validate_text("hola\x00mundo")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            _validate_text("a" * 10001)

    def test_exactly_max_len_accepted(self):
        assert len(_validate_text("a" * 10000)) == 10000

    def test_no_max_len(self):
        very_long = "a" * 50000
        assert _validate_text(very_long, max_len=None) == very_long


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


def _install_fake_load_model(monkeypatch, fake_load_model):
    """Patch mlx_audio.tts.load_model with *fake_load_model* in sys.modules.

    Works whether mlx_audio is installed (Apple Silicon) or absent (CI).
    Returns the patched eng module for convenience.
    """
    import sys
    import types

    import spanish_tts.engine as eng

    if "mlx_audio.tts" in sys.modules:
        monkeypatch.setattr(sys.modules["mlx_audio.tts"], "load_model", fake_load_model)
    else:
        fake_mlx = types.ModuleType("mlx_audio")
        fake_tts = types.ModuleType("mlx_audio.tts")
        fake_tts.load_model = fake_load_model
        fake_mlx.tts = fake_tts
        monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    return eng


class TestCacheLock:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Isolate each test by clearing the cache before and after."""
        _clear_cache()
        yield
        _clear_cache()

    def test_get_model_called_once_under_concurrency(self, monkeypatch):
        """Concurrent _get_model calls for the same model_id load the model
        exactly once, even under a ThreadPoolExecutor.
        """
        import time

        load_count = {"n": 0}

        class FakeModel:
            sample_rate = 24000

        def fake_load_model(model_id):
            load_count["n"] += 1
            # sleep releases the GIL, giving other threads a chance to race.
            time.sleep(0.02)
            return FakeModel()

        eng = _install_fake_load_model(monkeypatch, fake_load_model)
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

    def test_get_model_returns_cached_on_second_call(self, monkeypatch):
        """Sequential second call returns cached object; load_model not called again."""

        load_count = {"n": 0}

        class FakeModel:
            sample_rate = 24000

        def fake_load_model(model_id):
            load_count["n"] += 1
            return FakeModel()

        eng = _install_fake_load_model(monkeypatch, fake_load_model)
        model_id = MODELS["design"]

        first = eng._get_model(model_id)
        second = eng._get_model(model_id)
        assert load_count["n"] == 1
        assert first is second

    def test_clear_cache_evicts_all(self):
        """_clear_cache() empties the module-level _model_cache dict."""
        import spanish_tts.engine as eng

        # Mutate the real dict in place (no setattr — keeps _cache_lock consistent).
        with _cache_lock:
            eng._model_cache["fake_id"] = object()
        assert "fake_id" in eng._model_cache
        _clear_cache()
        assert eng._model_cache == {}


class TestGenerateLock:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Isolate each test."""
        _clear_cache()
        yield
        _clear_cache()

    def test_generate_lock_is_threading_lock(self):
        """_generate_lock must be a real threading.Lock (not a no-op)."""
        assert isinstance(_generate_lock, type(threading.Lock()))

    def _run_serialization_test(self, monkeypatch, tmp_path, generate_fn, model_key):
        """Shared helper: assert concurrent calls do not overlap in execution."""
        import time

        import spanish_tts.engine as eng

        intervals = []
        interval_lock = threading.Lock()

        class FakeModel:
            sample_rate = 24000

            def generate(self, **kwargs):
                # Record timestamps inside the _generate_lock critical section.
                start = time.monotonic()
                # sleep releases GIL so other threads can queue behind the lock.
                time.sleep(0.05)
                end = time.monotonic()
                with interval_lock:
                    intervals.append((start, end))

                class R:
                    audio = [0.0] * 100

                yield R()

        monkeypatch.setattr(eng, "_model_cache", {MODELS[model_key]: FakeModel()})

        errors = []

        def run(i):
            out = str(tmp_path / f"out_{i}.wav")
            try:
                generate_fn(eng, out, tmp_path)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run, i) for i in range(3)]
            for f in futures:
                f.result()

        assert not errors, f"generate_fn raised: {errors}"
        # Verify no two intervals overlap (serialized execution).
        # 1ms tolerance accounts for monotonic clock resolution on Linux (~1ms).
        sorted_ivs = sorted(intervals)
        for i in range(1, len(sorted_ivs)):
            prev_end = sorted_ivs[i - 1][1]
            cur_start = sorted_ivs[i][0]
            assert cur_start >= prev_end - 1e-3, (
                f"Overlap detected at position {i}: interval {i - 1} ends "
                f"{prev_end:.4f}s, interval {i} starts {cur_start:.4f}s"
            )

    def test_generate_design_serialized_under_concurrency(self, monkeypatch, tmp_path):
        """Concurrent generate_design calls must not overlap."""

        def fn(eng, out, tmp_path):
            eng.generate_design(
                text="hola", instruct="calm voice", output=out, output_dir=str(tmp_path)
            )

        self._run_serialization_test(monkeypatch, tmp_path, fn, "design")

    def test_generate_clone_serialized_under_concurrency(self, monkeypatch, tmp_path):
        """Concurrent generate_clone calls must not overlap."""
        import numpy as np
        import soundfile as sf

        # Create a minimal ref audio file for generate_clone.
        ref = str(tmp_path / "ref.wav")
        sf.write(ref, np.zeros(4096, dtype=np.float32), 24000)

        def fn(eng, out, tmp_path):
            eng.generate_clone(
                text="hola",
                ref_audio=ref,
                ref_text="hola",
                output=out,
                output_dir=str(tmp_path),
            )

        self._run_serialization_test(monkeypatch, tmp_path, fn, "clone")
