"""Tests for spanish_tts.engine module (no model loading)."""

import inspect

import pytest

from spanish_tts.engine import (
    MODELS,
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
