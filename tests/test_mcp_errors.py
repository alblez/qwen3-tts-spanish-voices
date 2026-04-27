"""MCP server error-branch unit tests (U3-6, Subagent A).

Covers lines in mcp_server.py that were uncovered by the existing test suite:
  - say(): empty/whitespace text, too-long text, voice not found, path OSError,
    generate() exception, stream=True on_chunk wiring
  - list_all_voices(): clone accent field, exception fallback
  - demo(): empty text, partial failure loop
"""

import importlib
import logging
import sys

import pytest
import yaml

from spanish_tts.engine import TtsResult

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def bundled_presets(tmp_path, monkeypatch):
    """Point SPANISH_TTS_CONFIG at an empty dir so load_voices() falls back
    to the bundled presets/voices.yaml.  Reloads modules to pick up new env.
    """
    monkeypatch.setenv("SPANISH_TTS_CONFIG", str(tmp_path))
    for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
        sys.modules.pop(mod, None)
    import spanish_tts.mcp_server as mcp_server

    importlib.reload(mcp_server)
    yield mcp_server
    for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
        sys.modules.pop(mod, None)


def _make_tts_result(path: str, duration: float = 1.23) -> TtsResult:
    """Minimal TtsResult for patching engine.generate in tests."""
    return TtsResult(path=path, duration_seconds=duration)


# ---------------------------------------------------------------------------
# say() — text validation
# ---------------------------------------------------------------------------


class TestSayTextValidation:
    def test_empty_string_rejected(self, bundled_presets):
        result = bundled_presets.say(text="")
        assert result == {"error": "text is empty", "code": "text_empty"}

    def test_whitespace_only_rejected(self, bundled_presets):
        result = bundled_presets.say(text="   \t\n")
        assert result == {"error": "text is empty", "code": "text_empty"}

    def test_nul_byte_rejected(self, bundled_presets):
        result = bundled_presets.say(text="hola\x00mundo")
        assert result == {"error": "text contains NUL byte", "code": "text_nul"}

    def test_text_too_long_rejected(self, bundled_presets):
        result = bundled_presets.say(text="a" * 10001)
        assert "error" in result
        assert "too long" in result["error"]
        assert "10001" in result["error"]
        assert result["code"] == "text_too_long"

    def test_text_exactly_10000_accepted(self, bundled_presets, monkeypatch, tmp_path):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        monkeypatch.setattr(
            mcp, "generate", lambda **kw: _make_tts_result(str(tmp_path / "out.wav"))
        )
        result = mcp.say(text="a" * 10000, voice="neutral_male")
        assert "error" not in result


# ---------------------------------------------------------------------------
# say() — voice not found
# ---------------------------------------------------------------------------


class TestSayVoiceNotFound:
    def test_nonexistent_voice_returns_error(self, bundled_presets):
        result = bundled_presets.say(text="hola", voice="__no_such_voice__")
        assert "error" in result
        assert "__no_such_voice__" in result["error"]
        assert result["code"] == "voice_not_found"

    def test_error_lists_available_voices(self, bundled_presets):
        result = bundled_presets.say(text="hola", voice="__no_such_voice__")
        assert "Available" in result["error"]


# ---------------------------------------------------------------------------
# say() — path OSError (lines 99-100)
# ---------------------------------------------------------------------------


class TestSayPathOsError:
    def test_oserror_from_resolve_returns_error(self, bundled_presets, tmp_path, monkeypatch):
        from pathlib import Path

        mcp = bundled_presets
        monkeypatch.setattr(
            mcp,
            "generate",
            lambda *a, **kw: pytest.fail("generate() must not be called"),
        )
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )

        real_resolve = Path.resolve

        def patched_resolve(self, strict=False):
            if "trigger_oserr" in str(self):
                raise OSError("simulated OS error during resolve")
            return real_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", patched_resolve)
        result = mcp.say(text="hola", voice="neutral_male", output="trigger_oserr/out.wav")
        assert "error" in result
        assert "cannot be resolved" in result["error"]


# ---------------------------------------------------------------------------
# say() — generate() exception (lines 137-139)
# ---------------------------------------------------------------------------


class TestSayGenerateException:
    def test_generate_exception_returns_error_dict(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )

        def _raise_boom(**kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(mcp, "generate", _raise_boom)
        result = mcp.say(text="hola", voice="neutral_male")
        assert "error" in result
        assert "Generation failed" in result["error"]
        assert "boom" in result["error"]
        assert result["code"] == "generation_failed"

    def test_generate_exception_is_logged(self, bundled_presets, tmp_path, monkeypatch, caplog):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )

        def _raise_kaboom(**kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(mcp, "generate", _raise_kaboom)
        with caplog.at_level(logging.ERROR, logger="spanish_tts.mcp_server"):
            mcp.say(text="hola", voice="neutral_male")
        assert any("generate() failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# say() — duration_seconds always present (U3-19: engine-returned, never None)
# ---------------------------------------------------------------------------


class TestSayDurationAlwaysPresent:
    def test_duration_seconds_is_float(self, bundled_presets, tmp_path, monkeypatch):
        """say() returns duration_seconds as a float, always — never None."""
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        monkeypatch.setattr(
            mcp, "generate", lambda **kw: _make_tts_result(str(tmp_path / "out.wav"), 2.5)
        )
        result = mcp.say(text="hola", voice="neutral_male")
        assert "error" not in result
        assert isinstance(result["duration_seconds"], float)
        assert result["duration_seconds"] == pytest.approx(2.5, abs=0.01)


# ---------------------------------------------------------------------------
# say() — stream wiring (on_chunk callback)
# ---------------------------------------------------------------------------


class TestSayStreamWiring:
    def test_stream_true_passes_on_chunk_to_generate(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )

        captured = {}

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return _make_tts_result(str(tmp_path / "out.wav"), 1.5)

        monkeypatch.setattr(mcp, "generate", fake_generate)

        mcp.say(text="hola", voice="neutral_male", stream=True)
        assert captured.get("on_chunk") is not None
        assert callable(captured["on_chunk"])
        # Verify on_chunk doesn't raise when invoked.
        captured["on_chunk"](0, 24000, 1.0)

    def test_stream_false_passes_none_on_chunk(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        captured = {}
        monkeypatch.setattr(
            mcp,
            "generate",
            lambda **kw: (captured.update(kw), _make_tts_result(str(tmp_path / "x.wav")))[1],
        )
        mcp.say(text="hola", voice="neutral_male", stream=False)
        assert captured["on_chunk"] is None


# ---------------------------------------------------------------------------
# list_all_voices() — clone voice accent + exception (lines 170, 179-181)
# ---------------------------------------------------------------------------


class TestListAllVoicesErrorBranches:
    def test_clone_voice_has_accent_in_summary(self, tmp_path, monkeypatch):
        """Clone voice accent field is included in list_all_voices output."""
        monkeypatch.setenv("SPANISH_TTS_CONFIG", str(tmp_path))
        for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
            sys.modules.pop(mod, None)

        voices_data = {
            "voices": {
                "test_clone": {
                    "type": "clone",
                    "gender": "male",
                    "language": "Spanish",
                    "ref_audio": "/tmp/ref.wav",
                    "accent": "mexico",
                }
            }
        }
        (tmp_path / "voices.yaml").write_text(yaml.dump(voices_data), encoding="utf-8")

        import spanish_tts.mcp_server as mcp_server

        importlib.reload(mcp_server)

        result = mcp_server.list_all_voices()
        for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
            sys.modules.pop(mod, None)

        assert "voices" in result
        assert "test_clone" in result["voices"]
        assert result["voices"]["test_clone"]["accent"] == "mexico"

    def test_list_voices_exception_returns_error_dict(self, bundled_presets, monkeypatch, caplog):
        mcp = bundled_presets

        def _raise_broken(**kw):
            raise RuntimeError("broken")

        monkeypatch.setattr(mcp, "list_voices", _raise_broken)
        with caplog.at_level(logging.ERROR, logger="spanish_tts.mcp_server"):
            result = mcp.list_all_voices()
        assert "error" in result
        assert "Failed to list voices" in result["error"]
        assert result["code"] == "internal_error"
        assert any("list_voices() failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# demo() — empty text + partial failure (lines 197, 207-224)
# ---------------------------------------------------------------------------


class TestDemoErrorBranches:
    def test_empty_text_returns_error(self, bundled_presets):
        result = bundled_presets.demo(text="")
        assert result == {"error": "text is empty", "code": "text_empty"}

    def test_whitespace_text_returns_error(self, bundled_presets):
        result = bundled_presets.demo(text="  ")
        assert result == {"error": "text is empty", "code": "text_empty"}

    def test_generates_result_per_voice(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        voices = mcp.list_voices()
        assert voices, "Need bundled voices for this test"

        def fake_generate(**kw):
            return _make_tts_result(kw["output"])

        monkeypatch.setattr(mcp, "generate", fake_generate)
        result = mcp.demo(text="hola", output_dir=str(tmp_path / "demo_out"))
        assert "results" in result
        assert len(result["results"]) == len(voices)
        for entry in result["results"]:
            assert entry["status"] == "ok"

    def test_partial_failure_continues(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        voices = list(mcp.list_voices().keys())
        assert len(voices) >= 2, "Need ≥2 voices for partial-failure test"
        first = voices[0]

        def fake_generate(**kw):
            name = kw["output"].split("/")[-1].replace(".wav", "")
            if name == first:
                raise RuntimeError(f"{first} simulated failure")
            return _make_tts_result(kw["output"])

        monkeypatch.setattr(mcp, "generate", fake_generate)
        result = mcp.demo(text="hola", output_dir=str(tmp_path / "partial"))
        statuses = {r["voice"]: r["status"] for r in result["results"]}
        assert statuses[first] == "failed"
        assert any(s == "ok" for s in statuses.values())

    def test_partial_failure_entry_has_error_field(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        first = list(mcp.list_voices().keys())[0]

        def _partial_fail(**kw):
            if kw["output"].endswith(f"{first}.wav"):
                raise RuntimeError("boom")
            return _make_tts_result(kw["output"])

        monkeypatch.setattr(mcp, "generate", _partial_fail)
        result = mcp.demo(text="hola", output_dir=str(tmp_path / "err_field"))
        failed = [r for r in result["results"] if r["voice"] == first]
        assert failed
        assert "error" in failed[0]

    def test_partial_failure_logs_error(self, bundled_presets, tmp_path, monkeypatch, caplog):
        mcp = bundled_presets
        first = list(mcp.list_voices().keys())[0]

        def boom(**kw):
            name = kw["output"].split("/")[-1].replace(".wav", "")
            if name == first:
                raise RuntimeError("demo kaboom")
            return _make_tts_result(kw["output"])

        monkeypatch.setattr(mcp, "generate", boom)
        with caplog.at_level(logging.ERROR, logger="spanish_tts.mcp_server"):
            mcp.demo(text="hola", output_dir=str(tmp_path / "log_demo"))
        assert any("demo generate for" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ECO-016: stable 'code' field on all error-return branches (U3-17)
# ---------------------------------------------------------------------------


class TestErrorCodeField:
    """Every MCP error return must include a stable 'code' enum value."""

    @pytest.mark.parametrize(
        "speed,expected_code",
        [
            (float("nan"), "speed_not_finite"),
            (float("inf"), "speed_not_finite"),
            (float("-inf"), "speed_not_finite"),
            (0.4999, "speed_out_of_range"),
            (2.0001, "speed_out_of_range"),
        ],
    )
    def test_say_speed_codes(self, bundled_presets, speed, expected_code):
        result = bundled_presets.say(text="hola", voice="neutral_male", speed=speed)
        assert "error" in result
        assert result["code"] == expected_code

    def test_say_path_escape_has_code(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        result = mcp.say(text="hola", voice="neutral_male", output="../escape.wav")
        assert result["code"] == "path_escape"

    def test_say_path_is_dir_has_code(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        result = mcp.say(text="hola", voice="neutral_male", output=".")
        assert result["code"] == "path_is_dir"

    def test_say_path_invalid_has_code(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        result = mcp.say(text="hola", voice="neutral_male", output="foo\x00.wav")
        assert result["code"] == "path_invalid"

    @pytest.mark.parametrize(
        "speed,expected_code",
        [
            (float("nan"), "speed_not_finite"),
            (float("inf"), "speed_not_finite"),
            (float("-inf"), "speed_not_finite"),
            (0.4999, "speed_out_of_range"),
        ],
    )
    def test_demo_speed_codes(self, bundled_presets, speed, expected_code):
        result = bundled_presets.demo(text="hola", speed=speed)
        assert "error" in result
        assert result["code"] == expected_code

    def test_demo_voices_empty_code(self, bundled_presets, monkeypatch):
        """demo() with no registered voices returns voices_empty code."""
        mcp = bundled_presets
        monkeypatch.setattr(mcp, "list_voices", lambda: {})
        result = mcp.demo(text="hola")
        assert "error" in result
        assert result["code"] == "voices_empty"


# ---------------------------------------------------------------------------
# U3-7: demo() sandbox + text cap (mirrors MCP-1 matrix)
# ---------------------------------------------------------------------------


class TestDemoSandbox:
    """U3-7: demo(output_dir=...) path-traversal guard mirrors MCP-1 say(output=...)."""

    def test_demo_rejects_nul_in_output_dir(self, bundled_presets):
        result = bundled_presets.demo(text="hola", output_dir="/tmp/foo\x00bar")
        assert "error" in result
        assert result["code"] == "path_invalid"

    def test_demo_rejects_absolute_escape(self, bundled_presets):
        result = bundled_presets.demo(text="hola", output_dir="/etc/cron.d")
        assert "error" in result
        assert result["code"] == "path_escape"

    def test_demo_accepts_default_tmp_path(self, bundled_presets, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(mcp, "list_voices", lambda: {})
        # Default /tmp/spanish-tts-demo must pass sandbox
        result = mcp.demo(text="hola", output_dir="/tmp/spanish-tts-demo")
        # sandbox passed → voices_empty (not a path error)
        assert result.get("code") != "path_escape", result
        assert result.get("code") != "path_invalid", result

    def test_demo_accepts_home_subdir(self, bundled_presets, monkeypatch, tmp_path):

        mcp = bundled_presets
        monkeypatch.setattr(mcp, "list_voices", lambda: {})
        # tmp_path is under $HOME on CI; use a path that's definitely under home
        home_sub = str(tmp_path)
        result = mcp.demo(text="hola", output_dir=home_sub)
        assert result.get("code") not in ("path_escape", "path_invalid"), result

    def test_demo_text_too_long_rejected(self, bundled_presets):
        result = bundled_presets.demo(text="a" * 10001)
        assert "error" in result
        assert result["code"] == "text_too_long"

    def test_demo_text_exactly_10000_accepted(self, bundled_presets, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(mcp, "list_voices", lambda: {})
        result = mcp.demo(text="a" * 10000)
        assert result.get("code") not in ("text_too_long", "text_empty"), result
