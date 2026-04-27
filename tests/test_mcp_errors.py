"""MCP server error-branch unit tests (U3-6, Subagent A).

Covers lines in mcp_server.py that were uncovered by the existing test suite:
  - say(): empty/whitespace text, too-long text, voice not found, path OSError,
    generate() exception, sf.info failure, stream=True on_chunk wiring
  - list_all_voices(): clone accent field, exception fallback
  - demo(): empty text, partial failure loop
"""

import importlib
import logging
import sys
import types

import pytest
import yaml

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


def _fake_sf_module(duration=2.0):
    """Return a minimal soundfile stand-in whose info() returns a given duration."""

    class _Info:
        pass

    inst = _Info()
    inst.duration = duration

    fake_sf = types.ModuleType("soundfile")
    fake_sf.info = lambda p: inst
    return fake_sf


# ---------------------------------------------------------------------------
# say() — text validation
# ---------------------------------------------------------------------------


class TestSayTextValidation:
    def test_empty_string_rejected(self, bundled_presets):
        result = bundled_presets.say(text="")
        assert result == {"error": "text is empty"}

    def test_whitespace_only_rejected(self, bundled_presets):
        result = bundled_presets.say(text="   \t\n")
        assert result == {"error": "text is empty"}

    def test_text_too_long_rejected(self, bundled_presets):
        result = bundled_presets.say(text="a" * 10001)
        assert "error" in result
        assert "too long" in result["error"]
        assert "10001" in result["error"]

    def test_text_exactly_10000_accepted(self, bundled_presets, monkeypatch, tmp_path):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        monkeypatch.setattr(mcp, "generate", lambda **kw: str(tmp_path / "out.wav"))
        monkeypatch.setitem(sys.modules, "soundfile", _fake_sf_module())
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
        monkeypatch.setattr(
            mcp, "generate", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        result = mcp.say(text="hola", voice="neutral_male")
        assert "error" in result
        assert "Generation failed" in result["error"]
        assert "boom" in result["error"]

    def test_generate_exception_is_logged(self, bundled_presets, tmp_path, monkeypatch, caplog):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        monkeypatch.setattr(
            mcp, "generate", lambda **kw: (_ for _ in ()).throw(RuntimeError("kaboom"))
        )
        with caplog.at_level(logging.ERROR, logger="spanish_tts.mcp_server"):
            mcp.say(text="hola", voice="neutral_male")
        assert any("generate() failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# say() — sf.info() failure (line 145 → duration_seconds=None)
# ---------------------------------------------------------------------------


class TestSaySfInfoFailure:
    def test_broken_wav_returns_duration_none(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        monkeypatch.setattr(
            mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        monkeypatch.setattr(mcp, "generate", lambda **kw: str(tmp_path / "out.wav"))

        bad_sf = types.ModuleType("soundfile")
        bad_sf.info = lambda p: (_ for _ in ()).throw(Exception("not a wav"))
        monkeypatch.setitem(sys.modules, "soundfile", bad_sf)

        result = mcp.say(text="hola", voice="neutral_male")
        assert "error" not in result
        assert result["duration_seconds"] is None


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
            return str(tmp_path / "out.wav")

        monkeypatch.setattr(mcp, "generate", fake_generate)
        monkeypatch.setitem(sys.modules, "soundfile", _fake_sf_module(1.5))

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
            mcp, "generate", lambda **kw: (captured.update(kw), str(tmp_path / "x.wav"))[1]
        )
        monkeypatch.setitem(sys.modules, "soundfile", _fake_sf_module())
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
        monkeypatch.setattr(
            mcp, "list_voices", lambda **kw: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        with caplog.at_level(logging.ERROR, logger="spanish_tts.mcp_server"):
            result = mcp.list_all_voices()
        assert "error" in result
        assert "Failed to list voices" in result["error"]
        assert any("list_voices() failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# demo() — empty text + partial failure (lines 197, 207-224)
# ---------------------------------------------------------------------------


class TestDemoErrorBranches:
    def test_empty_text_returns_error(self, bundled_presets):
        result = bundled_presets.demo(text="")
        assert result == {"error": "text is empty"}

    def test_whitespace_text_returns_error(self, bundled_presets):
        result = bundled_presets.demo(text="  ")
        assert result == {"error": "text is empty"}

    def test_generates_result_per_voice(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        voices = mcp.list_voices()
        assert voices, "Need bundled voices for this test"

        def fake_generate(**kw):
            return kw["output"]

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
            return kw["output"]

        monkeypatch.setattr(mcp, "generate", fake_generate)
        result = mcp.demo(text="hola", output_dir=str(tmp_path / "partial"))
        statuses = {r["voice"]: r["status"] for r in result["results"]}
        assert statuses[first] == "failed"
        assert any(s == "ok" for s in statuses.values())

    def test_partial_failure_entry_has_error_field(self, bundled_presets, tmp_path, monkeypatch):
        mcp = bundled_presets
        first = list(mcp.list_voices().keys())[0]

        monkeypatch.setattr(
            mcp,
            "generate",
            lambda **kw: (
                (_ for _ in ()).throw(RuntimeError("boom"))
                if kw["output"].endswith(f"{first}.wav")
                else kw["output"]
            ),
        )
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
            return kw["output"]

        monkeypatch.setattr(mcp, "generate", boom)
        with caplog.at_level(logging.ERROR, logger="spanish_tts.mcp_server"):
            mcp.demo(text="hola", output_dir=str(tmp_path / "log_demo"))
        assert any("demo generate for" in r.message for r in caplog.records)
