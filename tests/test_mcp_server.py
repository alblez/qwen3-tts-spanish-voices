"""Tests for MCP server tool behaviour."""

import importlib
import sys

import pytest


@pytest.fixture
def bundled_presets(tmp_path, monkeypatch):
    """Point SPANISH_TTS_CONFIG at an empty dir so load_voices() falls back
    to the bundled presets/voices.yaml (4 design + any clone presets).
    Reloads modules so they pick up the new env-driven config dir.
    """
    monkeypatch.setenv("SPANISH_TTS_CONFIG", str(tmp_path))
    # Drop cached modules so they re-read the env var.
    for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
        sys.modules.pop(mod, None)
    import spanish_tts.config  # noqa: F401
    import spanish_tts.mcp_server as mcp_server

    importlib.reload(mcp_server)
    yield mcp_server
    for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
        sys.modules.pop(mod, None)


def test_list_all_voices_returns_full_instruct(bundled_presets):
    """Design voices should return their complete instruct string,
    not a silent 80-char truncation. Regression test for M4.

    Bundled presets ship instructs in the 99-121 char range; clipping
    at 80 drops the trailing tone/pacing clause that defines the voice.
    """
    result = bundled_presets.list_all_voices()
    assert "voices" in result, result
    voices = result["voices"]

    expected = (
        "Hombre hispanohablante de 35 años con acento neutro y claro. "
        "Voz calmada y articulada, como un narrador profesional."
    )
    assert "neutral_male" in voices
    assert voices["neutral_male"]["type"] == "design"
    assert voices["neutral_male"]["description"] == expected
    assert len(voices["neutral_male"]["description"]) > 80

    # Every bundled design voice should come through unclipped.
    for name, info in voices.items():
        if info.get("type") == "design":
            assert "description" in info
            assert info["description"], f"{name} has empty description"


def test_say_rejects_absolute_path_outside_output_dir(bundled_presets, tmp_path, monkeypatch):
    """MCP-1 regression: absolute paths outside the configured
    output_dir must be rejected before engine.generate is called.
    """
    mcp = bundled_presets
    # Make sure engine.generate is NEVER reached.
    monkeypatch.setattr(
        mcp,
        "generate",
        lambda *a, **kw: pytest.fail("generate() must not be called for rejected path"),
    )
    # Force defaults.output_dir to a known tmp path.
    monkeypatch.setattr(mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)})

    result = mcp.say(text="hola", voice="neutral_male", output="/etc/passwd")
    assert "error" in result
    assert "escapes" in result["error"] or "traversal" in result["error"]


def test_say_rejects_dotdot_escape(bundled_presets, tmp_path, monkeypatch):
    """Relative `../../` escape attempts must also be rejected."""
    mcp = bundled_presets
    monkeypatch.setattr(
        mcp,
        "generate",
        lambda *a, **kw: pytest.fail("generate() must not be called for rejected path"),
    )
    monkeypatch.setattr(mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)})

    result = mcp.say(text="hola", voice="neutral_male", output="../../../etc/passwd")
    assert "error" in result
    assert "escapes" in result["error"] or "traversal" in result["error"]


def test_say_accepts_relative_path_inside_output_dir(bundled_presets, tmp_path, monkeypatch):
    """Relative paths that resolve inside output_dir must pass the
    guard and be handed to engine.generate with an absolute path.
    """
    mcp = bundled_presets
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(mcp, "generate", fake_generate)
    monkeypatch.setattr(mcp, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)})

    result = mcp.say(text="hola", voice="neutral_male", output="sub/out.wav")
    # No error returned. The path may or may not exist yet; sf.info
    # will fail which is fine — `duration_seconds` comes back None.
    assert "error" not in result, result
    resolved = captured["output"]
    assert resolved.startswith(str(tmp_path.resolve()))
    assert resolved.endswith("sub/out.wav")
