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
