"""MCP protocol shape tests (U3-5 part 2).

Verifies the stable JSON contract defined in CONTRACT.md using FastMCP's
in-process async API.  No MCP transport layer is involved — tool handlers
run directly in the test process.

FastMCP API used here (v1.27.0+):
  - mcp.list_tools()          → List[Tool]  (async)
  - mcp.call_tool(name, args) → List[TextContent]  (async, result[0].text is JSON)
  - Missing required args / wrong type → raises mcp.server.fastmcp.exceptions.ToolError
"""

import asyncio
import importlib
import json
import sys

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from spanish_tts.engine import TtsResult

# ---------------------------------------------------------------------------
# Fixture: reload mcp_server against bundled presets (no user voices.yaml)
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_module(tmp_path, monkeypatch):
    """Yield a freshly-imported mcp_server module pointing at an empty
    config dir so load_voices() falls back to the 4 bundled design voices.
    """
    monkeypatch.setenv("SPANISH_TTS_CONFIG", str(tmp_path))
    for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
        sys.modules.pop(mod, None)
    import spanish_tts.mcp_server as mcp_mod

    importlib.reload(mcp_mod)
    yield mcp_mod
    for mod in ("spanish_tts.config", "spanish_tts.mcp_server"):
        sys.modules.pop(mod, None)


def _call(mcp_obj, tool_name, args):
    """Run mcp_obj.call_tool synchronously; return parsed dict."""
    raw = asyncio.run(mcp_obj.mcp.call_tool(tool_name, args))
    return json.loads(raw[0].text)


def _list_tools(mcp_obj):
    return asyncio.run(mcp_obj.mcp.list_tools())


# ---------------------------------------------------------------------------
# Tool inventory
# ---------------------------------------------------------------------------


class TestToolInventory:
    def test_exactly_four_tools(self, mcp_module):
        """CONTRACT: exactly 4 tools — say, list_all_voices, demo, get_version."""
        tools = _list_tools(mcp_module)
        names = {t.name for t in tools}
        assert names == {"say", "list_all_voices", "demo", "get_version"}, (
            f"Unexpected tools: {names}"
        )

    def test_say_tool_description_present(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        assert tools["say"].description
        assert (
            "speech" in tools["say"].description.lower()
            or "text" in tools["say"].description.lower()
        )

    def test_list_all_voices_description_present(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        assert tools["list_all_voices"].description

    def test_demo_description_present(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        assert tools["demo"].description

    def test_get_version_description_present(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        assert tools["get_version"].description


class TestGetVersionShape:
    def test_get_version_returns_version_and_package(self, mcp_module):
        """CONTRACT: get_version returns version (str) and package (str)."""
        result = _call(mcp_module, "get_version", {})
        assert "version" in result, result
        assert "package" in result, result
        assert isinstance(result["version"], str)
        assert result["package"] == "qwen3-tts-spanish-voices"

    def test_get_version_no_required_params(self, mcp_module):
        """CONTRACT: get_version takes no required parameters."""
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["get_version"].inputSchema
        assert schema.get("required", []) == []


# ---------------------------------------------------------------------------
# say — input schema shape
# ---------------------------------------------------------------------------


class TestSayInputSchema:
    def test_say_text_required(self, mcp_module):
        """CONTRACT: say.text is required."""
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["say"].inputSchema
        assert "text" in schema.get("required", [])

    def test_say_text_is_string(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["say"].inputSchema
        assert schema["properties"]["text"]["type"] == "string"

    def test_say_voice_optional_string(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["say"].inputSchema
        assert "voice" not in schema.get("required", [])
        assert schema["properties"]["voice"]["type"] == "string"

    def test_say_speed_optional_nullable(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["say"].inputSchema
        assert "speed" not in schema.get("required", [])
        # anyOf nullable number
        speed_schema = schema["properties"]["speed"]
        types_in_schema = [s.get("type") for s in speed_schema.get("anyOf", [speed_schema])]
        assert "number" in types_in_schema or speed_schema.get("type") == "number"

    def test_say_stream_optional_boolean(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["say"].inputSchema
        assert "stream" not in schema.get("required", [])
        assert schema["properties"]["stream"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# say — malformed call rejection (pydantic validates before handler runs)
# ---------------------------------------------------------------------------


class TestSayRejection:
    def test_missing_text_raises_tool_error(self, mcp_module):
        """Missing required 'text' must be caught by pydantic before handler."""
        with pytest.raises(ToolError):
            asyncio.run(mcp_module.mcp.call_tool("say", {}))

    def test_wrong_type_for_stream_raises_tool_error(self, mcp_module):
        """stream must be boolean; passing a string should raise ToolError."""
        with pytest.raises(ToolError):
            asyncio.run(mcp_module.mcp.call_tool("say", {"text": "hola", "stream": "yes_please"}))


# ---------------------------------------------------------------------------
# say — success response shape (monkeypatched engine)
# ---------------------------------------------------------------------------


class TestSaySuccessShape:
    def test_say_returns_path_and_duration(self, mcp_module, monkeypatch, tmp_path):
        """CONTRACT: success response has 'path' (str) and 'duration_seconds' (float)."""
        out = str(tmp_path / "out.wav")

        monkeypatch.setattr(
            mcp_module, "generate", lambda **kw: TtsResult(path=out, duration_seconds=0.2)
        )
        monkeypatch.setattr(
            mcp_module, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )

        result = _call(mcp_module, "say", {"text": "hola", "voice": "neutral_male"})
        assert "error" not in result, result
        assert "path" in result
        assert isinstance(result["path"], str)
        assert "duration_seconds" in result
        assert isinstance(result["duration_seconds"], float)

    def test_say_error_response_has_error_key(self, mcp_module, monkeypatch, tmp_path):
        """CONTRACT: error response has 'error' (str)."""
        monkeypatch.setattr(
            mcp_module, "get_defaults", lambda: {"speed": 1.0, "output_dir": str(tmp_path)}
        )
        result = _call(mcp_module, "say", {"text": "hola", "voice": "__nonexistent__"})
        assert "error" in result
        assert isinstance(result["error"], str)


# ---------------------------------------------------------------------------
# list_all_voices — input schema + response shape
# ---------------------------------------------------------------------------


class TestListAllVoicesShape:
    def test_no_required_params(self, mcp_module):
        """CONTRACT: list_all_voices has no required params."""
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["list_all_voices"].inputSchema
        assert schema.get("required", []) == []

    def test_returns_voices_dict(self, mcp_module):
        """CONTRACT: success response has 'voices' mapping."""
        result = _call(mcp_module, "list_all_voices", {})
        assert "voices" in result, result
        assert isinstance(result["voices"], dict)

    def test_each_voice_has_required_fields(self, mcp_module):
        """CONTRACT: each voice entry has type, gender, language."""
        result = _call(mcp_module, "list_all_voices", {})
        for name, info in result["voices"].items():
            assert "type" in info, f"Voice {name!r} missing 'type'"
            assert "gender" in info, f"Voice {name!r} missing 'gender'"
            assert "language" in info, f"Voice {name!r} missing 'language'"

    def test_design_voice_has_description(self, mcp_module):
        """CONTRACT: design voices have 'description' field."""
        result = _call(mcp_module, "list_all_voices", {})
        for name, info in result["voices"].items():
            if info.get("type") == "design":
                assert "description" in info, f"Design voice {name!r} missing 'description'"

    def test_bundled_voices_present(self, mcp_module):
        """Bundled presets must include all 4 design voices."""
        result = _call(mcp_module, "list_all_voices", {})
        voices = result["voices"]
        for expected in ("neutral_male", "neutral_female", "energetic_male", "warm_female"):
            assert expected in voices, f"Bundled voice {expected!r} not found"

    def test_clone_voice_has_accent(self, mcp_module, monkeypatch):
        """CONTRACT: clone voice entries include 'accent' field."""
        import yaml

        clone_voices_yaml = """
voices:
  my_clone:
    type: clone
    ref_audio: /tmp/ref.wav
    gender: male
    language: Spanish
    accent: mexico
"""
        fake_voices = yaml.safe_load(clone_voices_yaml)
        monkeypatch.setattr(mcp_module, "list_voices", lambda: fake_voices["voices"])
        result = _call(mcp_module, "list_all_voices", {})
        assert "my_clone" in result["voices"]
        assert result["voices"]["my_clone"]["accent"] == "mexico"


# ---------------------------------------------------------------------------
# demo — input schema + response shape
# ---------------------------------------------------------------------------


class TestDemoShape:
    def test_demo_text_required(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["demo"].inputSchema
        assert "text" in schema.get("required", [])

    def test_demo_output_dir_optional(self, mcp_module):
        tools = {t.name: t for t in _list_tools(mcp_module)}
        schema = tools["demo"].inputSchema
        assert "output_dir" not in schema.get("required", [])

    def test_demo_missing_text_raises_tool_error(self, mcp_module):
        with pytest.raises(ToolError):
            asyncio.run(mcp_module.mcp.call_tool("demo", {}))

    def test_demo_success_has_results_list(self, mcp_module, monkeypatch, tmp_path):
        """CONTRACT: success response has 'results' list."""
        call_count = {"n": 0}

        def fake_generate(**kw):
            call_count["n"] += 1
            out = str(tmp_path / f"out_{call_count['n']}.wav")
            return TtsResult(path=out, duration_seconds=0.1)

        monkeypatch.setattr(mcp_module, "generate", fake_generate)
        result = _call(mcp_module, "demo", {"text": "hola", "output_dir": str(tmp_path)})
        assert "results" in result, result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0

    def test_demo_each_result_has_status(self, mcp_module, monkeypatch, tmp_path):
        """CONTRACT: each result entry has 'voice', 'status', and either 'path' or 'error'."""

        def fake_generate(**kw):
            out = str(tmp_path / "out.wav")
            return TtsResult(path=out, duration_seconds=0.1)

        monkeypatch.setattr(mcp_module, "generate", fake_generate)
        result = _call(mcp_module, "demo", {"text": "hola", "output_dir": str(tmp_path)})
        for entry in result["results"]:
            assert "voice" in entry, f"result entry missing 'voice': {entry}"
            assert "status" in entry, f"result entry missing 'status': {entry}"
            assert entry["status"] in ("ok", "failed"), f"unexpected status: {entry['status']}"
            if entry["status"] == "ok":
                assert "path" in entry
            else:
                assert "error" in entry

    def test_demo_partial_failure_continues(self, mcp_module, monkeypatch, tmp_path):
        """CONTRACT: one voice failure does not abort the whole demo."""
        voice_names = list(mcp_module.list_voices().keys())
        assert len(voice_names) >= 2, "Need at least 2 voices for partial-failure test"

        first_voice = voice_names[0]
        call_count = {"n": 0}

        def fake_generate(**kw):
            call_count["n"] += 1
            # Dispatch on output path suffix — identity check on voice_config dicts
            # would fail because list_voices() creates new objects on each call.
            name = kw["output"].split("/")[-1].replace(".wav", "")
            if name == first_voice:
                raise RuntimeError("simulated first-voice failure")
            out = str(tmp_path / f"out_{call_count['n']}.wav")
            return TtsResult(path=out, duration_seconds=0.1)

        monkeypatch.setattr(mcp_module, "generate", fake_generate)
        result = _call(mcp_module, "demo", {"text": "hola", "output_dir": str(tmp_path)})
        assert "results" in result
        statuses = {r["voice"]: r["status"] for r in result["results"]}
        # The first voice must have failed; at least one other must have succeeded.
        assert statuses.get(first_voice) == "failed", f"Expected {first_voice!r} to fail"
        assert "ok" in statuses.values(), "Expected at least one ok result"
