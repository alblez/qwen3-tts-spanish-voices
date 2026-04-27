# Architecture

## Module overview

```text
qwen3-tts-spanish-voices/
├── src/spanish_tts/
│   ├── __init__.py      # Public re-exports; __version__
│   ├── cli.py           # Click CLI (say, list, demo, add-ref, add-design, remove)
│   ├── config.py        # YAML voice registry: load, save, get, list, add, remove
│   ├── engine.py        # MLX Qwen3-TTS wrapper: clone + design generation, speed
│   └── mcp_server.py    # FastMCP server exposing say, list_all_voices, demo
├── presets/
│   └── voices.yaml      # Bundled preset voice definitions (shipped with package)
├── scripts/
│   └── curate.py        # VoxForge corpus browser: find + export reference audio
├── tests/               # pytest suite (fast and slow tiers)
├── CONTRACT.md          # Stable MCP JSON shapes + backward-compat policy
├── pyproject.toml
└── LICENSE
```

## Voice registry precedence

```
user  ~/.spanish-tts/voices.yaml   ← takes priority if it exists
           ↓  (fallback)
preset  presets/voices.yaml        ← shipped with package; read-only
```

`config.load_voices()` checks the user file first; if absent, returns the
bundled preset. `config.save_voices()` always writes to the user file
(creating `~/.spanish-tts/` if needed). The preset is never modified at
runtime.

## Synthesis pipeline

```
text + voice params
       │
       ▼
config.get_voice(name)      ← resolve instruct / ref_audio / defaults
       │
       ▼
engine._get_model(mode)     ← lazy-load; cache in _model_cache
       │
       ▼
model.generate(text, ...)   ← MLX inference (clone or design)
       │
       ▼
engine._collect_audio(...)  ← drain async/streaming result iterator
       │
       ▼
engine._apply_speed(...)    ← librosa time-stretch if speed ≠ 1.0
       │
       ▼
sf.write(output_path, ...)  ← write WAV; return path
```

Clone mode encodes reference audio before generation; design mode skips
that step and uses an instruct prompt instead.

## MCP security boundary

The MCP server is the only public entry point when the package is used as
an AI tool (Hermes, Claude Desktop, etc.). Security checks happen **at the
MCP seam** in `mcp_server.py` before any engine call:

- **Path-traversal guard** (`say(output=...)`): resolves the path and asserts
  it stays within `Path.home()`. Rejects `..`-escape attempts.
- **Text length cap**: 10 000 characters maximum.
- **Sandbox check** (`demo(output_dir=...)`): same path-resolve assertion.

The engine layer (`engine.py`) is considered trusted; it does not repeat
these checks. CLI users are assumed to be the owner of the machine.

## Test taxonomy

| Tier | Markers | Location | Runs in CI |
|------|---------|----------|------------|
| Fast | (none) | `tests/test_*.py` | Always (Ubuntu 3.11 + 3.12) |
| Slow | `@pytest.mark.slow` + `@pytest.mark.requires_mlx` | `tests/test_engine_integration.py` | macOS leg only (after U3-13) |

Fast tests mock the engine via `monkeypatch` or use synthetic audio arrays.
Slow tests load real MLX models; they require Apple Silicon and ~3 GB of
downloaded model weights.

CI filters with `-m "not slow"` on the ubuntu leg.

## Extension points

### Adding a new voice type

1. Add `type: <new_type>` entries to `voices.yaml`.
2. Add a branch in `engine.generate()` dispatching to a new `_generate_<type>` function.
3. Update `CONTRACT.md` `list_all_voices` schema if the voice shape changes.

### Adding a new MCP tool

1. Define a new `@mcp.tool()` function in `mcp_server.py`.
2. Add its schema and error contract to `CONTRACT.md`.
3. Add protocol-shape tests in `tests/test_mcp_protocol.py`.

## Key constants

| Name | Location | Purpose |
|------|----------|---------|
| `SPEED_MIN`, `SPEED_MAX` | `engine.py` | Single source of truth for speed range (0.5–2.0) |
| `MODELS` | `engine.py` | HF model IDs for clone and design modes |
| `DEFAULT_CONFIG_DIR` | `config.py` | `~/.spanish-tts/` |
| `DEFAULT_VOICES_FILE` | `config.py` | `presets/voices.yaml` (bundled) |
