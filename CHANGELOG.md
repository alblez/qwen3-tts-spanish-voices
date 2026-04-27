# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Rule:** every PR must add a line to `[Unreleased]` before merge.

---

## [Unreleased]

### Added
- `_validate_text` helper in `engine.py` — rejects empty/NUL/too-long text with `ValueError`. Called from `generate_clone` and `generate_design` (U3-17).
- `TtsResult` dataclass in `engine.py` — frozen, `path: str`, `duration_seconds: float`, `__str__` returns path for backward-compat (U3-19).
- Logger hygiene: `generate_design` logs instruct body at `DEBUG` (not `INFO`) to avoid voice-persona PII in production logs (U3-18).
- `_sandbox_path` helper in `mcp_server.py` — factors out MCP-1 path-traversal logic. Applied to `say(output=...)` and `demo(output_dir=...)` (U3-7).
- `demo` now rejects text > 10 000 chars (`text_too_long` code) and sandboxes `output_dir` to `$HOME` or the system temp directory (U3-7).
- `MODEL_REVISIONS` dict in `engine.py` with pinned HF commit SHAs for both models. `_get_model` passes `revision=` to `mlx_audio.tts.load` (U3-9).
- `CONTRIBUTING.md` documents the model-revision bump process (U3-9).
- `__version__` in `__init__.py` now read from `importlib.metadata` — single source of truth from `pyproject.toml` (U3-8).
- `get_version()` MCP tool returns `{"version": ..., "package": ...}` for skill compatibility probing (U3-8).
- `spanish-tts-mcp` console script added to `pyproject.toml` `[project.scripts]` (U3-14).
- `_preload_models` moved to background daemon thread so MCP stdio handshake (`list_tools`, `initialize`) responds immediately without blocking on model downloads (U3-14).
- `TtsResult` exported from `spanish_tts.__init__` (U3-19).
- `CONTRACT.md` — stable JSON shapes + backward-compat policy for the MCP server (U3-5 part 1).
- `LICENSE` file with full MIT text; PEP 639 migration in `pyproject.toml` (U3-1).
- `SECURITY.md` with supported versions, private advisory link, and 72h ack SLA (U3-11).
- `NOTICE` file attributing Qwen3-TTS (Apache-2.0), mlx-audio (MIT), VoxForge (GPL-3.0) (U3-16).
- `ARCHITECTURE.md`, `CONTRIBUTING.md`, `.github/CODEOWNERS` — module overview, contributor guide, code ownership (U3-12).
- `CHANGELOG.md` Keep-a-Changelog format with umbrella-2 backfill (U3-10).
- `tests/test_mcp_protocol.py` — 25 in-process FastMCP contract shape tests (U3-5 part 2).
- `tests/test_mcp_errors.py` — 20 unit tests covering all mcp_server.py error branches (U3-6).
- `tests/test_engine_integration.py` — 17 slow/mlx-marked integration tests: streaming, language mapping, cache hit, missing ref_audio (U3-6).
- `[project.urls]`, `classifiers`, `keywords` added to `pyproject.toml`; `requires_mlx` marker registered (U3-20).

### Changed
- `generate_clone`, `generate_design`, `generate` now return `TtsResult` instead of `str`. `str(result)` still returns the path — string-coercion callers unaffected; equality comparisons against plain path strings break (use `result.path`) (U3-19).
- MCP `say` `duration_seconds` field now always a float (engine-computed). The `sf.info` re-read and its bare-except dead branch are removed (U3-19).
- MCP `say`/`demo` error responses now include a stable `code` field alongside `error`. Stable enum documented in `CONTRACT.md` (U3-17, ECO-016).
- MCP `say` rejects NUL bytes in `text` (`text_nul` code) and adds `math.isfinite` guard on speed before the range check (`speed_not_finite` code) (U3-17).
- CLI `say` speed fallback: `speed or defaults.get(...)` replaced by `speed if speed is not None else ...` — prevents `speed=0.0` falling through (U3-17).
- Logger hygiene: `engine.py` 5 `print(..., file=sys.stderr)` replaced by `logger.info/debug`. Logger pinned to `"spanish_tts.engine"` (was `__name__`). `sys` import removed (U3-18).
- MCP `logger.error` calls now pass `exc_info` only at `DEBUG` level — avoids stack trace noise in production logs (U3-18).
- `pyproject.toml` version bumped to `0.3.0` (breaking engine public API).
- `CONTRACT.md` updated: `duration_seconds` documented as always-present finite float.
- `load_voices` now falls back to bundled presets (with a logged error) when the user's `voices.yaml` contains schema-invalid entries, in addition to the existing `YAMLError` fallback. The user's corrupt file is NOT overwritten.
- `config.py` `save_voices()` now writes atomically via `.yaml.tmp` + `os.replace` (U3-3).
- `config.py` `load_voices()` catches `yaml.YAMLError`, falls back to bundled presets, never overwrites corrupt user file (U3-3).
- `config.py` validates `SPANISH_TTS_CONFIG` env var resolves under `$HOME` before `mkdir` (U3-3).
- `engine.py` adds `_cache_lock` (thread-safe `_model_cache`) and `_generate_lock` (serialized synthesis pipeline) (U3-4).
- README voice tables rewritten: 4 bundled design voices listed; clone voices marked as build-locally-only (U3-2).
- README `Data Source` section corrects VoxForge license from "Creative Commons" to GPL-3.0 (U3-2).
- README adds "License inheritance" subsection and voice-likeness warning (U3-2).
- Logger names unified under `spanish_tts.*` namespace (U3-3).

### Fixed
- Wire `_validate_voices_schema` into `load_voices`/`save_voices` so malformed voice entries can no longer pass through unchecked. Previously the validator was defined and tested but never called in production.
- Speed boundary tests: parametrized NaN/±inf/edge-value rejection by `_apply_speed` (U3-6).

---

## [0.2.0] — 2026-04-26

Umbrella #2: speed correctness, CI, hardening, and developer tooling.

### Added
- Pitch-preserving time-stretch via `librosa.effects.time_stretch` in `_apply_speed`. Real `speed` control across 0.5–2.0 range (M1).
- `[speed]` optional extra; graceful degradation without librosa (M1).
- `speed` parameter on `demo` — CLI `--speed` and MCP `speed=` (M1).
- `tests/test_speed.py` — 18 tests covering boundaries, stretch ratios, pitch preservation, CLI rejection, MCP error shape, librosa-missing fallback (M1).
- GitHub Actions CI (`.github/workflows/ci.yml`) — Ubuntu, Python 3.11 and 3.12, `[mcp,speed]` extras (CI-1).
- SHA-pinned `actions/checkout` and `actions/setup-python`; Dependabot config for `dev-tools` group (CI-1).
- Librosa JIT warmup in MCP `_preload_models` (M1).
- `list_all_voices` returns full instruct-voice catalogue; dropped the clone-only clip (M4).
- `add_voice` collision warnings on duplicate name + duplicate ref_audio (M5).
- README install-from-source section with git-install instructions (REL-1).
- ruff + pre-commit config; `.git-blame-ignore-revs` for mechanical commits (DEV-1).
- CI lint job; Dependabot dev-tools group (DEV-2).
- Path-traversal guard on `say(output=...)` in MCP server (MCP-1).

### Changed
- `SPEED_MIN`, `SPEED_MAX` extracted as module-level constants — single source of truth for validators and docstrings (M2).
- CLI `--speed` uses `click.FloatRange(SPEED_MIN, SPEED_MAX)`; rejects out-of-range before model load (M2).
- All docstrings and help text unified on 0.5–2.0 range (was 0.8–1.3) (M2).
- MCP `say` default `speed` now `None`; honours `voices.yaml > defaults.speed` (M3).

### Removed
- Dead `MODELS["custom"]` entry (M7).

### Known limitations
- librosa time-stretch is phase-vocoder based; artefacts can appear outside 0.5–2.0.
- MLX Qwen3-TTS `speed=` kwarg upstream is a no-op; retained as a future hook.
- Package not on PyPI (Apple Silicon scope; git install covers all users) (REL-1 won't-fix).
- Hermes-skill architecture review deferred (M8 won't-fix).

---

## [0.1.0]

Initial release. See git history for feature set.
