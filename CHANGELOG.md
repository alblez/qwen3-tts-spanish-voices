# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Rule:** every PR must add a line to `[Unreleased]` before merge.

---

## [Unreleased]

### Added
- `CONTRACT.md` — stable JSON shapes + backward-compat policy for the MCP server (U3-5 part 1).
- `LICENSE` file with full MIT text; PEP 639 migration in `pyproject.toml` (U3-1).

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
