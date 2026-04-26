# Changelog

## 0.2.0 — speed correctness + CI

Fix the `speed` parameter that was effectively a no-op (upstream
mlx-audio limitation: the kwarg is accepted but does not alter playback
rate), plus related housekeeping.

### Added
- Pitch-preserving time-stretch via `librosa.effects.time_stretch`,
  applied post-synthesis in `_apply_speed`. Real `speed` control now
  works across the 0.5–2.0 range.
- New optional `[speed]` extra. Users who skip it get a graceful
  warning and unchanged audio rather than a crash.
- `speed` parameter on `demo` (both CLI `--speed` and MCP `speed=`).
- `tests/test_speed.py` — 18 tests covering boundaries, stretch ratios
  on synthetic waveforms, pitch preservation, CLI rejection before
  model load, MCP error shape, and graceful librosa-missing fallback.
- GitHub Actions CI (`.github/workflows/ci.yml`) — Ubuntu, Python 3.11
  and 3.12, `[mcp,speed]` extras. Installs skip `[mlx]` since MLX is
  Apple-Silicon only; `requires_mlx` / `slow` markers gate tests that
  need the model.
- Librosa JIT warmup in MCP `_preload_models` so the first user-facing
  `say` with a non-trivial speed does not pay the ~18s init cost.

### Changed
- `SPEED_MIN`, `SPEED_MAX` extracted as module-level constants in
  `engine.py` — single source of truth used by MCP/CLI validators and
  docstrings.
- CLI `--speed` option now uses `click.FloatRange(SPEED_MIN, SPEED_MAX)`
  and rejects out-of-range values before any model load.
- All docstrings and help text unified on the 0.5–2.0 range (was
  inconsistently documented as 0.8–1.3).

### Fixed
- MCP `say` and CLI `say` now behave identically: the speed arg is
  validated to the same range and actually affects output duration.
- MCP `say` default `speed` is now `None` (was `1.0`), matching the
  CLI. When omitted, both callers fall through to
  `voices.yaml > defaults.speed` before validation, so the registry
  default is honoured instead of being silently overridden by a
  truthy 1.0 default.

### Known limitations
- librosa time-stretch is phase-vocoder based; artefacts can creep in
  at speeds below 0.5 or above 2.0. The public range stays 0.5–2.0.
- MLX Qwen3-TTS `speed=` kwarg upstream remains a no-op; we keep it on
  the call site with a comment as a future hook.

## 0.1.0

Initial release. See git history for feature set.
