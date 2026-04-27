# Contributing

## Setup

```bash
conda activate qwen3-tts
pip install -e ".[dev,mcp,speed]"
pre-commit install                        # commit-stage hooks
pre-commit install --hook-type pre-push   # push-stage pytest smoke
```

The `[mlx]` extra is Apple Silicon-only; CI skips it and uses `requires_mlx` /
`slow` markers to gate model-dependent tests.

## Running tests

```bash
# Fast suite (CI-equivalent — no MLX, no downloads)
conda run -n qwen3-tts python -m pytest -q -m "not slow"

# Full suite including slow MLX integration tests (requires Apple Silicon + model files)
conda run -n qwen3-tts python -m pytest -q

# With coverage
conda run -n qwen3-tts python -m pytest --cov=spanish_tts --cov-report=term -q -m "not slow"
```

## Pytest marker conventions

| Marker | Meaning | Skipped in |
|--------|---------|------------|
| `slow` | Requires `[mlx]` extra or long synthesis run | CI ubuntu leg, pre-push hook |
| `requires_mlx` | Requires Apple Silicon + MLX model downloaded | CI ubuntu leg |

Every test that calls `engine.generate_*` against a real model must carry
both markers.

## Commit style

Caveman-full for commit messages, PR titles, and PR bodies: terse, shouty,
narrative. Code, identifiers, docstrings, and README prose use normal English.

Pattern: `type(scope): U3-N WHAT + WHY (brief)`

Examples:
```
feat(config): U3-3 ATOMIC SAVE + SAFE LOAD
fix(engine): U3-4 _model_cache CONCURRENCY LOCK
docs(license): U3-1 ADD LICENSE FILE + PEP 639 MIGRATION
```

## PR checklist

- [ ] `CHANGELOG.md [Unreleased]` entry added.
- [ ] Tests added or explicitly noted why not needed.
- [ ] `pre-commit run --all-files` clean.
- [ ] For HIGH behavior-changing code: full Phase 1 swarm (3 agents).
- [ ] CI green (3.11 + 3.12) before merge.

## Bumping HF model revisions (U3-9)

1. Find known-good commit SHA on HF for the target model.
2. Update the `MODELS` dict in `src/spanish_tts/engine.py`.
3. Add a `CHANGELOG.md [Unreleased]` entry.
4. Open a PR; CI validates the new revision on Ubuntu (mocked) and macOS (real).

## Voice schema

Each entry in `voices.yaml` (user config at `~/.spanish-tts/voices.yaml` or
preset at `presets/voices.yaml`):

```yaml
voices:
  <name>:
    type: clone | design           # required
    gender: male | female | ...    # required
    language: es | ...             # required
    accent: <string>               # optional
    description: <string>          # optional
    ref_audio: <path>              # required for clone voices
    instruct: <string>             # required for design voices
    source_license: <SPDX string>  # optional (e.g. GPL-3.0)
    source_url: <url>              # optional
```

Clone voices built from VoxForge-derived audio must set
`source_license: GPL-3.0` and `source_url` pointing to the HF dataset card.
