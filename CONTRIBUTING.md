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

`engine.py` pins both models to specific commit SHAs in `MODEL_REVISIONS`.
This prevents silent breakage when the upstream HF repo is updated.

### Bump process

1. **Find the new SHA** on the model's HF page → *Files and versions* → *Commits*
   (or use the HF API: `python -c "from huggingface_hub import model_info; print(model_info('mlx-community/<id>').sha)"`).
2. **Verify the commit** on the HF model card: check README diff, config.json,
   and tokenizer for unexpected changes.
3. **Update** `MODEL_REVISIONS` in `src/spanish_tts/engine.py`.
4. **Test locally** on Apple Silicon:
   ```bash
   conda run -n qwen3-tts pytest -m "requires_mlx" -q
   ```
5. **Open a PR** with a `CHANGELOG.md [Unreleased]` entry:
   ```
   ### Changed
   - Bump clone/design model revision to <new-sha> (tested on <date>).
   ```
6. CI validates the new revision on Ubuntu (mocked) and macOS (real, if CI macOS
   job is enabled after U3-13).

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
    source_license: <SPDX string>  # optional (e.g. "GPL-3.0", "CC-BY-4.0")
    source_url: <url>              # optional (HF dataset card or original source)
```

`source_license` and `source_url` are optional but **strongly recommended**
for any voice built from third-party audio.  `curate.py export` automatically
sets `source_license: "GPL-3.0"` and `source_url` to the VoxForge HF card.
The CLI `add-ref --license <SPDX>` flag writes `source_license`; omitting it
writes `"user-supplied-unspecified"` as a placeholder.

Clone voices built from VoxForge-derived audio must set
`source_license: GPL-3.0` and `source_url` pointing to the HF dataset card.
