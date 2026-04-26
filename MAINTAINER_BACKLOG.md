# spanish-tts — Maintainer Backlog & Fix Plan

Source-level findings and fix plan for the defects reported in
MCP_CLIENT_REPORT.md, plus issues only visible by reading the repo.

Scope of review: `src/spanish_tts/{mcp_server,engine,config,cli}.py`,
`presets/voices.yaml`, `~/.spanish-tts/voices.yaml` (runtime registry),
`README.md`, and a signature inspection of
`mlx_audio.tts.models.qwen3_tts.Model.generate`.

---

## Cross-reference

| Client-side (MCP_CLIENT_REPORT.md) | Maintainer issue |
|-----------------------------------|------------------|
| C1 speed no-op                    | M1               |
| C2 docstring drift                | M2               |
| C3 truncated descriptions         | M4               |
| C4 demo lacks speed               | M6               |
| C5 no prompts/resources           | M8               |
| —                                 | M3 (source-only) |
| —                                 | M5 (source-only) |
| —                                 | M7 (source-only) |

---

## M1 — `speed` is a no-op (HIGH, blocks M2 and M6)

**Root cause (confirmed)**
`mlx_audio.tts.models.qwen3_tts.Model.generate` accepts `speed:
float = 1.0` but the MLX port does not apply it to output timing.
Confirmed by:

  - `inspect.signature` shows the kwarg exists.
  - Three synthesized files with speed ∈ {0.5, 1.0, 2.0} have different
    sha256 (so the value reaches the model and perturbs sampling) but
    near-identical duration (3.92 / 3.28 / 3.68 s).

This is an upstream limitation, not a wiring bug. `engine.py` forwards
`speed` correctly.

**Fix strategy** — see Design Note at the bottom. Short version:
post-synthesis pitch-preserving time-stretch via `librosa.effects.
time_stretch` inside `_collect_audio` (or a new `_apply_speed` helper)
before `sf.write`.

**Files**
  - `src/spanish_tts/engine.py`     (add time-stretch step, add
                                    `SPEED_MIN/SPEED_MAX` constants)
  - `src/spanish_tts/mcp_server.py` (docstring, re-use constants)
  - `src/spanish_tts/cli.py`        (add range validator)
  - `pyproject.toml`                (add `librosa` to `[mlx]` extra or
                                    a new `[speed]` extra)

**Acceptance**
  - speed=0.5 produces duration ≥1.7× speed=1.0 (±15%).
  - speed=2.0 produces duration ≤0.6× speed=1.0 (±15%).
  - FFT centroid of speed=0.5 output within ±10 Hz of speed=1.0
    (pitch preservation).
  - Out-of-range values still rejected.

---

## M2 — Docstring/validator drift (MEDIUM)

Four spots advertise speed range 0.8-1.3 while the only validator in
place enforces 0.5-2.0. CLI has no validator at all.

**Locations**
  - `src/spanish_tts/mcp_server.py`  — `say` docstring.
  - `src/spanish_tts/cli.py`         — `--speed` click option help.
  - `src/spanish_tts/engine.py`      — `generate_clone` docstring.
  - `src/spanish_tts/engine.py`      — `generate_design` docstring.

**Fix**
  - Extract `SPEED_MIN, SPEED_MAX = 0.5, 2.0` in `engine.py`.
  - Reference them in every docstring and from both validators.
  - Add `click.FloatRange(SPEED_MIN, SPEED_MAX)` to the CLI.

**Acceptance**
  - `rg "0\.8" src/spanish_tts` → no hits referring to speed range.
  - CLI `--speed 3.0` exits non-zero before any model loads.
  - Parametrized boundary test covers both interfaces.

---

## M3 — MCP `effective_speed` fallback is dead code (LOW, source-only)

```python
effective_speed = speed or defaults.get("speed", 1.0)
```

in `mcp_server.say`. The MCP `speed` parameter has default `1.0`
(truthy) and `0.0` is excluded by the validator, so the fallback branch
never fires. Users who set `defaults.speed` in `voices.yaml` are
silently ignored by the MCP server even though the CLI (which uses
`default=None`) honors them.

**Fix (recommended: option a)**
  a) Change MCP param default to `None` and keep the fallback. Update
     docstring: "defaults to voices.yaml `defaults.speed` or 1.0."
  b) Remove the fallback; document that MCP ignores `defaults.speed`.

**Acceptance**
Behavior of MCP `say` matches CLI `say` for any given `voices.yaml`
`defaults.speed` when the user doesn't pass `speed`.

---

## M4 — `list_all_voices` truncates descriptions without indicator (LOW)

```python
summary[name]["description"] = instruct[:80]
```

in `mcp_server.list_all_voices`. No ellipsis, no `truncated` flag.
Clients cannot tell whether "Cheerful" is the end of the sentence or
the 80-char cutoff.

**Fix (pick one)**
  a) Return the full `instruct` — it's short (≤300 chars for current
     voices) and LLM clients handle longer strings fine.
  b) Append `…` when truncated and add `instruct_truncated: bool`.

Recommendation: (a). The reason truncation exists (avoiding token bloat
in tool responses) is negligible at the current voice count.

---

## M5 — README claims "14 voices out of the box" (MEDIUM, source-only)

`presets/voices.yaml` ships **4** design voices. The other 10 (clones)
exist only after a user runs `scripts/curate.py` against a VoxForge
download — a step not mentioned in the "Quick Start" section.

Also: `neutral_female` and `warm_female` exist in presets as **design**
voices and in the curated runtime registry as **clone** voices. The
curate step silently overwrites them. README's "Design Voices" table
lists only two designs, implying the preset entries for these names are
transient.

**Fix (pick one)**
  a) Rewrite README Quick Start to show the two-phase flow
     (install → `spanish-tts curate` or similar) honestly.
  b) Bundle pre-rendered VoxForge clips in the package (check licensing
     — VoxForge is GPL-ish; verify compatibility with MIT). Then `pip
     install` alone yields 14 voices.

Also:
  - Rename the preset versions of `neutral_female` / `warm_female` so
    the curate step doesn't shadow them, or document the overwrite
    explicitly.

**Acceptance**
A new user following README top-to-bottom sees the same voice count
promised.

---

## M6 — `demo` hardcodes `speed=1.0` (LOW)

Both `mcp_server.demo` and `cli.demo` pass `speed=1.0` unconditionally.
After M1 lands this is a noticeable feature gap.

**Fix**
Add `speed: float = 1.0` with the same range validator to both. Pass it
through to `generate`.

---

## M7 — `MODELS["custom"]` is dead code (TRIVIAL, source-only)

```python
MODELS = {
    "clone":  "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "custom": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
}
```

The `"custom"` key is never referenced. Either wire up a
`generate_custom` path (Qwen3-TTS CustomVoice uses a different prompt
format) or delete the entry.

Recommendation: delete. If the CustomVoice model is desired later, add
it back when the feature is actually implemented.

---

## M8 — No MCP prompts or resources (ENHANCEMENT)

Server registers tools only. Low-hanging fruit:

  - **Prompt** `narrate_story`: takes text + voice, emits a framed
    user+assistant template that encourages good prosody.
  - **Prompt** `compare_voices`: pre-baked A/B test over two voices.
  - **Resource** `spanish-tts://voices/{name}/preview.wav`: lazy-
    generate a short preview on first read, cache on disk.
  - **Resource** `spanish-tts://readme`: mirror README.md.

Not urgent; good first issue material.

---

## Design note — fixing `speed` with librosa

**Options considered**
  - A. Remove `speed` entirely. Rejected — breaks current API, users
    expect a speed knob.
  - B. Post-synthesis time-stretch via librosa. **Chosen.**
  - C. `pyrubberband` (better quality, system binary). Rejected as
    install burden.
  - D. `pysoundtouch`. Rejected — middling quality, extra dep.

**Why librosa**
  - Pure-Python phase vocoder (`librosa.effects.time_stretch`).
  - Pitch-preserving by construction. No chipmunk effect.
  - No system deps; works on Apple Silicon without extra setup.
  - Graceful fallback: wrap the import in try/except, skip time-stretch
    and log a warning if unavailable.

**Integration point**
`engine._collect_audio` already returns a numpy float array right
before `sf.write`. Add an `_apply_speed(audio, speed, sample_rate)`
helper called from both `generate_clone` and `generate_design` after
`_collect_audio` and before `sf.write`.

**Sample-rate awareness**
`librosa.effects.time_stretch` expects a mono float32 array and a
parameter `rate` where `rate > 1` speeds up. Our `speed` matches
that convention directly. Pass `audio.astype(np.float32)` if needed;
Qwen3-TTS already returns float.

**Stop passing `speed=speed` to `model.generate`**
Confirmed no-op upstream. Keep the keyword as a no-op with a
`# intentionally no-op; see librosa post-process` comment so a future
reader doesn't re-introduce it.

**Public range**
Keep 0.5–2.0. librosa tolerates 0.25–4.0 but artefacts get bad outside
0.5–2.0 for speech.

---

## Suggested delivery

  - Open 8 issues (M1–M8).
  - M1 blocks M2 and M6.
  - A single PR for M1+M2+M6 ("speed correctness") is reasonable.
  - M3, M4, M5, M7 are small independent PRs.
  - M8 is backlog / "good first issue".
