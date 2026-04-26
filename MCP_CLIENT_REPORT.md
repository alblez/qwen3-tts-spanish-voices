# spanish-tts MCP — Client-Side Report

> **Status (v0.2.0, 2026-04-26):** This is a point-in-time client-side
> probe taken before the speed-correctness fix. Defects C1 (speed no-op),
> C2 (docstring drift), and C4 (demo lacks speed) were addressed in PR #1
> (merged commit `cebf301`). C3 (truncated descriptions) and C5
> (no prompts/resources) remain open as M4 and M8 in MAINTAINER_BACKLOG.md.
> See CHANGELOG.md for the shipped set.

Findings from Hermes Agent acting purely as an MCP client of the
`spanish-tts` server. No source code was read to produce this document —
every claim is supported by a tool call and the response it returned.

Transport: native MCP client. Tools probed: `say`, `list_all_voices`,
`demo`. Prompts and resources: both empty on discovery.

---

## Summary of client-visible defects

| # | Severity | Surface | Defect |
|---|----------|---------|--------|
| C1 | HIGH   | say      | `speed` parameter appears to have no audible effect |
| C2 | MEDIUM | say      | Advertised `speed` range (0.8-1.3) contradicts server-enforced range (0.5-2.0) |
| C3 | LOW    | list_all_voices | Design-voice descriptions appear truncated with no indicator |
| C4 | LOW    | demo     | No `speed` parameter exposed |
| C5 | LOW    | server   | No MCP prompts or resources registered |

---

## C1 — `speed` parameter is (effectively) a no-op

**Repro**
Three calls with identical text, varying `speed`, voice `neutral_male`:

    say(text="El rápido zorro marrón salta sobre el perro perezoso.",
        voice="neutral_male", speed=0.5, output="/tmp/spd_050.wav")
    say(..., speed=1.0, output="/tmp/spd_100.wav")
    say(..., speed=2.0, output="/tmp/spd_200.wav")

**Observed**

| speed | reported duration | file size |
|-------|-------------------|-----------|
| 0.5   | 3.92 s            | 188 204 B |
| 1.0   | 3.28 s            | 157 484 B |
| 2.0   | 3.68 s            | 176 684 B |

**Expected**
If `speed` were a true time-scale factor, roughly 6.56 / 3.28 / 1.64 s.
Observed ratio across the full 4× range is well under 1.2× — within
normal run-to-run sampling variance.

**Cross-check on a clone voice** (`carlos_mx`): 0.5 → 3.60 s,
2.0 → 4.24 s. Same pattern.

**Files are not identical** — sha256 differs and byte sizes differ, so
the argument does reach the server. It just doesn't control playback
rate.

**Impact**
Users expecting slower pronunciation for accessibility or faster
narration for skimming get neither.

---

## C2 — Docstring / validator drift on `speed`

**Docstring (as exposed via MCP tool listing)**
"Speed factor 0.8-1.3 (default: 1.0)."

**Server behavior**
  - `say(speed=0.5)` → accepted
  - `say(speed=2.0)` → accepted
  - `say(speed=0.4)` → `{"error": "speed out of range 0.5-2.0: 0.4"}`
  - `say(speed=2.5)` → `{"error": "speed out of range 0.5-2.0: 2.5"}`

**Impact**
Clients that honor the schema will over-restrict users to 0.8-1.3.
Clients that don't will send values up to 2.0 that the server happily
accepts but (per C1) doesn't apply anyway. Contract is unreliable.

---

## C3 — `list_all_voices` returns truncated-looking descriptions

**Repro** `list_all_voices()` returns entries like:

    "energetic_male": { ...,
        "description": "A 28-year-old Spanish-speaking male with an
                        energetic, upbeat delivery. Cheerful" }
    "neutral_male": { ...,
        "description": "A 35-year-old native Spanish male narrator with a
                        deep, clear voice. Speaks with" }

Both end mid-sentence ("Cheerful", "Speaks with") with no ellipsis, no
`truncated: true` flag, and no way for a client to tell whether the
description is complete.

**Impact**
LLM clients asked "pick a voice for an energetic podcast intro" have to
guess whether "Cheerful" is the end of the description or a cutoff.

---

## C4 — `demo` tool has no `speed` parameter

`demo(text, output_dir)` does not accept a speed. If C1 gets fixed,
users will immediately want "generate a slow demo of every voice" for
pronunciation comparison. Currently impossible without calling `say`
in a loop.

---

## C5 — No MCP prompts or resources exposed

  - `prompts/list` → `[]`
  - `resources/list` → `[]`

The server is tool-only. A richer MCP surface (e.g. voice-preview
resources as `audio/wav`, a "narrate this long text with voice X"
prompt) would make discovery and onboarding smoother for agent clients.
Not a defect per se — logged for completeness.

---

## What this report does NOT cover

These require source access and are the maintainer's concern, not the
MCP client's:
  - Whether the speed bug is in the server, in `mlx-audio`, or in
    Qwen3-TTS itself (a client cannot tell).
  - README accuracy, dead code, internal API shape.
  - Fix design and test plan.

Those live in MAINTAINER_BACKLOG.md.
