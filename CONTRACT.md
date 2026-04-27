# MCP Contract

This document defines the stable JSON shapes returned by the `spanish-tts` MCP server.
**Shape changes are breaking — they bump the SemVer major version.**

## Tools

### `say`

Synthesise speech from text.

**Input schema**

```json
{
  "text":   {"type": "string",  "required": true},
  "voice":  {"type": "string",  "required": false, "default": null},
  "speed":  {"type": "number",  "required": false, "default": null},
  "stream": {"type": "boolean", "required": false, "default": false},
  "output": {"type": "string",  "required": false, "default": null}
}
```

**Success response**

```json
{"path": "<absolute-path-to-wav>", "duration_seconds": <float>}
```

> `duration_seconds` is computed in the engine layer (U3-19) from the raw
> audio array (`len(audio) / sample_rate`). It is **always a finite float**,
> never `null` / `None`. Callers may rely on this field being present and
> numeric on every successful `say` response.

**Error response** (U3-17: `code` field added)

```json
{"error": "<human-readable message>", "code": "<stable-enum>"}
```

The `error` string is human-readable and may change across versions.
The `code` field is **stable** — callers should branch on `code`, not `error`.

### Error sentinel — `code` values

| `code` | Returned when |
|---|---|
| `text_empty` | `text` is `None`, empty, or whitespace-only |
| `text_nul` | `text` contains a NUL byte (`\x00`) |
| `text_too_long` | `text` exceeds 10 000 characters |
| `voice_not_found` | `voice` is not registered |
| `speed_out_of_range` | `speed` outside 0.5–2.0 |
| `speed_not_finite` | `speed` is NaN or ±infinity |
| `path_invalid` | `output` is an empty string, contains NUL, or cannot be resolved |
| `path_escape` | `output` resolves outside `output_dir` (path-traversal attempt) |
| `path_is_dir` | `output` resolves to `output_dir` itself (a directory) |
| `generation_failed` | engine raised an unexpected exception |
| `voices_empty` | `demo` called with zero registered voices |

---

### `list_all_voices`

Return all registered voices.

**Input schema** — no required parameters.

**Success response**

```json
{
  "voices": {
    "<voice-name>": {
      "type":        "clone" | "design",
      "gender":      "<string>",
      "language":    "<string>",
      "accent":      "<string>",
      "description": "<string>"
    }
  }
}
```

`accent` and `description` are optional; callers must handle their absence.

**Error response**

```json
{"error": "<human-readable message>", "code": "<stable-enum>"}
```

---

### `demo`

Run `say` for every voice and return per-voice results.

**Input schema**

```json
{
  "text":       {"type": "string",  "required": false, "default": null},
  "speed":      {"type": "number",  "required": false, "default": null},
  "output_dir": {"type": "string",  "required": false, "default": null}
}
```

**Success response**

```json
{
  "results": [
    {"voice": "<name>", "path": "<absolute-path-to-wav>", "status": "ok"},
    {"voice": "<name>", "error": "<reason>", "status": "failed"}
  ]
}
```

Partial failure is normal: if one voice fails the others still run.
`status` is always present; `path` is present on `"ok"` entries,
`error` is present on `"failed"` entries.

**Error response** (whole-tool failure, not per-voice)

```json
{"error": "<human-readable message>", "code": "<stable-enum>"}
```

---

## Backward-compatibility policy

1. Adding a new **optional** field to a success response is **non-breaking**.
2. Adding a new tool is **non-breaking**.
3. Removing a field, renaming a field, or changing a field type is **breaking** → major version bump.
4. New `error` string values are **non-breaking**; callers must not switch-on them exhaustively.
5. Input schema changes that drop a previously-required field are **non-breaking**.
   Input schema changes that add a new required field are **breaking**.

---

## Version

Contract version: **0.2.x** (aligned with package version).
Last updated: 2026-04-26.
