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

**Error response**

```json
{"error": "<human-readable message>"}
```

Known error messages (stable strings):

| Condition | `error` value |
|-----------|--------------|
| Text is empty / whitespace-only | `"text is empty"` |
| Text contains NUL byte | `"text contains NUL byte"` |
| Text exceeds 10 000 characters | `"text too long (N chars, max 10000)"` |
| Voice not found in registry | `"voice '<name>' not found"` |
| Output path outside safe root | `"output path not allowed"` |
| Generation engine failure | `"generation failed: <reason>"` |

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
{"error": "<human-readable message>"}
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
{"error": "<human-readable message>"}
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
