# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| main (latest) | Yes |
| 0.2.x | Yes |
| < 0.2 | No |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:
<https://github.com/alblez/qwen3-tts-spanish-voices/security/advisories/new>

Please include:
- A description of the vulnerability and its potential impact.
- Steps to reproduce or a proof-of-concept.
- Affected versions.

**Response time:** acknowledgement within 72 hours; coordinated fix within
30 days where feasible. We will credit reporters unless they prefer anonymity.

Do **not** open a public issue for security vulnerabilities.

## Scope

This project runs as a local CLI and MCP server on the user's own machine.
The primary attack surface is the MCP `say` tool's `output=` parameter
(path-traversal risk). Report anything that allows writing or reading files
outside the user's home directory, or that allows code execution beyond what
the installed package already permits.
