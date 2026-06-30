# Security Policy

## Reporting a vulnerability

If you discover a security issue, please **do not open a public issue**. Instead,
use GitHub's private **"Report a vulnerability"** flow on the repository's
*Security* tab, or open a minimal private report. We'll acknowledge and respond
as soon as we can.

## Scope notes

`trainall` can execute model-generated content as part of verification and tool
use. Run these in a trusted/sandboxed environment:

- **`CodeVerifier` / `PythonTool`** run candidate code in a subprocess with a
  timeout. They are *not* a hardened sandbox — only run them on trusted inputs or
  inside an isolated container.
- **`SQLVerifier`** executes SQL against an in-memory SQLite database.

Treat any untrusted model output as untrusted code.

## Supported versions

The project is pre-1.0; security fixes target the latest `main`.
