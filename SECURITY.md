# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a Vulnerability

KinetiDiff is a research codebase. It runs AutoDock Vina and OpenBabel as subprocesses using paths and arguments derived from user-supplied configs and PDB files.

**Do not open a public GitHub issue for security vulnerabilities.**

To report a vulnerability, email **aaryanp0302@gmail.com** with:

1. A description of the vulnerability and potential impact.
2. Steps to reproduce (command line, config file, or minimal script).
3. Any suggested fix, if you have one.

You can expect an acknowledgment within 7 days and a resolution timeline within 30 days for confirmed issues.

## Threat Model

- **Subprocess injection**: `obabel` and `vina` are called with arguments derived from config files and intermediate file paths. Untrusted PDB files should not be passed to `VinaGradientGuidance` without validation.
- **Temporary files**: Intermediate PDB/PDBQT files are written to `tempfile.mkdtemp()` and cleaned up after each scoring call. Verify your system's `TMPDIR` is on a non-world-writable mount in shared-compute environments.
- **Network access**: The codebase makes no outbound network calls at runtime. The A3F SMILES lookup in the MD pipeline uses RCSB's public API and degrades gracefully if offline.
