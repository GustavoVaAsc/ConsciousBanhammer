# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `AGENTS.md` with the project's code hygiene rules (H1–H15), definition of done and layout; `CLAUDE.md` imports it.
- `docs/dependencies.md` dependency register.
- `docs/reviews/architecture/doc.md`: architecture assessment, decisions D1–D32.
- `docs/workstreams.md`: work breakdown along S1–S7 with dependencies and parallelism.
- Project bootstrap: `pyproject.toml` (Python 3.12, uv, per-platform torch sources for linux x86_64 CUDA 12.8 and linux aarch64 CPU, `gpu` extra, `dev` group, ruff, mypy, pytest and import-linter configuration), `uv.lock`, `.gitignore`, `.env.example`, root `conftest.py` registering the `gpu` and `integration` markers.
- `banhammer.core`: `Verdict` scale V0–V3 with the valid verdict–severity grid (D7), `BanhammerError` and `SettingsError`, frozen settings models read from `params.yaml` by `ParamsFile` (a settings model that declares a default is rejected at class creation, and the seed is bounded to `0 <= seed < 2**63`), and the `Seeder` helper that seeds Python, NumPy, torch, CUDA and `DataLoader` workers and hands out independent named generator streams.
- `tests/test_hygiene.py`: scans `src/`, `tests/`, `scripts/` and the root `conftest.py` for the H1 comment and string (and bytes) rules, `while True` (H4), the `Functional module:` line (H3) and cross-subpackage private imports in `src/` and in tests (H16), with its own tests on fixture snippets.
- import-linter contract that `core` imports from no other subpackage.
