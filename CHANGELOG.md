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
- `params.yaml` with the `core` section, and `dvc init` with an empty `dvc.yaml`.
- `tests/test_hygiene.py`: scans `src/`, `tests/`, `scripts/` and the root `conftest.py` for the H1 comment and string (and bytes) rules, `while True` (H4), the `Functional module:` line (H3) and cross-subpackage private imports in `src/` and in tests (H16), with its own tests on fixture snippets.
- import-linter contract that `core` imports from no other subpackage.
- `.pre-commit-config.yaml` (ruff, ruff-format, mypy, import-linter, detect-secrets with `.secrets.baseline`, nbstripout; hook repos pinned to commit SHAs) and the commit-msg hook `scripts/check_changelog_entry.py`, which rejects a commit touching `src/` without a `CHANGELOG.md` change unless it is typed refactor, test, style or chore (renames count both paths; an amend is checked against its newly staged files only).
- Docker on the GPU box (D3, D6, D27): `Dockerfile`, `.dockerignore`, `compose.yaml`, `docker/judge-proxy.conf` and `docs/docker.md`. The `judge` profile runs vLLM v0.34.0 with Qwen2.5-7B-Instruct-AWQ at a pinned revision, offline, on an internal network with `gateway_mode_ipv4: isolated` (Docker 28 or later), so the host cannot reach vLLM directly. An nginx proxy, started once the judge is healthy, publishes only `127.0.0.1:8000`, forwards only `/v1/*` and `/health` using the normalized URI, rejects ambiguous raw URIs (dot segments, percent escapes, double slashes) and unexpected `Host` headers. The `train` profile runs the `work` image (CUDA 12.8.1, uv 0.12.0, locked environment in `/opt/venv`) as the host user, with `.git` read-only and `.env` hidden. All services drop capabilities and set `no-new-privileges`.
- `params.yaml` `judge:` section. `tests/compose_files.py`, `tests/test_compose_params.py`, `tests/test_compose_hardening.py` and `tests/test_judge_proxy_conf.py` check `compose.yaml` against `params.yaml`, the hardening rules, and `docker/judge-proxy.conf` (Host allow-list, URI parts, raw-URI rejection).
- `docs/research/`: judge model assessment: local Stage-1 candidates for the 16 GB GPU, Stage-2 API judges under the $20 budget, and the evidence on LLM judges for rule-conditioned moderation.

### Changed

- `docs/workstreams.md`: steps #0 (bootstrap) and #1 (Docker on the GPU box) marked done; the parallelism notes now start from the code workstreams.
- `docs/reviews/architecture/doc.md`: the judge proxy recorded under "Deviations from the proposal".
