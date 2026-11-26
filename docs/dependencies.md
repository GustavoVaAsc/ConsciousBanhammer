# Dependency register

Every runtime and development dependency, with the reason it is here. See `AGENTS.md` H5 for the checks a dependency must pass before it is added. Remove the row when the dependency is removed.

- **Group:** `main`, `gpu` (extra), `dev` (dependency group), or `tool` (runs outside `uv.lock`: pre-commit hook repos, `uvx` tools; pinned version in the Purpose column).
- **Wheels:** whether wheels exist for linux aarch64 and linux x86_64, e.g. `yes / yes`.
- **Decision:** the architecture decision that chose it (D-id), or `H5` if it was added under the H5 checks alone.

| Package | Group | Purpose | Why not the standard library or an existing dependency | Licence | Wheels (aarch64 / x86_64) | Last release, maintenance | Decision |
|---|---|---|---|---|---|---|---|
| torch | main | Tensors, seeding of torch and CUDA, and every later model; per-platform build: CUDA 12.8 on x86_64, CPU on aarch64, pinned to 2.11 so both platforms run the same release | No standard-library or existing equivalent | BSD-3-Clause | yes (CPU index) / yes (cu128 index) | 2.11.0, 2026-04; PyTorch Foundation, active | D1, D3 |
| numpy | main | `numpy.random.Generator` and global seeding (H12), array maths | The standard library has no array type | BSD-3-Clause | yes / yes | 2.5.5, 2026-11; active | H5 |
| pydantic | main | Frozen settings and record models (H8), the pydantic mypy plugin | The standard library does not validate or parse into typed models | MIT | yes (pure Python) / yes | 2.14.4, 2026-11; active | D1 |
| pyyaml | main | Reads `params.yaml` into the settings models (H7) | The standard library has no YAML parser | MIT | yes / yes | 6.0.4, 2026-11; active | D4 |
| dvc | dev | Pipeline stages, `dvc repro`, `dvc exp`, parameter tracking; a command-line tool no `src` module imports, and both `uv sync --locked` and the `work` image install the `dev` group | No equivalent in the standard library or other dependencies | Apache-2.0 | yes (pure Python) / yes | 3.67.1, 2026-03; active | D4, D5 |
| bitsandbytes | gpu | 4-bit quantization for the Llama Guard QLoRA | Needed by QLoRA, which needs CUDA kernels | MIT | yes / yes | 0.50.2, 2026-08; active | D1 |
| peft | gpu | LoRA adapters for the Llama Guard QLoRA | Implements the adapter wrappers; writing them is out of scope | Apache-2.0 | yes (pure Python) / yes | 0.22.0, 2026-11; active | D1 |
| trl | gpu | Supervised fine-tuning trainer for the Llama Guard QLoRA | Implements the trainer loop on top of HF Trainer | Apache-2.0 | yes (pure Python) / yes | 1.19.0, 2026-11; active | D1 |
| ruff | dev | Linter and formatter (AGENTS.md, tooling configuration) | No standard-library linter or formatter | MIT | yes / yes | 0.17.6, 2026-11; active | H5 |
| mypy | dev | Strict type checking of `src` (H8) | No standard-library type checker | MIT | yes / yes | 2.4.1, 2026-10; active | H5 |
| pytest | dev | Test runner with markers (H13) | `unittest` lacks parametrization, fixtures and marker selection | MIT | yes (pure Python) / yes | 9.2.0, 2026-10; active | H5 |
| hypothesis | dev | Property tests of the verdict–severity grid and the policy engine (H13) | No equivalent | MPL-2.0 | yes / yes | 6.170.17, 2026-11; active | D7, D8 |
| pre-commit | dev | Runs the commit hooks (definition of done) | No equivalent | MIT | yes (pure Python) / yes | 4.7.0, 2026-10; active | H5 |
| import-linter | dev | Enforces that `core` imports from no other subpackage (H16) | The hygiene test cannot check layering across the whole import graph | BSD-2-Clause | yes (pure Python) / yes | 2.17, 2026-11; active | H16 |
| detect-secrets | dev | Creates and audits `.secrets.baseline` for the hook of the same name (H14) | No equivalent | Apache-2.0 | yes (pure Python) / yes | 1.5.0, 2024-05; infrequent releases, issue tracker open | H14 |
| types-pyyaml | dev | Type stubs for `pyyaml` so mypy strict passes (H8) | pyyaml ships no inline types | Apache-2.0 | yes (pure Python) / yes | 6.0.12.20261122, 2026-11; tracks typeshed | H8 |
| uv_build | tool | Build backend named in `[build-system]`; pinned `>=0.12.0,<0.13`, released with uv 0.12 | The `src` layout needs a PEP 517 backend to install `banhammer` | MIT OR Apache-2.0 | n/a (uv resolves it at build time) | Same cadence as uv; active | H5 |
| ruff-pre-commit | tool | Hook repo for ruff and ruff-format; pinned to commit `9026c9127e153371d7a78dfcb0c45ae2046a3d10` (tag `v0.17.6`, equal to the ruff in `uv.lock`) | Hook repos run in their own environments, outside `uv.lock` | MIT | yes / yes | v0.17.6, 2026-11; active | H5 |
| detect-secrets (hook) | tool | Hook repo `Yelp/detect-secrets`, pinned to commit `01886c8a910c64595c47f186ca1ffc0b77fa5458` (tag `v1.5.0`, equal to the dev package) | Same as the `detect-secrets` row | Apache-2.0 | yes (pure Python) / yes | v1.5.0, 2024-05; infrequent releases | H14 |
| nbstripout | tool | Hook repo `kynan/nbstripout`, pinned to commit `512fa4ac6e4674f3a748e7dd5a546c037ec39edf` (tag `0.9.2`); strips notebook outputs so no message text reaches git (H15) | No equivalent | MIT | yes (pure Python) / yes | 0.9.2, 2026-11; active, newest tag on 2026-11-25 | H15 |
| vllm/vllm-openai | tool | Docker image `v0.34.0` (`sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1`); local judge with structured outputs and prefix caching, kept out of the project venv | vLLM is never a project dependency (D3) | Apache-2.0 | yes / yes | v0.34.0, 2026-11-16; active | D3, D6 |
| nvidia/cuda | tool | Docker base image `12.8.1-cudnn-runtime-ubuntu24.04` (`sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc`) for `work`; CUDA minor matches the `cu128` torch wheels | No equivalent | NVIDIA Deep Learning Container License | yes / yes | 12.8.1, 2025-03; frozen minor series | D3 |
| ghcr.io/astral-sh/uv | tool | Docker image `0.12.0` (`sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105`) copied into `work`; same version that wrote `uv.lock` | Needed for `uv sync --locked` in the image | MIT OR Apache-2.0 | yes / yes | 0.12.0 pinned (latest 0.13.10); active | D3 |
| nginxinc/nginx-unprivileged | tool | Docker image `1.30.6-alpine` (`sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03`); reverse proxy in front of vLLM allowing only `/v1/*` and `/health` and checking `Host` | vLLM has no path or Host filtering of its own | BSD-2-Clause | yes / yes | 1.30.6, 2026-10-19; active | D3, H14 |
