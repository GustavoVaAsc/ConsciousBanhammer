# AGENTS.md

Rules for every agent and every human writing code in this repository. They are binding: a change that breaks one is not finished. Read-only reviewers report violations; they do not edit.

## Project

Rule-conditioned graduated content moderation in Spanish and English. Given a community ruleset and a message (plus its parent), a classifier returns a verdict V0–V3, an ordinal severity 0–3 and the cited rule. A deterministic policy engine turns that into a sanction.

Sources of truth, in order:

1. `docs/reviews/architecture/doc.md`: architecture decisions D1–D32 and the deviations from the proposal.
2. `docs/117002029_NLP_PropuestaProyecto.pdf`: the project proposal.
3. This file: how code is written.

If code and a decision disagree, stop and raise it. Do not quietly pick one.

## Commands

Python 3.12, managed with uv. These are the target commands; they work once `pyproject.toml` exists.

| Purpose | Command |
|---|---|
| Install (Pi) | `uv sync --locked` |
| Install (GPU box, inside the `work` container) | `uv sync --locked --extra gpu` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run mypy src` (authoritative run: the `work` container, see H8) |
| Import contracts | `uv run lint-imports` |
| Unit tests, any machine | `uv run pytest -m "not gpu and not integration"` |
| Unit + GPU tests, GPU box | `uv run pytest -m "not integration"` |
| Integration tests (real backends, may spend API budget) | `uv run pytest -m integration`, only when asked |
| Pipeline | `uv run dvc repro` / `uv run dvc exp run` (`dvc exp run` on the host only: `.git` is read-only in `work`) |
| Judge and its proxy on the GPU box (start / stop, D27) | `docker compose --profile judge up -d --wait --wait-timeout 1500` / `docker compose --profile judge down` |
| Build the `work` image | `docker compose --profile train build work` |
| Run a command in `work` | `docker compose --profile train run --rm work <cmd>` |

pytest runs with `--strict-markers`, so an unknown marker fails.

## Definition of done

A change is finished only when all of these hold:

1. Lint, format check, type check, import contracts and the unit tests pass.
2. New or changed behaviour has tests.
3. **`CHANGELOG.md` has an entry** under `## [Unreleased]` in one of the Keep a Changelog sections: Added, Changed, Deprecated, Removed, Fixed, Security. Commits typed `refactor:`, `test:`, `style:` or `chore:`, which by definition do not change behaviour, may skip the entry; every other commit needs one.
4. **This file is updated** when the change adds or alters a command, a top-level directory or subpackage, a dependency, a rule here, or an architecture decision.
5. The self-review checklist at the end of this file has been gone through.

`pre-commit` (dev group) is set up with the first code commit. It runs:

- ruff and ruff-format;
- mypy and import-linter as local hooks (`entry: uv run mypy src` and `entry: uv run lint-imports`, `language: system`, `pass_filenames: false`), so they see the project environment. They run in pre-commit rather than in a test because both are whole-project checks that need no fixtures, and a hook fails the commit before the code is shared;
- `detect-secrets` and `nbstripout`;
- a commit-msg hook that rejects a commit touching `src/` without a `CHANGELOG.md` change, unless its type is one of the exempt types in item 3. It checks the staged files only, so an `--amend` that adds `src/` files must stage the `CHANGELOG.md` entry again. Install it with `pre-commit install --hook-type pre-commit --hook-type commit-msg`.

Every hook repo is pinned to a commit SHA with a `# frozen: <tag>` comment (`pre-commit autoupdate --freeze`). The ruff hook's SHA is that of the tag equal to the ruff version in `uv.lock`. detect-secrets flags the SHAs, so `.secrets.baseline` records them as audited false positives (`is_secret: false`); after an autoupdate, rescan with `detect-secrets scan --baseline .secrets.baseline` and mark the new SHAs the same way.

`git commit --no-verify` is not allowed.

## Code hygiene

### H1. No comments except cited whys and pragmas

The code must explain itself through names, types and small functions. The only comments allowed are:

- **A cited why**: one line in the form `# <source>: <reason>`, where `<source>` is a proposal section (`Proposal 6.1`), a decision id (`D7`), `arXiv:<id>`, or a URL to a library issue. Example: `# D7: V0 admits severity 0 only`. A bare `# D7` is not enough. The hygiene test matches `^# (Proposal \d+(\.\d+)*|D\d+|arXiv:\S+|https?://\S+): \S.*$`.
- **Tool pragmas**: `# noqa: <code>` and `# type: ignore[<code>]`, always with a code and nothing after it (enforced by ruff `PGH003`, `PGH004`; ruff `RUF100` flags unused ones), `# pragma: no cover`, and `# pragma: allowlist secret` for `detect-secrets` false positives.
- A shebang line in an executable script.

When a pragma needs a reason, the cited why goes on the line directly above it, never on the same line:

```python
def build_lora_config(settings: QLoRASettings) -> LoraConfig:
    """Build the LoRA adapter configuration for the Llama Guard baseline."""
    # D3: gpu extra absent on the Pi
    from peft import LoraConfig  # noqa: PLC0415

    return LoraConfig(r=settings.rank, lora_alpha=settings.alpha)
```

Here `LoraConfig` in the signature comes from an `if TYPE_CHECKING:` import at module top.

Formatter pragmas (`# fmt: off`, `# fmt: skip`) are not allowed. Commented-out code is never allowed (ruff `ERA001` catches most of it). A string literal used as a comment is not allowed either. Docstrings and attribute docstrings (a string directly after a class-level assignment) are not comments.

Rationale that has no external source goes into a name (`stable_log_softmax`, `pad_to_multiple_of_eight`) or into the second paragraph of the docstring.

**Enforcement:** `tests/test_hygiene.py`, part of the first code commit, scans every `.py` file in `src/`, `tests/`, `scripts/` and the repository root (`conftest.py`) with `tokenize` and `ast`. It fails on:

- any comment outside the forms above; a cited why must stand alone, so one followed by a further ` # ` comment fails;
- a string or bytes statement that is neither a docstring nor an attribute docstring;
- `while True` (H4);
- in `src/` only, a module that defines public top-level functions and no classes (also counting those under a top-level `if` or `try`) but lacks the `Functional module:` line (H3); `__init__.py` is exempt;
- the import rules of H16 that import-linter does not cover. They apply to `src/` and to tests: a test under `tests/<pkg>/` counts as part of subpackage `<pkg>`.

### H2. Single Responsibility Principle

A class or function has one reason to change. Signs of a violation: "and" in the name, a docstring that lists several jobs, or a function that both computes and does I/O. Keep the pure logic in the core and put I/O (files, HTTP, GPU) at the edges, so the core can be tested without them.

### H3. Object-oriented by default

Code is organized into classes with clear responsibilities. Collaborators are passed to the constructor (dependency injection), not created inside, so tests can substitute them. Prefer composition to inheritance. Private helper functions inside a class module are fine.

A module may consist of plain functions only when that is clearly the better form, and its docstring then contains a line `Functional module: <reason>.` Accepted reasons:

- Stateless transformations: metric formulas, text normalization, statistical helpers.
- The library's own idiom is functional: Polars expression builders, DuckDB query modules, HF `datasets.map` callables, collate functions.
- Entry points: DVC stage scripts and CLI `main()`.

These are classes by library contract and need no justification: `nn.Module` (including losses with learned parameters, such as the cumulative-link loss, D18), `Dataset`, `TrainerCallback`, pydantic and pandera models.

A class with a single method and no state is a function in disguise; write the function and give the reason. Do not create an abstract base class or `Protocol` unless there are two implementations, or one implementation plus a test double (example: the judge backend). No speculative abstraction.

When H3 and H6 disagree, H3 wins unless the library's idiom is functional, in which case follow the idiom and add the `Functional module:` line.

### H4. No repetition, no unbounded loops

- Logic that appears a second time is extracted. The only exception is two copies that are expected to diverge; the docstring says so.
- No `while True`. Retries are bounded (`for attempt in range(settings.max_attempts)`), streams are iterators or generators, and polling loops have an explicit stop condition and limit.
- Prefer comprehensions, `itertools` and vectorized NumPy/Polars operations to manual index loops.

### H5. Dependencies

Before adding a dependency, answer:

1. Can the standard library or an existing dependency do it?
2. Is it maintained, and permissively licensed? Note its last release and the state of its issue tracker.
3. Does it ship wheels for both **linux aarch64** (the Pi) and **linux x86_64** (the GPU box)? GPU-only packages go in the `gpu` extra; development tools go in the `dev` dependency group.
4. Is the PyPI name exact, not a look-alike?

Add it with `uv add`, commit `uv.lock`, and record it in `docs/dependencies.md`. This includes packages already chosen by a decision; the D-id goes in the register. Remove the row when the dependency goes. `torch` is changed only by editing `[tool.uv.sources]` (see doc.md 3.1), never with a bare `uv add torch`, which would break the per-platform build. vLLM is never a project dependency; it runs in its own container and is called over HTTP (D3).

Before each milestone, audit the locked dependencies:

```
uv export --locked --all-extras --no-hashes --no-emit-project | uvx --from pip-audit==<pin> pip-audit -r /dev/stdin --no-deps --disable-pip
```

pip-audit evaluates environment markers for the machine it runs on, so run it on both the Pi and in the `work` container. The CUDA torch build is not on PyPI and is checked by hand against PyTorch security advisories. Verify the command on its first run.

Tools that run outside `uv.lock` (the pre-commit hook repos and `uvx` tools) are recorded in the register with group `tool` and a pinned version.

### H6. Challenge the code

Before finishing, ask whether there is a clearer way to write it. "Clearer" means readable on first read. It does not mean clever: nested comprehensions, chained walrus expressions and metaprogramming must earn their place. Prefer the standard idiom of the language and the library.

### H7. Constants and parameters

The test: a value is a **fixed fact** if changing it would change the meaning of the data or the schema. It is a **tunable** if changing it would change a result.

- **Fixed facts** live in a `constants.py` per subpackage: the verdict scale, schema field names, enums for closed sets, regular expressions, file names. Values shared across subpackages live in `banhammer/core/constants.py` (for example `Verdict`) and are never duplicated. `core.constants` is the one constants module every subpackage may import. A subpackage never imports another non-core subpackage's `constants.py`; values it needs move to `core`.
- **Tunables** (thresholds, hyperparameters, sample sizes, seeds, retry counts, timeouts, model ids and their revisions, data directories) live in `params.yaml`. They are loaded into plain pydantic `BaseModel` settings **with no defaults**. Not `BaseSettings`: it also reads environment variables, which would make `params.yaml` no longer the only source. Environment variables carry secrets only (H14).
- **Paths:** directories are tunables in `params.yaml`; file names inside them are fixed facts in `constants.py`. The `outs` in `dvc.yaml` build their directories with `${...}` interpolation from `params.yaml`, so each directory has one source.
- Every DVC stage lists exactly the `params.yaml` keys it reads under `params:` in `dvc.yaml`. Otherwise DVC does not rerun the stage when a key changes. A test checks that the declared and the read keys match.
- No magic numbers in logic. Ruff `PLR2004` catches them in comparisons; review catches the rest (arithmetic, arguments, tensor shapes).

### H8. Types and data

- Type hints on every function, method and class attribute. mypy runs with `strict = true`, `enable_error_code = ["ignore-without-code"]`, and the pydantic plugin (the pandera plugin too if pandera's own typing needs it). Ruff `ANN401` bans explicit `Any` except where a third-party signature forces it, which needs a `# noqa: ANN401` with a cited reason.
- Third-party libraries without stubs get a per-module `ignore_missing_imports` override in `pyproject.toml`, never a global loosening. The packages in the `gpu` extra (`peft`, `trl`, `bitsandbytes`) are the only ones overridden for being absent on the Pi. A per-module `implicit_reexport = true` override is allowed only where a library exposes a public name through a re-export that mypy strict does not see; today that is `huggingface_hub.utils` (`filter_repo_objects`).
- The authoritative type check runs in the `work` container, which has the full environment. On the Pi it must also pass.
- Records that cross a module boundary or come from outside (files, HTTP, judges) are **pydantic models** with `frozen=True, extra="forbid"`. Internal value objects that never leave a module may be frozen dataclasses. Closed sets are `Enum`s. No loose dicts crossing module boundaries.
- The five D28 boundary artifacts (rules, canonical clusters, messages, pairs with folds, labels) are Parquet files with a pandera schema in the subpackage that owns them. Other stage outputs (embeddings, call cache, predictions, metrics, checkpoints) use the format that fits them. If they are records, they are typed with pydantic.
- No mutable default arguments and no module-level mutable state. The module-level `logger` (H12) is the one exception.

### H9. Docstrings

Every module, package `__init__.py`, public class, public method and public function has a docstring: one line by default, a short paragraph at most, Google style. It says what the object is for.

- The `Args:` section is optional and lists only the parameters that are not obvious from name and type. Ruff `D417` is ignored for this reason.
- `__init__` is documented in the class docstring (`D107` ignored).
- Magic methods need no docstring (`D105` ignored).
- Tests need no docstrings.

Ruff `D` enforces the rest.

### H10. Size limits

Cyclomatic complexity at most 10 (`max-complexity = 10`) and at most 5 parameters per function (`max-args = 5`; group the rest into a settings object). Both are set explicitly in `pyproject.toml`. Functions should be short enough to read without scrolling. A long function means a missing abstraction.

### H11. Errors

Raise specific exceptions. Each subpackage defines its own in `errors.py`, deriving from `banhammer.core.errors.BanhammerError`. No bare `except`, no `except Exception: pass`, no swallowed errors. Validate at the boundaries and fail fast (D28).

### H12. Logging, paths and randomness

- Use `logging` through one module-level `logger = logging.getLogger(__name__)`, never `print` (ruff `T20`).
- Logs and notebooks never contain raw message text or user data; log ids and counts.
- Use `pathlib` for paths (ruff `PTH`). Never hard-code absolute paths; paths are built as described in H7.
- Randomness is explicit. Seeds come from `params.yaml`, and one seeding helper in `core` seeds Python, NumPy, torch, CUDA and `DataLoader` workers at the start of each stage. Code that draws random numbers receives a `numpy.random.Generator` or `torch.Generator`. Results on the Pi and the GPU box are not expected to match bit for bit.

### H13. Tests

- `tests/` mirrors `src/banhammer/`. Every public class has unit tests. Markers `gpu` and `integration` are registered in one root `conftest.py`.
- **Hypothesis property tests** cover policy engine monotonicity (D8) and the verdict–severity grid (D7).
- **The split invariant suite** (D28) is a set of pytest tests on the real artifacts, run as a DVC stage that blocks the pipeline. It is mandatory in `dvc.yaml`.
- Unit tests make no network calls and need no GPU. `gpu` tests need CUDA; `integration` tests call real backends and run only on request.

### H14. Security and data

- Secrets (API keys, `VLLM_API_KEY`) come from environment variables or an untracked `.env`. They are never committed or logged. A `detect-secrets` pre-commit hook checks commits.
- Datasets, model weights and caches are never committed to git; DVC tracks them.
- Services bind to `127.0.0.1` and are reached through SSH tunnels (D3).
- Respect the terms and anonymization conditions of every dataset (proposal 4.6), and record provenance in the manifests.

### H15. Notebooks

Notebooks live in `notebooks/` and are for exploration only. They are excluded from ruff and never imported. Their outputs are stripped before commit (`nbstripout`), so no raw message text reaches git. Logic the pipeline needs moves into `src/` with tests.

### H16. Pipeline and ML

- **DVC.** A stage is a pure function of its declared `deps`, `params` and `outs`; it reads no path it has not declared. `dvc.lock` is committed. Metrics go through DVCLive (D5).
- **Models and downloads.** Every Hugging Face model and dataset id in `params.yaml` has a pinned `revision`. `trust_remote_code=True` is not allowed. Weights load from safetensors, or with `torch.load(..., weights_only=True)`.
- **PyTorch.** Device and dtype are passed in, never hard-coded (`.cuda()` is not allowed), so code runs on the Pi too. Inference and metric computation run under `torch.inference_mode()`. Steps that fit parameters on predictions, such as temperature scaling (D9), do not. `forward` has no side effects.
- **Polars and DuckDB.** Prefer lazy frames and expressions. SQL is parameterized; never build it with f-strings from data. No `.to_pandas()` on the full message pool.
- **Language model calls.** Every judge call goes through the call cache (D24), with a timeout, bounded retries and the budget stop. Every HTTP client sets an explicit timeout.
- **Imports.**
  - `core` imports from no other subpackage. This is enforced by an import-linter `layers` contract (dev group).
  - Subpackages import each other only through their public `__init__` exports, never another non-core subpackage's `constants.py` or private modules. `core` modules, `core.constants` included, are importable everywhere. This is enforced by the hygiene test.
  - No relative imports beyond the current package (ruff `TID252`).
  - Imports of `gpu`-extra packages may sit inside functions with `# noqa: PLC0415` and the cited why on the line above (H1).
- **Library hooks.** Code whose signature a library forces, with unused arguments, uses the matching noqa with the cited why on the line above, naming the library:
  - `ARG002` for methods such as `TrainerCallback` hooks;
  - `ARG001` for free-function callbacks;
  - `ANN401` for a forced `**kwargs: Any`.

  Lambdas with unused arguments are replaced by named functions.

## Layout

```
src/banhammer/
  core/          constants, errors, settings, records, seeding, union-find, embedder, model snapshot, neighbour search
  rules/         S1 rule corpus: harvest, atomize, canonicalize
  messages/      S2 message pool: ingest, language ID, dedup, sampling
  pairing/       S3 pairs, folds, split invariants
  labeling/      S4 judge cascade, call cache, calibration gate
  annotation/    Label Studio export/import, Gradio review apps (D23)
  models/        S5 cross-encoder, flat baseline, Llama Guard QLoRA, keyword matcher
  policy/        S6 policy engine
  evaluation/    S7 metrics, bootstrap, slices, fairness
scripts/         pre-commit helper scripts (commit-msg changelog gate); tests/scripts/ covers them
tests/           mirrors src/banhammer/
notebooks/       exploration only
docs/            proposal, reviews, dependencies.md, docker.md, workstreams.md
docker/          judge reverse-proxy configuration (nginx)
Dockerfile       project image for the GPU box (D3; installs with uv sync --locked)
compose.yaml     judge and work services, build context is the repository root
params.yaml      tunable values (DVC)
dvc.yaml         pipeline stages
```

Each subpackage has `constants.py` and `errors.py` as needed. Data lives in DVC-tracked directories, never in git.

## Tooling configuration

Ruff, in `pyproject.toml`:

- **Selected rule sets:** `E`, `W`, `F`, `I`, `N`, `UP`, `B`, `A`, `C4`, `C90`, `SIM`, `RET`, `PTH`, `PL`, `ERA`, `T20`, `BLE`, `S`, `TRY`, `PERF`, `ARG`, `ANN`, `TID`, `PGH`, `D` (Google convention), `RUF`. Line length 100.
- **Ignored globally:**
  - `TRY003`: conflicts with the exception messages H11 asks for.
  - `D206`, `D300`: conflict with the ruff formatter. Any other formatter-conflict warning ruff prints on its first run is added here.
  - `D105`, `D107`, `D417`: see H9.
- **`tests/**`:** also ignore `S101`, `PLR2004`, `D`, `ARG`, `ANN`, and `RUF001`–`RUF003`, because moderation fixtures contain homoglyphs and typographic quotes on purpose.
- `S311` stays on: stdlib `random` is not used (H12).
- **`pep8-naming`:** `ignore-names` covers ML idiom: `X`, `X_*`, `F` (as in `import torch.nn.functional as F`).
- **`notebooks/`:** excluded.

mypy: `strict = true`, `enable_error_code = ["ignore-without-code"]`, the pydantic plugin, and the per-module overrides described in H8.

## Git

Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `style:`, `chore:`, `build:`). Dependency changes use `build(deps):`, which needs a CHANGELOG entry; `chore:` is for housekeeping that changes nothing users or the pipeline see. One concern per commit. Never commit to `main` directly once code exists; use a branch.

## Self-review checklist

Before declaring a change done:

- [ ] Each new class or function has one responsibility (H2), and any plain-function module states why (H3).
- [ ] Every comment is a cited why or a pragma with a code (H1).
- [ ] No duplicated logic, no `while True` (H4).
- [ ] Fixed facts are in `constants.py`, tunables in `params.yaml` and declared in `dvc.yaml` (H7).
- [ ] Any new dependency passed H5 and is in `docs/dependencies.md`.
- [ ] Types are complete; mypy strict passes (H8).
- [ ] I looked for a clearer way to write it (H6).
- [ ] Tests cover the change (H13); no raw message text in logs (H12).
- [ ] `CHANGELOG.md` has an entry; this file is updated if affected.
