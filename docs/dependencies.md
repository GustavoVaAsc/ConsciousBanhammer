# Dependency register

Every runtime and development dependency, with the reason it is here. See `AGENTS.md` H5 for the checks a dependency must pass before it is added. Remove the row when the dependency is removed.

- **Group:** `main`, `gpu` (extra), `dev` (dependency group), or `tool` (runs outside `uv.lock`: pre-commit hook repos, `uvx` tools; pinned version in the Purpose column).
- **Wheels:** whether wheels exist for linux aarch64 and linux x86_64, e.g. `yes / yes`.
- **Decision:** the architecture decision that chose it (D-id), or `H5` if it was added under the H5 checks alone.

| Package | Group | Purpose | Why not the standard library or an existing dependency | Licence | Wheels (aarch64 / x86_64) | Last release, maintenance | Decision |
|---|---|---|---|---|---|---|---|
