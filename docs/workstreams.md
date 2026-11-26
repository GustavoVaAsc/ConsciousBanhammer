# Workstreams

The work after the architecture review (`docs/reviews/architecture/doc.md`, section 9), split along S1–S7. The first milestone is the week-2 minimal pipeline (proposal 8.2): 15 English rulesets, 12 for training and 3 held out.

## Steps

| # | Step | Depends on | Parallel | Owner |
|---|---|---|---|---|
| 0 | **Bootstrap:** `pyproject.toml` (uv, per-platform torch sources, ruff and mypy configuration, `gpu` extra), `uv.lock`, pre-commit, `tests/test_hygiene.py`, root `conftest.py`, import-linter, `core/` (constants, errors, settings, seeding), `params.yaml`, `dvc.yaml`, DVC init | — | Blocks all code | code |
| 1 | **Docker on the GPU box (D3, D27):** pinned CUDA 12.8 and vLLM image tags with Blackwell (sm_120) support, `Dockerfile`, `compose.yaml`, smoke test of the `judge` and `work` profiles | `work` image: `uv.lock` from #0. `judge`: nothing | `judge` alongside #0 | code |
| 2 | Ethics and terms check for every external corpus (proposal 4.6) | — | Any time | human |
| 3 | Harvest the first 15 English rulesets, 3 held out; keep harvesting toward the full target | — | Any time | human |
| 4 | **S1 rules:** atomize, LaBSE, XNLI entailment, DSU, Gradio queue (D14, D15) | #0 | With 5, 8–11 | code |
| 5 | **S2 messages:** ingest, language ID, MinHash + LaBSE dedup, sampling (D13, D19) | #0 | With 4, 8–11 | code |
| 6 | Verdict Standard v0.1 and the 200 gate pilot items in Label Studio (D23) | Data from #4, #5; #3 | Writing the standard can start now | human |
| 7 | **S3 pairing:** pairs, grouped folds, split invariant suite as a blocking stage (D11, D28) | Schemas of #4, #5 | After 4 and 5 | code |
| 8 | **S4 labeling:** call cache and $20 budget stop, vLLM client, prompt builder, Stage-2 clients, calibration gate (D6, D17, D24, D29) | Code: #0. Run: #1, #6, #7 | Code yes; run later | code |
| 9 | **S5 models:** cross-encoder, cumulative-link loss, joint decode, flat baseline, keyword matcher (D7, D10, D18, D25) | Code: #0. Training: labels from #12 | Code yes; training later | code |
| 10 | **S6 policy engine** with Hypothesis property tests (D8, D31, D32) | #0 | Fully independent | code |
| 11 | **S7 evaluation:** metrics, cluster bootstrap, slices (D9, D22) | #0 | Yes | code |
| 12 | Judge run over the minimal-pipeline pairs | #1, #6, #7, #8 | No: GPU holds the judge alone (D27) | GPU box |
| 13 | Train and evaluate the baselines and the classifier; power simulation (4.4) | #9, #11, #12 | No: after labeling (D27) | GPU box |

## Parallelism

- **Now:** #0, the `judge` half of #1, and the human tracks #2, #3 and the standard in #6.
- **After #0:** #4, #5, #8, #9, #10 and #11 are independent subpackages. Each tests against doubles of the backends.
- **GPU:** labeling (#12) and training (#13) take turns (D27). Code is parallel; GPU runs are serial.
- **Merge hotspots:** `params.yaml`, `dvc.yaml`, `CHANGELOG.md`, `docs/dependencies.md`, `pyproject.toml` and `uv.lock` are shared by every branch. Add dependencies one branch at a time, and rebase before merging.

## After the milestone

Week 3 scales to the full corpus: Spanish pool and rulesets, the human pilot with the 300-message held-out sub-pilot, retraining, and the Llama Guard QLoRA. Week 4 runs the headline metrics, ablations, fairness slices and error analysis. The descope ladder is in proposal 8.3.
