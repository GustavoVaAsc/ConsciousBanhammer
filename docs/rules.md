# Rule corpus (S1)

How the rule corpus is written, validated and canonicalized. The decisions behind it are D14 and D15 in `docs/reviews/architecture/doc.md` (4.3) and D28; the code is `src/banhammer/rules/`.

## Ruleset files

A curator writes one YAML file per ruleset, `<ruleset_id>.yaml`, in `rules.rulesets_dir` (`params.yaml`). The file holds the provenance of the harvest and one record per **atomic** rule item: a sentence with several prohibitions is written as several records. No model splits rules.

```yaml
ruleset_id: askspain
platform: reddit
language: es
source_url: https://www.reddit.com/r/askspain/about/rules
harvest_date: 2026-11-02
terms_note: Public rule page; rule text quoted for research, no user data.
rules:
  - rule_id: askspain-01
    ruleset_id: askspain
    text: No se permiten insultos ni descalificaciones personales.
    scope: Comentarios y publicaciones
    exceptions: Citar un insulto para denunciarlo
    sanction_hint: null
    rule_type: conduct
    discretion_flag: false
```

| Field | Meaning |
|---|---|
| `ruleset_id` (file) | Lowercase slug (`a-z`, `0-9`, `_`, `-`); equals the file name without `.yaml`. |
| `platform` | Lowercase slug (`reddit`, `discord`, `wikipedia`, ...). Stratifies the held-out split (D11). |
| `language` | `en` or `es`; one language per ruleset. Stratifies the split and picks the labels of the embedded text. |
| `source_url`, `harvest_date`, `terms_note` | Provenance and terms (H14, proposal 4.6). `harvest_date` also fixes the order of the saturation curve. |
| `rule_id` | Unique across the whole corpus; lowercase slug that may contain `.`. |
| `ruleset_id` (rule) | Must equal the file's `ruleset_id`; it catches a record copied from another file. |
| `text` | What the rule says; not blank. |
| `scope`, `exceptions` | Where the rule applies and when it does not; leave out or `null` when the rule states none, never blank. |
| `sanction_hint` | A sanction the rule states itself (for example "ban"), free text for now; the policy engine (S6) decides how to read it. |
| `rule_type` | `harm`, `conduct`, `integrity` or `meta`. |
| `discretion_flag` | `true` when the rule leaves room for moderator discretion; required, so it is never forgotten. |

The loader (`RuleCorpusLoader`) reports every problem with the file name: unknown or missing keys, an unknown `rule_type`, empty text, a rule naming another ruleset, a rule id used twice (inside a file or across files), a ruleset id used twice, a file name that differs from its `ruleset_id`, a YAML key written twice, and malformed YAML. YAML is loaded with a safe loader.

YAML anchors and merge keys (`<<: *base`) are supported and resolve silently: a key given next to a merge overrides the inherited one without a duplicate-key error, and the merged result is validated like any other record. Use them sparingly in curated files; a reviewer reading the file does not see the merged value.

The text that is embedded and entailed for an item is its full text: the rule, then a `Scope:` line and an `Exceptions:` line when present (`Ámbito:` and `Excepciones:` in a Spanish ruleset).

## Stages

All paths and thresholds are in the `rules:` section of `params.yaml`. `dvc.yaml` is not wired yet; run the stages by hand from the repository root, in this order:

```
python -m banhammer.rules.stage_atomize
python -m banhammer.rules.stage_candidates
python -m banhammer.rules.stage_canonicalize
```

1. `stage_atomize`: validates the files and writes `rules.parquet` (the rules boundary: the eight fields) and `rulesets.parquet` (platform, language, provenance, rule count). Nothing is written if any file is invalid.
2. `stage_candidates`: embeds the items with LaBSE, takes, for each ruleset, the `neighbours_k` nearest items among the rules of all other rulesets with cosine at least `neighbours_min_cosine` (so same-ruleset items never use up the `k` places), scores entailment in both directions with the NLI model, and writes `candidate_pairs.parquet` (each pair once, `rule_a < rule_b`, with both probabilities and the outcome) and `entailment_review_queue.jsonl`. Both entailment probabilities at least `entail_high`: edge. Both at most `entail_low`: no edge. Anything else: review. Pairs inside one ruleset are not candidates. The outcome is decided on the float32 probabilities; the stored scores are rounded to 6 decimals, so a stored 0.8 can belong to a review pair that was just below the threshold.
3. `stage_canonicalize`: reads the candidate pairs and the decisions, merges edges with a union-find, checks each cluster's diameter, and writes `canonical_clusters.parquet` (one row per rule: `cluster_id`, `rule_id`, `ruleset_id`, `cluster_size`, `diameter`, `resolution`), `saturation_curve.parquet` and `split_review_queue.jsonl`. It removes the clusters and curve of an earlier run first, so they exist only after a run that succeeded, and it fails with `ReviewPendingError` while any review pair or flagged cluster has no decision. Review pairs come first: while one lacks a decision the split queue is left as it was (it cannot be computed yet); once every review pair is answered, the split queue is rewritten, empty when no cluster is flagged.

Every Parquet and JSONL file is written to a temporary file in its directory and moved into place, so an interrupted run never leaves a half-written artifact. The curve is written before the clusters boundary, and the rulesets manifest before the rules.

For the later DVC wiring: `stage_canonicalize` must declare `entailment_decisions.jsonl` and `split_decisions.jsonl` (in `rules.review_dir`) as `deps`, besides the Parquet inputs, the `rules`, `embedder` and `core` params it reads, and its outputs; otherwise DVC does not rerun it when a reviewer answers the queue. `stage_candidates` declares `entailment_review_queue.jsonl` as an out and reads no decisions.

A cluster id is `c-` plus the smallest `rule_id` of the cluster. `resolution` is `auto` (diameter within `diameter_max`), `kept` (over the limit, a human kept it whole) or `split` (a human split it).

## Review files

JSON Lines, UTF-8, in `rules.review_dir`. The queues are written by the stages; the decisions are written by the reviewer (or later by the Gradio app of `annotation/`, D23). A missing decisions file means no decisions yet. A decision for a pair or cluster that is not in the queue is an error, as are two decisions for one pair or cluster.

`entailment_review_queue.jsonl`, one line per unsettled pair (all of them, answered or not):

```json
{"rule_a":"askspain-01","rule_b":"mexico-03","text_a":"...","text_b":"...","cosine":0.81,"p_entail_ab":0.69,"p_entail_ba":0.99}
```

`entailment_decisions.jsonl`, `rule_a` before `rule_b` as in the queue; `accept: true` makes the pair an edge:

```json
{"rule_a":"askspain-01","rule_b":"mexico-03","accept":true}
```

`split_review_queue.jsonl`, one line per cluster over the diameter limit:

```json
{"cluster_id":"c-askspain-01","diameter":0.74,"members":[{"rule_id":"askspain-01","text":"..."},{"rule_id":"mexico-03","text":"..."}]}
```

`split_decisions.jsonl`: a partition of the members into groups. One group keeps the cluster whole; several groups split it into one cluster each. Every member appears in exactly one group.

```json
{"cluster_id":"c-askspain-01","groups":[["askspain-01","mexico-03"],["reddit-07"]]}
```

Review the entailment queue first; a change of entailment decisions changes the clusters, so split decisions made earlier may no longer match and are rejected.

## Saturation curve

`saturation_curve.parquet` has one row per `saturation_step` rulesets (and one for the last), adding rulesets in harvest order (`harvest_date`, then `ruleset_id`): `rulesets_added`, `rule_count`, `cluster_count` (canonical clusters reached so far) and `new_clusters` (growth since the previous row). Harvesting stops when the new clusters per additional 10 rulesets fall below 5% of the clusters so far (proposal 4.2). The curve uses the final clusters, so split decisions are reflected.

## The NLI model

`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, pinned to a commit in `params.yaml`: MIT licence, 278M parameters, one `model.safetensors` of 548 MB, and a `tokenizer.json`, so the sentencepiece file (`spm.model`) is not needed and the download allow-list (safetensors, JSON, text) is unchanged. It is loaded only through `download_snapshot` and offline, without remote code.

Measured on the pinned model with LaBSE (see the integration tests): translated rule pairs score 0.98 to 0.99 both ways; unrelated rules score below 0.03; but the model is unreliable on imperative rule text. "Spam is forbidden." against "No insultes a otros miembros." scored 0.65 and 0.88, and the general ban "No insults." did not entail "No insults toward streamers." (0.002) while the reverse scored 0.98. The mutual test and the review queue exist for this. LaBSE is a bitext model: translations score 0.92 to 0.95, but same-language paraphrases ("Spam is forbidden." against "No spam or self-promotion.") score 0.44 to 0.50, so a floor of 0.5 would drop them. `neighbours_min_cosine`, `entail_high`, `entail_low` and `diameter_max` are **provisional**: they must be calibrated on a labelled sample of candidate pairs, the way D13 calibrates `t2`, before the first full run.
