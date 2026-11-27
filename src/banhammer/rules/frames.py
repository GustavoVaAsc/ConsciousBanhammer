"""Conversion between the rule records and the Polars frames of the S1 artifacts.

Functional module: stateless transformations between records and columnar frames.

Every frame is built with explicit dtypes so that an empty frame still matches its schema.
The frames are not validated here; `RulesStore` validates them on every write and read.
"""

from collections import defaultdict
from collections.abc import Sequence

import polars as pl
from pydantic import ValidationError

from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import SchemaViolationError
from banhammer.rules.records import (
    Cluster,
    RuleRecord,
    Ruleset,
    SaturationPoint,
    ScoredPair,
    describe_validation_error,
)

RULES_COLUMNS = {
    "rule_id": pl.String,
    "ruleset_id": pl.String,
    "text": pl.String,
    "scope": pl.String,
    "exceptions": pl.String,
    "sanction_hint": pl.String,
    "rule_type": pl.String,
    "discretion_flag": pl.Boolean,
}
RULESETS_COLUMNS = {
    "ruleset_id": pl.String,
    "platform": pl.String,
    "language": pl.String,
    "source_url": pl.String,
    "harvest_date": pl.Date,
    "terms_note": pl.String,
    "rule_count": pl.Int64,
}
CANDIDATE_PAIRS_COLUMNS = {
    "rule_a": pl.String,
    "rule_b": pl.String,
    "cosine": pl.Float64,
    "p_entail_ab": pl.Float64,
    "p_entail_ba": pl.Float64,
    "outcome": pl.String,
}
CLUSTERS_COLUMNS = {
    "cluster_id": pl.String,
    "rule_id": pl.String,
    "ruleset_id": pl.String,
    "cluster_size": pl.Int64,
    "diameter": pl.Float64,
    "resolution": pl.String,
}
SATURATION_COLUMNS = {
    "rulesets_added": pl.Int64,
    "rule_count": pl.Int64,
    "cluster_count": pl.Int64,
    "new_clusters": pl.Int64,
}
RULESET_ONLY_FIELDS = {"rules"}


def rules_frame(corpus: RuleCorpus) -> pl.DataFrame:
    """One row per rule, in corpus order."""
    return pl.DataFrame(
        [rule.model_dump(mode="json") for rule in corpus.rules], schema=RULES_COLUMNS
    )


def rulesets_frame(corpus: RuleCorpus) -> pl.DataFrame:
    """One row per ruleset, in ruleset id order, with its rule count."""
    rows = [
        ruleset.model_dump(exclude=RULESET_ONLY_FIELDS) | {"rule_count": len(ruleset.rules)}
        for ruleset in corpus.rulesets
    ]
    return pl.DataFrame(rows, schema=RULESETS_COLUMNS)


def corpus_from_frames(rules: pl.DataFrame, rulesets: pl.DataFrame) -> RuleCorpus:
    """Rebuild the corpus from the rules and rulesets frames.

    Raises:
        SchemaViolationError: A rule belongs to a ruleset the rulesets frame lacks, a rule
            count disagrees with the rules, or the rebuilt records are invalid.
    """
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rules.iter_rows(named=True):
        grouped[row["ruleset_id"]].append(row)
    unknown = sorted(set(grouped) - set(rulesets["ruleset_id"]))
    if unknown:
        message = f"rules name rulesets missing from the rulesets table: {unknown}"
        raise SchemaViolationError(message)
    try:
        return RuleCorpus(
            rulesets=tuple(
                _ruleset_from_row(row, grouped[row["ruleset_id"]])
                for row in rulesets.iter_rows(named=True)
            )
        )
    except ValidationError as error:
        message = f"rules and rulesets tables are inconsistent: {describe_validation_error(error)}"
        raise SchemaViolationError(message) from error


def candidate_pairs_frame(pairs: Sequence[ScoredPair]) -> pl.DataFrame:
    """One row per candidate pair, in the order given."""
    return pl.DataFrame(
        [pair.model_dump(mode="json") for pair in pairs], schema=CANDIDATE_PAIRS_COLUMNS
    )


def candidate_pairs_from_frame(frame: pl.DataFrame) -> tuple[ScoredPair, ...]:
    """The candidate pairs of a validated frame, in row order."""
    return tuple(ScoredPair.model_validate(row) for row in frame.iter_rows(named=True))


def clusters_frame(clusters: Sequence[Cluster], corpus: RuleCorpus) -> pl.DataFrame:
    """One row per member rule of every cluster, in the order of the clusters."""
    rows = [
        {
            "cluster_id": cluster.cluster_id,
            "rule_id": rule_id,
            "ruleset_id": corpus.ruleset_of(rule_id).ruleset_id,
            "cluster_size": len(cluster.rule_ids),
            "diameter": cluster.diameter,
            "resolution": cluster.resolution.value,
        }
        for cluster in clusters
        for rule_id in cluster.rule_ids
    ]
    return pl.DataFrame(rows, schema=CLUSTERS_COLUMNS)


def saturation_frame(points: Sequence[SaturationPoint]) -> pl.DataFrame:
    """One row per point of the saturation curve."""
    return pl.DataFrame([point.model_dump() for point in points], schema=SATURATION_COLUMNS)


def _ruleset_from_row(row: dict[str, object], rule_rows: list[dict[str, object]]) -> Ruleset:
    rule_count = row["rule_count"]
    if rule_count != len(rule_rows):
        message = (
            f"rule_count of ruleset {row['ruleset_id']} is {rule_count} "
            f"but the rules table holds {len(rule_rows)}"
        )
        raise SchemaViolationError(message)
    fields = {key: value for key, value in row.items() if key != "rule_count"}
    rules = tuple(RuleRecord.model_validate(rule_row) for rule_row in rule_rows)
    return Ruleset.model_validate(fields | {"rules": rules})
