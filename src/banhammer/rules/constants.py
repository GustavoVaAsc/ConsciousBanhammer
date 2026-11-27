"""Fixed facts of the rules subpackage: closed sets, id patterns, labels and file names."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from banhammer.core.constants import Language

RULES_SECTION = "rules"

RULESET_FILE_SUFFIX = ".yaml"
RULES_FILE_NAME = "rules.parquet"
RULESETS_FILE_NAME = "rulesets.parquet"
CANDIDATE_PAIRS_FILE_NAME = "candidate_pairs.parquet"
CANONICAL_CLUSTERS_FILE_NAME = "canonical_clusters.parquet"
SATURATION_CURVE_FILE_NAME = "saturation_curve.parquet"
ENTAILMENT_QUEUE_FILE_NAME = "entailment_review_queue.jsonl"
ENTAILMENT_DECISIONS_FILE_NAME = "entailment_decisions.jsonl"
SPLIT_QUEUE_FILE_NAME = "split_review_queue.jsonl"
SPLIT_DECISIONS_FILE_NAME = "split_decisions.jsonl"

RULESET_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
RULE_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]*$"
PLATFORM_PATTERN = r"^[a-z][a-z0-9_]*$"
MERGE_TAG = "tag:yaml.org,2002:merge"
SOURCE_URL_PATTERN = r"^https?://\S+$"
CLUSTER_ID_PREFIX = "c-"
CLUSTER_ID_PATTERN = r"^c-[a-z0-9][a-z0-9_.-]*$"

ENTAILMENT_LABEL = "entailment"
COSINE_MIN = -1.0
COSINE_MAX = 1.0
COSINE_DISTANCE_MAX = 2.0
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
SCORE_DECIMALS = 6
MIN_MEMBERS_FOR_A_PAIR = 2


class RuleType(StrEnum):
    """What a rule protects, from the proposal's four rule types (proposal 4.1)."""

    HARM = "harm"
    CONDUCT = "conduct"
    INTEGRITY = "integrity"
    META = "meta"


class PairOutcome(StrEnum):
    """What the mutual-entailment test decides about a candidate pair (D15)."""

    EDGE = "edge"
    NO_EDGE = "no_edge"
    REVIEW = "review"


class ClusterResolution(StrEnum):
    """How a cluster stands with the diameter check (D15).

    `PENDING` marks a cluster over the diameter limit that awaits a split decision; it is
    never written to the canonical clusters artifact.
    """

    AUTO = "auto"
    KEPT = "kept"
    SPLIT = "split"
    PENDING = "pending"


FINAL_RESOLUTIONS = tuple(
    resolution for resolution in ClusterResolution if resolution is not ClusterResolution.PENDING
)


ITEM_SCOPE_LABELS: Mapping[Language, str] = MappingProxyType(
    {Language.EN: "Scope", Language.ES: "Ámbito"}
)
ITEM_EXCEPTIONS_LABELS: Mapping[Language, str] = MappingProxyType(
    {Language.EN: "Exceptions", Language.ES: "Excepciones"}
)
