"""S1 rule corpus: atomic rule records, canonical clusters and the saturation curve (D15, D28).

`rules` imports only `core` and `embeddings`. A curator writes one YAML file per ruleset with
one record per atomic rule item; the stages validate them (atomize), find candidate pairs and
score mutual entailment (candidates), and cluster the rules with manual review of the unsettled
pairs and of clusters that fail the diameter check (canonicalize). `CrossEncoderEntailmentScorer`
is exported lazily, so importing this package does not import `sentence_transformers`. The
stage modules (`stage_atomize`, `stage_candidates`, `stage_canonicalize`) are DVC entry points,
run as `python -m banhammer.rules.<stage>`, and are not re-exported.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from banhammer.core.constants import Language
from banhammer.rules.candidates import CandidateGenerator
from banhammer.rules.canonicalizer import CanonicalizationResult, Canonicalizer
from banhammer.rules.constants import (
    CANDIDATE_PAIRS_FILE_NAME,
    CANONICAL_CLUSTERS_FILE_NAME,
    RULES_FILE_NAME,
    RULES_SECTION,
    RULESETS_FILE_NAME,
    SATURATION_CURVE_FILE_NAME,
    ClusterResolution,
    PairOutcome,
    RuleType,
)
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.entailment import EntailmentScorer
from banhammer.rules.errors import (
    CanonicalizationError,
    EntailmentError,
    ReviewDecisionError,
    ReviewFileError,
    ReviewPendingError,
    RuleCorpusError,
    RulesError,
    RulesetFormatError,
    SchemaViolationError,
)
from banhammer.rules.loader import RuleCorpusLoader
from banhammer.rules.records import Cluster, RuleRecord, Ruleset, SaturationPoint, ScoredPair
from banhammer.rules.review import (
    EntailmentDecision,
    EntailmentReviewItem,
    ReviewDecisions,
    ReviewFiles,
    ReviewMember,
    ReviewQueueBuilder,
    SplitDecision,
    SplitReviewItem,
)
from banhammer.rules.saturation import SaturationCurveBuilder
from banhammer.rules.schemas import (
    CandidatePairsSchema,
    CanonicalClustersSchema,
    RulesetsSchema,
    RulesSchema,
    SaturationCurveSchema,
)
from banhammer.rules.settings import NliSettings, RulesSettings
from banhammer.rules.store import RulesStore

if TYPE_CHECKING:
    from banhammer.rules.cross_encoder_entailment_scorer import CrossEncoderEntailmentScorer

__all__ = [
    "CANDIDATE_PAIRS_FILE_NAME",
    "CANONICAL_CLUSTERS_FILE_NAME",
    "RULESETS_FILE_NAME",
    "RULES_FILE_NAME",
    "RULES_SECTION",
    "SATURATION_CURVE_FILE_NAME",
    "CandidateGenerator",
    "CandidatePairsSchema",
    "CanonicalClustersSchema",
    "CanonicalizationError",
    "CanonicalizationResult",
    "Canonicalizer",
    "Cluster",
    "ClusterResolution",
    "CrossEncoderEntailmentScorer",
    "EntailmentDecision",
    "EntailmentError",
    "EntailmentReviewItem",
    "EntailmentScorer",
    "Language",
    "NliSettings",
    "PairOutcome",
    "ReviewDecisionError",
    "ReviewDecisions",
    "ReviewFileError",
    "ReviewFiles",
    "ReviewMember",
    "ReviewPendingError",
    "ReviewQueueBuilder",
    "RuleCorpus",
    "RuleCorpusError",
    "RuleCorpusLoader",
    "RuleRecord",
    "RuleType",
    "RulesError",
    "RulesSchema",
    "RulesSettings",
    "RulesStore",
    "Ruleset",
    "RulesetFormatError",
    "RulesetsSchema",
    "SaturationCurveBuilder",
    "SaturationCurveSchema",
    "SaturationPoint",
    "SchemaViolationError",
    "ScoredPair",
    "SplitDecision",
    "SplitReviewItem",
]

_LAZY_BACKEND = "CrossEncoderEntailmentScorer"
_BACKEND_MODULE = "banhammer.rules.cross_encoder_entailment_scorer"


def __getattr__(name: str) -> object:
    """Import `CrossEncoderEntailmentScorer` on first access (PEP 562)."""
    if name == _LAZY_BACKEND:
        return getattr(import_module(_BACKEND_MODULE), name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
