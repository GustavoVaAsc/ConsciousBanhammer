import json
from pathlib import Path

import polars as pl
import pytest
import torch

from banhammer.embeddings import ExactNeighbourSearch
from banhammer.rules.candidates import CandidateGenerator
from banhammer.rules.canonicalizer import Canonicalizer
from banhammer.rules.constants import (
    CANDIDATE_PAIRS_FILE_NAME,
    CANONICAL_CLUSTERS_FILE_NAME,
    ENTAILMENT_DECISIONS_FILE_NAME,
    ENTAILMENT_QUEUE_FILE_NAME,
    RULES_FILE_NAME,
    RULESETS_FILE_NAME,
    SATURATION_CURVE_FILE_NAME,
    SPLIT_DECISIONS_FILE_NAME,
    SPLIT_QUEUE_FILE_NAME,
)
from banhammer.rules.errors import (
    ReviewDecisionError,
    ReviewFileError,
    ReviewPendingError,
    RulesetFormatError,
)
from banhammer.rules.loader import RuleCorpusLoader
from banhammer.rules.review import ReviewFiles
from banhammer.rules.settings import RulesSettings
from banhammer.rules.stage_atomize import AtomizeStage
from banhammer.rules.stage_candidates import CandidateStage
from banhammer.rules.stage_canonicalize import CanonicalizeStage
from banhammer.rules.store import RulesStore
from embedder_double import FakeEmbedder
from entailment_double import FakeEntailmentScorer

CPU = torch.device("cpu")
SETTINGS = RulesSettings.model_validate(
    {
        "rulesets_dir": "in",
        "output_dir": "out",
        "review_dir": "review",
        "neighbours_k": 5,
        "neighbours_min_cosine": 0.5,
        "neighbours_chunk_size": 4,
        "entail_high": 0.8,
        "entail_low": 0.2,
        "diameter_max": 0.5,
        "saturation_step": 1,
        "nli": {"model": "org/nli", "revision": "ab" * 20, "batch_size": 2, "max_length": 64},
    }
)

RULESET_TEMPLATE = """\
ruleset_id: {ruleset_id}
platform: reddit
language: {language}
source_url: https://example.org/{ruleset_id}
harvest_date: {date}
terms_note: Public rule page.
rules:
  - rule_id: {ruleset_id}-1
    ruleset_id: {ruleset_id}
    text: {text}
    rule_type: conduct
    discretion_flag: false
"""
RULESETS = {
    "one": ("en", "2026-11-01", "No insults."),
    "two": ("es", "2026-11-02", "No insultos."),
    "three": ("en", "2026-11-03", "No insults toward streamers."),
    "four": ("en", "2026-11-04", "Stay on topic."),
}
VECTORS = {
    "No insults.": [1.0, 0.0, 0.0],
    "No insultos.": [0.98, 0.2, 0.0],
    "No insults toward streamers.": [0.9, 0.0, 0.44],
    "Stay on topic.": [0.0, 0.0, 1.0],
}
PROBABILITIES = {
    ("No insults.", "No insultos."): 0.95,
    ("No insultos.", "No insults."): 0.9,
    ("No insults.", "No insults toward streamers."): 0.1,
    ("No insults toward streamers.", "No insults."): 0.9,
}


def write_rulesets(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for ruleset_id, (language, date, text) in RULESETS.items():
        content = RULESET_TEMPLATE.format(
            ruleset_id=ruleset_id, language=language, date=date, text=text
        )
        (directory / f"{ruleset_id}.yaml").write_text(content, encoding="utf-8")
    return directory


def candidate_stage(tmp_path: Path) -> CandidateStage:
    search = ExactNeighbourSearch(k=5, chunk_size=4, device=CPU, min_score=0.5)
    generator = CandidateGenerator(
        FakeEmbedder(VECTORS), search, FakeEntailmentScorer(PROBABILITIES), SETTINGS
    )
    return CandidateStage(RulesStore(tmp_path / "out"), generator, ReviewFiles(tmp_path / "review"))


def canonicalize_stage(tmp_path: Path) -> CanonicalizeStage:
    return CanonicalizeStage(
        RulesStore(tmp_path / "out"),
        Canonicalizer(FakeEmbedder(VECTORS), SETTINGS),
        ReviewFiles(tmp_path / "review"),
    )


def test_atomize_validates_the_rulesets_and_writes_the_two_tables(tmp_path: Path) -> None:
    rulesets = write_rulesets(tmp_path / "in")
    corpus = AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(rulesets)
    assert len(corpus.rules) == 4
    assert pl.read_parquet(tmp_path / "out" / RULES_FILE_NAME).height == 4
    assert pl.read_parquet(tmp_path / "out" / RULESETS_FILE_NAME).height == 4


def test_atomize_writes_nothing_when_a_ruleset_is_invalid(tmp_path: Path) -> None:
    rulesets = write_rulesets(tmp_path / "in")
    broken = (rulesets / "one.yaml").read_text(encoding="utf-8").replace("conduct", "spam")
    (rulesets / "one.yaml").write_text(broken, encoding="utf-8")
    with pytest.raises(RulesetFormatError, match=r"one\.yaml"):
        AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(rulesets)
    assert not (tmp_path / "out").exists()


def test_the_candidate_stage_writes_every_pair_and_queues_only_the_review_pairs(
    tmp_path: Path,
) -> None:
    AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(
        write_rulesets(tmp_path / "in")
    )
    pairs = candidate_stage(tmp_path).run()
    assert pl.read_parquet(tmp_path / "out" / CANDIDATE_PAIRS_FILE_NAME).height == len(pairs)
    queue = (tmp_path / "review" / ENTAILMENT_QUEUE_FILE_NAME).read_text(encoding="utf-8")
    queued = [json.loads(line) for line in queue.splitlines()]
    assert [(item["rule_a"], item["rule_b"]) for item in queued] == [("one-1", "three-1")]
    assert queued[0]["text_b"] == "No insults toward streamers."


def test_canonicalization_waits_for_the_review_pairs_then_finishes_with_decisions(
    tmp_path: Path,
) -> None:
    AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(
        write_rulesets(tmp_path / "in")
    )
    candidate_stage(tmp_path).run()
    with pytest.raises(ReviewPendingError, match="1 review pair"):
        canonicalize_stage(tmp_path).run()
    assert not (tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME).exists()

    decision = {"rule_a": "one-1", "rule_b": "three-1", "accept": False}
    (tmp_path / "review" / ENTAILMENT_DECISIONS_FILE_NAME).write_text(
        json.dumps(decision) + "\n", encoding="utf-8"
    )
    result = canonicalize_stage(tmp_path).run()
    assert result.is_final
    clusters = pl.read_parquet(tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME)
    assert clusters.group_by("cluster_id").len().sort("cluster_id")["cluster_id"].to_list() == [
        "c-four-1",
        "c-one-1",
        "c-three-1",
    ]
    assert clusters.filter(pl.col("cluster_id") == "c-one-1")["rule_id"].to_list() == [
        "one-1",
        "two-1",
    ]
    assert pl.read_parquet(tmp_path / "out" / SATURATION_CURVE_FILE_NAME).height == 4
    assert (tmp_path / "review" / SPLIT_QUEUE_FILE_NAME).read_text(encoding="utf-8") == ""


def test_a_flagged_cluster_is_queued_and_blocks_the_outputs_until_decided(
    tmp_path: Path,
) -> None:
    AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(
        write_rulesets(tmp_path / "in")
    )
    candidate_stage(tmp_path).run()
    accept = {"rule_a": "one-1", "rule_b": "three-1", "accept": True}
    (tmp_path / "review" / ENTAILMENT_DECISIONS_FILE_NAME).write_text(
        json.dumps(accept) + "\n", encoding="utf-8"
    )
    tight = SETTINGS.model_copy(update={"diameter_max": 0.05})
    stage = CanonicalizeStage(
        RulesStore(tmp_path / "out"),
        Canonicalizer(FakeEmbedder(VECTORS), tight),
        ReviewFiles(tmp_path / "review"),
    )
    with pytest.raises(ReviewPendingError, match="1 cluster"):
        stage.run()
    queue = (tmp_path / "review" / SPLIT_QUEUE_FILE_NAME).read_text(encoding="utf-8")
    (item,) = [json.loads(line) for line in queue.splitlines()]
    assert item["cluster_id"] == "c-one-1"
    assert [member["rule_id"] for member in item["members"]] == ["one-1", "three-1", "two-1"]
    assert not (tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME).exists()

    split = {"cluster_id": "c-one-1", "groups": [["one-1", "two-1"], ["three-1"]]}
    (tmp_path / "review" / SPLIT_DECISIONS_FILE_NAME).write_text(
        json.dumps(split) + "\n", encoding="utf-8"
    )
    assert stage.run().is_final
    clusters = pl.read_parquet(tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME)
    assert set(clusters["resolution"]) == {"split", "auto"}


def test_a_stale_canonical_output_is_removed_when_review_is_pending(tmp_path: Path) -> None:
    AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(
        write_rulesets(tmp_path / "in")
    )
    candidate_stage(tmp_path).run()
    stale = tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME
    stale.write_bytes(b"stale")
    with pytest.raises(ReviewPendingError):
        canonicalize_stage(tmp_path).run()
    assert not stale.exists()


def test_canonical_outputs_exist_only_after_a_successful_run(tmp_path: Path) -> None:
    AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(
        write_rulesets(tmp_path / "in")
    )
    candidate_stage(tmp_path).run()
    decisions = tmp_path / "review" / ENTAILMENT_DECISIONS_FILE_NAME
    reject = {"rule_a": "one-1", "rule_b": "three-1", "accept": False}
    decisions.write_text(json.dumps(reject) + "\n", encoding="utf-8")
    canonicalize_stage(tmp_path).run()
    clusters = tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME
    curve = tmp_path / "out" / SATURATION_CURVE_FILE_NAME
    assert clusters.is_file()
    assert curve.is_file()

    stale = {"rule_a": "four-1", "rule_b": "one-1", "accept": True}
    decisions.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(ReviewDecisionError, match="four-1"):
        canonicalize_stage(tmp_path).run()
    assert not clusters.exists()
    assert not curve.exists()

    decisions.write_text("{not json\n", encoding="utf-8")
    clusters.write_bytes(b"stale")
    with pytest.raises(ReviewFileError):
        canonicalize_stage(tmp_path).run()
    assert not clusters.exists()


def test_pending_review_pairs_leave_an_existing_split_queue_alone(tmp_path: Path) -> None:
    AtomizeStage(RuleCorpusLoader(), RulesStore(tmp_path / "out")).run(
        write_rulesets(tmp_path / "in")
    )
    candidate_stage(tmp_path).run()
    queue = tmp_path / "review" / SPLIT_QUEUE_FILE_NAME
    queue.write_text("earlier queue\n", encoding="utf-8")
    with pytest.raises(ReviewPendingError):
        canonicalize_stage(tmp_path).run()
    assert queue.read_text(encoding="utf-8") == "earlier queue\n"
