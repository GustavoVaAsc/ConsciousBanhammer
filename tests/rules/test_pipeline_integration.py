import json
from pathlib import Path

import polars as pl
import pytest
import torch

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.embeddings import ExactNeighbourSearch, SentenceTransformerEmbedder
from banhammer.rules import (
    CandidateGenerator,
    Canonicalizer,
    CrossEncoderEntailmentScorer,
    PairOutcome,
    ReviewFiles,
    RuleCorpusLoader,
    RulesStore,
)
from banhammer.rules.constants import (
    CANONICAL_CLUSTERS_FILE_NAME,
    ENTAILMENT_DECISIONS_FILE_NAME,
    ENTAILMENT_QUEUE_FILE_NAME,
)
from banhammer.rules.stage_atomize import AtomizeStage
from banhammer.rules.stage_candidates import CandidateStage
from banhammer.rules.stage_canonicalize import CanonicalizeStage
from banhammer.rules.stage_runtime import StageSettings

REPO_ROOT = Path(__file__).resolve().parents[2]

HEADER = """\
ruleset_id: {ruleset_id}
platform: reddit
language: {language}
source_url: https://example.org/{ruleset_id}
harvest_date: 2026-11-0{day}
terms_note: Public rule page.
rules:
"""
RULE = """\
  - rule_id: {ruleset_id}-{number}
    ruleset_id: {ruleset_id}
    text: {text}
    rule_type: conduct
    discretion_flag: false
"""
RULESETS = {
    "enone": (
        "en",
        1,
        ["Spam is forbidden.", "Do not insult other members.", "Memes are allowed on weekends."],
    ),
    "esone": (
        "es",
        2,
        [
            "Está prohibido el spam.",
            "No insultes a otros miembros.",
            "Los memes solo se permiten los fines de semana.",
        ],
    ),
    "entwo": (
        "en",
        3,
        ["No spam or self-promotion.", "No insults toward streamers.", "Stay on topic."],
    ),
}


def write_rulesets(directory: Path) -> None:
    directory.mkdir()
    for ruleset_id, (language, day, texts) in RULESETS.items():
        rules = "".join(
            RULE.format(ruleset_id=ruleset_id, number=number, text=text)
            for number, text in enumerate(texts, start=1)
        )
        header = HEADER.format(ruleset_id=ruleset_id, language=language, day=day)
        (directory / f"{ruleset_id}.yaml").write_text(header + rules, encoding="utf-8")


@pytest.mark.integration
def test_the_three_stages_cluster_translated_rules_with_the_real_models(tmp_path: Path) -> None:
    settings = StageSettings.load(REPO_ROOT / PARAMS_FILE_NAME)
    device = torch.device("cpu")
    embedder = SentenceTransformerEmbedder.from_hub(settings.embedder, device)
    scorer = CrossEncoderEntailmentScorer.from_hub(settings.rules.nli, device, torch.float32)
    search = ExactNeighbourSearch(
        k=settings.rules.neighbours_k,
        chunk_size=settings.rules.neighbours_chunk_size,
        device=device,
        min_score=settings.rules.neighbours_min_cosine,
    )
    store, review = RulesStore(tmp_path / "out"), ReviewFiles(tmp_path / "review")
    write_rulesets(tmp_path / "in")

    AtomizeStage(RuleCorpusLoader(), store).run(tmp_path / "in")
    pairs = CandidateStage(
        store, CandidateGenerator(embedder, search, scorer, settings.rules), review
    ).run()
    edges = {(p.rule_a, p.rule_b) for p in pairs if p.outcome is PairOutcome.EDGE}
    assert {("enone-1", "esone-1"), ("enone-2", "esone-2")} <= edges

    queue = (tmp_path / "review" / ENTAILMENT_QUEUE_FILE_NAME).read_text(encoding="utf-8")
    rejections = [
        {"rule_a": item["rule_a"], "rule_b": item["rule_b"], "accept": False}
        for item in map(json.loads, queue.splitlines())
    ]
    (tmp_path / "review" / ENTAILMENT_DECISIONS_FILE_NAME).write_text(
        "".join(f"{json.dumps(rejection)}\n" for rejection in rejections), encoding="utf-8"
    )
    result = CanonicalizeStage(store, Canonicalizer(embedder, settings.rules), review).run()

    assert result.is_final
    clusters = pl.read_parquet(tmp_path / "out" / CANONICAL_CLUSTERS_FILE_NAME)
    by_cluster = clusters.group_by("cluster_id").agg(pl.col("rule_id").sort()).sort("cluster_id")
    assert ["enone-1", "esone-1"] in by_cluster["rule_id"].to_list()
    assert ["enone-2", "esone-2"] in by_cluster["rule_id"].to_list()
    assert by_cluster.height == sum(len(texts) for _, _, texts in RULESETS.values()) - 2
