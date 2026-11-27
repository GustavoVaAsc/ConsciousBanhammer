from pathlib import Path
from typing import Any

import polars as pl
import pytest

from banhammer.rules.constants import (
    CANDIDATE_PAIRS_FILE_NAME,
    CANONICAL_CLUSTERS_FILE_NAME,
    RULES_FILE_NAME,
    RULESETS_FILE_NAME,
    SATURATION_CURVE_FILE_NAME,
    ClusterResolution,
    PairOutcome,
)
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import SchemaViolationError
from banhammer.rules.records import Cluster, SaturationPoint, ScoredPair
from banhammer.rules.schemas import CanonicalClustersSchema, RulesSchema, SaturationCurveSchema
from banhammer.rules.store import RulesStore
from rule_corpus_builders import make_ruleset

CORPUS = RuleCorpus(
    rulesets=(
        make_ruleset("a", [("a-1", "No insults."), ("a-2", "Be kind.")]),
        make_ruleset("b", [("b-1", "No insults at all.")]),
    )
)
PAIR = ScoredPair(
    rule_a="a-1",
    rule_b="b-1",
    cosine=0.9,
    p_entail_ab=0.9,
    p_entail_ba=0.9,
    outcome=PairOutcome.EDGE,
)
CLUSTERS = (
    Cluster(
        cluster_id="c-a-1", rule_ids=("a-1", "b-1"), diameter=0.2, resolution=ClusterResolution.AUTO
    ),
    Cluster(cluster_id="c-a-2", rule_ids=("a-2",), diameter=0.0, resolution=ClusterResolution.AUTO),
)
CURVE = (SaturationPoint(rulesets_added=2, rule_count=3, cluster_count=2, new_clusters=2),)


def test_the_corpus_is_written_as_two_parquet_files_and_read_back(tmp_path: Path) -> None:
    store = RulesStore(tmp_path / "out")
    store.write_corpus(CORPUS)
    assert (tmp_path / "out" / RULES_FILE_NAME).is_file()
    assert (tmp_path / "out" / RULESETS_FILE_NAME).is_file()
    assert RulesSchema.validate(pl.read_parquet(tmp_path / "out" / RULES_FILE_NAME)).height == 3
    assert store.read_corpus() == CORPUS


def test_candidate_pairs_are_written_and_read_back(tmp_path: Path) -> None:
    store = RulesStore(tmp_path)
    store.write_candidate_pairs([PAIR])
    assert (tmp_path / CANDIDATE_PAIRS_FILE_NAME).is_file()
    assert store.read_candidate_pairs() == (PAIR,)


def test_canonical_outputs_are_written_with_their_schemas(tmp_path: Path) -> None:
    store = RulesStore(tmp_path)
    store.write_canonical_outputs(CLUSTERS, CURVE, CORPUS)
    clusters = pl.read_parquet(tmp_path / CANONICAL_CLUSTERS_FILE_NAME)
    curve = pl.read_parquet(tmp_path / SATURATION_CURVE_FILE_NAME)
    assert CanonicalClustersSchema.validate(clusters, lazy=True).height == 3
    assert SaturationCurveSchema.validate(curve, lazy=True).height == 1


def test_removing_canonical_outputs_deletes_both_files_and_tolerates_absence(
    tmp_path: Path,
) -> None:
    store = RulesStore(tmp_path)
    store.remove_canonical_outputs()
    store.write_canonical_outputs(CLUSTERS, CURVE, CORPUS)
    store.remove_canonical_outputs()
    assert not (tmp_path / CANONICAL_CLUSTERS_FILE_NAME).exists()
    assert not (tmp_path / SATURATION_CURVE_FILE_NAME).exists()


def test_a_frame_that_breaks_its_schema_is_not_written(tmp_path: Path) -> None:
    store = RulesStore(tmp_path)
    broken = Cluster.model_construct(
        cluster_id="c-a-1", rule_ids=("a-1", "b-1"), diameter=5.0, resolution=ClusterResolution.AUTO
    )
    with pytest.raises(SchemaViolationError, match="canonical clusters"):
        store.write_canonical_outputs([broken], CURVE, CORPUS)
    assert not (tmp_path / CANONICAL_CLUSTERS_FILE_NAME).exists()


def test_a_tampered_file_is_rejected_when_read(tmp_path: Path) -> None:
    store = RulesStore(tmp_path)
    store.write_corpus(CORPUS)
    path = tmp_path / RULES_FILE_NAME
    pl.read_parquet(path).with_columns(pl.lit("spam").alias("rule_type")).write_parquet(path)
    with pytest.raises(SchemaViolationError, match="rules"):
        store.read_corpus()


def test_a_missing_file_is_reported_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(SchemaViolationError, match=RULES_FILE_NAME):
        RulesStore(tmp_path).read_corpus()


def test_a_file_that_is_not_parquet_is_reported(tmp_path: Path) -> None:
    (tmp_path / CANDIDATE_PAIRS_FILE_NAME).write_text("not parquet", encoding="utf-8")
    with pytest.raises(SchemaViolationError, match=CANDIDATE_PAIRS_FILE_NAME):
        RulesStore(tmp_path).read_candidate_pairs()


def test_the_saturation_curve_is_written_before_the_clusters_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: list[str] = []
    original = pl.DataFrame.write_parquet

    def recording(self: pl.DataFrame, file: Path, *args: Any, **kwargs: Any) -> None:
        written.append(Path(file).name.split(".")[1] + ".parquet")
        original(self, file, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", recording)
    store = RulesStore(tmp_path)
    store.write_canonical_outputs(CLUSTERS, CURVE, CORPUS)
    store.write_corpus(CORPUS)
    assert written == [
        SATURATION_CURVE_FILE_NAME,
        CANONICAL_CLUSTERS_FILE_NAME,
        RULESETS_FILE_NAME,
        RULES_FILE_NAME,
    ]


def test_a_failed_write_keeps_the_previous_file_and_leaves_no_temporary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = RulesStore(tmp_path)
    store.write_candidate_pairs([PAIR])
    before = (tmp_path / CANDIDATE_PAIRS_FILE_NAME).read_bytes()

    def failing(self: pl.DataFrame, file: Path, *args: Any, **kwargs: Any) -> None:
        Path(file).write_bytes(b"partial")
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", failing)
    with pytest.raises(OSError, match="disk full"):
        store.write_candidate_pairs([])
    assert (tmp_path / CANDIDATE_PAIRS_FILE_NAME).read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == [CANDIDATE_PAIRS_FILE_NAME]
