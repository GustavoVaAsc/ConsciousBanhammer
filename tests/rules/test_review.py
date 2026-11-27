import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from banhammer.core.constants import Language
from banhammer.rules.constants import (
    ENTAILMENT_DECISIONS_FILE_NAME,
    ENTAILMENT_QUEUE_FILE_NAME,
    SPLIT_DECISIONS_FILE_NAME,
    SPLIT_QUEUE_FILE_NAME,
    ClusterResolution,
    PairOutcome,
)
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import ReviewFileError
from banhammer.rules.records import Cluster, ScoredPair
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
from rule_corpus_builders import make_rule, make_ruleset

SPANISH = make_ruleset("b", [("b-1", "No insultos.")], language=Language.ES).model_copy(
    update={"rules": (make_rule("b-1", "b", "No insultos.", scope="Comentarios"),)}
)
CORPUS = RuleCorpus(rulesets=(make_ruleset("a", [("a-1", "No insults.")]), SPANISH))
TEXT_B = "No insultos.\nÁmbito: Comentarios"


def write_lines(directory: Path, name: str, lines: list[str]) -> None:
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_builder_attaches_item_texts_and_scores_to_review_pairs() -> None:
    pair = ScoredPair(
        rule_a="a-1",
        rule_b="b-1",
        cosine=0.8,
        p_entail_ab=0.9,
        p_entail_ba=0.4,
        outcome=PairOutcome.REVIEW,
    )
    (item,) = ReviewQueueBuilder(CORPUS).entailment_items([pair])
    assert item == EntailmentReviewItem(
        rule_a="a-1",
        rule_b="b-1",
        text_a="No insults.",
        text_b=TEXT_B,
        cosine=0.8,
        p_entail_ab=0.9,
        p_entail_ba=0.4,
    )


def test_the_builder_lists_cluster_members_with_their_texts() -> None:
    cluster = Cluster(
        cluster_id="c-a-1",
        rule_ids=("a-1", "b-1"),
        diameter=0.7,
        resolution=ClusterResolution.AUTO,
    )
    (item,) = ReviewQueueBuilder(CORPUS).split_items([cluster])
    assert item.cluster_id == "c-a-1"
    assert item.diameter == pytest.approx(0.7)
    assert item.members == (
        ReviewMember(rule_id="a-1", text="No insults."),
        ReviewMember(rule_id="b-1", text=TEXT_B),
    )


def test_queues_are_written_as_json_lines_with_unescaped_unicode(tmp_path: Path) -> None:
    items = [
        EntailmentReviewItem(
            rule_a="a-1",
            rule_b="b-1",
            text_a="No insults.",
            text_b=TEXT_B,
            cosine=0.8,
            p_entail_ab=0.9,
            p_entail_ba=0.4,
        )
    ]
    ReviewFiles(tmp_path / "review").write_entailment_queue(items)
    text = (tmp_path / "review" / ENTAILMENT_QUEUE_FILE_NAME).read_text(encoding="utf-8")
    assert "Ámbito" in text
    assert [json.loads(line)["rule_b"] for line in text.splitlines()] == ["b-1"]


def test_writing_a_queue_replaces_the_previous_one(tmp_path: Path) -> None:
    files = ReviewFiles(tmp_path)
    member = ReviewMember(rule_id="a-1", text="x")
    item = SplitReviewItem(cluster_id="c-a-1", diameter=0.9, members=(member, member))
    files.write_split_queue([item])
    files.write_split_queue([])
    assert (tmp_path / SPLIT_QUEUE_FILE_NAME).read_text(encoding="utf-8") == ""


def test_missing_decision_files_mean_no_decisions_yet(tmp_path: Path) -> None:
    assert ReviewFiles(tmp_path).read_decisions() == ReviewDecisions(entailment=(), splits=())


def test_decisions_are_read_skipping_blank_lines(tmp_path: Path) -> None:
    write_lines(
        tmp_path,
        ENTAILMENT_DECISIONS_FILE_NAME,
        [
            '{"rule_a": "a-1", "rule_b": "b-1", "accept": true}',
            "",
            '{"rule_a": "a-2", "rule_b": "b-2", "accept": false}',
        ],
    )
    write_lines(
        tmp_path,
        SPLIT_DECISIONS_FILE_NAME,
        ['{"cluster_id": "c-a-1", "groups": [["a-1", "b-1"], ["c-1"]]}'],
    )
    decisions = ReviewFiles(tmp_path).read_decisions()
    assert decisions.entailment == (
        EntailmentDecision(rule_a="a-1", rule_b="b-1", accept=True),
        EntailmentDecision(rule_a="a-2", rule_b="b-2", accept=False),
    )
    assert decisions.splits == (
        SplitDecision(cluster_id="c-a-1", groups=(("a-1", "b-1"), ("c-1",))),
    )


@pytest.mark.parametrize(
    ("name", "line", "reason"),
    [
        (ENTAILMENT_DECISIONS_FILE_NAME, "{not json", "JSON"),
        (ENTAILMENT_DECISIONS_FILE_NAME, '["a-1", "b-1"]', "should be an object"),
        (ENTAILMENT_DECISIONS_FILE_NAME, '{"rule_a": "a-1", "rule_b": "b-1"}', "accept"),
        (
            ENTAILMENT_DECISIONS_FILE_NAME,
            '{"rule_a": "b-1", "rule_b": "a-1", "accept": true}',
            "sort before",
        ),
        (
            ENTAILMENT_DECISIONS_FILE_NAME,
            '{"rule_a": "a-1", "rule_b": "b-1", "accept": true, "note": "x"}',
            "note",
        ),
        (SPLIT_DECISIONS_FILE_NAME, '{"cluster_id": "c-a-1", "groups": []}', "groups"),
        (
            SPLIT_DECISIONS_FILE_NAME,
            '{"cluster_id": "c-a-1", "groups": [[]]}',
            "at least one rule id",
        ),
        (
            SPLIT_DECISIONS_FILE_NAME,
            '{"cluster_id": "c-a-1", "groups": [["a-1"], ["a-1", "b-1"]]}',
            "a-1",
        ),
    ],
)
def test_malformed_decision_lines_are_reported_with_file_and_line(
    tmp_path: Path, name: str, line: str, reason: str
) -> None:
    write_lines(tmp_path, name, ["", line])
    with pytest.raises(ReviewFileError, match=rf"{name}:2.*{reason}"):
        ReviewFiles(tmp_path).read_decisions()


def test_two_decisions_for_one_pair_are_reported(tmp_path: Path) -> None:
    line = '{"rule_a": "a-1", "rule_b": "b-1", "accept": true}'
    write_lines(tmp_path, ENTAILMENT_DECISIONS_FILE_NAME, [line, line])
    with pytest.raises(ReviewFileError, match=r"a-1.*b-1"):
        ReviewFiles(tmp_path).read_decisions()


def test_two_decisions_for_one_cluster_are_reported(tmp_path: Path) -> None:
    line = '{"cluster_id": "c-a-1", "groups": [["a-1"]]}'
    write_lines(tmp_path, SPLIT_DECISIONS_FILE_NAME, [line, line])
    with pytest.raises(ReviewFileError, match="c-a-1"):
        ReviewFiles(tmp_path).read_decisions()


def test_an_unreadable_decision_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / ENTAILMENT_DECISIONS_FILE_NAME).write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ReviewFileError, match=ENTAILMENT_DECISIONS_FILE_NAME):
        ReviewFiles(tmp_path).read_decisions()


def test_a_split_decision_into_one_group_keeps_the_cluster_whole() -> None:
    decision = SplitDecision(cluster_id="c-a-1", groups=(("a-1", "b-1"),))
    assert decision.splits_cluster is False
    assert SplitDecision(cluster_id="c-a-1", groups=(("a-1",), ("b-1",))).splits_cluster is True


@pytest.mark.parametrize(
    "fields", [{"cluster_id": "x y"}, {"cluster_id": "a-1"}, {"groups": (("A B",),)}]
)
def test_split_decision_ids_must_be_well_formed(fields: dict[str, Any]) -> None:
    base: dict[str, Any] = {"cluster_id": "c-a-1", "groups": (("a-1",),)}
    with pytest.raises(ValidationError):
        SplitDecision.model_validate(base | fields)


def test_lines_are_split_on_newlines_only_not_on_other_unicode_separators(tmp_path: Path) -> None:
    line = '{"rule_a": "a-1", "rule_b": "b-1", "accept": true} '
    write_lines(tmp_path, ENTAILMENT_DECISIONS_FILE_NAME, [line])
    with pytest.raises(ReviewFileError, match=rf"{ENTAILMENT_DECISIONS_FILE_NAME}:1"):
        ReviewFiles(tmp_path).read_decisions()


def test_a_failed_queue_write_keeps_the_previous_queue_and_leaves_no_temporary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = ReviewFiles(tmp_path)
    member = ReviewMember(rule_id="a-1", text="x")
    files.write_split_queue(
        [SplitReviewItem(cluster_id="c-a-1", diameter=0.9, members=(member,) * 2)]
    )
    before = (tmp_path / SPLIT_QUEUE_FILE_NAME).read_bytes()

    def failing(self: Path, *args: Any, **kwargs: Any) -> int:
        with self.open("wb") as handle:
            handle.write(b"partial")
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr(Path, "write_text", failing)
    with pytest.raises(OSError, match="disk full"):
        files.write_split_queue([])
    assert (tmp_path / SPLIT_QUEUE_FILE_NAME).read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == [SPLIT_QUEUE_FILE_NAME]
