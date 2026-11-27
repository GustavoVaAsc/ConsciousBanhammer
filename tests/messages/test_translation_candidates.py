import polars as pl
import pytest

from banhammer.messages import TranslationCandidateFinder
from dedup_corpus import E1, E2, E3, S1, S2, S3, cosine, embedder, search


def side(*rows: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"message_id": pl.String, "text": pl.String}, orient="row")


def test_candidates_are_the_cross_language_pairs_above_the_floor() -> None:
    finder = TranslationCandidateFinder(embedder(), search())
    result = finder.find(
        side(("e1", E1), ("e2", E2), ("e3", E3)), side(("s1", S1), ("s2", S2), ("s3", S3))
    )
    pairs = {
        (row["en_message_id"], row["es_message_id"]): row["cosine"]
        for row in result.iter_rows(named=True)
    }
    assert pairs.keys() == {("e1", "s1"), ("e2", "s2")}
    assert pairs[("e1", "s1")] == pytest.approx(cosine(E1, S1), abs=1e-5)
    assert pairs[("e2", "s2")] == pytest.approx(cosine(E2, S2), abs=1e-5)
    assert result["cosine"].to_list() == sorted(result["cosine"], reverse=True)


def test_text_is_cleaned_before_it_is_embedded() -> None:
    model = embedder()
    TranslationCandidateFinder(model, search()).find(
        side(("e1", f"{E1} https://example.com 😂")), side(("s1", f"<@12> {S1}"))
    )
    assert sorted(model.calls, key=len) == sorted([(E1,), (S1,)], key=len)


def test_an_empty_side_gives_no_candidates_and_no_embedding_calls() -> None:
    model = embedder()
    finder = TranslationCandidateFinder(model, search())
    for english, spanish in [
        (side(), side(("s1", S1))),
        (side(("e1", E1)), side()),
        (side(), side()),
    ]:
        result = finder.find(english, spanish)
        assert result.height == 0
        assert result.columns == ["en_message_id", "es_message_id", "cosine"]
    assert model.calls == []
