"""Cross-language candidate pairs by LaBSE cosine similarity (D13, D14)."""

import numpy as np
import polars as pl
from numpy.typing import NDArray

from banhammer.embeddings import Embedder, ExactNeighbourSearch
from banhammer.messages.constants import Column, PairColumn
from banhammer.messages.text_cleaning import clean_text

CANDIDATE_SCHEMA = {
    PairColumn.EN_MESSAGE_ID.value: pl.String,
    PairColumn.ES_MESSAGE_ID.value: pl.String,
    PairColumn.COSINE.value: pl.Float64,
}


class TranslationCandidateFinder:
    """Finds English-Spanish message pairs whose LaBSE embeddings are close.

    English messages are the queries and Spanish messages the corpus of an exact top-k search.
    The search's `min_score` is the candidate floor, below the translation threshold t2, so the
    result also holds the pairs a human needs to see to calibrate t2. Same-language pairs are
    never searched: they are paraphrases more often than duplicates (D13).

    Callers pass one message per distinct text: exact copies would tie in the top-k.
    """

    def __init__(self, embedder: Embedder, search: ExactNeighbourSearch) -> None:
        """Take the embedder and a configured search (k, chunk size, device, floor)."""
        self._embedder = embedder
        self._search = search

    def find(self, english: pl.DataFrame, spanish: pl.DataFrame) -> pl.DataFrame:
        """Pairs of `message_id`s with their cosine, best first.

        Both frames need `message_id` and `text` columns. The result has the columns
        `en_message_id`, `es_message_id` and `cosine`.
        """
        if english.is_empty() or spanish.is_empty():
            return pl.DataFrame(schema=CANDIDATE_SCHEMA)
        pairs = self._search.between(self._encode(english), self._encode(spanish))
        return pl.DataFrame(
            {
                PairColumn.EN_MESSAGE_ID: english[Column.MESSAGE_ID].gather(pairs.query_indices),
                PairColumn.ES_MESSAGE_ID: spanish[Column.MESSAGE_ID].gather(pairs.corpus_indices),
                PairColumn.COSINE: pairs.scores.astype("float64"),
            },
            schema=CANDIDATE_SCHEMA,
        ).sort(
            PairColumn.COSINE,
            PairColumn.EN_MESSAGE_ID,
            PairColumn.ES_MESSAGE_ID,
            descending=[True, False, False],
        )

    def _encode(self, frame: pl.DataFrame) -> NDArray[np.float32]:
        return self._embedder.encode([clean_text(text) for text in frame[Column.TEXT]])
