"""The candidate set a human labels to calibrate the translation threshold t2 (D13)."""

import numpy as np
import polars as pl

from banhammer.messages.constants import CalibrationOrigin, Column, PairColumn
from banhammer.messages.errors import DeduplicationError
from banhammer.messages.settings import CalibrationSettings

EXPORT_COLUMNS = [column.value for column in PairColumn]


class CalibrationSetBuilder:
    """Builds the translation pairs a human labels to place t2.

    The set holds the `top_count` highest-scoring candidates, which show what clear translations
    look like, and `band_count` candidates drawn at random from the band `t2 +- band_half_width`
    that is not already in the top, which is where the decision point is decided. The rows come
    out in random order so the score order does not guide the labeller; `cosine` and `origin`
    are there for the analysis and should be hidden while labelling. The text of each message is
    included, so the file is data and never goes into logs.
    """

    def __init__(
        self, settings: CalibrationSettings, translation_threshold: float, rng: np.random.Generator
    ) -> None:
        """Take the calibration sizes, the provisional t2 the band is centred on, and the rng."""
        self._settings = settings
        self._threshold = translation_threshold
        self._rng = rng

    def build(self, candidates: pl.DataFrame, messages: pl.DataFrame) -> pl.DataFrame:
        """The labelling set with the columns of `PairColumn`; `is_translation` is left empty.

        `candidates` has `en_message_id`, `es_message_id` and `cosine`; `messages` has
        `message_id` and `text`.

        Raises:
            DeduplicationError: A candidate's message has no text in `messages`.
        """
        ranked = candidates.sort(
            PairColumn.COSINE,
            PairColumn.EN_MESSAGE_ID,
            PairColumn.ES_MESSAGE_ID,
            descending=[True, False, False],
        )
        top = ranked.head(self._settings.top_count)
        remaining = ranked.slice(self._settings.top_count)
        in_band = remaining.filter(
            (pl.col(PairColumn.COSINE) - self._threshold).abs() <= self._settings.band_half_width
        )
        band = in_band[self._rng.permutation(in_band.height)[: self._settings.band_count]]
        chosen = pl.concat(
            [
                top.with_columns(pl.lit(CalibrationOrigin.TOP).alias(PairColumn.ORIGIN)),
                band.with_columns(pl.lit(CalibrationOrigin.BAND).alias(PairColumn.ORIGIN)),
            ]
        )
        shuffled = chosen[self._rng.permutation(chosen.height)]
        return self._with_texts(shuffled, messages)

    def _with_texts(self, pairs: pl.DataFrame, messages: pl.DataFrame) -> pl.DataFrame:
        texts = messages.select(Column.MESSAGE_ID, Column.TEXT)
        joined = pairs.join(
            texts.rename(
                {Column.MESSAGE_ID: PairColumn.EN_MESSAGE_ID, Column.TEXT: PairColumn.EN_TEXT}
            ),
            on=PairColumn.EN_MESSAGE_ID,
            how="left",
            maintain_order="left",
        ).join(
            texts.rename(
                {Column.MESSAGE_ID: PairColumn.ES_MESSAGE_ID, Column.TEXT: PairColumn.ES_TEXT}
            ),
            on=PairColumn.ES_MESSAGE_ID,
            how="left",
            maintain_order="left",
        )
        untexted = joined.filter(
            pl.col(PairColumn.EN_TEXT).is_null() | pl.col(PairColumn.ES_TEXT).is_null()
        )
        if untexted.height:
            message = f"{untexted.height} candidate pair(s) have a message without text"
            raise DeduplicationError(message)
        return joined.with_columns(
            pl.lit(None, dtype=pl.String).alias(PairColumn.IS_TRANSLATION)
        ).select(EXPORT_COLUMNS)
