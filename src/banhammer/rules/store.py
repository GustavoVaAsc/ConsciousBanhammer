"""Parquet files of the S1 artifacts, validated against their schemas on every write and read."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandera.errors
import pandera.polars as pa
import polars as pl

from banhammer.core.atomic_write import atomic_destination
from banhammer.rules.constants import (
    CANDIDATE_PAIRS_FILE_NAME,
    CANONICAL_CLUSTERS_FILE_NAME,
    RULES_FILE_NAME,
    RULESETS_FILE_NAME,
    SATURATION_CURVE_FILE_NAME,
)
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import SchemaViolationError
from banhammer.rules.frames import (
    candidate_pairs_frame,
    candidate_pairs_from_frame,
    clusters_frame,
    corpus_from_frames,
    rules_frame,
    rulesets_frame,
    saturation_frame,
)
from banhammer.rules.records import Cluster, SaturationPoint, ScoredPair
from banhammer.rules.schemas import (
    CandidatePairsSchema,
    CanonicalClustersSchema,
    RulesetsSchema,
    RulesSchema,
    SaturationCurveSchema,
)

logger = logging.getLogger(__name__)


class RulesStore:
    """Reads and writes the S1 artifacts in one directory under their fixed file names.

    A frame that does not satisfy its schema is never written, and a file that does not
    satisfy it is never returned (D28).
    """

    def __init__(self, directory: Path) -> None:
        """Use `directory` for every artifact; it is created on the first write."""
        self._directory = directory

    def write_corpus(self, corpus: RuleCorpus) -> None:
        """Write the rules boundary and the rulesets manifest."""
        rules = self._validated(RulesSchema, rules_frame(corpus), "rules")
        rulesets = self._validated(RulesetsSchema, rulesets_frame(corpus), "rulesets")
        self._write(rulesets, RULESETS_FILE_NAME)
        self._write(rules, RULES_FILE_NAME)

    def read_corpus(self) -> RuleCorpus:
        """Read the rules and rulesets files back into the corpus.

        Raises:
            SchemaViolationError: A file is missing, unreadable, breaks its schema, or the two
                files disagree.
        """
        rules = self._validated(RulesSchema, self._read(RULES_FILE_NAME), "rules")
        rulesets = self._validated(RulesetsSchema, self._read(RULESETS_FILE_NAME), "rulesets")
        return corpus_from_frames(rules, rulesets)

    def write_candidate_pairs(self, pairs: Sequence[ScoredPair]) -> None:
        """Write the scored candidate pairs."""
        frame = self._validated(
            CandidatePairsSchema, candidate_pairs_frame(pairs), "candidate pairs"
        )
        self._write(frame, CANDIDATE_PAIRS_FILE_NAME)

    def read_candidate_pairs(self) -> tuple[ScoredPair, ...]:
        """Read the scored candidate pairs back.

        Raises:
            SchemaViolationError: The file is missing, unreadable or breaks its schema.
        """
        frame = self._read(CANDIDATE_PAIRS_FILE_NAME)
        return candidate_pairs_from_frame(
            self._validated(CandidatePairsSchema, frame, "candidate pairs")
        )

    def write_canonical_outputs(
        self, clusters: Sequence[Cluster], curve: Sequence[SaturationPoint], corpus: RuleCorpus
    ) -> None:
        """Write the canonical clusters boundary and the saturation curve.

        Both frames are validated before either file is written. Each file is replaced
        atomically, the curve first and the boundary last, so a reader that finds the clusters
        also finds the curve of the same run.
        """
        cluster_rows = self._validated(
            CanonicalClustersSchema, clusters_frame(clusters, corpus), "canonical clusters"
        )
        curve_rows = self._validated(
            SaturationCurveSchema, saturation_frame(curve), "saturation curve"
        )
        self._write(curve_rows, SATURATION_CURVE_FILE_NAME)
        self._write(cluster_rows, CANONICAL_CLUSTERS_FILE_NAME)

    def remove_canonical_outputs(self) -> None:
        """Delete the canonical clusters and the curve, so stale ones never outlive their inputs."""
        for name in (CANONICAL_CLUSTERS_FILE_NAME, SATURATION_CURVE_FILE_NAME):
            (self._directory / name).unlink(missing_ok=True)

    def _write(self, frame: pl.DataFrame, name: str) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        with atomic_destination(self._directory / name) as temporary:
            frame.write_parquet(temporary)
        logger.info("wrote %s: %d rows", name, frame.height)

    def _read(self, name: str) -> pl.DataFrame:
        path = self._directory / name
        try:
            return pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError) as error:
            message = f"{path}: cannot read artifact: {error}"
            raise SchemaViolationError(message) from error

    @staticmethod
    def _validated(schema: type[pa.DataFrameModel], frame: pl.DataFrame, name: str) -> pl.DataFrame:
        try:
            return schema.validate(frame, lazy=True)
        except pandera.errors.SchemaErrors as error:
            message = f"{name} do not satisfy their schema: {error}"
            raise SchemaViolationError(message) from error
