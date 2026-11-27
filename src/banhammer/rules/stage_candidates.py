"""Stage "candidates": embed the rule items, score candidate pairs, queue the unsettled ones."""

import logging
from pathlib import Path

import torch

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.seeding import Seeder
from banhammer.embeddings import ExactNeighbourSearch, SentenceTransformerEmbedder
from banhammer.rules.candidates import CandidateGenerator
from banhammer.rules.constants import PairOutcome
from banhammer.rules.cross_encoder_entailment_scorer import CrossEncoderEntailmentScorer
from banhammer.rules.records import ScoredPair
from banhammer.rules.review import ReviewFiles, ReviewQueueBuilder
from banhammer.rules.stage_runtime import StageSettings, select_device
from banhammer.rules.store import RulesStore

logger = logging.getLogger(__name__)


class CandidateStage:
    """Writes `candidate_pairs.parquet` and the entailment review queue."""

    def __init__(
        self, store: RulesStore, generator: CandidateGenerator, review: ReviewFiles
    ) -> None:
        """Read the corpus from `store`, which also gets the pairs; the queue goes to `review`."""
        self._store = store
        self._generator = generator
        self._review = review

    def run(self) -> tuple[ScoredPair, ...]:
        """Score the candidate pairs of the stored corpus and queue those needing a human.

        The queue holds every review pair, answered or not: decisions are read later, by the
        canonicalize stage, so this stage depends on nothing a reviewer writes.
        """
        corpus = self._store.read_corpus()
        pairs = self._generator.generate(corpus)
        self._store.write_candidate_pairs(pairs)
        queued = [pair for pair in pairs if pair.outcome is PairOutcome.REVIEW]
        self._review.write_entailment_queue(ReviewQueueBuilder(corpus).entailment_items(queued))
        logger.info("%d candidate pairs, %d queued for review", len(pairs), len(queued))
        return pairs


def main() -> None:
    """Entry point of the DVC stage."""
    logging.basicConfig(level=logging.INFO)
    settings = StageSettings.load(Path(PARAMS_FILE_NAME))
    Seeder(settings.core.seed).seed_everything()
    device = select_device()
    search = ExactNeighbourSearch(
        k=settings.rules.neighbours_k,
        chunk_size=settings.rules.neighbours_chunk_size,
        device=device,
        min_score=settings.rules.neighbours_min_cosine,
    )
    generator = CandidateGenerator(
        SentenceTransformerEmbedder.from_hub(settings.embedder, device),
        search,
        CrossEncoderEntailmentScorer.from_hub(settings.rules.nli, device, torch.float32),
        settings.rules,
    )
    CandidateStage(
        RulesStore(settings.rules.output_dir), generator, ReviewFiles(settings.rules.review_dir)
    ).run()


if __name__ == "__main__":
    main()
