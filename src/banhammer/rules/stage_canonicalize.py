"""Stage "canonicalize": cluster the rules and write the canonical clusters and saturation curve."""

import logging
from pathlib import Path

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.seeding import Seeder
from banhammer.embeddings import SentenceTransformerEmbedder
from banhammer.rules.canonicalizer import CanonicalizationResult, Canonicalizer
from banhammer.rules.errors import ReviewPendingError
from banhammer.rules.review import ReviewFiles, ReviewQueueBuilder
from banhammer.rules.stage_runtime import StageSettings, select_device
from banhammer.rules.store import RulesStore

logger = logging.getLogger(__name__)


class CanonicalizeStage:
    """Runs the canonicalizer on the stored corpus, pairs and the reviewer's decisions."""

    def __init__(
        self, store: RulesStore, canonicalizer: Canonicalizer, review: ReviewFiles
    ) -> None:
        """Read the corpus and pairs from `store`, which also receives the clusters."""
        self._store = store
        self._canonicalizer = canonicalizer
        self._review = review

    def run(self) -> CanonicalizationResult:
        """Canonicalize and write the outputs once nothing is pending.

        The canonical outputs of an earlier run are removed first, so they exist only after a
        run that succeeded, whatever way it fails. While review pairs lack a decision the split
        queue is left as it is, because it cannot be computed yet.

        Raises:
            ReviewPendingError: Review pairs or flagged clusters still lack a decision. The
                split queue is refreshed first when only flagged clusters are pending.
        """
        self._store.remove_canonical_outputs()
        corpus = self._store.read_corpus()
        result = self._canonicalizer.canonicalize(
            corpus, self._store.read_candidate_pairs(), self._review.read_decisions()
        )
        if result.pending_pairs:
            raise self._pending(result)
        self._review.write_split_queue(
            ReviewQueueBuilder(corpus).split_items(result.pending_clusters)
        )
        if result.pending_clusters:
            raise self._pending(result)
        self._store.write_canonical_outputs(result.clusters, result.saturation, corpus)
        return result

    @staticmethod
    def _pending(result: CanonicalizationResult) -> ReviewPendingError:
        message = (
            f"{len(result.pending_pairs)} review pair(s) and {len(result.pending_clusters)} "
            "cluster(s) await a decision"
        )
        return ReviewPendingError(message)


def main() -> None:
    """Entry point of the DVC stage."""
    logging.basicConfig(level=logging.INFO)
    settings = StageSettings.load(Path(PARAMS_FILE_NAME))
    Seeder(settings.core.seed).seed_everything()
    embedder = SentenceTransformerEmbedder.from_hub(settings.embedder, select_device())
    CanonicalizeStage(
        RulesStore(settings.rules.output_dir),
        Canonicalizer(embedder, settings.rules),
        ReviewFiles(settings.rules.review_dir),
    ).run()


if __name__ == "__main__":
    main()
