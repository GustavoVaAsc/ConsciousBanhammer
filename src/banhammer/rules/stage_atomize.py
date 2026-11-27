"""Stage "atomize": validate the curated ruleset files and write the rules artifacts.

The splitting into atomic items is done by the curator when writing the files (the human
writes one record per atomic rule item); this stage checks them and fixes them as Parquet.
"""

import logging
from pathlib import Path

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.rules.constants import RULES_SECTION
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.loader import RuleCorpusLoader
from banhammer.rules.settings import RulesSettings
from banhammer.rules.store import RulesStore

logger = logging.getLogger(__name__)


class AtomizeStage:
    """Loads the ruleset files and writes `rules.parquet` and `rulesets.parquet`."""

    def __init__(self, loader: RuleCorpusLoader, store: RulesStore) -> None:
        """Read with `loader`, write with `store`."""
        self._loader = loader
        self._store = store

    def run(self, rulesets_dir: Path) -> RuleCorpus:
        """Validate every ruleset file in `rulesets_dir` and write the artifacts.

        Nothing is written when any file is invalid.
        """
        corpus = self._loader.load_directory(rulesets_dir)
        self._store.write_corpus(corpus)
        logger.info("atomized %d rulesets into %d rules", len(corpus.rulesets), len(corpus.rules))
        return corpus


def main() -> None:
    """Entry point of the DVC stage."""
    logging.basicConfig(level=logging.INFO)
    settings = ParamsFile(Path(PARAMS_FILE_NAME)).section(RULES_SECTION, RulesSettings)
    AtomizeStage(RuleCorpusLoader(), RulesStore(settings.output_dir)).run(settings.rulesets_dir)


if __name__ == "__main__":
    main()
