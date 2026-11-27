"""Entry points of the S2 pipeline stages: `ingest` and `build`.

Functional module: stage entry points (`main`) wiring settings, models and files together.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from banhammer.core.constants import CORE_SECTION, PARAMS_FILE_NAME
from banhammer.core.seeding import Seeder
from banhammer.core.settings import CoreSettings, ParamsFile
from banhammer.embeddings import (
    EMBEDDER_SECTION,
    Embedder,
    EmbedderSettings,
    SentenceTransformerEmbedder,
)
from banhammer.messages.constants import MESSAGES_SECTION
from banhammer.messages.discord_adapter import DiscordDataAdapter
from banhammer.messages.language import LanguageDetector, LinguaLanguageDetector
from banhammer.messages.pool_builder import MessagePoolBuilder, Provenance
from banhammer.messages.settings import MessagesSettings
from banhammer.messages.source_adapter import SourceAdapter
from banhammer.messages.source_download import download_source
from banhammer.messages.store import MessageStore

logger = logging.getLogger(__name__)

INGEST = "ingest"
BUILD = "build"


def run_ingest(adapter: SourceAdapter, store: MessageStore) -> int:
    """Stream the messages of `adapter` into `store` as the raw messages, with the ingest report.

    Returns:
        The number of raw messages written.

    Raises:
        SourceFormatError: A timestamp or bot flag cannot be parsed (raised before any write).
    """
    report = adapter.report()
    store.write_raw(adapter.scan())
    store.write_ingest_report(report)
    logger.info(
        "ingested %d raw messages, dropped %d oversize, replaced %d mention(s)",
        report.rows,
        report.dropped_oversize,
        sum(report.mentions_replaced.values()),
    )
    return report.rows


def run_build(
    provenance: Provenance, embedder: Embedder, detector: LanguageDetector, device: torch.device
) -> int:
    """Build the messages boundary from the stored raw messages and write it.

    Writes the messages, the manifest and the translation calibration candidates next to the
    raw messages. The raw messages are read lazily.

    Returns:
        The number of messages in the boundary.
    """
    store = MessageStore(provenance.settings.output_dir)
    builder = MessagePoolBuilder.assemble(provenance, embedder, detector, device)
    result = builder.build(store.read_raw(), store.read_ingest_report())
    store.write_messages(result.messages)
    store.write_manifest(result.manifest)
    store.write_calibration(result.calibration)
    logger.info("built %d messages", result.messages.height)
    return result.messages.height


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `ingest` or `build` stage with the settings of `params.yaml`."""
    parser = argparse.ArgumentParser(description="S2 message pool stages")
    parser.add_argument("stage", choices=[INGEST, BUILD])
    parser.add_argument("--params", type=Path, default=Path(PARAMS_FILE_NAME))
    arguments = parser.parse_args(argv)
    params = ParamsFile(arguments.params)
    settings = params.section(MESSAGES_SECTION, MessagesSettings)
    seed = params.section(CORE_SECTION, CoreSettings).seed
    Seeder(seed).seed_everything()
    if arguments.stage == INGEST:
        files = download_source(settings.source, settings.raw_dir)
        adapter = DiscordDataAdapter(files, settings.source.columns, settings.source.repo_id)
        run_ingest(adapter, MessageStore(settings.output_dir))
        return 0
    embedder_settings = params.section(EMBEDDER_SECTION, EmbedderSettings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_build(
        Provenance(seed, settings, embedder_settings),
        SentenceTransformerEmbedder.from_hub(embedder_settings, device),
        LinguaLanguageDetector(settings.language.detector_languages),
        device,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
