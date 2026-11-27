"""S2 message pool: ingest, language identification, deduplication and sampling.

`messages` imports only `core` and `embeddings`.
"""

from banhammer.messages.calibration import CalibrationSetBuilder
from banhammer.messages.constants import MESSAGES_SECTION, DropReason
from banhammer.messages.discord_adapter import DiscordDataAdapter
from banhammer.messages.duplicate_groups import DuplicateGrouper, DuplicateGroups, EdgeCounts
from banhammer.messages.errors import (
    DeduplicationError,
    LanguageDetectionError,
    MessageSchemaError,
    SamplingError,
    SourceDownloadError,
    SourceFormatError,
)
from banhammer.messages.language import (
    LanguageDetector,
    LanguageFilter,
    LanguageResult,
    LinguaLanguageDetector,
)
from banhammer.messages.manifest import IngestReport, MessagesManifest, PoolCounts
from banhammer.messages.near_duplicates import NearDuplicateFinder, NearDuplicates, shingles
from banhammer.messages.pool_builder import MessagePoolBuilder, PoolBuild, Provenance
from banhammer.messages.sampling import ChannelSampler, SampleResult
from banhammer.messages.schemas import (
    MessagesSchema,
    RawMessageSchema,
    validate_frame,
    validate_lazy_data,
)
from banhammer.messages.settings import (
    CalibrationSettings,
    DedupSettings,
    LanguageSettings,
    MessagesSettings,
    SamplingSettings,
    SourceColumns,
    SourceSettings,
)
from banhammer.messages.source_adapter import SourceAdapter
from banhammer.messages.source_download import download_source
from banhammer.messages.store import MessageStore
from banhammer.messages.translation_candidates import TranslationCandidateFinder

__all__ = [
    "MESSAGES_SECTION",
    "CalibrationSetBuilder",
    "CalibrationSettings",
    "ChannelSampler",
    "DedupSettings",
    "DeduplicationError",
    "DiscordDataAdapter",
    "DropReason",
    "DuplicateGrouper",
    "DuplicateGroups",
    "EdgeCounts",
    "IngestReport",
    "LanguageDetectionError",
    "LanguageDetector",
    "LanguageFilter",
    "LanguageResult",
    "LanguageSettings",
    "LinguaLanguageDetector",
    "MessagePoolBuilder",
    "MessageSchemaError",
    "MessageStore",
    "MessagesManifest",
    "MessagesSchema",
    "MessagesSettings",
    "NearDuplicateFinder",
    "NearDuplicates",
    "PoolBuild",
    "PoolCounts",
    "Provenance",
    "RawMessageSchema",
    "SampleResult",
    "SamplingError",
    "SamplingSettings",
    "SourceAdapter",
    "SourceColumns",
    "SourceDownloadError",
    "SourceFormatError",
    "SourceSettings",
    "TranslationCandidateFinder",
    "download_source",
    "shingles",
    "validate_frame",
    "validate_lazy_data",
]
