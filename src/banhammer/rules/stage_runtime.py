"""What the three S1 stage entry points share: the settings they read and the device they use."""

from pathlib import Path
from typing import Self

import torch

from banhammer.core.constants import CORE_SECTION
from banhammer.core.settings import CoreSettings, ParamsFile
from banhammer.embeddings import EMBEDDER_SECTION, EmbedderSettings
from banhammer.rules.constants import RULES_SECTION
from banhammer.rules.records import RulesRecord
from banhammer.rules.settings import RulesSettings


class StageSettings(RulesRecord):
    """The `params.yaml` sections the model-backed S1 stages read: core, rules and embedder."""

    core: CoreSettings
    rules: RulesSettings
    embedder: EmbedderSettings

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read and validate the three sections of the params file at `path`.

        Raises:
            SettingsError: The file or a section is missing or invalid.
        """
        params = ParamsFile(path)
        return cls(
            core=params.section(CORE_SECTION, CoreSettings),
            rules=params.section(RULES_SECTION, RulesSettings),
            embedder=params.section(EMBEDDER_SECTION, EmbedderSettings),
        )


def select_device() -> torch.device:
    """The CUDA device when there is one, else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
