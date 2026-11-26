"""Fixed facts shared by every subpackage: the verdict scale and the severity scale."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

PARAMS_FILE_NAME = "params.yaml"
CORE_SECTION = "core"

NUMPY_SEED_MODULUS = 2**32
SEED_LIMIT = 2**63
STREAM_KEY_BYTES = 8

COMMIT_HASH_PATTERN = r"^[0-9a-f]{40}$"
MODEL_SNAPSHOT_PATTERNS = ("*.safetensors", "*.json", "*.txt")

SEVERITY_MIN = 0
SEVERITY_MAX = 3
SEVERITIES = tuple(range(SEVERITY_MIN, SEVERITY_MAX + 1))


class Verdict(StrEnum):
    """Ordinal verdict of a message against a ruleset, from V0 (no violation) to V3."""

    V0 = "V0"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"

    @property
    def ordinal(self) -> int:
        """Position on the scale: 0 for V0 up to 3 for V3."""
        return VERDICT_ORDINALS[self]

    @property
    def valid_severities(self) -> frozenset[int]:
        """Severities this verdict admits (D7)."""
        return VALID_SEVERITIES[self]

    def admits(self, severity: int) -> bool:
        """Whether the pair (this verdict, severity) lies on the valid grid (D7)."""
        return severity in self.valid_severities


VALID_SEVERITIES: Mapping[Verdict, frozenset[int]] = MappingProxyType(
    {
        Verdict.V0: frozenset({0}),
        Verdict.V1: frozenset({1, 2}),
        Verdict.V2: frozenset({1, 2, 3}),
        Verdict.V3: frozenset({1, 2, 3}),
    }
)

VERDICT_ORDINALS: Mapping[Verdict, int] = MappingProxyType(
    {verdict: ordinal for ordinal, verdict in enumerate(Verdict)}
)
