"""Same-language near-duplicates by MinHash, LSH and exact Jaccard (D13)."""

from collections.abc import Sequence
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from banhammer.messages.errors import DeduplicationError
from banhammer.messages.settings import DedupSettings


def shingles(key: str, size: int) -> frozenset[str]:
    """Character `size`-grams of `key`; a key shorter than `size` is its own single shingle."""
    if len(key) <= size:
        return frozenset({key})
    return frozenset(key[start : start + size] for start in range(len(key) - size + 1))


@dataclass(frozen=True)
class NearDuplicates:
    """Near-duplicate pairs: the verified `edges`, and how many LSH `candidates` were checked."""

    edges: list[tuple[str, str]]
    candidates: int


class NearDuplicateFinder:
    """Finds pairs of keys whose shingle sets have a Jaccard similarity of at least t1.

    MinHash signatures and LSH propose candidate pairs; each candidate is then checked against
    the exact Jaccard of the two shingle sets, so every returned pair really is at or above the
    threshold. LSH is probabilistic: a pair just above the threshold can be missed, which
    `lsh_recall_weight` makes rare by choosing bands and rows that favour recall.
    """

    def __init__(self, settings: DedupSettings) -> None:
        """Take the dedup settings."""
        self._settings = settings

    def find(self, ids: Sequence[str], keys: Sequence[str]) -> NearDuplicates:
        """Pairs `(first, second)` of ids, `first < second`, sorted, whose keys are near-equal.

        Raises:
            DeduplicationError: `ids` and `keys` differ in length, or `ids` repeat.
        """
        if len(ids) != len(keys) or len(set(ids)) != len(ids):
            message = (
                f"ids must be unique and match the keys in length, got {len(ids)} ids "
                f"({len(set(ids))} unique) and {len(keys)} keys"
            )
            raise DeduplicationError(message)
        sets = {
            identifier: shingles(key, self._settings.shingle_size)
            for identifier, key in zip(ids, keys, strict=True)
        }
        signatures = {
            identifier: self._signature(shingle_set) for identifier, shingle_set in sets.items()
        }
        index = self._index(signatures)
        candidates = {
            (min(identifier, other), max(identifier, other))
            for identifier, signature in signatures.items()
            for other in index.query(signature)
            if other != identifier
        }
        edges = sorted(pair for pair in candidates if self._similar(sets[pair[0]], sets[pair[1]]))
        return NearDuplicates(edges, len(candidates))

    def _signature(self, shingle_set: frozenset[str]) -> MinHash:
        signature = MinHash(num_perm=self._settings.num_perm, seed=self._settings.minhash_seed)
        signature.update_batch([shingle.encode() for shingle in sorted(shingle_set)])
        return signature

    def _index(self, signatures: dict[str, MinHash]) -> MinHashLSH:
        recall = self._settings.lsh_recall_weight
        index = MinHashLSH(
            threshold=self._settings.jaccard_threshold,
            num_perm=self._settings.num_perm,
            weights=(1 - recall, recall),
        )
        for identifier, signature in signatures.items():
            index.insert(identifier, signature)
        return index

    def _similar(self, first: frozenset[str], second: frozenset[str]) -> bool:
        return len(first & second) / len(first | second) >= self._settings.jaccard_threshold
