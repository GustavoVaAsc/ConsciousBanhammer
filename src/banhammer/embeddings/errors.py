"""Exceptions of the embeddings subpackage."""

from banhammer.core.errors import BanhammerError


class NeighbourSearchError(BanhammerError):
    """The neighbour search received an embedding matrix or a parameter it cannot work with."""


class EmbeddingError(BanhammerError):
    """The embedding model cannot report its output dimension."""
