"""Shared LaBSE embedder and exact neighbour search (D13, D14, D15).

`embeddings` imports only `core`. `SentenceTransformerEmbedder` is exported lazily, so importing
this package does not import `sentence_transformers`.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from banhammer.embeddings.constants import EMBEDDER_SECTION
from banhammer.embeddings.errors import EmbeddingError, NeighbourSearchError
from banhammer.embeddings.neighbours import ExactNeighbourSearch, NeighbourPairs
from banhammer.embeddings.protocol import Embedder
from banhammer.embeddings.settings import EmbedderSettings

if TYPE_CHECKING:
    from banhammer.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

__all__ = [
    "EMBEDDER_SECTION",
    "Embedder",
    "EmbedderSettings",
    "EmbeddingError",
    "ExactNeighbourSearch",
    "NeighbourPairs",
    "NeighbourSearchError",
    "SentenceTransformerEmbedder",
]

_LAZY_BACKEND = "SentenceTransformerEmbedder"
_BACKEND_MODULE = "banhammer.embeddings.sentence_transformer_embedder"


def __getattr__(name: str) -> object:
    """Import `SentenceTransformerEmbedder` on first access (PEP 562)."""
    if name == _LAZY_BACKEND:
        return getattr(import_module(_BACKEND_MODULE), name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
