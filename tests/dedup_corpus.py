import numpy as np
import torch

from banhammer.embeddings import ExactNeighbourSearch
from banhammer.messages import DedupSettings
from embedder_double import FakeEmbedder

CPU = torch.device("cpu")

E1 = "the weather is lovely today my friend, I really hope it stays like this for the whole weekend"
E1_SHOUTED = (
    "THE WEATHER is lovely today my friend, I really hope it stays like this for the whole weekend"
)
E1_EDITED = (
    "the weather is lovely today, my friend, I really hope it stays like this for the whole weekend"
)
E1_PARAPHRASE = "what a beautiful day we are having"
E2 = "I will see you at the station tomorrow"
E3 = "nothing in common with the others here"
S1 = "hace un tiempo precioso hoy mi amigo, ojalá se quede así durante todo el fin de semana"
S1_EDITED = (
    "hace un tiempo precioso hoy, mi amigo, ojalá se quede así durante todo el fin de semana"
)
S2 = "te veré en la estación mañana temprano"
S3 = "completamente distinto a todo lo anterior"

VECTORS = {
    E1: [1, 0, 0, 0],
    E1_SHOUTED: [1, 0, 0, 0],
    E1_EDITED: [1, 0, 0, 0],
    E1_PARAPHRASE: [0.95, 0.31, 0, 0],
    E2: [0, 1, 0, 0],
    E3: [0, 0, 1, 0],
    S1: [0.97, 0, 0, 0.24],
    S1_EDITED: [0.97, 0, 0, 0.24],
    S2: [0, 0.9, 0.43, 0],
    S3: [0, 0, 0, 1],
}


def embedder() -> FakeEmbedder:
    return FakeEmbedder(VECTORS)


def search(k: int = 3, floor: float = 0.7) -> ExactNeighbourSearch:
    return ExactNeighbourSearch(k=k, chunk_size=4, device=CPU, min_score=floor)


def dedup_settings(**overrides: object) -> DedupSettings:
    fields: dict[str, object] = {
        "shingle_size": 5,
        "num_perm": 128,
        "minhash_seed": 1,
        "lsh_recall_weight": 0.95,
        "jaccard_threshold": 0.8,
        "translation_threshold": 0.85,
        "candidate_floor": 0.7,
        "k": 3,
        "chunk_size": 4,
        "max_group_size_warning": 50,
    }
    return DedupSettings.model_validate(fields | overrides)


def cosine(first: str, second: str) -> float:
    a, b = np.asarray(VECTORS[first], float), np.asarray(VECTORS[second], float)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
