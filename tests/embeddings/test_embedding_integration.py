from pathlib import Path

import numpy as np
import pytest
import torch

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.embeddings import EMBEDDER_SECTION, EmbedderSettings, SentenceTransformerEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_labse_at_the_pinned_revision_embeds_translations_close_together() -> None:
    settings = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(EMBEDDER_SECTION, EmbedderSettings)
    embedder = SentenceTransformerEmbedder.from_hub(settings, torch.device("cpu"))
    vectors = embedder.encode(["You are an idiot", "Eres un idiota", "The weather is nice today"])
    assert vectors.shape == (3, 768)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
