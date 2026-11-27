from pathlib import Path

import pytest
import torch

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.embeddings import EMBEDDER_SECTION, EmbedderSettings, SentenceTransformerEmbedder
from banhammer.messages import (
    IngestReport,
    LinguaLanguageDetector,
    MessagePoolBuilder,
    Provenance,
)
from message_frames import raw_frame
from messages_params import messages_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSLATIONS = [
    (
        "Does anyone know when the next event starts tonight?",
        "¿Alguien sabe a qué hora empieza el próximo evento esta noche?",
    ),
    (
        "I really hope it stays like this for the whole weekend",
        "Ojalá se quede así durante todo el fin de semana",
    ),
    (
        "Please check the server rules before writing here",
        "Por favor revisa las reglas del servidor antes de escribir aquí",
    ),
]
UNRELATED = [
    "Great game yesterday, we almost won the match",
    "Nos vemos mañana en la estación de tren a las ocho",
]
PARAPHRASES = ["Stop posting that here, please", "Please stop posting that in this channel"]


@pytest.mark.integration
def test_real_models_group_translations_but_not_paraphrases_or_unrelated_messages() -> None:
    params = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME)
    settings = messages_settings()
    embedder_settings = params.section(EMBEDDER_SECTION, EmbedderSettings)
    device = torch.device("cpu")
    texts = [text for pair in TRANSLATIONS for text in pair] + UNRELATED + PARAPHRASES
    raw = raw_frame([{"message_id": f"m{i:02d}", "text": text} for i, text in enumerate(texts)])
    builder = MessagePoolBuilder.assemble(
        Provenance(42, settings, embedder_settings),
        SentenceTransformerEmbedder.from_hub(embedder_settings, device),
        LinguaLanguageDetector(settings.language.detector_languages),
        device,
    )
    result = builder.build(
        raw.lazy(), IngestReport(rows=raw.height, dropped_oversize=0, mentions_replaced={})
    )
    group = dict(zip(result.messages["message_id"], result.messages["dup_group_id"], strict=True))
    assert len(group) == len(texts)
    assert group["m00"] == group["m01"]
    assert group["m04"] == group["m05"]
    assert len({group[f"m{i:02d}"] for i in range(len(texts))}) == len(texts) - 3
