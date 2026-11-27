from pathlib import Path

import pytest
import torch

from banhammer.core.constants import PARAMS_FILE_NAME
from banhammer.core.settings import ParamsFile
from banhammer.rules import RULES_SECTION, CrossEncoderEntailmentScorer, RulesSettings

REPO_ROOT = Path(__file__).resolve().parents[2]

PARAPHRASES = [
    ("No se permiten insultos.", "Insults are not allowed."),
    ("Spam is forbidden.", "Está prohibido el spam."),
    ("Se prohíbe publicar publicidad no solicitada.", "Unsolicited advertising may not be posted."),
    ("Do not insult other members.", "No insultes a otros miembros."),
]
UNRELATED = [
    ("No insults.", "Posting memes is allowed."),
    ("Está prohibido el spam.", "Users may advertise freely."),
]
ONE_WAY = ("Be respectful and do not insult other members.", "Do not insult other members.")


@pytest.mark.integration
def test_the_pinned_nli_model_scores_mutual_and_one_way_entailment_across_languages() -> None:
    settings = ParamsFile(REPO_ROOT / PARAMS_FILE_NAME).section(RULES_SECTION, RulesSettings)
    scorer = CrossEncoderEntailmentScorer.from_hub(settings.nli, torch.device("cpu"), torch.float32)

    def both_ways(pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        premises, hypotheses = [a for a, _ in pairs], [b for _, b in pairs]
        forward = scorer.entailment_probabilities(premises, hypotheses).tolist()
        backward = scorer.entailment_probabilities(hypotheses, premises).tolist()
        return list(zip(forward, backward, strict=True))

    for forward, backward in both_ways(PARAPHRASES):
        assert min(forward, backward) >= settings.entail_high, (forward, backward)
    for forward, backward in both_ways(UNRELATED):
        assert max(forward, backward) <= settings.entail_low, (forward, backward)
    ((forward, backward),) = both_ways([ONE_WAY])
    assert forward >= settings.entail_high
    assert backward < settings.entail_high
