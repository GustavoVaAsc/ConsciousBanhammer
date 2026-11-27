# Research: Which LLMs are reliable judges for rule-conditioned moderation (esp. Spanish), and which protocol choices matter?
Answer: The evidence is thin and mostly indirect. Small open models (4-8B) are near the trivial baseline on rule-conditioned moderation across communities, and frontier models are only moderately better. Reasoning toggles do not reliably help, and LLM labels agree only moderately with humans on Spanish and English hate speech. The design (a strong API pair for the verdict, a calibration gate on human labels, a Stage 1 upgrade) is sensible, but no source ranks late-2026 models for this task. Pick models by running them on the ~200 pilot items.

## Findings
- **F1 [medium] On a cross-community rule-violation benchmark (PluRule, 9 languages), small open models are near chance and reasoning variants did worse.**
  Qwen3-VL-4B and 8B Instruct scored 48.4% and 49.8% with full context. The trivial always-"no violation" baseline is 50%. The 30B Instruct reached 52.3%. GPT-5.2 high reasoning reached about 57.6%. Thinking variants scored lower than Instruct at every size (8B: 44.9 vs 49.8). GPT-5.2 low and high effort did not differ significantly. Rule types needing context (low-effort, relevance, evidence) were below baseline, while civility-type rules reached about 69%. Caveats: the benchmark is hard and noisy (May 2026), it uses Reddit rather than Discord, and the models are VL variants. The per-language (Spanish) numbers were not readable in the extract.
  Sources: [PluRule arXiv 2605.17187](https://arxiv.org/html/2605.17187v1)
  Support: direct

- **F2 [medium] LLM labels agree only moderately with humans on Spanish and English hate speech, and the result depends on the model.**
  Piot et al. (Dec 2025) tested 4-bit Llama-3.1-8B, Mistral-Nemo-12B and R1-Distill-Llama-8B on DETESTS (Spanish) and HateXplain/MHS (English). Only Nemo reached kappa of about 0.30-0.34 against the human majority. Llama and DeepSeek were near 0 and DeepSeek was negative on MHS. Human inter-annotator kappa was 0.84 on DETESTS and about 0.47 on the English sets. Model rankings of classifiers were preserved for Nemo (Kendall tau about 0.95) but inverted for DeepSeek on MHS. The authors conclude LLMs are not reliable substitutes for human annotators, though they can proxy system rankings. They advise checking correlation on the target dataset. This supports a calibration gate and a model-specific check. It also suggests that an R1-distill 8B judge is a poor choice, and that Nemo-class 12B models may be better at that size (interpretation).
  Sources: [Piot et al., arXiv 2512.09662](https://arxiv.org/html/2512.09662v1)
  Support: direct (the Nemo and DeepSeek interpretation is mine)

- **F3 [medium] gpt-oss-safeguard (20b/120b, Oct 2025) is the only open policy-in-prompt guard with a reasoning design, but its evidence is vendor-only.**
  OpenAI's report gives multi-policy accuracy of 46.3% (120b), 43.6% (20b) and 43.2% for gpt-5-thinking. It gives F1 of about 82.9% on the OpenAI Moderation set and about 79-80% on ToxicChat. The 20b model nearly matches the 120b. OpenAI says classifiers trained on tens of thousands of labels can still beat it. The multi-policy accuracy being below 50% shows that the task is hard even for the vendor's own model. The numbers came through a search summary of a partly garbled PDF, so verify them at the source before quoting. The 20b model is a plausible Stage 1 candidate. Whether it fits a 16 GB GPU is outside my angle (MXFP4 weights are reportedly about 16 GB, unchecked). Nothing in what I read tests Spanish.
  Sources: [gpt-oss-safeguard technical report](https://openai.com/index/gpt-oss-safeguard-technical-report), [launch post](https://openai.com/index/introducing-gpt-oss-safeguard/)
  Support: direct for the report's numbers, inference for suitability

- **F4 [medium] Reasoning evidence is mixed.**
  Pro: guard-model papers (Safety Through Reasoning, Findings of EMNLP 2025; ThinkGuard, ACL Findings 2025) report better generalization to custom policies and better sample efficiency. Dual-mode models with short traces keep accuracy at lower cost. These compare trained reasoning guards against label-only baselines, so training differences are confounded. Con: PluRule (F1) found Thinking worse than Instruct at 4B-30B. LeanGuard argues moderation labeling does not need reasoning (I did not verify its claims). Practical implication: do not assume thinking helps. Use thinking mode only if it wins on the pilot items (inference).
  Sources: [Safety Through Reasoning](https://arxiv.org/html/2505.20087v1), [ThinkGuard](https://arxiv.org/html/2502.13458v1), [PluRule](https://arxiv.org/html/2605.17187v1), [LeanGuard](https://arxiv.org/pdf/2606.26686)
  Support: direct, with the conflict noted

- **F5 [medium] Fixed-taxonomy guards (Llama Guard, ShieldGemma) are weaker outside English and inflexible across policies.**
  Papers report weaker non-English performance, with Granite Guardian relatively better across languages (CultureGuard). Qwen3Guard claims 119 languages and state-of-the-art results, but that is vendor-reported and I did not extract a Spanish comparison. I found no head-to-head Spanish comparison on custom rulesets. For 8-20 community rules with graded severity, a general instruct or reasoning model with the policy in the prompt is the closer fit than a fixed-taxonomy guard (inference).
  Sources: [CultureGuard](https://arxiv.org/html/2508.01710v1), [Qwen3Guard report](https://arxiv.org/pdf/2510.14276)
  Support: interpretation

- **F6 [medium] Holistic or graded toxicity judgments agree less with humans than binary accuracy suggests.**
  RTP-LX (28 languages, AAAI 2025) found acceptable accuracy but low agreement with human judges when scoring toxicity holistically. A 6-language study reports high LLM-human agreement for GPT-4 in safety evaluation (Multilingual Alignment Prism). The setups differ, so this is a mild contradiction. Implication: measure agreement on the severity scale (weighted kappa) and on the cited rule, not only on the binary verdict (inference). I found no direct study of ordinal vs binary reliability.
  Sources: [RTP-LX](https://arxiv.org/html/2404.14397v2), [Multilingual Alignment Prism](https://arxiv.org/pdf/2406.18682)
  Support: direct for the two papers, inference for the implication

- **F7 [medium] Position bias is real, varies by judge and task, and is not random. Repeated-query agreement is a calibration-free confidence signal.**
  "Judging the Judges" measures position bias for pairwise comparison (the pairwise setup is a limit on how far it transfers). The 2026 AACL paper argues that repeated queries (and order swaps) catch failure modes that single-shot consistency misses. Other work it cites reports that judges are overconfident and have low intra-rater reliability. Your option shuffling is consistent with this. Add 2-3 resamples or reorderings to the sampling and treat disagreement as low confidence. Do not use verbalized confidence as a gate (inference). I did not open the AACL paper itself.
  Sources: [Judging the Judges](https://arxiv.org/html/2406.07791v9), [AACL 2026 long.207](https://aclanthology.org/2026.aacl-long.207.pdf)
  Support: interpretation

- **F8 [low] Inter-judge agreement as a quality signal.**
  Agreement between two judges is a reasonable filter, but F2 shows models can share systematic errors or differ widely (LLM-LLM kappa only 0.17-0.24 there). Agreement does not prove correctness, so measure the error rate among agreed items on the human pilot (inference). Source: [Piot et al.](https://arxiv.org/html/2512.09662v1)

- **F9 [low] Model recommendations for late 2026 (inference, no direct benchmark).**
  Stage 1: replace Qwen2.5-7B with a current 14-20B-class open model, such as gpt-oss-safeguard-20b, Qwen3 or Mistral Nemo-class, if the 16 GB limit allows. F1 and F2 say 4-8B models are near baseline, so use Stage 1 only as a tiebreaker or prefilter. Prefer non-thinking mode unless the pilot shows otherwise. Stage 2: two frontier or near-frontier API judges from different vendors (to reduce shared bias), chosen by pilot agreement per language under the $20 budget. GPT-5.2 is the only frontier model with a measured number in what I read (F1).

## Contradictions
- Reasoning helps guard models (Safety Through Reasoning, ThinkGuard) but hurts open instruct models on PluRule. The setups differ: trained guards vs general models, and safety vs community rules.
- LLM-human agreement is "high" in Multilingual Alignment Prism (GPT-4, safety) but low in RTP-LX (holistic toxicity) and Piot et al. (hate speech).

## Missing evidence
- No source found with Spanish-specific rule-conditioned moderation results for named current models (PluRule's per-language figure was unreadable in the extract).
- No controlled study of ordinal vs binary label reliability, verbalized-confidence calibration for moderation, or position bias in single-item rule classification (as opposed to pairwise).
- Nothing on Discord-style message-plus-parent context, or on models newer than GPT-5.2 and Qwen3.
- gpt-oss-safeguard figures are vendor-only and I did not read the PDF directly. Qwen3Guard, ThinkGuard and similar claims are also vendor- or author-reported.

## Sources
- Kept: PluRule (arxiv.org/html/2605.17187v1, May 2026): closest benchmark, fetched.
- Kept: Piot et al. (arxiv.org/html/2512.09662v1, Dec 2025): Spanish and English, fetched.
- Kept: gpt-oss-safeguard report (openai.com, Oct 2025): vendor, search-summary only.
- Kept, from search results only: Safety Through Reasoning, ThinkGuard, LeanGuard, CultureGuard, Qwen3Guard, RTP-LX, Multilingual Alignment Prism, Judging the Judges, AACL 2026 long.207.
- Not used: Kumar et al. "Watch Your Language" (arXiv 2309.14517, GPT-3.5 on subreddit rules, median accuracy 64%) and "Toxicity Detection is NOT all you Need" (arXiv 2311.07879). Both predate the relevant models. They are background showing that rule-specific moderation is hard.
- Rejected: SEO and explainer pages (Analytics Vidhya, personal-site notes, Wikipedia on LLM-as-a-Judge) as support for any claim.

## Not covered
- Hardware and quantization fit (16 GB) and API pricing against the $20 budget.
- Check against the code: that the prompt's rule-citation output and V0-V3 mapping can be scored per rule, and how tie-breaking by Stage 1 (D29) is evaluated. Check also how many pilot items per language there are (about 200 gives wide confidence intervals).

## Open questions
- [unverified] gpt-oss-safeguard accuracy and F1 figures were read from a search summary, not the PDF.
- [unverified] PluRule per-language (Spanish) accuracy was not obtained.
- [decision] Whether to spend pilot budget on a thinking vs non-thinking comparison for Stage 1 and Stage 2, given the mixed evidence in F4.
