# Research: which open-weight LLM should replace Qwen2.5-7B-Instruct-AWQ as local Stage-1 judge (RTX 5060 Ti 16 GB, vLLM 0.34, es+en, as of 2026-11-25)
Answer: Shortlist, in order: (1) Qwen3.5-9B (Apache-2.0, 201 languages, thinking switchable off) in AWQ-4bit or FP8, the lowest-risk drop-in; (2) gpt-oss-safeguard-20b (Apache-2.0, built to take a written policy, MXFP4 about 13 GB) as the policy-conditioned candidate, with a reasoning-cost and harmony-format caveat; (3) Gemma 4 12B or E4B at 4-bit (Apache-2.0 per model cards). Spanish and judge-quality evidence is mostly vendor or secondary, so a local bake-off must decide. Several details are unverified (see Missing evidence).

## Findings
- **F1 [medium] Qwen3.5-9B is the best-fitting general replacement.** It has 9B parameters (safetensors metadata says 10B), Apache-2.0, a "February 2026" citation, 201 languages and dialects, IFEval 91.5 and MMLU-Pro 82.5. Thinking is on by default and is disabled per request with `chat_template_kwargs: {enable_thinking: False}` (the `/nothink` soft switch is not supported). Architecture is hybrid Gated DeltaNet (3 linear layers per 1 full-attention layer), so KV cache is small. That helps 8K context on 16 GB. 4-bit weights are about 5-6 GB and FP8 about 10 GB, so both fit with room for KV. Caveat: the hybrid/linear-attention layout interacts with prefix caching, and the card says to use a nightly vLLM. Check that prefix caching and guided JSON work together on 0.34.
  Sources: [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)
  Support: direct for the card facts; inference for the VRAM figures and the prefix-caching risk.
- **F2 [medium] gpt-oss-safeguard-20b is the only candidate purpose-built for "policy at inference".** It is a gpt-oss fine-tune with 21B total and 3.6B active parameters, Apache-2.0, and OpenAI says it fits 16 GB. Reasoning effort is low/medium/high. It was released as a research preview on 2025-10-29. It must be used with the harmony response format, and it returns a conclusion plus reasoning. OpenAI states it outperforms gpt-oss and gpt-5-thinking on multi-policy accuracy, and a dedicated classifier trained on tens of thousands of labels can still beat it. OpenAI calls it "time and compute-intensive". On ToxicChat it was marginally behind gpt-5-thinking and OpenAI's internal Safety Reasoner. No multilingual results were found in OpenAI's post or the card, so Spanish is unproven. It is a safety classifier by design, so refusal on toxic input is expected to be low, but this is unverified. Whether JSON-schema guided decoding is compatible with harmony and the reasoning channel on vLLM is not documented in what I read. Test it. Reasoning can be bounded (low effort) but not disabled.
  Sources: [HF card](https://huggingface.co/openai/gpt-oss-safeguard-20b), [OpenAI announcement](https://openai.com/index/introducing-gpt-oss-safeguard/), [OpenAI cookbook guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide)
  Support: direct; the refusal and structured-output points are inference.
- **F3 [medium] gpt-oss-20b MXFP4 has run under vLLM on a single 5060 Ti, but on an old version.** An arXiv paper (January 2026, vLLM 0.12, CUDA 12.9) lists GPT-OSS-20B MXFP4 at about 488 TPS on 1x 5060 Ti with API-style concurrency of 32-64. The official vLLM recipe, though, lists data-center and workstation GPUs and RTX 5090 as targets, and SM120 MXFP4 problems are documented on RTX PRO 6000 (needed custom backend flags). Nothing found confirms vLLM 0.34 on a 5060 Ti. Weights are about 13 GB, which leaves roughly 1 GB for KV on a 14.8 GB budget at 0.9 utilisation, so 8K context at low concurrency is tight. Treat as a test, not a given.
  Sources: [arXiv 2601.09527](https://arxiv.org/html/2601.09527), [vLLM gpt-oss recipe results via search](https://recipes.vllm.ai/openai/gpt-oss-20b), [NVIDIA forum SM120 report](https://forums.developer.nvidia.com/t/gpt-oss-120b-mxfp4-on-rtx-pro-6000-blackwell-max-q-sm120-full-debug-path-what-was-actually-broken-and-what-finally-worked/365333)
  Support: direct for the paper; interpretation for the fit.
- **F4 [medium] Gemma 4 (2026) is Apache-2.0 per the model cards, not the old Gemma terms.** Sizes: E2B, E4B (4.5B effective, 8B with embeddings, 128K context), 12B "Unified" (11.95B), 26B-A4B MoE (25.2B total, 3.8B active) and 31B dense. It is pre-trained on 140+ languages with out-of-the-box support for 35+. Spanish is not named in what I read, but it is very likely among them. Thinking is toggled with `<|think|>` in the system prompt, so off by default. The 26B-A4B has MMLU-Pro 82.6, but its 4-bit weights are about 14-15 GB (inference from parameter count), which leaves no KV room on 14.8 GB. Drop it, and the 31B too. The 12B (4-bit about 7-8 GB) and E4B fit. An Unsloth gemma-4-12b NVFP4 checkpoint exists (needs vLLM 0.25+). Artificial Analysis (April 2026) ranks Gemma 4 31B and 26B-A4B below Qwen3.5 27B and 35B-A3B on its index. In the January paper's vLLM 0.12 test, Gemma3-12B single-GPU runs on the 5060 Ti failed at 8K RAG. Check for a repeat on 0.34. Gemma has a history of refusing less than Llama, but I have no source on toxic-input refusal.
  Sources: [Gemma 4 26B-A4B card](https://huggingface.co/google/gemma-4-26B-A4B-it), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), [Unsloth 12B NVFP4](https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4), [Artificial Analysis sub-32B](https://artificialanalysis.ai/articles/sub-32b-open-weights), [arXiv 2601.09527](https://arxiv.org/html/2601.09527)
  Support: direct for the card facts; inference for footprints; the 12B/E4B numbers come from a search snippet, not a fetched card.
- **F5 [medium] Larger Qwen options (Qwen3.5-27B, Qwen3.6-27B, 35B-A3B) do not fit comfortably.** Qwen3.6-27B (2026-04-22) and Qwen3.6-35B-A3B are Apache-2.0, but a 27B at 4-bit is about 14-16 GB, which is too tight with KV and the 1.2 GB desktop overhead. A secondary report claims Qwen3.6-27B reaches 40 tok/s on a 5060 Ti 16 GB, presumably on llama.cpp with heavy quantization and not vLLM. Do not use for a vLLM 8K-context judge. I found no Qwen3.6 small model (9B-class) in the results, so Qwen3.5-9B is the newest small Qwen I could confirm.
  Sources: [NYU RITS write-up](https://rits.shanghai.nyu.edu/ai/qwen3-6-27b-a-dense-27b-model-that-beats-a-397b-moe-on-coding), [AI Weekly](https://aiweekly.co/alerts/qwen36-27b-hits-40-toks-on-single-rtx-5060-ti-16gb)
  Support: secondary (low-medium); fit is inference.
- **F6 [medium] Quantization on sm_120 under vLLM: AWQ (awq_marlin) is the best-evidenced; NVFP4 and MXFP4 run on dense models; FP8, GPTQ and bitsandbytes are not confirmed here.**
  - A vLLM forum field report (vLLM 0.17.2rc1, WSL2) says AWQ on a 5060 Ti works with `awq_marlin` plus `TRITON_ATTN`. The same thread says FlashAttention lacks SM120, and that GPTQ and BnB were broken at that time. I could not open the page (404 on fetch), so this is search-snippet evidence only.
  - The arXiv paper ran NVFP4, W4A16 (AWQ) and MXFP4 on the 5060 Ti under vLLM 0.12. Qwen3-8B NVFP4 lost about 2.2 MMLU, 1.9 GSM8K and 2.0 HellaSwag points versus BF16. Gemma3-12B NVFP4 lost about 4.1 MMLU and 3.6 GSM8K points. Those quality numbers were measured on a reference GPU, not the 5060 Ti.
  - NVFP4 MoE models had SM120 backend failures in vLLM 0.15 (issue 33416). Dense NVFP4 is safer. Some SM120 reports show a Marlin FP4 weight-only fallback.
  - vLLM's quantization hardware table has no Blackwell column, so it confirms nothing for SM120. FP8 W8A8 is listed for Ada and Hopper only, while FP8 via Marlin (weight-only) covers Ampere and up. Whether FP8 is native on SM120 in 0.34 is not confirmed. SM120 has FP8 tensor cores in hardware, so I expect it to work (inference). `VLLM_STRICT_QUANT_SCHEME=1` stops silent fallbacks.
  - AWQ 4-bit typically costs about 1-3 points on benchmarks, and FP8 under 1 point (general knowledge, not from a fetched source; low confidence).
  Sources: [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/), [vLLM forum field report](https://discuss.vllm.ai/t/field-report-awq-on-rtx-5060-ti-sm-120-blackwell-awq-marlin-triton-attn-working/2463), [arXiv 2601.09527](https://arxiv.org/html/2601.09527), [vLLM issue 33416](https://github.com/vllm-project/vllm/issues/33416)
  Support: direct for the paper and docs; interpretation for the rest. The notes are dated against vLLM 0.12-0.17, and 0.34 is much newer.
- **F7 [low] Other families are weaker fits.** Mistral Small 3.2 (24B) needs about 18 GB with overhead, per a blog, so it does not fit. Mistral Small 4 is 119B total with 6B active, so it does not fit either. I did not research Ministral, Llama 3.x/4, Phi or Nemotron individually. Artificial Analysis puts gpt-oss-20B and Nemotron 3 Nano 30B-A3B at about 24 on its index, below Qwen3.5 and Gemma 4. Llama community licence and older Gemma terms are not needed here, as the shortlist is all Apache-2.0.
  Sources: [Codersera/other blogs via search](https://dev.to/jovan_chan_9500711396d4e6/open-source-llm-shootout-2026-qwen36-vs-gemma-4-vs-llama-4-vs-glm-51-vs-deepseek-v4-which-fits-56nn), [Artificial Analysis](https://artificialanalysis.ai/articles/sub-32b-open-weights)
  Support: secondary.

## Ranked shortlist
1. **Qwen3.5-9B (AWQ 4-bit, or FP8 if it works natively)**, with thinking off. Lowest risk, same family and tooling as the current judge, Apache-2.0.
2. **gpt-oss-safeguard-20b (MXFP4), reasoning effort low.** Best conceptual match to rule-conditioned labeling, but costly per call, harmony-format only, Spanish unproven, tight on VRAM.
3. **Gemma 4 12B (4-bit AWQ or NVFP4), or E4B as a fast fallback**, thinking off. Strong multilingual pre-training claim, Apache-2.0.

## Bake-off should measure
- Agreement with a human-labelled gold set of a few hundred items, split es / en / Latin American slang. Report V0-V3 accuracy, severity MAE, rule-id precision and recall, and Cohen's kappa against humans and against Qwen2.5 as the baseline.
- Refusal and empty-output rate on the most toxic items, per model.
- JSON-schema validity under vLLM guided decoding. For safeguard, check that structured outputs work alongside the harmony reasoning channel.
- Whether prefix caching hits on the shared ruleset prefix (watch the vLLM prefix-cache hit-rate metric), especially for hybrid-attention Qwen3.5.
- Throughput and cost per call: output tokens per call (reasoning tokens for safeguard, and thinking on versus off for Qwen3.5 and Gemma 4), tail latency and peak VRAM at 8K context. Extrapolate to 50K calls.
- Quantization A/B: AWQ vs FP8 vs NVFP4 on the same model on the gold set, to price the quality loss.
- Run-to-run stability at temperature 0 on the same items, and sensitivity to a rewording of the ruleset.

## Contradictions
- Gemma 4's size list differs by source. One blog lists a 12B that "needs up to 16 GB", while the Google model card (via search) lists E2B/E4B/12B/26B-A4B/31B. I used the model card.
- The Qwen3.5-9B card says "sparse Mixture-of-Experts" in its highlights but describes a dense FFN in the overview. I treated it as dense, with 9B parameters.

## Missing evidence
- No measured vLLM 0.34 results on a 5060 Ti for any candidate. The evidence is vLLM 0.12 (paper) and 0.17 (forum, unfetched).
- No Spanish or Latin American slang benchmarks for gpt-oss-safeguard, Gemma 4 or Qwen3.5 beyond language-count claims.
- No source on toxic-input refusal for any candidate. Not checked.
- Whether FP8 and bitsandbytes run natively on sm_120 in 0.34, and whether prefix caching is supported for Qwen3.5's hybrid layers. Check the vLLM 0.34 release notes.
- Official Gemma 4 12B/E4B cards were not fetched, so those figures rest on a search summary. Gemma 4 release date was not found (the 26B card cites a "Jul 2" technical report, year unstated).
- Qwen3.5-9B quantized VRAM figures are estimates, not measured.

## Sources
- Kept: [gpt-oss-safeguard-20b card](https://huggingface.co/openai/gpt-oss-safeguard-20b): primary, licence and specs.
- Kept: [OpenAI announcement](https://openai.com/index/introducing-gpt-oss-safeguard/): primary, evaluations and limits.
- Kept: [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B): primary.
- Kept: [Gemma 4 26B-A4B card](https://huggingface.co/google/gemma-4-26B-A4B-it): primary.
- Kept: [arXiv 2601.09527](https://arxiv.org/html/2601.09527): the only measured consumer-Blackwell vLLM data (old vLLM).
- Kept: [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/): no SM120 column.
- Kept: [Artificial Analysis sub-32B](https://artificialanalysis.ai/articles/sub-32b-open-weights): dated 2026-04-13, third-party index.
- Kept, low: the vLLM forum field report and GitHub issue 33416 (search snippets, page not opened), NYU RITS and AI Weekly (secondary), dev.to blog (secondary).
- Rejected: SEO comparison blogs (tech-insider, mindstudio, codersera and similar) as support for any decision claim: unsourced and inconsistent.

## Not covered
- Check against the code and Docker setup: the vLLM image tag and flags (attention backend, `--enable-prefix-caching`, structured-outputs backend), `--max-model-len 8192`, the gpu-memory-utilization setting (about 14.8 GB usable), and the existing prompt and schema (does it use `response_format` JSON schema, and does the harmony format break it for safeguard).
- Not researched: Ministral, Llama 3.x/4, Phi-4 and Nemotron in depth, plus any other policy-conditioned classifiers (for example Llama Guard or ShieldGemma, which do not take free-form rules the same way).
- Hardware reality check: run `vllm bench` yourself.

## Open questions
- [unverified] Does vLLM 0.34 run gpt-oss-safeguard-20b (MXFP4) on sm_120 with enough KV cache for 8K context, and does JSON-schema guided decoding work with its harmony reasoning output?
- [unverified] Does prefix caching work with Qwen3.5's hybrid Gated DeltaNet layers in vLLM 0.34?
- [unverified] Spanish and slang quality, and refusal rate on toxic input, for all three shortlisted models.
- [decision] Whether to accept the extra per-call cost of a reasoning judge (safeguard) for possibly better rule adherence, or restrict the bake-off to non-reasoning modes.
