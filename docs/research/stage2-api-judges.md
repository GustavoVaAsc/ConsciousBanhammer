# Research: best two API Stage-2 judges for a moderation labeling cascade, $20 budget, late Nov 2026
Answer: Pair OpenAI gpt-6-luna (or gpt-5.6-luna) with Google Gemini 3.5 Flash-Lite (or 3.8 Flash if you want a stronger model). Use Anthropic Claude Haiku 5.5 as the third option or swap-in. All three cost under about $3 for 3,000 calls with the Batch API, so the $20 cap is not binding. Refusal and retention behaviour is NOT verified (see Missing evidence), so run a small refusal pilot before choosing.

## Findings
- **F1 [high] OpenAI cheap tier (Standard / cached / output per 1M tokens, batch is 50% off all of them):** gpt-5-nano $0.05/$0.005/$0.40; gpt-6-luna $0.10/$0.01/$0.50; gpt-5.6-luna $0.20/$0.02/$1.20; gpt-5-mini $0.25/$0.025/$2.00; gpt-5.4-nano $0.20/$0.02/$1.25. Batch rates for gpt-6-luna are $0.05 in / $0.25 out.
  Sources: [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
  Support: direct (read through a small-model extract; date not shown)
- **F2 [high] Anthropic:** Claude Haiku 5.5 costs $0.10 in / $0.01 cache read / $0.50 out per 1M tokens for prompts up to 100k. Batch is 50% off ($0.05 in / $0.25 out). Haiku 4.5 is $1/$5. Sonnet 5.5 is $2/$10, batch $1/$5, cache read $0.10. Claude 4.7+ tokenizers produce about 30% more tokens for the same text, so budget about 1.3x the token counts.
  Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
  Support: direct
- **F3 [high] Google (paid tier):** Gemini 3.5 Flash-Lite is $0.30 in / $0.03 cached (plus hourly storage) / $2.50 out, batch $0.15/$1.25. Gemini 3.1 Flash-Lite is $0.25/$1.50, batch $0.125/$0.75. Gemini 3.8 Flash is $0.75 in / $3.75 out through 2026-12-31, then $1.50/$7.50, with batch at half price. Output prices include thinking tokens. The free tier may use content to improve Google products. The paid tier does not. gemini-2.5-flash is auto-routed to 3.5 Flash-Lite from 2026-12-01.
  Sources: [Gemini pricing](https://aistudio.google.com/docs/pricing) (moved from ai.google.dev, cross-host 301; the extract covered only the first 100k of 118k characters)
  Support: direct
- **F4 [medium] Cost for 3,000 calls (6.0M input tokens, 0.3M output tokens, no caching credit, thinking tokens excluded):**
  - gpt-6-luna: standard $0.75, batch about $0.38.
  - Haiku 5.5: standard $0.75, batch about $0.38. With the 1.3x tokenizer, about $0.50.
  - Gemini 3.5 Flash-Lite: standard $2.55, batch about $1.28.
  - Gemini 3.8 Flash: batch about $2.8 (about $5.6 standard).
  - Sonnet 5.5: batch about $7.50.
  - Any pair of these totals under $10 with batch. Caching would cut input cost further, but is minor at this scale.
  Sources: F1-F3
  Support: interpretation (my arithmetic)
- **F5 [medium] Recommended pair, from different families:** (A) gpt-6-luna and Gemini 3.5 Flash-Lite (or 3.8 Flash). Both have native batch, both support JSON-schema structured output as I recall (not re-verified), and Google exposes an OpenAI-compatible endpoint as I recall (not verified here). (B) Claude Haiku 5.5 is the cheapest at similar quality tier and a good third family. It needs the native Anthropic SDK for batch, which breaks the "openai SDK only" preference. Spare budget (more than $15) allows upgrading one judge to Sonnet 5.5 or Gemini 3.8 Flash for the hard cases.
  Support: inference; confidence medium-low, because quality on Spanish/English toxic chat was not benchmarked.
- F6 [low] Mistral, DeepSeek and OpenAI-compatible hosts were not checked (out of quick depth). They are unlikely to be needed given the costs above.

## Contradictions
none found. Prices come from single vendor pages each.

## Missing evidence
- Refusal behaviour on toxic/hateful content for each candidate (not looked up). Run about 50 real pairs through each candidate before committing.
- Data retention and training terms for the OpenAI and Anthropic APIs (not fetched). Only the Google paid vs free-tier distinction is verified. Anthropic and OpenAI API data is generally not trained on by default, but I did not confirm that. Retention and abuse-monitoring windows are unknown here.
- Structured-output and batch-endpoint details per vendor, and the OpenAI-compat layer for Gemini and Anthropic, were not verified.
- Whether gpt-6-luna and the Haiku 5.5 / Flash-Lite thinking tokens inflate output cost. Gemini bills thinking as output.

## Sources
- Kept: OpenAI API pricing (https://developers.openai.com/api/docs/pricing): vendor page.
- Kept: Anthropic pricing (https://platform.claude.com/docs/en/about-claude/pricing): vendor page, full table.
- Kept: Gemini pricing (https://aistudio.google.com/docs/pricing): vendor page.
- Rejected: openai.com/api/pricing: had no per-token prices.

## Not covered
Mistral, DeepSeek, OpenAI-compatible hosts. Check against the code: the `openai` SDK base_url support for the second vendor, JSON-schema strictness, and the batch-file format per vendor.

## Open questions
- [unverified] Refusal rates on toxic Spanish/English content for gpt-6-luna, Gemini 3.5 Flash-Lite and Haiku 5.5.
- [unverified] Data-retention and no-training terms for submitted content at OpenAI and Anthropic.
- [decision] Accept a non-OpenAI-SDK batch client (Anthropic), or keep to OpenAI plus Google (OpenAI-compatible).
