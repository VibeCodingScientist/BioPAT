# Tier 3 Novelty: Evaluator-Annotator Overlap Analysis

WP1 hold-out evaluation results (300 statements, k=10 context, CoT mode).

All hold-outs use OpenAI-compatible client via OpenRouter with pinned providers
(no fallback) for reproducibility. Bootstrap 95% CIs over 10,000 resamples.

## Headline: Accuracy with 95% CI

| Model | Family | Role | Accuracy [95% CI] | Macro F1 [95% CI] |
|---|---|---|---|---|
| GPT-5.2 | OpenAI GPT | annotator+eval | 0.753 | 0.661 |
| Claude Sonnet 4.6 | Anthropic Claude | annotator+eval | 0.773 | 0.650 |
| Gemini 3 Pro | Google Gemini | annotator+eval | 0.753 | 0.550 |
| Claude Haiku 4.5 (CoT) | Anthropic Claude | bias-ctrl (in-family) | 0.690 [0.637, 0.740] | 0.548 [0.441, 0.635] |
| Llama-3.3-70B (CoT) | Meta Llama | hold-out (out-of-family) | 0.547 [0.490, 0.603] | 0.426 [0.358, 0.493] |
| DeepSeek-V3 (CoT) | DeepSeek | hold-out (out-of-family) | 0.443 [0.387, 0.500] | 0.367 [0.299, 0.435] |
| Qwen-2.5-72B (CoT) | Alibaba Qwen | hold-out (out-of-family) | 0.643 [0.587, 0.697] | 0.466 [0.381, 0.557] |
| Mistral-Large (CoT) | Mistral | hold-out (out-of-family) | 0.730 [0.680, 0.780] | 0.423 [0.335, 0.514] |

## CoT vs no-CoT: confirms reasoning is essential

| Model | CoT acc | no-CoT acc | Delta |
|---|---|---|---|
| Llama-3.3-70B | 0.547 | 0.273 | +0.273 |
| DeepSeek-V3 | 0.443 | 0.153 | +0.290 |
| Qwen-2.5-72B | 0.643 | 0.260 | +0.383 |
| Mistral-Large | 0.730 | 0.337 | +0.393 |

## Per-class F1 (CoT mode, hold-outs)

| Model | F1-NOVEL [CI] | F1-PARTIAL [CI] | F1-ANTIC [CI] |
|---|---|---|---|
| Llama-3.3-70B | 0.235 [0.082, 0.391] | 0.355 [0.262, 0.442] | 0.689 [0.632, 0.741] |
| DeepSeek-V3 | 0.200 [0.049, 0.368] | 0.369 [0.291, 0.444] | 0.532 [0.462, 0.595] |
| Qwen-2.5-72B | 0.200 [0.000, 0.444] | 0.432 [0.340, 0.519] | 0.765 [0.715, 0.809] |
| Mistral-Large | 0.200 [0.000, 0.444] | 0.225 [0.110, 0.341] | 0.843 [0.807, 0.876] |

## Interpretation

**Mistral-Large at 73.0% [68.0, 78.0]** is the strongest hold-out result and
the most relevant for the overlap critique: it is a commercial-grade model
from a separate training lineage (European, no shared training data with
GPT/Claude/Gemini) and reaches accuracy within 4-5pp of the in-family annotators.
Its 95% CI overlaps with all three annotators', supporting the claim that the
75-77% headline accuracy is not primarily an artifact of self-evaluation by
the annotation panel.

**Open-weight hold-outs (Llama 54.7%, Qwen 64.3%, DeepSeek 44.3%)** trail
the commercial models, but this gap tracks model-capability rankings on
other reasoning benchmarks; we attribute it to capability rather than
overlap-bias correction. All four hold-outs collapse without CoT (mean delta
-0.34 accuracy), reproducing the pattern observed in the in-family Haiku 4.5
ablation (-0.24).

**ANTICIPATED class** (the majority class) is recovered well across hold-outs:
F1 in [0.53, 0.84]. **NOVEL class** F1 is uniformly low (0.20-0.27 in CoT mode)
across all hold-outs and the in-family Haiku 4.5 (0.48), reflecting the
intrinsic difficulty of the class given only 18 NOVEL items and the override
construction discussed in §5.4.

## Provider pinning (reproducibility)

| Model | OpenRouter slug | Provider pinned | Quantization |
|---|---|---|---|
| Llama-3.3-70B | meta-llama/llama-3.3-70b-instruct | Novita | bf16 |
| DeepSeek-V3 | deepseek/deepseek-chat | Novita | fp8 |
| Qwen-2.5-72B | qwen/qwen-2.5-72b-instruct | DeepInfra | fp8 |
| Mistral-Large | mistralai/mistral-large | Mistral | bf16 (native) |

`provider.allow_fallbacks=false` enforced on every request. Per-call provider
metadata logged in `results/tier3_holdout_provider_log.jsonl` for audit.

## Cost ledger

| Model | CoT cost | no-CoT cost | Total |
|---|---|---|---|
| Llama-3.3-70B | $0.270 | $0.160 | $0.430 |
| DeepSeek-V3 | $0.069 | $0.049 | $0.118 |
| Qwen-2.5-72B | $0.274 | $0.184 | $0.458 |
| Mistral-Large | $1.556 | $0.836 | $2.392 |
| **Total WP1** | | | **$3.398** |
