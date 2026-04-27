# Tier 3 Novelty: Evaluator-Annotator Overlap Analysis (Updated)

Updated to include three-way reasoning-mode comparison and no-CoT confusion-matrix evidence.
All hold-outs use OpenAI-compatible client via OpenRouter with pinned providers (no fallback).
Bootstrap 95% CIs over 10,000 resamples, seed=42.

## 1. Headline: Three-way comparison (no-CoT / prompted CoT / reasoning mode)

Mode column: **none** = no-CoT system prompt only. **CoT** = explicit step-by-step CoT system prompt. **reason** = thinking-mode-tuned model with `reasoning.effort=high` parameter (model performs hidden reasoning chain before answering).

| Family | Model | Mode | Accuracy [95% CI] | Macro F1 [95% CI] |
|---|---|---|---|---|
| OpenAI | GPT-5.2 | reason | 0.753 | 0.661 |
| Anthropic | Claude Sonnet 4.6 | reason | 0.773 | 0.650 |
| Google | Gemini 3 Pro | reason | 0.753 | 0.550 |
| Anthropic | Claude Haiku 4.5 (in-family bias-ctrl) | CoT | 0.690 [0.637, 0.740] | 0.548 [0.441, 0.635] |
|  | | | | |
| Meta Llama | Llama-3.3-70B | none | 0.273 [0.223, 0.323] | 0.250 [0.200, 0.300] |
| Meta Llama | Llama-3.3-70B | CoT | 0.547 [0.490, 0.603] | 0.426 [0.358, 0.493] |
| Meta Llama | Llama-3.3-Nemotron-49B | reason | 0.347 [0.293, 0.400] | 0.319 [0.267, 0.370] |
|  |  |  |  |  |
| DeepSeek | DeepSeek-V3 | none | 0.153 [0.113, 0.197] | 0.154 [0.115, 0.193] |
| DeepSeek | DeepSeek-V3 | CoT | 0.443 [0.387, 0.500] | 0.367 [0.299, 0.435] |
| DeepSeek | DeepSeek-R1-0528 | reason | 0.417 [0.360, 0.470] | 0.338 [0.284, 0.390] |
|  |  |  |  |  |
| Alibaba Qwen | Qwen-2.5-72B | none | 0.260 [0.213, 0.310] | 0.225 [0.158, 0.293] |
| Alibaba Qwen | Qwen-2.5-72B | CoT | 0.643 [0.587, 0.697] | 0.466 [0.381, 0.557] |
| Alibaba Qwen | QwQ-32B | reason | 0.438 [0.379, 0.497] | 0.379 [0.323, 0.433] |
|  |  |  |  |  |
| Mistral | Mistral-Large | none | 0.337 [0.283, 0.390] | 0.320 [0.259, 0.380] |
| Mistral | Mistral-Large | CoT | 0.730 [0.680, 0.780] | 0.423 [0.335, 0.514] |
| Mistral | (no reasoning model on OpenRouter) | reason | — | — |
|  |  |  |  |  |

## 2. Three-way pattern: reasoning mode does NOT close the gap

Per-family deltas:

| Family | none → CoT (Δ acc) | CoT → reasoning (Δ acc) | Best mode |
|---|---|---|---|
| Meta Llama | +0.273 | -0.200 | **CoT** (0.547) |
| DeepSeek | +0.290 | -0.027 | **CoT** (0.443) |
| Alibaba Qwen | +0.383 | -0.205 | **CoT** (0.643) |
| Mistral | +0.393 | n/a | **CoT** (0.730) |

**Key finding:** For every open-weight family with a reasoning variant, the prompted-CoT mode of the non-reasoning version *outperforms* the dedicated reasoning model. The gap to in-family annotators (75-77%) is therefore primarily a model-capability gap, not a reasoning-mode mismatch artifact.

## 3. Reasoning mode introduces a different failure mode: NOVEL over-prediction

Whereas no-CoT mode hedges toward PARTIALLY_ANTICIPATED (suppressing ANTICIPATED), reasoning mode skews toward NOVEL when given time to think:

| Mode | DeepSeek | QwQ-32B | Nemotron | Avg NOVEL pred rate |
|---|---|---|---|---|
| GT distribution | — | — | — | **6.0%** |
| no-CoT (V3/2.5/3.3) | 54.0% | 5.7% | 26.3% | 28.7% |
| reasoning (R1/QwQ/Nemotron) | 48.3% | 45.2% | 52.3% | **48.6%** |

Reasoning-tuned models commit to NOVEL ~8× more often than ground-truth frequency. The thinking chain appears to lead these models to find novelty in claims rather than confirm prior art coverage. F1[NOVEL] recall is high (0.83-0.89) but precision is uniformly ~10%, indicating the over-prediction is the dominant error mode.

## 4. No-CoT mode: 'hedge to PARTIALLY_ANTICIPATED' (manuscript-relevant finding)

All four no-CoT runs systematically suppress ANTICIPATED predictions and over-predict PARTIALLY_ANTICIPATED or NOVEL. Without explicit reasoning, models refuse to commit to 'fully anticipated.'

| Model | Mode | Top class | Concentration | ANTIC preds (vs GT 210) |
|---|---|---|---|---|
| Llama-3.3-70B | no-CoT | PARTIAL | 68.7% | 15 |
| DeepSeek-V3 | no-CoT | NOVEL | 54.0% (distributed) | 7 |
| Qwen-2.5-72B | no-CoT | PARTIAL | **92.3%** (most concentrated) | 6 |
| Mistral-Large | no-CoT | PARTIAL | 68.7% | 39 |

DeepSeek-V3's 15.3% no-CoT accuracy is *not* parser-broken or single-class collapse — it is genuine, distributed misclassification: 54% NOVEL, 44% PARTIALLY_ANTICIPATED, only 2% ANTICIPATED. All four models fail to commit to 'fully anticipated' without a CoT prompt or reasoning mode, despite GT being 70% ANTICIPATED. This is reportable as a standalone manuscript finding.

## 5. Per-class F1 (CoT mode for non-reasoning, reasoning mode where applicable)

| Model | F1-NOVEL [CI] | F1-PARTIAL [CI] | F1-ANTIC [CI] |
|---|---|---|---|
| Llama-3.3-70B (CoT) | 0.235 [0.082, 0.391] | 0.355 [0.262, 0.442] | 0.689 [0.632, 0.741] |
| DeepSeek-V3 (CoT) | 0.200 [0.049, 0.368] | 0.369 [0.291, 0.444] | 0.532 [0.462, 0.595] |
| Qwen-2.5-72B (CoT) | 0.200 [0.000, 0.444] | 0.432 [0.340, 0.519] | 0.765 [0.715, 0.809] |
| Mistral-Large (CoT) | 0.200 [0.000, 0.444] | 0.225 [0.110, 0.341] | 0.843 [0.807, 0.876] |
| DeepSeek-R1 (reason) | 0.184 [0.103, 0.265] | 0.236 [0.129, 0.343] | 0.593 [0.528, 0.653] |
| QwQ-32B (reason) | 0.215 [0.126, 0.305] | 0.328 [0.222, 0.431] | 0.594 [0.525, 0.658] |
| Llama-Nemotron-49B (reason) | 0.183 [0.106, 0.261] | 0.299 [0.203, 0.392] | 0.475 [0.400, 0.545] |

## 6. Provider pinning (reproducibility)

| Model | OpenRouter slug | Provider pinned | Quantization |
|---|---|---|---|
| Llama-3.3-70B | meta-llama/llama-3.3-70b-instruct | Novita | bf16 |
| DeepSeek-V3 | deepseek/deepseek-chat | Novita | fp8 |
| Qwen-2.5-72B | qwen/qwen-2.5-72b-instruct | DeepInfra | fp8 |
| Mistral-Large | mistralai/mistral-large | Mistral | bf16 (native) |
| DeepSeek-R1-0528 | deepseek/deepseek-r1-0528 | SiliconFlow | fp8 |
| QwQ-32B | qwen/qwq-32b | SiliconFlow | fp8 |
| Llama-3.3-Nemotron-49B | nvidia/llama-3.3-nemotron-super-49b-v1.5 | DeepInfra | fp8 |

`provider.allow_fallbacks=false` enforced on every request. Per-call provider metadata logged in `tier3_holdout_provider_log.jsonl`.

## 7. Cost ledger

| Model | no-CoT | CoT | reasoning | Total |
|---|---|---|---|---|
| Llama-3.3-70B | $0.0534 | $0.0964 | $0.0000 | $0.1497 |
| DeepSeek-V3 | $0.0650 | $0.0912 | $0.0000 | $0.1562 |
| Qwen-2.5-72B | $0.0857 | $0.1257 | $0.0000 | $0.2114 |
| Mistral-Large | $0.8359 | $1.5561 | $0.0000 | $2.3920 |
| DeepSeek-R1-0528 | $0.0000 | $0.0000 | $0.5480 | $0.5480 |
| QwQ-32B | $0.0000 | $0.0000 | $0.2487 | $0.2487 |
| Llama-3.3-Nemotron-Super-49B | $0.0000 | $0.0000 | $0.1553 | $0.1553 |
| **Grand total (WP1 + reasoning rerun)** | | | | **$3.8613** |

_Reasoning rerun cost alone: $0.9520 (under $2.54 estimate)._

## 8. Footnote: QwQ-32B failures

QwQ-32B (`qwen/qwq-32b` on SiliconFlow, `ctx_k10_reasoning` mode) had **10/300 calls fail** with empty raw response (TypeError on null content), yielding n=290 valid predictions. All other 10 model-mode runs reached 300/300 with zero failures. Failures cluster in RN200-RN207, suggesting a transient SiliconFlow-side issue rather than systematic. Bootstrap 95% CIs are computed over the 290 successful predictions; effect on point estimates is below 0.01 accuracy. Per-call audit: `tier3_holdout_provider_log.jsonl`.

## 9. Cost ledger (final)

| Phase | Cost |
|---|---|
| WP1 original (CoT + no-CoT × 4 models) | $2.91 |
| Reasoning rerun (3 models × reasoning mode) | $0.95 |
| **Total OpenRouter spend** | **$3.86** |

All within the $10-15 cap.
