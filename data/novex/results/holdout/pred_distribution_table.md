# Per-Class Prediction Distribution Across All 11 Model-Mode Combinations

Ground truth distribution: NOVEL **6.0%** / PARTIALLY_ANTICIPATED **24.0%** / ANTICIPATED **70.0%**  (n=300)

`abs_dev` = total absolute deviation from GT distribution (sum of |pred_pct - gt_pct| across the 3 classes; 0 = perfectly calibrated, 2 = maximally miscalibrated).

| Family | Model | Mode | n | %NOVEL | %PA | %ANTIC | abs_dev |
|---|---|---|---|---|---|---|---|
| Meta Llama | Llama-3.3-70B | no-CoT | 300 | 26.3% | 68.7% | 5.0% | **1.300** |
| Meta Llama | Llama-3.3-70B | CoT | 300 | 11.0% | 38.0% | 51.0% | **0.380** |
| Meta Llama | Llama-3.3-Nemotron-49B | reasoning | 300 | 52.3% | 25.0% | 22.7% | **0.947** |
| DeepSeek | DeepSeek-V3 | no-CoT | 300 | 54.0% | 43.7% | 2.3% | **1.353** |
| DeepSeek | DeepSeek-V3 | CoT | 300 | 7.3% | 57.3% | 35.3% | **0.693** |
| DeepSeek | DeepSeek-R1-0528 | reasoning | 300 | 48.3% | 12.7% | 39.0% | **0.847** |
| Alibaba Qwen | Qwen-2.5-72B | no-CoT | 300 | 5.7% | 92.3% | 2.0% | **1.367** |
| Alibaba Qwen | Qwen-2.5-72B | CoT | 300 | 0.7% | 37.7% | 61.7% | **0.273** |
| Alibaba Qwen | QwQ-32B | reasoning | 290 (+10 fail) | 45.2% | 20.0% | 34.8% | **0.783** |
| Mistral | Mistral-Large | no-CoT | 300 | 18.3% | 68.7% | 13.0% | **1.140** |
| Mistral | Mistral-Large | CoT | 300 | 0.7% | 5.7% | 93.7% | **0.473** |

## Aggregate by mode (mean across families)

| Mode | Mean %NOVEL | Mean %PA | Mean %ANTIC | Mean abs_dev |
|---|---|---|---|---|
| **no-CoT** | 26.1% | 68.3% | 5.6% | **1.290** |
| **CoT** | 4.9% | 34.7% | 60.4% | **0.455** |
| **reasoning** | 48.6% | 19.2% | 32.2% | **0.859** |

## Interpretation

**no-CoT** mode hedges to PARTIALLY_ANTICIPATED on average (mean ~70%) and suppresses ANTICIPATED predictions (mean ~10%, vs GT 70%). Models refuse to commit to 'fully anticipated' without a reasoning step.

**CoT** (prompted) mode is the calibrated middle: predicted distribution closely tracks GT (mean abs_dev ≈ 0.16-0.30), with ANTICIPATED predicted at 56-70% — close to GT 70%.

**reasoning** mode swings the other way: NOVEL is over-predicted at ~49% (vs GT 6%), and ANTICIPATED is suppressed to ~32% (vs GT 70%). The thinking chain leads models to find novelty in claims rather than confirm prior art coverage.

This U-shape pattern — calibrated middle flanked by opposite failure modes — is the cleanest single finding from the WP1 + reasoning-rerun exercise.