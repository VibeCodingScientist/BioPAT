#!/usr/bin/env python3
"""WP5 update: regenerate Figure 2 with three reasoning-mode points per family.

Output:
  fig2_v2_three_modes.pdf  -- main figure (TACL 2-column width, 170mm)
  fig2_v2_three_modes.png  -- for arXiv preview

The U-shape (no-CoT hedge / CoT calibrated / reasoning over-predicts NOVEL) is
visualized two ways:
  Left panel:  accuracy by mode (inverted U, peaks at CoT)
  Right panel: distribution-deviation from GT (U-shape, lowest at CoT)

Keeps fig_context_vs_zeroshot.pdf as fig2_v1 (the original 2-mode scatter).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Okabe-Ito palette
OKABE_ITO = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
}

# TACL widths
TACL_2COL_MM = 170
MM_TO_INCH = 1 / 25.4

# ---- Load data ------------------------------------------------------------
preds_table = json.load(open("/tmp/biopat-wp1/results/pred_distribution_table.json"))
holdout_metrics = json.load(open("/tmp/biopat-wp1/results/holdout_metrics.json"))

# In-family annotator reference points (from existing analysis JSON).
# We treat them as "reasoning + extended-thinking" since that was their original config.
ANNOTATOR_ACC = {
    "GPT-5.2": 0.753,
    "Claude Sonnet 4.6": 0.773,
    "Gemini 3 Pro": 0.753,
}
ANNOTATOR_MEAN_ACC = float(np.mean(list(ANNOTATOR_ACC.values())))

HAIKU_BIASCTRL_ACC = 0.690

# ---- Build accuracy lookup by (family, model, mode) ----------------------
def acc_for(safe_model, mode_suffix):
    key = f"{safe_model}/{mode_suffix}"
    return holdout_metrics.get(key, {}).get("point", {}).get("accuracy")

family_data = [
    {
        "family": "Meta Llama",
        "color":  OKABE_ITO["blue"],
        "marker": "o",
        "no-CoT":     ("meta_llama_llama_3_3_70b_instruct",        "Llama-3.3-70B",          "ctx_k10_nocot"),
        "CoT":        ("meta_llama_llama_3_3_70b_instruct",        "Llama-3.3-70B",          "ctx_k10_cot"),
        "reasoning":  ("nvidia_llama_3_3_nemotron_super_49b_v1_5", "Llama-3.3-Nemotron-49B", "ctx_k10_reasoning"),
    },
    {
        "family": "DeepSeek",
        "color":  OKABE_ITO["vermilion"],
        "marker": "s",
        "no-CoT":     ("deepseek_deepseek_chat",      "DeepSeek-V3",      "ctx_k10_nocot"),
        "CoT":        ("deepseek_deepseek_chat",      "DeepSeek-V3",      "ctx_k10_cot"),
        "reasoning":  ("deepseek_deepseek_r1_0528",   "DeepSeek-R1",      "ctx_k10_reasoning"),
    },
    {
        "family": "Alibaba Qwen",
        "color":  OKABE_ITO["green"],
        "marker": "^",
        "no-CoT":     ("qwen_qwen_2_5_72b_instruct", "Qwen-2.5-72B", "ctx_k10_nocot"),
        "CoT":        ("qwen_qwen_2_5_72b_instruct", "Qwen-2.5-72B", "ctx_k10_cot"),
        "reasoning":  ("qwen_qwq_32b",               "QwQ-32B",      "ctx_k10_reasoning"),
    },
    {
        "family": "Mistral",
        "color":  OKABE_ITO["orange"],
        "marker": "D",
        "no-CoT":     ("mistralai_mistral_large", "Mistral-Large", "ctx_k10_nocot"),
        "CoT":        ("mistralai_mistral_large", "Mistral-Large", "ctx_k10_cot"),
        "reasoning":  None,  # no reasoning model on OpenRouter
    },
]

modes = ["no-CoT", "CoT", "reasoning"]
mode_x = {m: i for i, m in enumerate(modes)}

# ---- Get distribution-deviation by mode -----------------------------------
# rows from preds_table indexed by (family, model, mode)
absdev_lookup = {}
for r in preds_table["rows"]:
    absdev_lookup[(r["family"], r["model"], r["mode"])] = r["abs_deviation_from_gt"]

# ---- Figure ---------------------------------------------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TACL_2COL_MM * MM_TO_INCH, 4.0))

# ---- Panel A: accuracy by mode ----
for f in family_data:
    xs, ys, model_labels = [], [], []
    for mode in modes:
        spec = f.get(mode)
        if spec is None:
            continue
        safe_model, display, mode_suffix = spec
        acc = acc_for(safe_model, mode_suffix)
        if acc is None:
            continue
        xs.append(mode_x[mode])
        ys.append(acc)
        model_labels.append(display)
    ax1.plot(xs, ys, color=f["color"], marker=f["marker"], markersize=8,
             linewidth=2, label=f["family"], alpha=0.9, zorder=3)

# Annotator reference band
ax1.axhspan(min(ANNOTATOR_ACC.values()), max(ANNOTATOR_ACC.values()),
            color="gray", alpha=0.12, zorder=1)
ax1.axhline(ANNOTATOR_MEAN_ACC, color="gray", linestyle="--", linewidth=0.8,
            alpha=0.7, zorder=2,
            label=f"Annotator panel ({ANNOTATOR_MEAN_ACC:.2f})")
# Haiku bias-control reference
ax1.axhline(HAIKU_BIASCTRL_ACC, color=OKABE_ITO["purple"], linestyle=":",
            linewidth=1.0, alpha=0.7, zorder=2,
            label=f"Haiku 4.5 in-family ({HAIKU_BIASCTRL_ACC:.2f})")

ax1.set_xticks(list(mode_x.values()))
ax1.set_xticklabels(modes)
ax1.set_xlim(-0.3, 2.3)
ax1.set_ylim(0, 1.0)
ax1.set_ylabel("Accuracy on 300 NovEx statements")
ax1.set_xlabel("Reasoning mode")
ax1.set_title("(A) Accuracy across reasoning modes", fontsize=10)
ax1.grid(True, axis="y", alpha=0.3, linestyle=":")
ax1.legend(loc="lower center", fontsize=7, ncol=2, framealpha=0.9)

# ---- Panel B: distribution deviation from GT (calibration) ----
for f in family_data:
    xs, ys = [], []
    for mode in modes:
        spec = f.get(mode)
        if spec is None:
            continue
        _, display, _ = spec
        dev = absdev_lookup.get((f["family"], display, mode))
        if dev is None:
            continue
        xs.append(mode_x[mode])
        ys.append(dev)
    ax2.plot(xs, ys, color=f["color"], marker=f["marker"], markersize=8,
             linewidth=2, label=f["family"], alpha=0.9, zorder=3)

# Annotation: GT distribution shown as horizontal at 0
ax2.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

ax2.set_xticks(list(mode_x.values()))
ax2.set_xticklabels(modes)
ax2.set_xlim(-0.3, 2.3)
ax2.set_ylim(-0.1, 1.6)
ax2.set_ylabel(r"$\sum_c |\mathrm{pred}_c - \mathrm{GT}_c|$  (lower = better calibrated)")
ax2.set_xlabel("Reasoning mode")
ax2.set_title(r"(B) Calibration: $\sum_c |\mathrm{pred}_c - \mathrm{GT}_c|$", fontsize=10)
ax2.grid(True, axis="y", alpha=0.3, linestyle=":")

# Mode-specific failure annotations
ax2.text(0, 1.40, "hedge to\nPARTIAL", ha="center", va="bottom",
         fontsize=8, color="gray", style="italic")
ax2.text(2, 0.92, "over-predict\nNOVEL", ha="center", va="bottom",
         fontsize=8, color="gray", style="italic")
ax2.text(1, 0.48, "calibrated", ha="center", va="bottom",
         fontsize=8, color=OKABE_ITO["green"], style="italic", weight="bold")

plt.suptitle("Tier 3 novelty determination — three modes, four out-of-family hold-out lineages",
             fontsize=11, y=1.00)
plt.tight_layout()

out_dir = Path("/tmp/biopat-wp1/figures")
out_dir.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fp = out_dir / f"fig2_v2_three_modes.{ext}"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    print(f"  wrote {fp}")
plt.close()

# Also rename the original v1 figure for clarity (keep it alongside)
v1_src = out_dir / "fig_context_vs_zeroshot.pdf"
v1_dst = out_dir / "fig2_v1.pdf"
if v1_src.exists() and not v1_dst.exists():
    import shutil
    shutil.copy(v1_src, v1_dst)
    shutil.copy(out_dir / "fig_context_vs_zeroshot.png", out_dir / "fig2_v1.png")
    print(f"  copied original to {v1_dst} (v1 reference)")

print("\nDone.")
