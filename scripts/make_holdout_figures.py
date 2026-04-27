#!/usr/bin/env python3
"""WP5: Generate publication figures with hold-out points integrated.

Outputs:
  fig2_context_vs_zeroshot.{pdf,png}     -- accuracy with 95% CIs, includes hold-outs
  fig3_confusion_matrices.{pdf,png}      -- 4 panels: top 3 annotators + best hold-out
  fig_overlap_comparison.{pdf,png}       -- in-family vs out-of-family overlap analysis

TACL-friendly: vector PDF, font-embedded, colorblind palette (Okabe-Ito).
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermilion": "#D55E00",
    "purple":  "#CC79A7",
}

# TACL column widths (mm)
TACL_1COL_MM = 85
TACL_2COL_MM = 170
MM_TO_INCH = 1 / 25.4


# ---------------------------------------------------------------------------

NOVELTY_LABELS = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
NOVELTY_MAP = {0: "NOVEL", 1: "PARTIALLY_ANTICIPATED", 2: "ANTICIPATED"}


def load_gt(path: Path) -> dict:
    gt = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gt[parts[0]] = NOVELTY_MAP.get(int(parts[2]), "NOVEL")
    return gt


def bootstrap_ci(values, n_boot=10000, seed=42):
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = arr[rng.randint(0, n, size=n)].mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metric_with_ci(preds, gt, metric="accuracy"):
    pairs = [(preds[q], gt[q]) for q in preds if q in gt]
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if metric == "accuracy":
        vals = [1 if p == g else 0 for p, g in pairs]
    elif metric == "balanced_accuracy":
        # per-class recall, sample binary
        def boot_balanced_acc(boot_pairs):
            recs = []
            for lab in NOVELTY_LABELS:
                tp = sum(1 for p, g in boot_pairs if p == lab and g == lab)
                fn = sum(1 for p, g in boot_pairs if p != lab and g == lab)
                recs.append(tp / (tp + fn) if tp + fn else 0)
            return float(np.mean(recs))
        rng = np.random.RandomState(42)
        n_boot = 10000
        baseline = boot_balanced_acc(pairs)
        boots = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boots.append(boot_balanced_acc([pairs[i] for i in idx]))
        return baseline, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return bootstrap_ci(vals)


def load_holdout_preds(results_dir: Path) -> dict:
    """Return {model_safe: {mode: preds_dict}}."""
    out = {}
    for f in sorted(results_dir.glob("tier3_holdout_*.jsonl")):
        if "provider_log" in f.name:
            continue
        stem = f.stem.replace("tier3_holdout_", "")
        if stem.endswith("_ctx_k10_cot"):
            mode, model_safe = "cot", stem[:-len("_ctx_k10_cot")]
        elif stem.endswith("_ctx_k10_nocot"):
            mode, model_safe = "nocot", stem[:-len("_ctx_k10_nocot")]
        else:
            continue
        preds = {}
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok") and r.get("label"):
                        preds[r["qid"]] = r["label"]
                except json.JSONDecodeError:
                    continue
        out.setdefault(model_safe, {})[mode] = preds
    return out


# ---------------------------------------------------------------------------

def fig_overlap_comparison(gt, holdout_preds, output_dir: Path):
    """Bar chart: accuracy with 95% CI for in-family vs out-of-family models."""
    # In-family: pull from existing analysis JSON
    in_family_data = {
        "GPT-5.2":           {"acc": 0.753, "ci": (0.703, 0.803), "type": "in"},
        "Claude Sonnet 4.6": {"acc": 0.773, "ci": (0.722, 0.820), "type": "in"},
        "Gemini 3 Pro":      {"acc": 0.753, "ci": (0.703, 0.803), "type": "in"},
        "Claude Haiku 4.5":  {"acc": 0.690, "ci": (0.637, 0.740), "type": "in"},  # in-family with Sonnet
    }

    # Out-of-family: compute from hold-out preds (CoT mode)
    out_family_data = {}
    label_map = {
        "meta_llama_llama_3_3_70b_instruct": "Llama-3.3-70B",
        "deepseek_deepseek_chat":            "DeepSeek-V3",
        "qwen_qwen_2_5_72b_instruct":        "Qwen-2.5-72B",
        "mistralai_mistral_large":           "Mistral-Large",
    }
    for ms, modes in holdout_preds.items():
        cot_preds = modes.get("cot", {})
        if not cot_preds:
            continue
        m, lo, hi = metric_with_ci(cot_preds, gt)
        display = label_map.get(ms, ms)
        out_family_data[display] = {"acc": m, "ci": (lo, hi), "type": "out"}

    # Build figure
    width_in = TACL_2COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(width_in, 4.5))

    # Order: in-family first, then out-of-family
    models = list(in_family_data.keys()) + list(out_family_data.keys())
    accs = [in_family_data.get(m, out_family_data.get(m))["acc"] for m in models]
    cis = [in_family_data.get(m, out_family_data.get(m))["ci"] for m in models]
    types = [in_family_data.get(m, {}).get("type") or out_family_data[m]["type"] for m in models]

    colors = [OKABE_ITO["blue"] if t == "in" else OKABE_ITO["vermilion"] for t in types]
    x = np.arange(len(models))
    err = np.array([[a - lo, hi - a] for a, (lo, hi) in zip(accs, cis)]).T

    bars = ax.bar(x, accs, yerr=err, color=colors, edgecolor="black",
                   linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Accuracy [95% CI]")
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=OKABE_ITO["blue"], edgecolor="black", label="In-family (annotator/relative)"),
        Patch(facecolor=OKABE_ITO["vermilion"], edgecolor="black", label="Out-of-family (held-out)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_title("Tier 3 novelty accuracy: in-family vs out-of-family models", fontsize=10)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out_path = output_dir / f"fig_overlap_comparison.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  wrote {out_path}")
    plt.close()


def fig_context_vs_zeroshot(gt, holdout_preds, output_dir: Path):
    """Scatter: context (CoT) accuracy vs zero-shot ablation, all models."""
    width_in = TACL_1COL_MM * MM_TO_INCH * 1.4
    fig, ax = plt.subplots(figsize=(width_in, 3.5))

    # Existing data points (from prior analysis)
    in_family = {
        "GPT-5.2":           (0.753, 0.193),
        "Claude Sonnet 4.6": (0.773, 0.515),
        "Gemini 3 Pro":      (0.753, 0.064),
        "Claude Haiku 4.5":  (0.690, 0.447),  # no-CoT proxy = "zero-shot like"
    }

    label_map = {
        "meta_llama_llama_3_3_70b_instruct": "Llama-3.3-70B",
        "deepseek_deepseek_chat":            "DeepSeek-V3",
        "qwen_qwen_2_5_72b_instruct":        "Qwen-2.5-72B",
        "mistralai_mistral_large":           "Mistral-Large",
    }

    out_family = {}
    for ms, modes in holdout_preds.items():
        cot_preds = modes.get("cot", {})
        nocot_preds = modes.get("nocot", {})
        if not cot_preds or not nocot_preds:
            continue
        cot_m, _, _ = metric_with_ci(cot_preds, gt)
        nocot_m, _, _ = metric_with_ci(nocot_preds, gt)
        out_family[label_map.get(ms, ms)] = (cot_m, nocot_m)

    for name, (ctx_acc, zs_acc) in in_family.items():
        ax.scatter(ctx_acc, zs_acc, s=60, c=OKABE_ITO["blue"], edgecolor="black",
                    linewidth=0.5, zorder=3)
        ax.annotate(name, (ctx_acc, zs_acc), xytext=(5, -5),
                     textcoords="offset points", fontsize=7)
    for name, (ctx_acc, zs_acc) in out_family.items():
        ax.scatter(ctx_acc, zs_acc, s=60, c=OKABE_ITO["vermilion"], edgecolor="black",
                    linewidth=0.5, zorder=3, marker="s")
        ax.annotate(name, (ctx_acc, zs_acc), xytext=(5, -5),
                     textcoords="offset points", fontsize=7)

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Accuracy with k=10 context (CoT)")
    ax.set_ylabel("Accuracy without context / no-CoT")
    ax.set_title("Context contribution to novelty determination", fontsize=10)
    ax.legend(handles=[
        Patch(facecolor=OKABE_ITO["blue"], edgecolor="black", label="In-family"),
        Patch(facecolor=OKABE_ITO["vermilion"], edgecolor="black", label="Hold-out"),
    ], loc="upper left", fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(output_dir / f"fig_context_vs_zeroshot.{ext}", dpi=300, bbox_inches="tight")
        print(f"  wrote fig_context_vs_zeroshot.{ext}")
    plt.close()


def fig_confusion_matrices(gt, holdout_preds, output_dir: Path):
    """4 panels: 3 main annotators + best-performing hold-out."""
    # Pick best hold-out by macro-F1 in CoT mode
    best_holdout = None
    best_f1 = -1
    label_map = {
        "meta_llama_llama_3_3_70b_instruct": "Llama-3.3-70B",
        "deepseek_deepseek_chat":            "DeepSeek-V3",
        "qwen_qwen_2_5_72b_instruct":        "Qwen-2.5-72B",
        "mistralai_mistral_large":           "Mistral-Large",
    }
    for ms, modes in holdout_preds.items():
        cot_preds = modes.get("cot", {})
        if not cot_preds:
            continue
        pairs = [(cot_preds[q], gt[q]) for q in cot_preds if q in gt]
        f1s = []
        for lab in NOVELTY_LABELS:
            tp = sum(1 for p, g in pairs if p == lab and g == lab)
            fp = sum(1 for p, g in pairs if p == lab and g != lab)
            fn = sum(1 for p, g in pairs if p != lab and g == lab)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        macro = float(np.mean(f1s))
        if macro > best_f1:
            best_f1 = macro
            best_holdout = (ms, cot_preds)

    if not best_holdout:
        print("  No hold-outs available yet for confusion matrix figure")
        return

    # Build confusion matrices for the 3 main + best hold-out
    # NOTE: we don't have stored preds for the main 3, so we use canonical metrics
    # from the analysis JSON. Confusion matrices for them come from data/novex/analysis/confusion_matrices.json
    cm_main_path = Path("/tmp/biopat-explore/data/novex/analysis/confusion_matrices.json")
    if not cm_main_path.exists():
        print(f"  Missing {cm_main_path}; skipping main-3 panels")
        return
    cm_data = json.load(open(cm_main_path))

    # Construct the 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(TACL_2COL_MM * MM_TO_INCH, 3))

    panels = [
        ("GPT-5.2", cm_data.get("gpt-5.2_ctx", {})),
        ("Claude Sonnet 4.6", cm_data.get("claude-sonnet-4-6_ctx", {})),
        ("Gemini 3 Pro", cm_data.get("gemini-3-pro-preview_ctx", {})),
    ]
    # Best hold-out: build from preds
    best_preds = best_holdout[1]
    cm_holdout = {p: {g: 0 for g in NOVELTY_LABELS} for p in NOVELTY_LABELS}
    for q, p in best_preds.items():
        if q in gt:
            cm_holdout[p][gt[q]] += 1
    panels.append((label_map.get(best_holdout[0], best_holdout[0]) + " (CoT)", cm_holdout))

    for ax, (title, cm) in zip(axes, panels):
        # Convert to matrix (rows=pred, cols=gt)
        mat = np.array([[cm.get(p, {}).get(g, 0) for g in NOVELTY_LABELS] for p in NOVELTY_LABELS])
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([l[:6] for l in NOVELTY_LABELS], fontsize=7, rotation=30)
        ax.set_yticklabels([l[:6] for l in NOVELTY_LABELS], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("GT", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Predicted", fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center",
                         fontsize=7, color="white" if mat[i, j] > mat.max() / 2 else "black")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(output_dir / f"fig_confusion_matrices.{ext}", dpi=300, bbox_inches="tight")
        print(f"  wrote fig_confusion_matrices.{ext}")
    plt.close()


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-path", default="/tmp/biopat-explore/data/novex/qrels/tier3.tsv")
    p.add_argument("--holdout-results", default="/tmp/biopat-wp1/results")
    p.add_argument("--output-dir", default="/tmp/biopat-wp1/figures")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(Path(args.gt_path))
    print(f"GT: {len(gt)} statements")

    holdout_preds = load_holdout_preds(Path(args.holdout_results))
    for ms, modes in holdout_preds.items():
        for mode, preds in modes.items():
            print(f"  hold-out {ms} ({mode}): {len(preds)} preds")

    plt.rcParams["pdf.fonttype"] = 42  # TrueType (font embedding)
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig_overlap_comparison(gt, holdout_preds, output_dir)
    fig_context_vs_zeroshot(gt, holdout_preds, output_dir)
    fig_confusion_matrices(gt, holdout_preds, output_dir)

    print(f"\nAll figures written to {output_dir}/")


if __name__ == "__main__":
    main()
