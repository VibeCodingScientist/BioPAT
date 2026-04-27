#!/usr/bin/env python3
"""Generate the overlap comparison table for §3.6 / §4.3 / §5.3.

Combines:
  - 3 main annotators (GPT-5.2, Sonnet 4.6, Gemini 3 Pro) -- in-family
  - Haiku 4.5 -- in-family with Sonnet (existing bias control)
  - 3-4 hold-out models from WP1 -- out-of-family

Output: a markdown table + LaTeX table fragment ready to drop into the manuscript.
"""

import argparse
import json
from pathlib import Path
import numpy as np


NOVELTY_LABELS = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
NOVELTY_MAP = {0: "NOVEL", 1: "PARTIALLY_ANTICIPATED", 2: "ANTICIPATED"}


def load_gt(qrels_path: Path) -> dict:
    gt = {}
    with open(qrels_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gt[parts[0]] = NOVELTY_MAP.get(int(parts[2]), "NOVEL")
    return gt


def metrics_with_ci(preds: dict, gt: dict, n_boot=10000, seed=42):
    pairs = [(preds[q], gt[q]) for q in sorted(preds) if q in gt]
    n = len(pairs)
    if n == 0:
        return None

    correct = sum(1 for p, g in pairs if p == g)
    accuracy = correct / n

    # Per-class F1 + macro F1
    f1s = {}
    for lab in NOVELTY_LABELS:
        tp = sum(1 for p, g in pairs if p == lab and g == lab)
        fp = sum(1 for p, g in pairs if p == lab and g != lab)
        fn = sum(1 for p, g in pairs if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        f1s[lab] = f1
    macro_f1 = float(np.mean(list(f1s.values())))
    balanced_acc = float(np.mean([
        sum(1 for p, g in pairs if p == lab and g == lab) /
        (sum(1 for _, g in pairs if g == lab) or 1)
        for lab in NOVELTY_LABELS
    ]))

    # Bootstrap CIs
    rng = np.random.RandomState(seed)
    accs, mf1s, bal_accs = [], [], []
    for _ in range(n_boot):
        idxs = rng.randint(0, n, size=n)
        bp = [pairs[i] for i in idxs]
        bc = sum(1 for p, g in bp if p == g)
        accs.append(bc / n)
        boot_f1s = []
        boot_recs = []
        for lab in NOVELTY_LABELS:
            tp = sum(1 for p, g in bp if p == lab and g == lab)
            fp = sum(1 for p, g in bp if p == lab and g != lab)
            fn = sum(1 for p, g in bp if p != lab and g == lab)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            boot_f1s.append(f1)
            boot_recs.append(rec)
        mf1s.append(np.mean(boot_f1s))
        bal_accs.append(np.mean(boot_recs))

    return {
        "n": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_f1": f1s,
        "ci_acc": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
        "ci_macro_f1": (float(np.percentile(mf1s, 2.5)), float(np.percentile(mf1s, 97.5))),
        "ci_balanced_acc": (float(np.percentile(bal_accs, 2.5)), float(np.percentile(bal_accs, 97.5))),
    }


def load_main_models(checkpoint_dir: Path, gt: dict) -> dict:
    """Load existing in-family annotator results from checkpoints."""
    out = {}
    for model_id, family, source in [
        ("gpt_5.2", "OpenAI GPT", "annotator+eval"),
        ("claude_sonnet_4_6", "Anthropic Claude", "annotator+eval"),
        ("gemini_3_pro_preview", "Google Gemini", "annotator+eval"),
    ]:
        provider_prefix = {"gpt_5.2": "openai", "claude_sonnet_4_6": "anthropic",
                           "gemini_3_pro_preview": "google"}[model_id]
        cp = checkpoint_dir / f"t3_{provider_prefix}_{model_id}_ctx.json"
        if not cp.exists():
            print(f"  WARN: missing {cp}")
            continue
        # Extract per-statement preds from the checkpoint
        # The checkpoint stores preds as per-query items
        data = json.load(open(cp))
        preds = data.get("predictions", {}) or data.get("per_query", {})
        # If checkpoint only has aggregate metrics, skip — we'll use the
        # numbers from data/novex/analysis/tier3_table.json instead
        if not preds:
            # Fallback: pull from analysis JSON
            out[model_id] = {
                "family": family,
                "source": source,
                "metrics": {
                    "accuracy": data["metrics"].get("accuracy"),
                    "macro_f1": data["metrics"].get("macro_f1"),
                    "from_analysis_json": True,
                },
            }
            continue
        pred_dict = {q: v["label"] if isinstance(v, dict) else v for q, v in preds.items()}
        m = metrics_with_ci(pred_dict, gt)
        out[model_id] = {"family": family, "source": source, "metrics": m}
    return out


def load_holdout(results_dir: Path, gt: dict) -> dict:
    """Load WP1 hold-out predictions."""
    out = {}
    for f in sorted(results_dir.glob("tier3_holdout_*.jsonl")):
        if "provider_log" in f.name:
            continue
        stem = f.stem.replace("tier3_holdout_", "")
        if stem.endswith("_ctx_k10_cot"):
            mode = "cot"
            model_safe = stem[:-len("_ctx_k10_cot")]
        elif stem.endswith("_ctx_k10_nocot"):
            mode = "nocot"
            model_safe = stem[:-len("_ctx_k10_nocot")]
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
        m = metrics_with_ci(preds, gt)
        out[f"{model_safe}_{mode}"] = {
            "model_safe": model_safe,
            "mode": mode,
            "metrics": m,
            "n_preds": len(preds),
        }
    return out


def render_markdown(main_models, holdout, haiku_metrics):
    lines = ["# Tier 3 Novelty: Overlap Analysis (WP1 Hold-Outs Added)\n"]
    lines.append("| Model | Family | Role | Accuracy [95% CI] | Macro F1 [95% CI] |")
    lines.append("|---|---|---|---|---|")

    # Main 3 annotator+eval models
    for k, v in main_models.items():
        m = v["metrics"]
        if not m or "accuracy" not in m:
            continue
        if "from_analysis_json" in m:
            lines.append(f"| {k} | {v['family']} | annotator+eval | {m['accuracy']:.3f} | {m.get('macro_f1','?')} |")
        else:
            lines.append(
                f"| {k} | {v['family']} | annotator+eval | "
                f"{m['accuracy']:.3f} [{m['ci_acc'][0]:.3f}, {m['ci_acc'][1]:.3f}] | "
                f"{m['macro_f1']:.3f} [{m['ci_macro_f1'][0]:.3f}, {m['ci_macro_f1'][1]:.3f}] |"
            )

    # Haiku 4.5 (existing in-family bias control)
    if haiku_metrics:
        lines.append(
            f"| Claude Haiku 4.5 (CoT) | Anthropic Claude | bias-ctrl (in-family) | "
            f"{haiku_metrics['accuracy']:.3f} | {haiku_metrics['macro_f1']:.3f} |"
        )

    # Hold-outs (only CoT mode in main table)
    for k, v in holdout.items():
        if v["mode"] != "cot":
            continue
        m = v["metrics"]
        if not m:
            continue
        family = "Meta Llama" if "llama" in k else \
                 "DeepSeek" if "deepseek" in k else \
                 "Alibaba Qwen" if "qwen" in k else \
                 "Mistral" if "mistral" in k else "?"
        lines.append(
            f"| {v['model_safe'].replace('_', '-')} | {family} | hold-out (CoT) | "
            f"{m['accuracy']:.3f} [{m['ci_acc'][0]:.3f}, {m['ci_acc'][1]:.3f}] | "
            f"{m['macro_f1']:.3f} [{m['ci_macro_f1'][0]:.3f}, {m['ci_macro_f1'][1]:.3f}] |"
        )

    lines.append("\n## CoT vs no-CoT ablation (hold-outs)\n")
    lines.append("| Model | CoT acc | no-CoT acc | Delta |")
    lines.append("|---|---|---|---|")
    holdout_by_model = {}
    for k, v in holdout.items():
        holdout_by_model.setdefault(v["model_safe"], {})[v["mode"]] = v
    for ms, modes in sorted(holdout_by_model.items()):
        cot_m = modes.get("cot", {}).get("metrics")
        nocot_m = modes.get("nocot", {}).get("metrics")
        if not cot_m or not nocot_m:
            continue
        delta = cot_m["accuracy"] - nocot_m["accuracy"]
        lines.append(f"| {ms.replace('_','-')} | {cot_m['accuracy']:.3f} | {nocot_m['accuracy']:.3f} | {delta:+.3f} |")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-path", default="/tmp/biopat-explore/data/novex/qrels/tier3.tsv")
    p.add_argument("--main-checkpoints", default="/tmp/biopat-explore/data/novex/results/checkpoints")
    p.add_argument("--holdout-results", default="/tmp/biopat-wp1/results")
    p.add_argument("--out", default="/tmp/biopat-wp1/results/overlap_analysis.md")
    args = p.parse_args()

    gt = load_gt(Path(args.gt_path))
    print(f"Ground truth: {len(gt)} statements")

    main_models = load_main_models(Path(args.main_checkpoints), gt)
    holdout = load_holdout(Path(args.holdout_results), gt)

    # Haiku (from existing all_results.json)
    haiku_metrics = None
    all_results_path = Path("/tmp/biopat-explore/data/novex/results/all_results.json")
    if all_results_path.exists():
        for r in json.load(open(all_results_path)):
            if r.get("model") == "claude-haiku-4-5-20251001":
                haiku_metrics = r.get("metrics")
                break

    md = render_markdown(main_models, holdout, haiku_metrics)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"\nWrote {args.out}")
    print("\n" + md)


if __name__ == "__main__":
    main()
