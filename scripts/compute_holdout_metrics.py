#!/usr/bin/env python3
"""Compute Tier 3 metrics + bootstrap CIs for hold-out runs.

Reads results/*.jsonl and emits:
  - Per-model metrics (accuracy, balanced acc, macro F1, per-class P/R/F1)
  - 95% bootstrap CIs (10k resamples, seed=42)
  - Confusion matrices
  - Comparison table including the 3 main annotators (from data/novex)
  - Summary JSON for figure generation
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

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


def load_predictions(jsonl_path: Path) -> dict:
    preds = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("ok") and r.get("label"):
                    preds[r["qid"]] = r["label"]
            except json.JSONDecodeError:
                continue
    return preds


def compute_metrics(preds: dict, gt: dict) -> dict:
    pairs = [(preds[q], gt[q]) for q in preds if q in gt]
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    correct = sum(1 for p, g in pairs if p == g)
    accuracy = correct / n

    # Per-class P/R/F1
    f1s = {}
    for lab in NOVELTY_LABELS:
        tp = sum(1 for p, g in pairs if p == lab and g == lab)
        fp = sum(1 for p, g in pairs if p == lab and g != lab)
        fn = sum(1 for p, g in pairs if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
    macro_f1 = np.mean([f1s[l]["f1"] for l in NOVELTY_LABELS])

    # Balanced accuracy = mean per-class recall
    balanced_acc = np.mean([f1s[l]["recall"] for l in NOVELTY_LABELS])

    # Confusion matrix
    cm = {p: {g: 0 for g in NOVELTY_LABELS} for p in NOVELTY_LABELS}
    for p, g in pairs:
        if p in cm and g in cm[p]:
            cm[p][g] += 1

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": f1s,
        "confusion_matrix": cm,
    }


def bootstrap_ci(values: list, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05):
    """Returns (mean, lower, upper) for a 95% CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = arr[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    return float(arr.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def bootstrap_metrics(preds: dict, gt: dict, n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap CIs for accuracy + macro F1 + per-class F1."""
    qids = sorted(set(preds) & set(gt))
    pairs = [(preds[q], gt[q]) for q in qids]
    n = len(pairs)
    if n == 0:
        return {}

    rng = np.random.RandomState(seed)
    accs, balanced_accs, macro_f1s = [], [], []
    per_class_f1s = {l: [] for l in NOVELTY_LABELS}

    for _ in range(n_boot):
        idxs = rng.randint(0, n, size=n)
        boot_pairs = [pairs[i] for i in idxs]
        boot_correct = sum(1 for p, g in boot_pairs if p == g)
        accs.append(boot_correct / n)

        rec_per_class = {}
        f1_per_class = {}
        for lab in NOVELTY_LABELS:
            tp = sum(1 for p, g in boot_pairs if p == lab and g == lab)
            fp = sum(1 for p, g in boot_pairs if p == lab and g != lab)
            fn = sum(1 for p, g in boot_pairs if p != lab and g == lab)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            rec_per_class[lab] = rec
            f1_per_class[lab] = f1
            per_class_f1s[lab].append(f1)
        balanced_accs.append(np.mean(list(rec_per_class.values())))
        macro_f1s.append(np.mean(list(f1_per_class.values())))

    def ci(vals):
        return {
            "mean": round(float(np.mean(vals)), 4),
            "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
            "ci_upper": round(float(np.percentile(vals, 97.5)), 4),
        }

    return {
        "accuracy": ci(accs),
        "balanced_accuracy": ci(balanced_accs),
        "macro_f1": ci(macro_f1s),
        "per_class_f1": {l: ci(per_class_f1s[l]) for l in NOVELTY_LABELS},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--gt-path", default="/tmp/biopat-explore/data/novex/qrels/tier3.tsv")
    p.add_argument("--out-path", default="results/holdout_metrics.json")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    gt = load_gt(Path(args.gt_path))
    print(f"Ground truth: {len(gt)} statements")
    print(f"  GT distribution: {dict(Counter(gt.values()))}\n")

    results_dir = Path(args.results_dir)
    all_metrics = {}

    for f in sorted(results_dir.glob("tier3_holdout_*_ctx_k10*.jsonl")):
        # Skip the provider log
        if "provider_log" in f.name:
            continue
        # Parse model + mode from filename
        # Format: tier3_holdout_<safe_model>_ctx_k10_<mode>.jsonl
        stem = f.stem.replace("tier3_holdout_", "")
        if stem.endswith("_ctx_k10_cot"):
            model_safe = stem[:-len("_ctx_k10_cot")]
            mode = "ctx_k10_cot"
        elif stem.endswith("_ctx_k10_nocot"):
            model_safe = stem[:-len("_ctx_k10_nocot")]
            mode = "ctx_k10_nocot"
        else:
            continue

        preds = load_predictions(f)
        m = compute_metrics(preds, gt)
        bm = bootstrap_metrics(preds, gt, n_boot=args.n_boot)

        key = f"{model_safe}/{mode}"
        all_metrics[key] = {"point": m, "bootstrap": bm}

        print(f"=== {key} ({m['n']} preds) ===")
        if m["n"] == 0:
            print("  No predictions")
            continue
        print(f"  Accuracy:          {m['accuracy']:.3f}  [{bm['accuracy']['ci_lower']:.3f}, {bm['accuracy']['ci_upper']:.3f}]")
        print(f"  Balanced accuracy: {m['balanced_accuracy']:.3f}  [{bm['balanced_accuracy']['ci_lower']:.3f}, {bm['balanced_accuracy']['ci_upper']:.3f}]")
        print(f"  Macro F1:          {m['macro_f1']:.3f}  [{bm['macro_f1']['ci_lower']:.3f}, {bm['macro_f1']['ci_upper']:.3f}]")
        for lab in NOVELTY_LABELS:
            pc = m["per_class"][lab]
            ci = bm["per_class_f1"][lab]
            print(f"  F1[{lab[:14]:14s}] {pc['f1']:.3f}  [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]   P={pc['precision']:.3f} R={pc['recall']:.3f}")
        print()

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nWrote {args.out_path}")


if __name__ == "__main__":
    main()
