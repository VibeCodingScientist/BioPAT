#!/usr/bin/env python3
"""Phase 4: Final gap-filling — stale checkpoints, new local analyses, figures.

Part A: Pure computation (no API calls)
  - GT agreement stratification (model accuracy by unanimous/majority/override GT)
  - Fleiss' kappa for the 3 LLM annotators who created GT labels

Part B: API re-runs (~$1-2)
  - Delete stale Claude ctx_k10 (101 queries, needs 300)
  - Delete stale ZS checkpoints (no predicted field)
  - Re-run Claude ctx k=10
  - Re-run 3 ZS conditions (all 3 models)

Part C: Regenerate all figures (PDF + PNG)

Usage (VPS):
    cd ~/BioPAT && set -a && source .env && set +a
    nohup env PYTHONPATH=src venv/bin/python3 scripts/run_phase4.py >> logs/phase4.log 2>&1 &
"""

import importlib.util
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase4")

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "novex"
CHECKPOINTS = DATA / "results" / "checkpoints"
ANALYSIS = DATA / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

_src = ROOT / "src"

MODELS = [
    ("openai", "gpt-5.2", "GPT-5.2"),
    ("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("google", "gemini-3-pro-preview", "Gemini 3 Pro"),
]


def load_statements():
    stmts = {}
    with open(DATA / "statements.jsonl") as f:
        for line in f:
            s = json.loads(line)
            stmts[s["statement_id"]] = s
    return stmts


def load_tier3_qrels():
    labels = {}
    label_map = {0: "NOVEL", 1: "PARTIALLY_ANTICIPATED", 2: "ANTICIPATED"}
    with open(DATA / "qrels" / "tier3.tsv") as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("query_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                labels[parts[0]] = label_map.get(int(parts[2]), "UNKNOWN")
    return labels


def load_checkpoint(name):
    p = CHECKPOINTS / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ============================================================
# Part A: Pure Computation
# ============================================================

def gt_agreement_stratification():
    """Stratify T3 model accuracy by GT agreement level (unanimous/majority/override)."""
    logger.info("=== Part A.1: GT Agreement Stratification ===")
    stmts = load_statements()
    gt_labels = load_tier3_qrels()

    # Group queries by agreement level
    agreement_groups = defaultdict(list)
    for sid, s in stmts.items():
        agreement = s.get("ground_truth", {}).get("tier3_agreement", "unknown")
        agreement_groups[agreement].append(sid)

    logger.info("  Agreement groups: %s",
                {k: len(v) for k, v in agreement_groups.items()})

    results = {}
    for provider, model_safe_raw, model_name in MODELS:
        model_safe = f"{provider}_{model_safe_raw}".replace("-", "_")
        # Try k=10 first, then k=20, then k=5
        cp = None
        for k in [10, 20, 5, 3, 1]:
            cp = load_checkpoint(f"t3_{model_safe}_ctx_k{k}")
            if cp and len(cp.get("per_query", {})) > 200:
                break
        if cp is None:
            logger.warning("  No checkpoint for %s", model_name)
            continue

        pq = cp.get("per_query", {})
        model_results = {}

        for agreement_level, qids in sorted(agreement_groups.items()):
            correct = 0
            total = 0
            label_correct = Counter()
            label_total = Counter()

            for qid in qids:
                if qid not in pq:
                    continue
                gt = gt_labels.get(qid, "?")
                pred = pq[qid].get("predicted", "?")
                is_correct = pq[qid].get("correct", 0)

                total += 1
                correct += is_correct
                label_total[gt] += 1
                if is_correct:
                    label_correct[gt] += 1

            acc = correct / total if total else 0
            model_results[agreement_level] = {
                "n_queries": total,
                "accuracy": round(acc, 4),
                "by_label": {
                    label: {
                        "n": label_total[label],
                        "correct": label_correct[label],
                        "accuracy": round(label_correct[label] / label_total[label], 4) if label_total[label] else 0,
                    }
                    for label in ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
                },
            }
            logger.info("  %s / %s: acc=%.4f (%d/%d)",
                        model_name, agreement_level, acc, int(correct), total)

        results[model_name] = model_results

    out = ANALYSIS / "gt_agreement_stratification.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  Saved: %s", out)
    return results


def fleiss_kappa_gt():
    """Compute Fleiss' kappa for the 3 LLM annotators who created GT labels."""
    logger.info("=== Part A.2: Fleiss' Kappa for GT Annotators ===")
    stmts = load_statements()

    categories = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Build rating matrix: n_subjects × n_categories
    # Each cell = number of raters who assigned that category
    n_raters = 3  # openai, anthropic, google
    matrix = []
    valid_ids = []

    for sid, s in sorted(stmts.items()):
        individual = s.get("ground_truth", {}).get("tier3_individual", {})
        if len(individual) < 3:
            continue

        row = [0] * len(categories)
        for rater, label in individual.items():
            label_norm = label.upper().replace(" ", "_")
            if label_norm in cat_idx:
                row[cat_idx[label_norm]] += 1

        if sum(row) == n_raters:
            matrix.append(row)
            valid_ids.append(sid)

    N = len(matrix)  # number of subjects
    n = n_raters
    k = len(categories)

    logger.info("  Subjects: %d, Raters: %d, Categories: %d", N, n, k)

    # Distribution of ratings
    rating_dist = Counter()
    for sid, s in sorted(stmts.items()):
        individual = s.get("ground_truth", {}).get("tier3_individual", {})
        for rater, label in individual.items():
            rating_dist[label.upper().replace(" ", "_")] += 1
    logger.info("  Rating distribution: %s", dict(rating_dist))

    # Fleiss' kappa computation
    # P_i = (1 / (n*(n-1))) * sum_j(n_ij^2) - n / (n*(n-1))
    # P_bar = mean(P_i)
    # P_e = sum_j(p_j^2)
    # kappa = (P_bar - P_e) / (1 - P_e)

    P_i_list = []
    for row in matrix:
        sum_sq = sum(r ** 2 for r in row)
        P_i = (sum_sq - n) / (n * (n - 1))
        P_i_list.append(P_i)

    P_bar = sum(P_i_list) / N

    # Column proportions
    col_sums = [0] * k
    for row in matrix:
        for j in range(k):
            col_sums[j] += row[j]
    total_ratings = N * n
    p_j = [col_sums[j] / total_ratings for j in range(k)]
    P_e = sum(pj ** 2 for pj in p_j)

    kappa = (P_bar - P_e) / (1 - P_e) if abs(1 - P_e) > 1e-10 else 1.0

    logger.info("  P_bar=%.4f, P_e=%.4f, kappa=%.4f", P_bar, P_e, kappa)

    # Per-category kappa (specific agreement)
    per_cat_kappa = {}
    for j, cat in enumerate(categories):
        # Proportion of ratings in this category
        p_j_val = p_j[j]
        # Mean of P_ij for this category
        sum_nij = sum(row[j] for row in matrix)
        sum_nij_sq = sum(row[j] ** 2 for row in matrix)
        P_j = (sum_nij_sq - sum_nij) / (sum_nij * (n - 1)) if sum_nij > 0 else 0
        kappa_j = (P_j - p_j_val) / (1 - p_j_val) if abs(1 - p_j_val) > 1e-10 else 1.0
        per_cat_kappa[cat] = round(kappa_j, 4)
        logger.info("  %s: specific kappa=%.4f (p_j=%.4f)", cat, kappa_j, p_j_val)

    # Also compute pairwise Cohen's kappa for each annotator pair
    pairwise = {}
    rater_names = ["openai", "anthropic", "google"]
    rater_labels = {}
    for sid, s in sorted(stmts.items()):
        individual = s.get("ground_truth", {}).get("tier3_individual", {})
        for rater, label in individual.items():
            if rater not in rater_labels:
                rater_labels[rater] = {}
            rater_labels[rater][sid] = label.upper().replace(" ", "_")

    for i in range(len(rater_names)):
        for j in range(i + 1, len(rater_names)):
            r1, r2 = rater_names[i], rater_names[j]
            common = set(rater_labels.get(r1, {}).keys()) & set(rater_labels.get(r2, {}).keys())
            if not common:
                continue

            # Cohen's kappa
            agree = sum(1 for sid in common if rater_labels[r1][sid] == rater_labels[r2][sid])
            n_common = len(common)
            p_o = agree / n_common

            # Expected agreement
            counts_1 = Counter(rater_labels[r1][sid] for sid in common)
            counts_2 = Counter(rater_labels[r2][sid] for sid in common)
            p_e = sum(
                (counts_1.get(cat, 0) / n_common) * (counts_2.get(cat, 0) / n_common)
                for cat in categories
            )

            ck = (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-10 else 1.0
            pairwise[f"{r1}_vs_{r2}"] = {
                "n_common": n_common,
                "observed_agreement": round(p_o, 4),
                "expected_agreement": round(p_e, 4),
                "cohens_kappa": round(ck, 4),
            }
            logger.info("  %s vs %s: kappa=%.4f (n=%d)", r1, r2, ck, n_common)

    result = {
        "n_subjects": N,
        "n_raters": n,
        "categories": categories,
        "category_proportions": {categories[j]: round(p_j[j], 4) for j in range(k)},
        "fleiss_kappa": round(kappa, 4),
        "P_bar": round(P_bar, 4),
        "P_e": round(P_e, 4),
        "per_category_kappa": per_cat_kappa,
        "pairwise_cohens_kappa": pairwise,
        "interpretation": _interpret_kappa(kappa),
    }

    out = ANALYSIS / "fleiss_kappa_gt.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("  Saved: %s", out)
    return result


def _interpret_kappa(k):
    if k < 0:
        return "poor (below chance)"
    elif k < 0.20:
        return "slight agreement"
    elif k < 0.40:
        return "fair agreement"
    elif k < 0.60:
        return "moderate agreement"
    elif k < 0.80:
        return "substantial agreement"
    else:
        return "almost perfect agreement"


# ============================================================
# Part B: API Re-runs (VPS only)
# ============================================================

def _load_mod(name, filepath):
    """Direct module loading — bypasses heavy __init__.py."""
    import types
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = [str(_src / parent.replace(".", "/"))]
            pkg.__package__ = parent
            sys.modules[parent] = pkg
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def api_reruns():
    """Delete stale checkpoints and re-run them with predicted field."""
    logger.info("=== Part B: API Re-runs ===")

    # Load evaluator modules
    _load_mod("biopat.novex._util", _src / "biopat/novex/_util.py")
    _load_mod("biopat.novex.benchmark", _src / "biopat/novex/benchmark.py")
    _load_mod("biopat.novex.evaluator", _src / "biopat/novex/evaluator.py")

    from biopat.novex.benchmark import NovExBenchmark
    from biopat.novex.evaluator import NovExEvaluator

    # --- Step 1: Delete stale checkpoints ---
    stale = []

    # Claude ctx_k10 — only 101 queries (old Phase 1)
    cp_claude_k10 = CHECKPOINTS / "t3_anthropic_claude_sonnet_4_6_ctx_k10.json"
    if cp_claude_k10.exists():
        with open(cp_claude_k10) as f:
            d = json.load(f)
        n = len(d.get("per_query", {}))
        if n < 200:
            stale.append(("Claude ctx k=10", cp_claude_k10, n))

    # ZS checkpoints — no predicted field
    for provider, model_safe_raw, model_name in MODELS:
        model_safe = f"{provider}_{model_safe_raw}".replace("-", "_")
        cp_zs = CHECKPOINTS / f"t3_{model_safe}_zs.json"
        if cp_zs.exists():
            with open(cp_zs) as f:
                d = json.load(f)
            pq = d.get("per_query", {})
            sample = next(iter(pq.values()), {})
            if "predicted" not in sample:
                stale.append((f"{model_name} ZS", cp_zs, len(pq)))

    for desc, path, n in stale:
        logger.info("  Deleting stale: %s (%d queries) — %s", desc, n, path.name)
        path.unlink()

    if not stale:
        logger.info("  No stale checkpoints found (all re-runs already done)")

    # --- Step 2: Load benchmark and evaluator ---
    logger.info("  Loading benchmark...")
    b = NovExBenchmark(data_dir=str(DATA))
    b.load()
    logger.info("  Loaded: %d statements, %d docs", len(b.statements), len(b.corpus))

    ev = NovExEvaluator(
        benchmark=b,
        results_dir=str(DATA / "results"),
        budget_usd=500.0,
        seed=42,
    )

    # --- Step 3: Re-run Claude ctx k=10 ---
    logger.info("  Re-running Claude ctx k=10...")
    try:
        r = ev.run_tier3("anthropic", "claude-sonnet-4-6", with_context=True, context_k=10)
        logger.info("    → acc=%.4f, %d queries, $%.2f",
                     r.metrics.get("accuracy", 0), len(r.per_query), r.cost_usd)
    except Exception as exc:
        logger.error("    FAILED: %s", exc)

    # --- Step 4: Re-run ZS for all 3 models ---
    for provider_raw, model_id, model_name in MODELS:
        logger.info("  Re-running %s ZS...", model_name)
        try:
            r = ev.run_tier3(provider_raw, model_id, with_context=False)
            logger.info("    → acc=%.4f, %d queries, $%.2f",
                         r.metrics.get("accuracy", 0), len(r.per_query), r.cost_usd)
        except Exception as exc:
            logger.error("    FAILED: %s", exc)

    total_cost = ev.cost_tracker.total_cost
    logger.info("  Part B complete: $%.2f total", total_cost)
    return total_cost


# ============================================================
# Part C: Regenerate Figures (PDF + PNG)
# ============================================================

def regenerate_figures():
    """Generate all figures in both PDF and PNG formats."""
    logger.info("=== Part C: Regenerate Figures ===")

    try:
        import matplotlib as mpl
        mpl.use("Agg")
    except ImportError:
        logger.error("matplotlib not available — skipping figure generation")
        return

    # Import figures module directly
    spec = importlib.util.spec_from_file_location(
        "figures", _src / "biopat" / "novex" / "figures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    NovExFigureGenerator = mod.NovExFigureGenerator

    fig_dir = str(ANALYSIS / "figures")
    gen = NovExFigureGenerator(analysis_dir=str(ANALYSIS))

    for fmt in ["pdf", "png"]:
        logger.info("  Generating %s figures...", fmt.upper())
        mpl.rcParams["savefig.dpi"] = 300
        paths = gen.generate_all(output_dir=fig_dir, fmt=fmt)
        logger.info("    → %d %s figures", len(paths), fmt.upper())
        for p in paths:
            logger.info("      %s", p.name)

    return fig_dir


# ============================================================
# Part D: Re-run Phase 3 analyses with fresh checkpoints
# ============================================================

def rerun_analyses():
    """Re-run confusion_matrices and context_ablation with fresh Claude k=10 data."""
    logger.info("=== Part D: Re-run analyses with fresh checkpoints ===")

    # Confusion matrices (now includes ZS with predicted field)
    logger.info("  Re-computing confusion matrices...")
    from collections import Counter as Ctr

    gt_labels = load_tier3_qrels()
    labels_order = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]

    confusion_results = {}
    for provider, model_safe_raw, model_name in MODELS:
        model_safe = f"{provider}_{model_safe_raw}".replace("-", "_")

        for mode, suffix in [("ctx", "_k10"), ("zs", "")]:
            cp = load_checkpoint(f"t3_{model_safe}_{mode}{suffix}")
            if cp is None:
                continue
            pq = cp.get("per_query", {})
            if not pq or "predicted" not in next(iter(pq.values()), {}):
                continue

            matrix = {gt: {pred: 0 for pred in labels_order} for gt in labels_order}
            for qid, data in pq.items():
                gt = gt_labels.get(qid, "?")
                pred = data.get("predicted", "?")
                if gt in matrix and pred in matrix[gt]:
                    matrix[gt][pred] += 1

            # Per-class stats
            class_stats = {}
            for label in labels_order:
                tp = matrix[label][label]
                fp = sum(matrix[gt][label] for gt in labels_order if gt != label)
                fn = sum(matrix[label][pred] for pred in labels_order if pred != label)
                prec = tp / (tp + fp) if (tp + fp) else 0
                rec = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
                class_stats[label] = {
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                }

            key = f"{model_safe_raw.replace('-', '_')}_{mode}"
            confusion_results[key] = {
                "matrix": matrix,
                "class_stats": class_stats,
                "n_queries": len(pq),
            }
            logger.info("    %s: %d queries", key, len(pq))

    out = ANALYSIS / "confusion_matrices.json"
    with open(out, "w") as f:
        json.dump(confusion_results, f, indent=2)
    logger.info("  Updated: %s (%d entries)", out, len(confusion_results))

    # Context ablation (now with fresh Claude k=10)
    logger.info("  Re-computing context ablation...")
    ablation = {}
    for provider, model_safe_raw, model_name in MODELS:
        model_safe = f"{provider}_{model_safe_raw}".replace("-", "_")
        model_key = model_safe_raw.replace("_", "-") if "_" in model_safe_raw else model_safe_raw
        model_data = {}
        for k in [1, 3, 5, 10, 20]:
            cp = load_checkpoint(f"t3_{model_safe}_ctx_k{k}")
            if cp is None:
                continue
            model_data[str(k)] = round(cp["metrics"]["accuracy"], 4)
        if model_data:
            ablation[model_key] = model_data

    out = ANALYSIS / "context_ablation.json"
    with open(out, "w") as f:
        json.dump(ablation, f, indent=2)
    logger.info("  Updated: %s", out)

    # Re-run qualitative examples with fresh ZS data
    logger.info("  Re-computing qualitative examples with ZS data...")
    stmts = load_statements()

    # Gather ZS predictions
    zs_preds = {}
    for provider, model_safe_raw, model_name in MODELS:
        model_safe = f"{provider}_{model_safe_raw}".replace("-", "_")
        cp = load_checkpoint(f"t3_{model_safe}_zs")
        if cp and "predicted" in next(iter(cp.get("per_query", {}).values()), {}):
            pq = cp.get("per_query", {})
            zs_preds[model_name] = {qid: d["predicted"] for qid, d in pq.items()}
            logger.info("    ZS %s: %d queries with predicted", model_name, len(pq))

    # ZS confusion matrices
    if zs_preds:
        for model_name, preds in zs_preds.items():
            model_key = {
                "GPT-5.2": "gpt_5.2_zs",
                "Claude Sonnet 4.6": "claude_sonnet_4_6_zs",
                "Gemini 3 Pro": "gemini_3_pro_preview_zs",
            }.get(model_name)
            if not model_key:
                continue
            # Already computed in confusion_results above
            if model_key in confusion_results:
                logger.info("    ZS confusion matrix for %s: already computed", model_name)

    return True


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    logger.info("Phase 4: Final gap-filling")

    # Part A: Pure computation
    logger.info("\n" + "=" * 60)
    logger.info("PART A: Pure Computation")
    logger.info("=" * 60)
    try:
        gt_agreement_stratification()
    except Exception as exc:
        logger.error("FAILED gt_agreement_stratification: %s", exc, exc_info=True)

    try:
        fleiss_kappa_gt()
    except Exception as exc:
        logger.error("FAILED fleiss_kappa_gt: %s", exc, exc_info=True)

    # Part B: API re-runs (only works on VPS with SDKs)
    api_cost = 0
    logger.info("\n" + "=" * 60)
    logger.info("PART B: API Re-runs")
    logger.info("=" * 60)
    try:
        api_cost = api_reruns()
    except ImportError as exc:
        logger.warning("Skipping API re-runs (missing SDK): %s", exc)
    except Exception as exc:
        logger.error("FAILED api_reruns: %s", exc, exc_info=True)

    # Part D: Re-run analyses with fresh checkpoints (before figures)
    logger.info("\n" + "=" * 60)
    logger.info("PART D: Re-run analyses with fresh data")
    logger.info("=" * 60)
    try:
        rerun_analyses()
    except Exception as exc:
        logger.error("FAILED rerun_analyses: %s", exc, exc_info=True)

    # Part C: Regenerate figures
    logger.info("\n" + "=" * 60)
    logger.info("PART C: Regenerate Figures")
    logger.info("=" * 60)
    try:
        regenerate_figures()
    except Exception as exc:
        logger.error("FAILED regenerate_figures: %s", exc, exc_info=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PHASE 4 COMPLETE")
    print(f"{'=' * 60}")
    print(f"Time:     {elapsed:.0f}s")
    print(f"API cost: ${api_cost:.2f}")
    print(f"{'=' * 60}")
    print(f"\nNew analysis files:")
    for f in ["gt_agreement_stratification.json", "fleiss_kappa_gt.json",
              "confusion_matrices.json", "context_ablation.json"]:
        p = ANALYSIS / f
        if p.exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f}")
    print(f"\nFigures in: {ANALYSIS / 'figures'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
