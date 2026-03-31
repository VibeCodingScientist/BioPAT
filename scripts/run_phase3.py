#!/usr/bin/env python3
"""Phase 3: Pre-writing analyses from existing checkpoint data.

Pure computation — no API calls. Reads checkpoints + benchmark files to produce:
  1. ablation_significance.json  — bootstrap paired tests between k values
  2. qualitative_examples.json   — hardest queries, interesting disagreements
  3. dataset_statistics.json     — domain dist, doc counts, lengths, label balance
  4. calibration_note.json       — documents that confidence scores are unavailable

Usage:
    PYTHONPATH=src python3 scripts/run_phase3.py
"""

import importlib.util
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase3")

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "novex"
CHECKPOINTS = DATA / "results" / "checkpoints"
ANALYSIS = DATA / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("openai", "gpt_5.2", "GPT-5.2"),
    ("anthropic", "claude_sonnet_4_6", "Claude Sonnet 4.6"),
    ("google", "gemini_3_pro_preview", "Gemini 3 Pro"),
]
K_VALUES = [1, 3, 5, 10, 20]
SEED = 42


def load_checkpoint(name):
    """Load a checkpoint JSON file."""
    p = CHECKPOINTS / f"{name}.json"
    if not p.exists():
        logger.warning("Checkpoint not found: %s", p)
        return None
    with open(p) as f:
        return json.load(f)


def load_statements():
    """Load statements from JSONL."""
    stmts = {}
    p = DATA / "statements.jsonl"
    with open(p) as f:
        for line in f:
            s = json.loads(line)
            stmts[s["statement_id"]] = s
    return stmts


def load_tier3_qrels():
    """Load tier3 ground truth labels from qrels/tier3.tsv."""
    labels = {}
    label_map = {0: "NOVEL", 1: "PARTIALLY_ANTICIPATED", 2: "ANTICIPATED"}
    p = DATA / "qrels" / "tier3.tsv"
    with open(p) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("query_id"):
                continue  # skip header
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                score = int(parts[2])
                labels[qid] = label_map.get(score, "UNKNOWN")
    return labels


def load_tier1_qrels():
    """Load tier1 ground truth from qrels/tier1.tsv."""
    qrels = defaultdict(dict)
    p = DATA / "qrels" / "tier1.tsv"
    with open(p) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("query_id"):
                continue  # skip header
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did = parts[0], parts[1]
                grade = int(parts[2]) if len(parts) >= 3 else 0
                qrels[qid][did] = grade
    return dict(qrels)


# ============================================================
# Analysis 1: Ablation Significance Tests
# ============================================================
def ablation_significance():
    """Bootstrap paired test between consecutive k values for each model."""
    logger.info("=== Analysis 1: Ablation Significance Tests ===")
    rng = random.Random(SEED)
    n_boot = 10000

    results = {}

    for provider, model_safe, model_name in MODELS:
        logger.info("  Model: %s", model_name)
        model_results = {}

        # Load per-query accuracy vectors for each k
        k_vectors = {}
        for k in K_VALUES:
            cp = load_checkpoint(f"t3_{provider}_{model_safe}_ctx_k{k}")
            if cp is None:
                continue
            pq = cp.get("per_query", {})
            # Build ordered accuracy vector
            qids = sorted(pq.keys())
            vec = [pq[qid]["correct"] for qid in qids]
            k_vectors[k] = (qids, vec)

        def _paired_bootstrap(k_a, k_b):
            """Bootstrap paired test on intersection of queries for k_a vs k_b."""
            if k_a not in k_vectors or k_b not in k_vectors:
                return None
            qids_a_set = set(k_vectors[k_a][0])
            qids_b_set = set(k_vectors[k_b][0])
            common = sorted(qids_a_set & qids_b_set)
            if len(common) < 10:
                logger.warning("    k%d vs k%d: only %d common queries, skipping", k_a, k_b, len(common))
                return None

            pq_a = {q: k_vectors[k_a][1][k_vectors[k_a][0].index(q)] for q in common}
            pq_b = {q: k_vectors[k_b][1][k_vectors[k_b][0].index(q)] for q in common}
            n = len(common)

            mean_a = sum(pq_a[q] for q in common) / n
            mean_b = sum(pq_b[q] for q in common) / n
            obs_diff = mean_b - mean_a

            diffs = [pq_b[q] - pq_a[q] for q in common]
            boot_means = []
            for _ in range(n_boot):
                sample = rng.choices(diffs, k=n)
                boot_means.append(sum(sample) / n)

            boot_means.sort()
            ci_lo = boot_means[int(0.025 * n_boot)]
            ci_hi = boot_means[int(0.975 * n_boot)]

            if obs_diff >= 0:
                p_val = sum(1 for b in boot_means if b <= 0) / n_boot
            else:
                p_val = sum(1 for b in boot_means if b >= 0) / n_boot
            p_val = min(p_val * 2, 1.0)

            significant = ci_lo > 0 or ci_hi < 0
            return {
                "k_from": k_a,
                "k_to": k_b,
                "n_queries": n,
                "acc_from": round(mean_a, 4),
                "acc_to": round(mean_b, 4),
                "diff": round(obs_diff, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "p_value": round(p_val, 4),
                "significant_at_05": significant,
            }

        # Pairwise tests: consecutive + key jumps
        all_pairs = []
        for i in range(len(K_VALUES) - 1):
            all_pairs.append((K_VALUES[i], K_VALUES[i + 1]))
        all_pairs.extend([(1, 10), (1, 20), (5, 20)])

        for k_a, k_b in all_pairs:
            pair_key = f"k{k_a}_vs_k{k_b}"
            if pair_key in model_results:
                continue
            r = _paired_bootstrap(k_a, k_b)
            if r is None:
                continue
            model_results[pair_key] = r
            logger.info("    %s (n=%d): diff=%.4f, CI=[%.4f, %.4f], p=%.4f %s",
                        pair_key, r["n_queries"], r["diff"],
                        r["ci_95"][0], r["ci_95"][1], r["p_value"],
                        "*" if r["significant_at_05"] else "")

        results[model_name] = model_results

    out = ANALYSIS / "ablation_significance.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("  Saved: %s", out)
    return results


# ============================================================
# Analysis 2: Qualitative Examples Table
# ============================================================
def qualitative_examples():
    """Find hardest queries, interesting disagreements, model-specific strengths."""
    logger.info("=== Analysis 2: Qualitative Examples ===")
    stmts = load_statements()
    gt_labels = load_tier3_qrels()

    # Load predictions — prefer k=10, fallback to largest available checkpoint
    model_preds = {}
    for provider, model_safe, model_name in MODELS:
        best_cp = None
        best_n = 0
        for k in [10, 20, 5, 3, 1]:
            cp = load_checkpoint(f"t3_{provider}_{model_safe}_ctx_k{k}")
            if cp and len(cp.get("per_query", {})) > best_n:
                best_cp = cp
                best_n = len(cp.get("per_query", {}))
        if best_cp is None:
            continue
        pq = best_cp.get("per_query", {})
        model_preds[model_name] = {qid: d["predicted"] for qid, d in pq.items()}
        logger.info("  Loaded %s: %d queries", model_name, len(pq))

    # Use queries covered by ALL models
    common_qids = set(gt_labels.keys())
    for mp in model_preds.values():
        common_qids &= set(mp.keys())
    qids = sorted(common_qids)
    logger.info("  Common queries across all models: %d", len(qids))

    # --- Category 1: All models wrong ---
    all_wrong = []
    for qid in qids:
        gt = gt_labels[qid]
        wrong_count = 0
        preds = {}
        for mname, mp in model_preds.items():
            pred = mp.get(qid, "?")
            preds[mname] = pred
            if pred != gt:
                wrong_count += 1
        if wrong_count == len(model_preds):
            all_wrong.append({
                "statement_id": qid,
                "text": stmts[qid]["text"][:200] + ("..." if len(stmts[qid]["text"]) > 200 else ""),
                "domain": stmts[qid].get("domain", "?"),
                "category": stmts[qid].get("category", "?"),
                "ground_truth": gt,
                "predictions": preds,
            })
    logger.info("  All-wrong: %d statements", len(all_wrong))

    # --- Category 2: Exactly one model correct ---
    one_right = defaultdict(list)
    for qid in qids:
        gt = gt_labels[qid]
        correct_models = []
        preds = {}
        for mname, mp in model_preds.items():
            pred = mp.get(qid, "?")
            preds[mname] = pred
            if pred == gt:
                correct_models.append(mname)
        if len(correct_models) == 1:
            one_right[correct_models[0]].append({
                "statement_id": qid,
                "text": stmts[qid]["text"][:200] + ("..." if len(stmts[qid]["text"]) > 200 else ""),
                "domain": stmts[qid].get("domain", "?"),
                "category": stmts[qid].get("category", "?"),
                "ground_truth": gt,
                "predictions": preds,
            })
    for mname, items in one_right.items():
        logger.info("  Only %s correct: %d statements", mname, len(items))

    # --- Category 3: All models correct ---
    all_correct = []
    for qid in qids:
        gt = gt_labels[qid]
        preds = {}
        all_right = True
        for mname, mp in model_preds.items():
            pred = mp.get(qid, "?")
            preds[mname] = pred
            if pred != gt:
                all_right = False
        if all_right:
            all_correct.append(qid)

    # --- Category 4: Interesting NOVEL cases (most misclassified) ---
    novel_cases = []
    for qid in qids:
        gt = gt_labels[qid]
        if gt != "NOVEL":
            continue
        preds = {}
        for mname, mp in model_preds.items():
            preds[mname] = mp.get(qid, "?")
        novel_cases.append({
            "statement_id": qid,
            "text": stmts[qid]["text"][:200] + ("..." if len(stmts[qid]["text"]) > 200 else ""),
            "domain": stmts[qid].get("domain", "?"),
            "category": stmts[qid].get("category", "?"),
            "ground_truth": gt,
            "predictions": preds,
            "num_correct": sum(1 for p in preds.values() if p == gt),
        })

    # --- Category 5: Context sensitivity (big delta between k=1 and k=20) ---
    # Use k=1 vs k=20 since Claude k=10 has fewer queries
    context_sensitive = []
    for qid in qids:
        gt = gt_labels[qid]
        for provider, model_safe, model_name in MODELS:
            cp_lo = load_checkpoint(f"t3_{provider}_{model_safe}_ctx_k1")
            cp_hi = load_checkpoint(f"t3_{provider}_{model_safe}_ctx_k20")
            if cp_lo is None or cp_hi is None:
                continue
            pq_lo = cp_lo.get("per_query", {})
            pq_hi = cp_hi.get("per_query", {})
            if qid not in pq_lo or qid not in pq_hi:
                continue
            pred_lo = pq_lo[qid].get("predicted", "?")
            pred_hi = pq_hi[qid].get("predicted", "?")
            corr_lo = pq_lo[qid].get("correct", 0)
            corr_hi = pq_hi[qid].get("correct", 0)
            if corr_lo != corr_hi:
                context_sensitive.append({
                    "statement_id": qid,
                    "model": model_name,
                    "ground_truth": gt,
                    "pred_k1": pred_lo,
                    "pred_k20": pred_hi,
                    "improved": corr_hi > corr_lo,
                })

    # Count improvements vs regressions
    improved = sum(1 for c in context_sensitive if c["improved"])
    regressed = len(context_sensitive) - improved
    logger.info("  Context-sensitive: %d improved, %d regressed (k1→k10)", improved, regressed)

    # Select top examples for each category
    result = {
        "all_wrong": all_wrong[:10],
        "all_wrong_count": len(all_wrong),
        "one_model_correct": {
            mname: items[:5] for mname, items in one_right.items()
        },
        "one_model_correct_counts": {mname: len(items) for mname, items in one_right.items()},
        "all_correct_count": len(all_correct),
        "novel_cases": sorted(novel_cases, key=lambda x: x["num_correct"])[:10],
        "novel_total": len(novel_cases),
        "context_sensitivity": {
            "improved_count": improved,
            "regressed_count": regressed,
            "examples_improved": [c for c in context_sensitive if c["improved"]][:5],
            "examples_regressed": [c for c in context_sensitive if not c["improved"]][:5],
        },
    }

    out = ANALYSIS / "qualitative_examples.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("  Saved: %s", out)
    return result


# ============================================================
# Analysis 3: Dataset Statistics Table
# ============================================================
def dataset_statistics():
    """Comprehensive dataset statistics for the paper."""
    logger.info("=== Analysis 3: Dataset Statistics ===")
    stmts = load_statements()
    gt_labels = load_tier3_qrels()
    t1_qrels = load_tier1_qrels()

    # --- Domain distribution ---
    domain_counts = Counter()
    for s in stmts.values():
        domain_counts[s.get("domain", "unknown")] += 1

    # --- Category distribution ---
    cat_counts = Counter()
    for s in stmts.values():
        cat_counts[s.get("category", "unknown")] += 1

    # --- Statement text lengths ---
    lengths = [len(s["text"]) for s in stmts.values()]
    word_counts = [len(s["text"].split()) for s in stmts.values()]
    lengths.sort()
    word_counts.sort()

    def percentiles(arr):
        if not arr:
            return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0, "std": 0}
        n = len(arr)
        mean = sum(arr) / n
        return {
            "min": arr[0],
            "p25": arr[int(0.25 * n)],
            "median": arr[int(0.5 * n)],
            "p75": arr[int(0.75 * n)],
            "max": arr[-1],
            "mean": round(mean, 1),
            "std": round((sum((x - mean)**2 for x in arr) / n) ** 0.5, 1),
        }

    # --- Novelty label distribution ---
    label_counts = Counter(gt_labels.values())

    # --- T1 relevant docs per query ---
    docs_per_query = [len(docs) for docs in t1_qrels.values()]
    docs_per_query.sort()

    # --- Grade distribution in T1 ---
    grade_counts = Counter()
    for docs in t1_qrels.values():
        for grade in docs.values():
            grade_counts[grade] += 1

    # --- Source type ---
    source_types = Counter()
    for s in stmts.values():
        has_patent = bool(s.get("source_patent_id"))
        has_paper = bool(s.get("source_paper_id"))
        if has_patent and has_paper:
            source_types["both"] += 1
        elif has_patent:
            source_types["patent"] += 1
        elif has_paper:
            source_types["paper"] += 1
        else:
            source_types["unknown"] += 1

    # --- Agreement distribution ---
    agreement_counts = Counter()
    for s in stmts.values():
        agreement_counts[s["ground_truth"].get("tier3_agreement", "unknown")] += 1

    # --- Corpus size (count unique docs in T1 qrels) ---
    all_docs = set()
    for docs in t1_qrels.values():
        all_docs.update(docs.keys())

    result = {
        "num_statements": len(stmts),
        "num_unique_corpus_docs": len(all_docs),
        "domain_distribution": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
        "category_distribution": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        "source_type_distribution": dict(source_types),
        "novelty_label_distribution": {
            "NOVEL": label_counts.get("NOVEL", 0),
            "PARTIALLY_ANTICIPATED": label_counts.get("PARTIALLY_ANTICIPATED", 0),
            "ANTICIPATED": label_counts.get("ANTICIPATED", 0),
        },
        "novelty_label_proportions": {
            k: round(v / sum(label_counts.values()), 4)
            for k, v in sorted(label_counts.items())
        },
        "tier3_agreement": dict(sorted(agreement_counts.items(), key=lambda x: -x[1])),
        "statement_length_chars": percentiles(lengths),
        "statement_length_words": percentiles(word_counts),
        "relevant_docs_per_query": percentiles(docs_per_query),
        "tier1_grade_distribution": {
            str(k): v for k, v in sorted(grade_counts.items())
        },
        "total_relevance_judgments": sum(grade_counts.values()),
    }

    out = ANALYSIS / "dataset_statistics.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("  Saved: %s", out)

    # Print summary
    logger.info("  Statements: %d", result["num_statements"])
    logger.info("  Corpus docs: %d", result["num_unique_corpus_docs"])
    logger.info("  Novelty: NOVEL=%d, PARTIAL=%d, ANTICIPATED=%d",
                label_counts.get("NOVEL", 0),
                label_counts.get("PARTIALLY_ANTICIPATED", 0),
                label_counts.get("ANTICIPATED", 0))
    logger.info("  Domains: %s", dict(domain_counts.most_common(5)))
    logger.info("  Statement length: median=%d words", percentiles(word_counts)["median"])
    logger.info("  Docs per query: median=%d", percentiles(docs_per_query)["median"])

    return result


# ============================================================
# Analysis 4: Calibration Note
# ============================================================
def calibration_note():
    """Document that confidence scores are not available in current checkpoints."""
    logger.info("=== Analysis 4: Calibration Note ===")

    # Verify checkpoint structure
    sample_keys = set()
    for provider, model_safe, model_name in MODELS:
        cp = load_checkpoint(f"t3_{provider}_{model_safe}_ctx_k10")
        if cp:
            pq = cp.get("per_query", {})
            for qid, data in pq.items():
                sample_keys.update(data.keys())
                break

    # Proxy: compute "implicit calibration" from agreement patterns
    # If a model's prediction matches the majority vote more often, it's better calibrated
    stmts = load_statements()
    proxy_calibration = {}

    for provider, model_safe, model_name in MODELS:
        cp = load_checkpoint(f"t3_{provider}_{model_safe}_ctx_k10")
        if cp is None:
            continue
        pq = cp.get("per_query", {})

        # Distribution of predictions vs GT distribution
        pred_dist = Counter()
        gt_dist = Counter()
        for qid, data in pq.items():
            pred_dist[data["predicted"]] += 1
            # Get GT from statements
            s = stmts.get(qid, {})
            gt_label = s.get("ground_truth", {}).get("tier3_novelty_label", "?")
            gt_dist[gt_label] += 1

        # Jensen-Shannon-like divergence proxy (simple version)
        total = sum(pred_dist.values())
        labels = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
        pred_props = {l: pred_dist.get(l, 0) / total for l in labels}
        gt_props = {l: gt_dist.get(l, 0) / total for l in labels}

        # KL-divergence (pred || gt), with smoothing
        eps = 1e-10
        kl = sum(
            pred_props[l] * math.log((pred_props[l] + eps) / (gt_props[l] + eps))
            for l in labels
        )

        proxy_calibration[model_name] = {
            "prediction_distribution": {l: pred_dist.get(l, 0) for l in labels},
            "ground_truth_distribution": {l: gt_dist.get(l, 0) for l in labels},
            "prediction_proportions": {l: round(pred_props[l], 4) for l in labels},
            "gt_proportions": {l: round(gt_props[l], 4) for l in labels},
            "kl_divergence_pred_vs_gt": round(kl, 4),
        }
        logger.info("  %s: pred_dist=%s, KL=%.4f",
                     model_name, dict(pred_dist), kl)

    result = {
        "checkpoint_keys_available": sorted(sample_keys),
        "confidence_scores_available": False,
        "note": "T3 checkpoints store 'correct' (0/1) and 'predicted' (label string) but not confidence/probability scores. Full calibration analysis (ECE, reliability diagrams) requires re-running T3 with logprob extraction. As a proxy, we report prediction distribution alignment with ground truth distribution.",
        "proxy_calibration": proxy_calibration,
    }

    out = ANALYSIS / "calibration_note.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("  Saved: %s", out)
    return result


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    logger.info("Phase 3: Pre-writing analyses")

    analyses = [
        ("ablation_significance", ablation_significance),
        ("qualitative_examples", qualitative_examples),
        ("dataset_statistics", dataset_statistics),
        ("calibration_note", calibration_note),
    ]

    results = {}
    for name, fn in analyses:
        try:
            results[name] = fn()
        except Exception as exc:
            logger.error("FAILED %s: %s", name, exc, exc_info=True)

    elapsed = time.time() - t0
    logger.info("Phase 3 complete: %d/%d analyses in %.1fs",
                len(results), len(analyses), elapsed)

    print(f"\n{'='*60}")
    print(f"PHASE 3 COMPLETE")
    print(f"{'='*60}")
    print(f"Analyses: {len(results)}/{len(analyses)}")
    print(f"Time:     {elapsed:.1f}s")
    for name in results:
        print(f"  ✓ {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
