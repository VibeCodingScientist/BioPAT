"""NovEx Analysis — vocabulary gap, per-domain, cross-domain, agreement, paper tables."""

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from biopat.evaluation.statistical_tests import (
    bootstrap_confidence_interval,
    bootstrap_paired_test,
)
from biopat.novex.benchmark import NovExBenchmark
from biopat.novex.evaluator import TierResult

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would could "
    "should may might shall can to of in for on with at by from as into through during "
    "before after between out off over under again then once here there when where why "
    "how all both each few more most other some such no nor not only own same so than "
    "too very and but if or because until while that which this these those it its we "
    "our they their".split()
)


def _tokenize(text: str) -> set:
    return set(text.lower().split()) - STOPWORDS


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _compute_ci(vals: List[float]) -> Dict[str, float]:
    """Return mean + 95% bootstrap CI as {mean, ci_lower, ci_upper}."""
    if not vals:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    mean, lower, upper = bootstrap_confidence_interval(vals, n_bootstrap=10000, seed=42)
    return {"mean": round(mean, 4), "ci_lower": round(lower, 4), "ci_upper": round(upper, 4)}


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a-mx)**2 for a in x))
    sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov / (sx * sy) if sx and sy else 0.0


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation (ties handled by average rank)."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    return _pearson(_rank(x), _rank(y))


def _cohens_kappa(a: List[int], b: List[int]) -> float:
    """Cohen's kappa for two lists of categorical labels."""
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    conf = [[0] * k for _ in range(k)]
    for ai, bi in zip(a, b):
        conf[idx[ai]][idx[bi]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row_sums = [sum(conf[i]) for i in range(k)]
    col_sums = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-10 else 1.0


def _fleiss_kappa(ratings: List[List[int]], categories: List[int]) -> float:
    """Fleiss' kappa for multiple raters. ratings[i] = list of rater labels for item i."""
    n = len(ratings)
    if n == 0:
        return 0.0
    k = len(categories)
    cat_idx = {c: j for j, c in enumerate(categories)}
    num_raters = len(ratings[0])
    # Count matrix: n_ij = number of raters who assigned category j to item i
    nij = [[0] * k for _ in range(n)]
    for i, item_ratings in enumerate(ratings):
        for r in item_ratings:
            if r in cat_idx:
                nij[i][cat_idx[r]] += 1
    p_j = [sum(nij[i][j] for i in range(n)) / (n * num_raters) for j in range(k)]
    p_e = sum(pj * pj for pj in p_j)
    p_i = [(sum(nij[i][j] * nij[i][j] for j in range(k)) - num_raters) /
           (num_raters * (num_raters - 1)) if num_raters > 1 else 0 for i in range(n)]
    p_bar = sum(p_i) / n
    return (p_bar - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-10 else 1.0


class NovExAnalyzer:
    """Generate all analysis outputs for the NovEx paper."""

    def __init__(self, benchmark: NovExBenchmark, results: List[TierResult],
                 output_dir: str = "data/novex/analysis"):
        self.b = benchmark
        self.results = results
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t1 = [r for r in results if r.tier == 1]
        self.t2 = [r for r in results if r.tier == 2]
        self.t3 = [r for r in results if r.tier == 3]

    def _load_or_run(self, name: str, fn):
        """Checkpoint wrapper: load from file if exists, else run and save."""
        path = self.out / name
        if path.exists():
            logger.info("Checkpoint hit: %s", name)
            with open(path) as f:
                return json.load(f)
        logger.info("Running: %s", name)
        result = fn()
        return result

    def run_all(self) -> Dict[str, Any]:
        analyses = [
            ("vocabulary_gap.json", self.vocabulary_gap),
            ("per_domain.json", self.per_domain),
            ("cross_domain.json", self.cross_domain),
            ("doc_type_split.json", self.doc_type_split),
            ("tier1_table.json", self.tier1_table),
            ("tier2_table.json", self.tier2_table),
            ("tier3_table.json", self.tier3_table),
            ("inter_model_agreement.json", self.inter_model_agreement),
            ("error_analysis.json", self.error_analysis),
            ("significance_tests.json", self.significance_tests),
            ("tier2_grade_distribution.json", self.tier2_grade_distribution),
            ("tier_correlation.json", self.tier_correlation),
            ("difficulty_stratification.json", self.difficulty_stratification),
            ("cost_performance.json", self.cost_performance),
            ("confusion_matrices.json", self.confusion_matrices),
            ("context_ablation.json", self.context_ablation),
            ("summary.json", self.summary),
        ]
        out = {}
        for name, fn in analyses:
            key = name.replace(".json", "")
            out[key] = self._load_or_run(name, fn)
        self._save("full_analysis.json", out)
        logger.info("Analysis complete — %d outputs saved to %s", len(out), self.out)
        return out

    def vocabulary_gap(self) -> Dict:
        rows = []
        bm25 = next((r for r in self.t1 if r.method == "bm25"), None)
        for sid, s in self.b.statements.items():
            sw = _tokenize(s.text)
            rel = self.b.tier1_qrels.get(sid, {})
            if not sw or not rel:
                continue
            overlaps = []
            for did in rel:
                doc = self.b.corpus.get(did, {})
                dw = _tokenize(doc.get("title", "") + " " + doc.get("text", ""))
                if dw:
                    overlaps.append(len(sw & dw) / len(sw | dw))
            pq = bm25.per_query.get(sid, {}) if bm25 else {}
            rows.append({
                "id": sid, "domain": s.domain,
                "overlap": _mean(overlaps),
                "bm25_r10": pq.get("recall@10", 0),
                "bm25_r100": pq.get("recall@100", 0),
                "bm25_ndcg10": pq.get("ndcg@10", 0),
            })
        overlaps = [r["overlap"] for r in rows]
        corr_r10 = _pearson(overlaps, [r["bm25_r10"] for r in rows])
        corr_r100 = _pearson(overlaps, [r["bm25_r100"] for r in rows])
        corr_ndcg10 = _pearson(overlaps, [r["bm25_ndcg10"] for r in rows])
        # Per-domain correlations
        per_domain = {}
        for d in sorted(set(r["domain"][:3] for r in rows)):
            dr = [r for r in rows if r["domain"].startswith(d)]
            per_domain[d] = {
                "n": len(dr),
                "corr_recall@10": round(_pearson([r["overlap"] for r in dr], [r["bm25_r10"] for r in dr]), 4),
                "corr_ndcg@10": round(_pearson([r["overlap"] for r in dr], [r["bm25_ndcg10"] for r in dr]), 4),
            }
        out = {
            "per_statement": rows,
            "correlation_recall@10": round(corr_r10, 4),
            "correlation_recall@100": round(corr_r100, 4),
            "correlation_ndcg@10": round(corr_ndcg10, 4),
            "per_domain": per_domain,
        }
        self._save("vocabulary_gap.json", out)
        return out

    def per_domain(self) -> Dict:
        domains = sorted(set(s.domain[:3] for s in self.b.statements.values()))
        out = {}
        for d in domains:
            ids = {sid for sid, s in self.b.statements.items() if s.domain.startswith(d)}
            dm = {}
            for r in self.t1:
                vals = [m for qid, m in r.per_query.items() if qid in ids]
                if vals:
                    dm[f"{r.method}/{r.model}"] = {
                        k: _compute_ci([v[k] for v in vals]) for k in vals[0]
                    }
            out[d] = {"count": len(ids), "metrics": dm}
        self._save("per_domain.json", out)
        return out

    def cross_domain(self) -> Dict:
        """Compare retrieval performance across prior-art source categories."""
        cats = {"patents_only", "papers_only", "both", "novel"}
        out = {}
        for r in self.t1:
            entry = {}
            for cat in sorted(cats):
                vals = [m.get("recall@10", 0) for qid, m in r.per_query.items()
                        if self.b.statements.get(qid) and self.b.statements[qid].category == cat]
                if vals:
                    entry[cat] = {**_compute_ci(vals), "count": len(vals)}
            if entry:
                out[f"{r.method}/{r.model}"] = entry
        self._save("cross_domain.json", out)
        return out

    def doc_type_split(self) -> Dict:
        out = {f"{r.method}/{r.model}": {k: v for k, v in r.metrics.items()
               if "paper_recall" in k or "patent_recall" in k or "recall@" in k}
               for r in self.t1}
        self._save("doc_type_split.json", out)
        return out

    def tier1_table(self) -> List[Dict]:
        rows = []
        for r in self.t1:
            row = {"method": r.method, "model": r.model}
            for k, v in r.metrics.items():
                row[k] = round(v, 4)
                per_q_vals = [pq.get(k, 0) for pq in r.per_query.values() if k in pq]
                if per_q_vals:
                    ci = _compute_ci(per_q_vals)
                    row[f"{k}_ci_lower"] = ci["ci_lower"]
                    row[f"{k}_ci_upper"] = ci["ci_upper"]
            rows.append(row)
        self._save("tier1_table.json", rows)
        metric_keys = ["recall@10", "recall@50", "recall@100", "ndcg@10", "map"]
        self._latex("tier1.tex",
                    ["Method", "Model", "R@10", "R@50", "R@100", "NDCG@10", "MAP"],
                    [[r["method"], r["model"]] + [
                        self._fmt_ci(r, k) for k in metric_keys
                    ] for r in rows])
        return rows

    def tier2_table(self) -> List[Dict]:
        rows = []
        for r in self.t2:
            row = {"model": r.model}
            for k, v in r.metrics.items():
                row[k] = round(v, 4)
            for k in ("accuracy", "mae"):
                per_q_vals = [pq.get(k, 0) for pq in r.per_query.values() if k in pq]
                if per_q_vals:
                    ci = _compute_ci(per_q_vals)
                    row[f"{k}_ci_lower"] = ci["ci_lower"]
                    row[f"{k}_ci_upper"] = ci["ci_upper"]
            rows.append(row)
        self._save("tier2_table.json", rows)
        metric_keys = ["accuracy", "mae", "weighted_kappa"]
        self._latex("tier2.tex",
                    ["Model", "Accuracy", "MAE", "QW-$\\kappa$"],
                    [[r["model"]] + [self._fmt_ci(r, k) for k in metric_keys]
                     for r in rows])
        return rows

    def tier3_table(self) -> List[Dict]:
        rows = []
        for r in self.t3:
            row = {"model": r.model, "context": r.metadata.get("with_context", True)}
            for k, v in r.metrics.items():
                row[k] = round(v, 4)
            per_q_vals = [pq.get("correct", 0) for pq in r.per_query.values()]
            if per_q_vals:
                ci = _compute_ci(per_q_vals)
                row["accuracy_ci_lower"] = ci["ci_lower"]
                row["accuracy_ci_upper"] = ci["ci_upper"]
            # Balanced accuracy: mean of per-class recall
            per_class_recall = []
            for label in ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]:
                total = sum(1 for qid in r.per_query if self.b.tier3_labels.get(qid) == label)
                correct = sum(1 for qid, pq in r.per_query.items()
                              if self.b.tier3_labels.get(qid) == label and pq.get("correct", 0) == 1)
                if total > 0:
                    per_class_recall.append(correct / total)
            row["balanced_accuracy"] = round(_mean(per_class_recall), 4) if per_class_recall else 0.0
            rows.append(row)
        self._save("tier3_table.json", rows)
        metric_keys = ["accuracy", "balanced_accuracy", "macro_f1",
                        "f1_novel", "f1_partially_anticipated", "f1_anticipated"]
        self._latex("tier3.tex",
                    ["Model", "Ctx", "Acc", "BAcc", "F1$_{macro}$", "F1$_{N}$", "F1$_{PA}$", "F1$_{A}$"],
                    [[r["model"], "\\checkmark" if r["context"] else "--"] + [
                        self._fmt_ci(r, k) for k in metric_keys
                    ] for r in rows])
        return rows

    def inter_model_agreement(self) -> Dict:
        """Pairwise and multi-rater agreement across LLMs for T2 and T3."""
        out = {}

        # --- T2: Pairwise agreement on per-query accuracy (binarized: above/below median) ---
        if len(self.t2) >= 2:
            t2_agree = {}
            # Get common query IDs across all T2 models
            common_t2 = set.intersection(*(set(r.per_query.keys()) for r in self.t2))
            qids = sorted(common_t2)
            # Per-model binary correctness vectors (accuracy >= 0.5)
            model_vecs = {}
            for r in self.t2:
                model_vecs[r.model] = [int(r.per_query[q]["accuracy"] >= 0.5) for q in qids]
            # Pairwise Cohen's kappa
            models = sorted(model_vecs.keys())
            pairwise = {}
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    kappa = _cohens_kappa(model_vecs[m1], model_vecs[m2])
                    agree_pct = sum(a == b for a, b in zip(model_vecs[m1], model_vecs[m2])) / len(qids)
                    pairwise[f"{m1}_vs_{m2}"] = {
                        "cohens_kappa": round(kappa, 4),
                        "agreement_pct": round(agree_pct, 4),
                        "n_queries": len(qids),
                    }
            # Fleiss' kappa (all raters)
            if len(models) >= 3:
                ratings = [[model_vecs[m][i] for m in models] for i in range(len(qids))]
                fk = _fleiss_kappa(ratings, [0, 1])
                t2_agree["fleiss_kappa"] = round(fk, 4)
            t2_agree["pairwise"] = pairwise
            # Correlation of per-query accuracy (continuous)
            corr_pairs = {}
            for i, m1 in enumerate(models):
                r1 = next(r for r in self.t2 if r.model == m1)
                v1 = [r1.per_query[q]["accuracy"] for q in qids]
                for m2 in models[i+1:]:
                    r2 = next(r for r in self.t2 if r.model == m2)
                    v2 = [r2.per_query[q]["accuracy"] for q in qids]
                    corr_pairs[f"{m1}_vs_{m2}"] = round(_pearson(v1, v2), 4)
            t2_agree["accuracy_correlation"] = corr_pairs
            out["tier2"] = t2_agree

        # --- T3: Pairwise agreement on correctness (binary) ---
        if len(self.t3) >= 2:
            # Group by context mode
            for ctx_label, ctx_val in [("ctx", True), ("zs", False)]:
                t3_sub = [r for r in self.t3 if r.metadata.get("with_context", True) == ctx_val]
                if len(t3_sub) < 2:
                    continue
                common_t3 = set.intersection(*(set(r.per_query.keys()) for r in t3_sub))
                qids = sorted(common_t3)
                model_vecs = {}
                for r in t3_sub:
                    model_vecs[r.model] = [int(r.per_query[q]["correct"]) for q in qids]
                models = sorted(model_vecs.keys())
                pairwise = {}
                for i, m1 in enumerate(models):
                    for m2 in models[i+1:]:
                        kappa = _cohens_kappa(model_vecs[m1], model_vecs[m2])
                        agree_pct = sum(a == b for a, b in zip(model_vecs[m1], model_vecs[m2])) / len(qids)
                        pairwise[f"{m1}_vs_{m2}"] = {
                            "cohens_kappa": round(kappa, 4),
                            "agreement_pct": round(agree_pct, 4),
                            "n_queries": len(qids),
                        }
                t3_entry = {"pairwise": pairwise}
                if len(models) >= 3:
                    ratings = [[model_vecs[m][i] for m in models] for i in range(len(qids))]
                    t3_entry["fleiss_kappa"] = round(_fleiss_kappa(ratings, [0, 1]), 4)
                # Unanimous agreement stats
                all_correct = sum(all(model_vecs[m][i] == 1 for m in models) for i in range(len(qids)))
                all_wrong = sum(all(model_vecs[m][i] == 0 for m in models) for i in range(len(qids)))
                t3_entry["unanimous_correct"] = all_correct
                t3_entry["unanimous_wrong"] = all_wrong
                t3_entry["n_queries"] = len(qids)
                out[f"tier3_{ctx_label}"] = t3_entry

        self._save("inter_model_agreement.json", out)
        return out

    def error_analysis(self) -> Dict:
        """Per-domain, per-category, per-difficulty error breakdowns for T2 and T3."""
        out = {}

        # --- T2 error analysis: accuracy by domain/category ---
        if self.t2:
            t2_errors = {}
            for r in self.t2:
                by_domain = defaultdict(list)
                by_category = defaultdict(list)
                by_difficulty = defaultdict(list)
                for qid, pq in r.per_query.items():
                    stmt = self.b.statements.get(qid)
                    if not stmt:
                        continue
                    acc = pq.get("accuracy", 0)
                    mae = pq.get("mae", 0)
                    by_domain[stmt.domain[:3]].append({"accuracy": acc, "mae": mae})
                    by_category[stmt.category].append({"accuracy": acc, "mae": mae})
                    by_difficulty[stmt.difficulty].append({"accuracy": acc, "mae": mae})
                t2_errors[r.model] = {
                    "by_domain": {d: {"accuracy": _compute_ci([v["accuracy"] for v in vals]),
                                      "mae": _compute_ci([v["mae"] for v in vals]),
                                      "count": len(vals)}
                                  for d, vals in sorted(by_domain.items())},
                    "by_category": {c: {"accuracy": _compute_ci([v["accuracy"] for v in vals]),
                                        "count": len(vals)}
                                    for c, vals in sorted(by_category.items())},
                    "by_difficulty": {d: {"accuracy": _compute_ci([v["accuracy"] for v in vals]),
                                          "count": len(vals)}
                                      for d, vals in sorted(by_difficulty.items())},
                }
            out["tier2"] = t2_errors

        # --- T3 error analysis: per-domain/category + confusion-style breakdown ---
        if self.t3:
            t3_errors = {}
            for r in self.t3:
                ctx_label = "ctx" if r.metadata.get("with_context", True) else "zs"
                key = f"{r.model}_{ctx_label}"
                by_domain = defaultdict(list)
                by_category = defaultdict(list)
                by_difficulty = defaultdict(list)
                # Ground truth class breakdown
                by_gt_label = defaultdict(lambda: {"correct": 0, "total": 0})
                for qid, pq in r.per_query.items():
                    stmt = self.b.statements.get(qid)
                    if not stmt:
                        continue
                    correct = pq.get("correct", 0)
                    by_domain[stmt.domain[:3]].append(correct)
                    by_category[stmt.category].append(correct)
                    by_difficulty[stmt.difficulty].append(correct)
                    gt_label = self.b.tier3_labels.get(qid, "UNKNOWN")
                    by_gt_label[gt_label]["total"] += 1
                    by_gt_label[gt_label]["correct"] += int(correct)

                t3_errors[key] = {
                    "by_domain": {d: {"accuracy": _compute_ci(vals), "count": len(vals)}
                                  for d, vals in sorted(by_domain.items())},
                    "by_category": {c: {"accuracy": _compute_ci(vals), "count": len(vals)}
                                    for c, vals in sorted(by_category.items())},
                    "by_difficulty": {d: {"accuracy": _compute_ci(vals), "count": len(vals)}
                                      for d, vals in sorted(by_difficulty.items())},
                    "by_ground_truth_label": {
                        label: {
                            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
                            "correct": v["correct"],
                            "total": v["total"],
                        }
                        for label, v in sorted(by_gt_label.items())
                    },
                }

            # Hardest queries: wrong by all ctx models
            ctx_results = [r for r in self.t3 if r.metadata.get("with_context", True)]
            if ctx_results:
                all_qids = set.intersection(*(set(r.per_query.keys()) for r in ctx_results))
                hardest = []
                for qid in sorted(all_qids):
                    n_wrong = sum(1 for r in ctx_results if r.per_query[qid]["correct"] == 0)
                    if n_wrong >= 2:  # wrong by majority
                        stmt = self.b.statements.get(qid)
                        hardest.append({
                            "id": qid,
                            "domain": stmt.domain if stmt else "?",
                            "category": stmt.category if stmt else "?",
                            "gt_label": self.b.tier3_labels.get(qid, "?"),
                            "n_models_wrong": n_wrong,
                            "n_models_total": len(ctx_results),
                        })
                t3_errors["hardest_queries"] = sorted(hardest, key=lambda x: -x["n_models_wrong"])

            out["tier3"] = t3_errors

        self._save("error_analysis.json", out)
        return out

    def significance_tests(self) -> Dict:
        """Pairwise bootstrap significance tests between methods/models."""
        out = {}

        # --- T1: pairwise significance on recall@10 ---
        if len(self.t1) >= 2:
            t1_sig = {}
            common = set.intersection(*(set(r.per_query.keys()) for r in self.t1))
            qids = sorted(common)
            for i, ra in enumerate(self.t1):
                for rb in self.t1[i+1:]:
                    a_vals = [ra.per_query[q].get("recall@10", 0) for q in qids]
                    b_vals = [rb.per_query[q].get("recall@10", 0) for q in qids]
                    diff, lo, hi = bootstrap_paired_test(a_vals, b_vals, n_bootstrap=10000, seed=42)
                    sig = lo > 0 or hi < 0  # CI excludes 0
                    key_a = f"{ra.method}/{ra.model}"
                    key_b = f"{rb.method}/{rb.model}"
                    t1_sig[f"{key_a}_vs_{key_b}"] = {
                        "metric": "recall@10",
                        "mean_diff": round(diff, 4),
                        "ci_lower": round(lo, 4),
                        "ci_upper": round(hi, 4),
                        "significant": sig,
                    }
            out["tier1"] = t1_sig

        # --- T2: pairwise significance on accuracy ---
        if len(self.t2) >= 2:
            t2_sig = {}
            common = set.intersection(*(set(r.per_query.keys()) for r in self.t2))
            qids = sorted(common)
            for i, ra in enumerate(self.t2):
                for rb in self.t2[i+1:]:
                    a_vals = [ra.per_query[q].get("accuracy", 0) for q in qids]
                    b_vals = [rb.per_query[q].get("accuracy", 0) for q in qids]
                    diff, lo, hi = bootstrap_paired_test(a_vals, b_vals, n_bootstrap=10000, seed=42)
                    sig = lo > 0 or hi < 0
                    t2_sig[f"{ra.model}_vs_{rb.model}"] = {
                        "metric": "accuracy",
                        "mean_diff": round(diff, 4),
                        "ci_lower": round(lo, 4),
                        "ci_upper": round(hi, 4),
                        "significant": sig,
                    }
            out["tier2"] = t2_sig

        # --- T3: pairwise significance on correctness (ctx only) ---
        ctx_results = [r for r in self.t3 if r.metadata.get("with_context", True)]
        if len(ctx_results) >= 2:
            t3_sig = {}
            common = set.intersection(*(set(r.per_query.keys()) for r in ctx_results))
            qids = sorted(common)
            for i, ra in enumerate(ctx_results):
                for rb in ctx_results[i+1:]:
                    a_vals = [ra.per_query[q].get("correct", 0) for q in qids]
                    b_vals = [rb.per_query[q].get("correct", 0) for q in qids]
                    diff, lo, hi = bootstrap_paired_test(a_vals, b_vals, n_bootstrap=10000, seed=42)
                    sig = lo > 0 or hi < 0
                    t3_sig[f"{ra.model}_vs_{rb.model}"] = {
                        "metric": "accuracy (ctx)",
                        "mean_diff": round(diff, 4),
                        "ci_lower": round(lo, 4),
                        "ci_upper": round(hi, 4),
                        "significant": sig,
                    }
            out["tier3_ctx"] = t3_sig

        self._save("significance_tests.json", out)
        return out

    def summary(self) -> Dict:
        best_t1 = max(self.t1, key=lambda r: r.metrics.get("recall@100", 0)) if self.t1 else None
        best_t2 = max(self.t2, key=lambda r: r.metrics.get("weighted_kappa", 0)) if self.t2 else None
        best_t3 = max(self.t3, key=lambda r: r.metrics.get("macro_f1", 0)) if self.t3 else None
        out = {
            "benchmark": self.b.get_stats(),
            "best_t1": self._summary_ci(best_t1, "recall@100") if best_t1 else None,
            "best_t2": self._summary_ci(best_t2, "weighted_kappa") if best_t2 else None,
            "best_t3": self._summary_ci(best_t3, "macro_f1") if best_t3 else None,
            "total_cost": sum(r.cost_usd for r in self.results),
        }
        self._save("summary.json", out)
        return out

    # ==================== New analyses (VPS-strengthening) ====================

    def tier2_grade_distribution(self) -> Dict:
        """Compute GT grade distribution for T2 — explains high raw agreement + low kappa."""
        grade_counts = Counter()
        total_pairs = 0
        for qid, docs in self.b.tier2_qrels.items():
            for did, grade in docs.items():
                grade_counts[grade] += 1
                total_pairs += 1
        if total_pairs == 0:
            out = {"total_pairs": 0, "grade_distribution": {}, "interpretation": "No T2 data"}
            self._save("tier2_grade_distribution.json", out)
            return out
        dist = {str(g): {"count": c, "proportion": round(c / total_pairs, 4)}
                for g, c in sorted(grade_counts.items())}
        # Prevalence-adjusted interpretation
        max_grade = max(grade_counts, key=grade_counts.get)
        max_prop = grade_counts[max_grade] / total_pairs
        interpretation = (
            f"Grade {max_grade} dominates ({max_prop:.1%} of pairs). "
            f"High raw agreement reflects prevalence effect: two raters agreeing by chance "
            f"on the majority class inflates raw agreement while kappa corrects for this, "
            f"explaining the apparent agreement-kappa paradox."
        )
        out = {
            "total_pairs": total_pairs,
            "grade_distribution": dist,
            "prevalence_index": round(max_prop, 4),
            "dominant_grade": int(max_grade),
            "interpretation": interpretation,
        }
        self._save("tier2_grade_distribution.json", out)
        return out

    def tier_correlation(self) -> Dict:
        """Correlate BM25 recall@10 with T3 context accuracy per query."""
        bm25 = next((r for r in self.t1 if r.method == "bm25"), None)
        if not bm25:
            out = {"error": "No BM25 T1 results found"}
            self._save("tier_correlation.json", out)
            return out
        ctx_results = [r for r in self.t3 if r.metadata.get("with_context", True)]
        if not ctx_results:
            out = {"error": "No T3 context results found"}
            self._save("tier_correlation.json", out)
            return out
        per_model = {}
        all_r10 = []
        all_acc = []
        for r in ctx_results:
            common = sorted(set(bm25.per_query) & set(r.per_query))
            if not common:
                continue
            r10_vals = [bm25.per_query[q].get("recall@10", 0) for q in common]
            acc_vals = [r.per_query[q].get("correct", 0) for q in common]
            per_model[r.model] = {
                "n_queries": len(common),
                "pearson": round(_pearson(r10_vals, acc_vals), 4),
                "spearman": round(_spearman(r10_vals, acc_vals), 4),
                "per_query": [{"qid": q, "bm25_r10": round(r10, 4), "t3_correct": acc}
                              for q, r10, acc in zip(common, r10_vals, acc_vals)],
            }
            all_r10.extend(r10_vals)
            all_acc.extend(acc_vals)
        out = {
            "per_model": per_model,
            "pooled_pearson": round(_pearson(all_r10, all_acc), 4),
            "pooled_spearman": round(_spearman(all_r10, all_acc), 4),
            "pooled_n": len(all_r10),
        }
        self._save("tier_correlation.json", out)
        return out

    def difficulty_stratification(self) -> Dict:
        """Post-hoc difficulty from BM25 recall@10: Hard <0.3, Medium 0.3-0.7, Easy >0.7."""
        bm25 = next((r for r in self.t1 if r.method == "bm25"), None)
        if not bm25:
            out = {"error": "No BM25 T1 results found"}
            self._save("difficulty_stratification.json", out)
            return out
        # Assign difficulty buckets
        buckets: Dict[str, str] = {}
        for qid, pq in bm25.per_query.items():
            r10 = pq.get("recall@10", 0)
            if r10 < 0.3:
                buckets[qid] = "hard"
            elif r10 <= 0.7:
                buckets[qid] = "medium"
            else:
                buckets[qid] = "easy"
        bucket_counts = Counter(buckets.values())
        # T1 stratification
        t1_strat = {}
        for r in self.t1:
            by_bucket: Dict[str, list] = defaultdict(list)
            for qid, pq in r.per_query.items():
                if qid in buckets:
                    by_bucket[buckets[qid]].append(pq.get("recall@10", 0))
            t1_strat[f"{r.method}/{r.model}"] = {
                b: {**_compute_ci(vals), "count": len(vals)}
                for b, vals in sorted(by_bucket.items())
            }
        # T2 stratification
        t2_strat = {}
        for r in self.t2:
            by_bucket: Dict[str, list] = defaultdict(list)
            for qid, pq in r.per_query.items():
                if qid in buckets:
                    by_bucket[buckets[qid]].append(pq.get("accuracy", 0))
            t2_strat[r.model] = {
                b: {**_compute_ci(vals), "count": len(vals)}
                for b, vals in sorted(by_bucket.items())
            }
        # T3 stratification
        t3_strat = {}
        for r in self.t3:
            ctx_label = "ctx" if r.metadata.get("with_context", True) else "zs"
            by_bucket: Dict[str, list] = defaultdict(list)
            for qid, pq in r.per_query.items():
                if qid in buckets:
                    by_bucket[buckets[qid]].append(pq.get("correct", 0))
            t3_strat[f"{r.model}_{ctx_label}"] = {
                b: {**_compute_ci(vals), "count": len(vals)}
                for b, vals in sorted(by_bucket.items())
            }
        out = {
            "bucket_thresholds": {"hard": "<0.3", "medium": "0.3-0.7", "easy": ">0.7"},
            "bucket_counts": dict(bucket_counts),
            "tier1": t1_strat,
            "tier2": t2_strat,
            "tier3": t3_strat,
        }
        self._save("difficulty_stratification.json", out)
        return out

    def cost_performance(self) -> Dict:
        """Cost vs primary metric for each tier/method — cost-effectiveness analysis."""
        out = {"tier1": [], "tier2": [], "tier3": []}
        # T1: cost by method (BM25 is free; rerank/agent have LLM costs)
        prev_cost = 0.0
        for r in sorted(self.results, key=lambda x: x.cost_usd):
            if r.tier == 1:
                incr_cost = max(0, r.cost_usd - prev_cost) if r.cost_usd > 0 else 0.0
                out["tier1"].append({
                    "method": r.method, "model": r.model,
                    "recall@10": round(r.metrics.get("recall@10", 0), 4),
                    "ndcg@10": round(r.metrics.get("ndcg@10", 0), 4),
                    "cost_usd": round(r.cost_usd, 2),
                    "incremental_cost": round(incr_cost, 2),
                })
        # T2
        for r in self.t2:
            out["tier2"].append({
                "model": r.model,
                "accuracy": round(r.metrics.get("accuracy", 0), 4),
                "weighted_kappa": round(r.metrics.get("weighted_kappa", 0), 4),
                "cost_usd": round(r.cost_usd, 2),
            })
        # T3
        for r in self.t3:
            ctx_label = "ctx" if r.metadata.get("with_context", True) else "zs"
            out["tier3"].append({
                "model": r.model, "context": ctx_label,
                "accuracy": round(r.metrics.get("accuracy", 0), 4),
                "macro_f1": round(r.metrics.get("macro_f1", 0), 4),
                "cost_usd": round(r.cost_usd, 2),
            })
        total = sum(r.cost_usd for r in self.results)
        out["total_cost_usd"] = round(total, 2)
        self._save("cost_performance.json", out)
        return out

    def confusion_matrices(self) -> Dict:
        """T3 confusion matrices from per_query predicted labels (Phase 2 — requires re-run)."""
        labels = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
        out = {}
        for r in self.t3:
            ctx_label = "ctx" if r.metadata.get("with_context", True) else "zs"
            key = f"{r.model}_{ctx_label}"
            # Check if predicted labels are stored
            has_predicted = any("predicted" in pq for pq in r.per_query.values())
            if not has_predicted:
                continue
            matrix = {gt: {pr: 0 for pr in labels} for gt in labels}
            for qid, pq in r.per_query.items():
                gt = self.b.tier3_labels.get(qid)
                pred = pq.get("predicted")
                if gt in labels and pred in labels:
                    matrix[gt][pred] += 1
            # Per-class precision/recall
            class_stats = {}
            for lab in labels:
                tp = matrix[lab][lab]
                fp = sum(matrix[gt][lab] for gt in labels if gt != lab)
                fn = sum(matrix[lab][pr] for pr in labels if pr != lab)
                prec = tp / (tp + fp) if (tp + fp) else 0
                rec = tp / (tp + fn) if (tp + fn) else 0
                class_stats[lab] = {
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0,
                }
            out[key] = {"matrix": matrix, "class_stats": class_stats}
        self._save("confusion_matrices.json", out)
        return out

    def context_ablation(self) -> Dict:
        """Context quantity ablation: accuracy at varying k values (Phase 2 — requires runs)."""
        # Look for T3 results with context_k metadata
        ablation_data: Dict[str, Dict[int, float]] = defaultdict(dict)
        for r in self.t3:
            if not r.metadata.get("with_context", True):
                continue
            k = r.metadata.get("context_k", 10)
            acc = r.metrics.get("accuracy", 0)
            ablation_data[r.model][k] = round(acc, 4)
        out = {}
        for model, kv in sorted(ablation_data.items()):
            out[model] = dict(sorted(kv.items()))
        self._save("context_ablation.json", out)
        return out

    @staticmethod
    def _summary_ci(result: TierResult, metric: str) -> Dict:
        entry = {"method": result.method, "model": result.model, metric: result.metrics.get(metric)}
        per_q_vals = [pq.get(metric, 0) for pq in result.per_query.values() if metric in pq]
        if not per_q_vals and metric in ("macro_f1", "weighted_kappa"):
            # For aggregate metrics without per-query breakdown, use "correct" or "accuracy"
            fallback = "correct" if result.tier == 3 else "accuracy"
            per_q_vals = [pq.get(fallback, 0) for pq in result.per_query.values() if fallback in pq]
        if per_q_vals:
            ci = _compute_ci(per_q_vals)
            entry[f"{metric}_ci_lower"] = ci["ci_lower"]
            entry[f"{metric}_ci_upper"] = ci["ci_upper"]
        return entry

    @staticmethod
    def _fmt_ci(row: Dict, metric: str) -> str:
        """Format metric as 'mean [lower, upper]' for LaTeX."""
        val = row.get(metric, 0)
        lo = row.get(f"{metric}_ci_lower")
        hi = row.get(f"{metric}_ci_upper")
        if lo is not None and hi is not None:
            return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"
        return f"{val:.3f}"

    def _save(self, name: str, data):
        with open(self.out / name, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _latex(self, name: str, headers: List[str], rows: List[List[str]]):
        lines = ["\\begin{table}[t]", "\\centering",
                 f"\\begin{{tabular}}{{{'l' + 'c' * (len(headers)-1)}}}",
                 "\\toprule",
                 " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
                 "\\midrule"]
        for row in rows:
            lines.append(" & ".join(str(c) for c in row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        with open(self.out / name, "w") as f:
            f.write("\n".join(lines))
