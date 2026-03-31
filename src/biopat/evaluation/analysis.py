"""Results analysis for BioPAT benchmark experiments."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes experiment results for publication.

    Provides per-domain analysis, cost-effectiveness comparison,
    vocabulary gap quantification, and rank correlations between models.
    """

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all experiment results from summary file."""
        summary_path = self.results_dir / "experiment_summary.json"
        if not summary_path.exists():
            logger.warning("No experiment summary found at %s", summary_path)
            return []
        with open(summary_path) as f:
            data = json.load(f)
        return data.get("results", [])

    def per_domain_analysis(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        query_domains: Dict[str, str],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Breakdown metrics by IPC domain (A61/C07/C12).

        Args:
            results: {query_id: {doc_id: score}} retrieval results.
            qrels: {query_id: {doc_id: relevance}} ground truth.
            query_domains: {query_id: domain} mapping.
            k_values: Values of k for metrics.

        Returns:
            {domain: {metric_name: value}} nested dict.
        """
        k_values = k_values or [10, 50, 100]

        # Group queries by domain
        domain_queries: Dict[str, List[str]] = {}
        for qid, domain in query_domains.items():
            domain_queries.setdefault(domain, []).append(qid)

        from biopat.evaluation.metrics import MetricsComputer

        mc = MetricsComputer()
        domain_metrics: Dict[str, Dict[str, float]] = {}

        for domain, qids in sorted(domain_queries.items()):
            domain_results = {qid: results[qid] for qid in qids if qid in results}
            domain_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}

            if domain_results and domain_qrels:
                metrics = mc.compute_all_metrics(domain_results, domain_qrels, k_values)
                domain_metrics[domain] = metrics

        return domain_metrics

    def cost_effectiveness(
        self,
        experiment_results: List[Dict[str, Any]],
        metric_key: str = "NDCG@10",
    ) -> List[Dict[str, Any]]:
        """Compute metric-per-dollar for each model.

        Args:
            experiment_results: List of ExperimentResult dicts.
            metric_key: Which metric to use for comparison.

        Returns:
            Sorted list of {model, metric, cost, metric_per_dollar}.
        """
        analysis = []
        for r in experiment_results:
            metric_val = r.get("metrics", {}).get(metric_key, 0)
            cost = r.get("cost_usd", 0)
            analysis.append({
                "experiment": r.get("experiment", ""),
                "model": r.get("model", ""),
                "metric": metric_val,
                "cost_usd": cost,
                "metric_per_dollar": metric_val / cost if cost > 0 else float("inf"),
            })

        analysis.sort(key=lambda x: x["metric_per_dollar"], reverse=True)
        return analysis

    def vocabulary_gap_analysis(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, dict],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        """Quantify the patent-vs-paper vocabulary gap.

        Computes:
        - Jaccard similarity between query and relevant doc vocabularies
        - Unique term ratio (terms appearing only in queries or only in docs)
        - Average overlap per query
        """
        import re

        def tokenize(text: str) -> set:
            return set(re.findall(r'\b[a-z][a-z0-9-]{2,}\b', text.lower()))

        overlaps = []
        unique_query_ratios = []

        for qid, query_text in queries.items():
            query_tokens = tokenize(query_text)
            if not query_tokens or qid not in qrels:
                continue

            # Get relevant doc tokens
            rel_docs = qrels[qid]
            doc_tokens: set = set()
            for did, score in rel_docs.items():
                if score > 0 and did in corpus:
                    doc = corpus[did]
                    text = f"{doc.get('title', '')} {doc.get('text', '')}" if isinstance(doc, dict) else str(doc)
                    doc_tokens |= tokenize(text)

            if not doc_tokens:
                continue

            # Jaccard similarity
            intersection = query_tokens & doc_tokens
            union = query_tokens | doc_tokens
            jaccard = len(intersection) / len(union) if union else 0

            overlaps.append(jaccard)

            # Unique query term ratio
            unique_q = query_tokens - doc_tokens
            unique_query_ratios.append(len(unique_q) / len(query_tokens) if query_tokens else 0)

        import numpy as np

        return {
            "mean_jaccard_similarity": float(np.mean(overlaps)) if overlaps else 0,
            "median_jaccard_similarity": float(np.median(overlaps)) if overlaps else 0,
            "mean_unique_query_term_ratio": float(np.mean(unique_query_ratios)),
            "num_queries_analyzed": len(overlaps),
        }

    def correlation_analysis(
        self,
        system_rankings: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute rank correlations (Kendall's tau) between systems.

        Args:
            system_rankings: {system_name: {query_id: score}}.

        Returns:
            {sys_a: {sys_b: tau}} pairwise correlations.
        """
        from scipy.stats import kendalltau

        systems = list(system_rankings.keys())
        # Get common query IDs
        all_qids = set()
        for scores in system_rankings.values():
            all_qids |= set(scores.keys())
        common_qids = sorted(all_qids)

        correlations: Dict[str, Dict[str, float]] = {}
        for sys_a in systems:
            correlations[sys_a] = {}
            scores_a = [system_rankings[sys_a].get(qid, 0) for qid in common_qids]
            for sys_b in systems:
                scores_b = [system_rankings[sys_b].get(qid, 0) for qid in common_qids]
                tau, _ = kendalltau(scores_a, scores_b)
                correlations[sys_a][sys_b] = float(tau)

        return correlations
