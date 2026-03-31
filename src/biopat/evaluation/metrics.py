"""Metrics computation module.

Computes standard IR evaluation metrics for the benchmark.

Phase 5 (v2.0): Extended to support multi-dimensional reporting
with breakdowns by document type (papers vs patents).

Phase 6 (v3.0): Extended to support per-jurisdiction reporting
(US, EP, WO) for international patent coverage benchmark.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import math

logger = logging.getLogger(__name__)

# Document type constants for v2.0
DOC_TYPE_PAPER = "paper"
DOC_TYPE_PATENT = "patent"

# Jurisdiction constants for v3.0
JURISDICTION_US = "US"
JURISDICTION_EP = "EP"
JURISDICTION_WO = "WO"
ALL_JURISDICTIONS = [JURISDICTION_US, JURISDICTION_EP, JURISDICTION_WO]


class MetricsComputer:
    """Computes IR evaluation metrics."""

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """Compute Precision@k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.
            k: Cutoff rank.

        Returns:
            Precision@k value.
        """
        if k <= 0:
            return 0.0

        retrieved_k = retrieved[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """Compute Recall@k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.
            k: Cutoff rank.

        Returns:
            Recall@k value.
        """
        if not relevant:
            return 0.0

        retrieved_k = set(retrieved[:k])
        hits = len(retrieved_k & relevant)
        return hits / len(relevant)

    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        """Compute Average Precision.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.

        Returns:
            Average Precision value.
        """
        if not relevant:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                hits += 1
                precision_at_i = hits / i
                sum_precisions += precision_at_i

        return sum_precisions / len(relevant)

    @staticmethod
    def dcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, int],
        k: int,
    ) -> float:
        """Compute Discounted Cumulative Gain at k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevance_scores: Dictionary mapping doc_id to relevance score.
            k: Cutoff rank.

        Returns:
            DCG@k value.
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 1)
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, int],
        k: int,
    ) -> float:
        """Compute Normalized DCG at k.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevance_scores: Dictionary mapping doc_id to relevance score.
            k: Cutoff rank.

        Returns:
            NDCG@k value.
        """
        dcg = MetricsComputer.dcg_at_k(retrieved, relevance_scores, k)

        # Compute ideal DCG
        sorted_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(sorted_rels, 1):
            idcg += (2 ** rel - 1) / math.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        """Compute Mean Reciprocal Rank for a single query.

        Args:
            retrieved: List of retrieved document IDs (ranked).
            relevant: Set of relevant document IDs.

        Returns:
            Reciprocal rank (1/position of first relevant doc).
        """
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    def compute_all_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Compute all metrics for a result set.

        Args:
            results: Retrieved results {query_id: {doc_id: score}}.
            qrels: Ground truth {query_id: {doc_id: relevance}}.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of aggregated metrics.
        """
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_ndcgs = {k: [] for k in k_values}
        all_aps = []
        all_mrrs = []

        for query_id, retrieved_scores in results.items():
            if query_id not in qrels:
                continue

            query_qrels = qrels[query_id]
            relevant = set(query_qrels.keys())

            # Sort retrieved by score
            retrieved = sorted(
                retrieved_scores.keys(),
                key=lambda x: retrieved_scores[x],
                reverse=True
            )

            # Compute metrics at each k
            for k in k_values:
                all_precisions[k].append(self.precision_at_k(retrieved, relevant, k))
                all_recalls[k].append(self.recall_at_k(retrieved, relevant, k))
                all_ndcgs[k].append(self.ndcg_at_k(retrieved, query_qrels, k))

            # MAP and MRR
            all_aps.append(self.average_precision(retrieved, relevant))
            all_mrrs.append(self.mrr(retrieved, relevant))

        # Aggregate
        metrics = {}

        for k in k_values:
            if all_precisions[k]:
                metrics[f"P@{k}"] = sum(all_precisions[k]) / len(all_precisions[k])
            if all_recalls[k]:
                metrics[f"Recall@{k}"] = sum(all_recalls[k]) / len(all_recalls[k])
            if all_ndcgs[k]:
                metrics[f"NDCG@{k}"] = sum(all_ndcgs[k]) / len(all_ndcgs[k])

        if all_aps:
            metrics["MAP"] = sum(all_aps) / len(all_aps)
        if all_mrrs:
            metrics["MRR"] = sum(all_mrrs) / len(all_mrrs)

        return metrics

    def compute_per_domain_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        query_domains: Dict[str, str],
        k_values: List[int] = [10, 100],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by domain.

        Args:
            results: Retrieved results.
            qrels: Ground truth.
            query_domains: Mapping of query_id to domain.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of domain to metrics.
        """
        # Group queries by domain
        domain_results = {}
        domain_qrels = {}

        for query_id in results:
            if query_id not in qrels:
                continue

            domain = query_domains.get(query_id, "unknown")

            if domain not in domain_results:
                domain_results[domain] = {}
                domain_qrels[domain] = {}

            domain_results[domain][query_id] = results[query_id]
            domain_qrels[domain][query_id] = qrels[query_id]

        # Compute metrics per domain
        domain_metrics = {}
        for domain in domain_results:
            domain_metrics[domain] = self.compute_all_metrics(
                domain_results[domain],
                domain_qrels[domain],
                k_values
            )
            domain_metrics[domain]["num_queries"] = len(domain_results[domain])

        return domain_metrics

    def format_metrics_table(
        self,
        metrics: Dict[str, float],
        name: str = "Model",
    ) -> str:
        """Format metrics as a readable table.

        Args:
            metrics: Dictionary of metrics.
            name: Model/run name.

        Returns:
            Formatted string.
        """
        lines = [f"\n{'=' * 50}", f"Results for: {name}", "=" * 50]

        # Sort metrics for consistent display
        sorted_metrics = sorted(metrics.items())

        for metric_name, value in sorted_metrics:
            lines.append(f"{metric_name:15s}: {value:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def compute_metrics_by_doc_type(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        doc_types: Dict[str, str],
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by document type (v2.0).

        Multi-dimensional reporting for dual-corpus benchmark:
        - Overall: Combined performance across all documents
        - Papers Only: Performance on scientific literature subset
        - Patents Only: Performance on prior patent subset

        Args:
            results: Retrieved results {query_id: {doc_id: score}}.
            qrels: Ground truth {query_id: {doc_id: relevance}}.
            doc_types: Mapping of doc_id to doc_type ("paper" or "patent").
            k_values: Values of k for metrics.

        Returns:
            Dictionary with metrics for each subset:
            {"overall": {...}, "papers": {...}, "patents": {...}}
        """
        # Overall metrics
        overall_metrics = self.compute_all_metrics(results, qrels, k_values)

        # Filter qrels by doc_type
        paper_qrels = {}
        patent_qrels = {}

        for query_id, doc_rels in qrels.items():
            paper_docs = {
                doc_id: rel for doc_id, rel in doc_rels.items()
                if doc_types.get(doc_id) == DOC_TYPE_PAPER
            }
            patent_docs = {
                doc_id: rel for doc_id, rel in doc_rels.items()
                if doc_types.get(doc_id) == DOC_TYPE_PATENT
            }

            if paper_docs:
                paper_qrels[query_id] = paper_docs
            if patent_docs:
                patent_qrels[query_id] = patent_docs

        # Compute metrics for each subset
        paper_metrics = self.compute_all_metrics(results, paper_qrels, k_values) if paper_qrels else {}
        patent_metrics = self.compute_all_metrics(results, patent_qrels, k_values) if patent_qrels else {}

        # Add counts
        overall_metrics["num_queries"] = len(qrels)
        overall_metrics["num_relevant_docs"] = sum(len(docs) for docs in qrels.values())

        paper_metrics["num_queries"] = len(paper_qrels)
        paper_metrics["num_relevant_docs"] = sum(len(docs) for docs in paper_qrels.values())

        patent_metrics["num_queries"] = len(patent_qrels)
        patent_metrics["num_relevant_docs"] = sum(len(docs) for docs in patent_qrels.values())

        return {
            "overall": overall_metrics,
            "papers": paper_metrics,
            "patents": patent_metrics,
        }

    def compute_cross_type_retrieval_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        doc_types: Dict[str, str],
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, Any]:
        """Compute detailed cross-type retrieval analysis (v2.0).

        Analyzes how well the system retrieves each document type
        and whether there's bias toward papers or patents.

        Args:
            results: Retrieved results {query_id: {doc_id: score}}.
            qrels: Ground truth {query_id: {doc_id: relevance}}.
            doc_types: Mapping of doc_id to doc_type.
            k_values: Values of k for metrics.

        Returns:
            Detailed analysis dictionary.
        """
        analysis = {
            "by_doc_type": self.compute_metrics_by_doc_type(
                results, qrels, doc_types, k_values
            ),
            "retrieval_bias": {},
            "per_query_analysis": [],
        }

        # Analyze retrieval bias
        for k in k_values:
            paper_retrieved = 0
            patent_retrieved = 0
            total_relevant_papers = 0
            total_relevant_patents = 0

            for query_id, retrieved_scores in results.items():
                if query_id not in qrels:
                    continue

                query_qrels = qrels[query_id]

                # Count relevant by type
                for doc_id, rel in query_qrels.items():
                    if rel > 0:
                        if doc_types.get(doc_id) == DOC_TYPE_PAPER:
                            total_relevant_papers += 1
                        elif doc_types.get(doc_id) == DOC_TYPE_PATENT:
                            total_relevant_patents += 1

                # Count retrieved by type at k
                sorted_docs = sorted(
                    retrieved_scores.keys(),
                    key=lambda x: retrieved_scores[x],
                    reverse=True
                )[:k]

                for doc_id in sorted_docs:
                    if doc_id in query_qrels and query_qrels[doc_id] > 0:
                        if doc_types.get(doc_id) == DOC_TYPE_PAPER:
                            paper_retrieved += 1
                        elif doc_types.get(doc_id) == DOC_TYPE_PATENT:
                            patent_retrieved += 1

            # Calculate bias metrics
            paper_recall = paper_retrieved / total_relevant_papers if total_relevant_papers > 0 else 0
            patent_recall = patent_retrieved / total_relevant_patents if total_relevant_patents > 0 else 0

            analysis["retrieval_bias"][f"@{k}"] = {
                "paper_recall": paper_recall,
                "patent_recall": patent_recall,
                "bias_ratio": paper_recall / patent_recall if patent_recall > 0 else float('inf'),
                "papers_retrieved": paper_retrieved,
                "patents_retrieved": patent_retrieved,
                "total_relevant_papers": total_relevant_papers,
                "total_relevant_patents": total_relevant_patents,
            }

        return analysis

    def format_dual_corpus_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        name: str = "Model",
    ) -> str:
        """Format dual-corpus metrics as a detailed report (v2.0).

        Args:
            metrics: Metrics dictionary from compute_metrics_by_doc_type.
            name: Model/run name.

        Returns:
            Formatted report string.
        """
        lines = [
            f"\n{'=' * 60}",
            f"DUAL-CORPUS EVALUATION REPORT: {name}",
            "=" * 60,
        ]

        # Overall metrics
        lines.append("\n## OVERALL METRICS")
        lines.append("-" * 40)
        for metric_name, value in sorted(metrics.get("overall", {}).items()):
            if isinstance(value, float):
                lines.append(f"  {metric_name:20s}: {value:.4f}")
            else:
                lines.append(f"  {metric_name:20s}: {value}")

        # Papers metrics
        lines.append("\n## PAPERS ONLY")
        lines.append("-" * 40)
        for metric_name, value in sorted(metrics.get("papers", {}).items()):
            if isinstance(value, float):
                lines.append(f"  {metric_name:20s}: {value:.4f}")
            else:
                lines.append(f"  {metric_name:20s}: {value}")

        # Patents metrics
        lines.append("\n## PATENTS ONLY")
        lines.append("-" * 40)
        for metric_name, value in sorted(metrics.get("patents", {}).items()):
            if isinstance(value, float):
                lines.append(f"  {metric_name:20s}: {value:.4f}")
            else:
                lines.append(f"  {metric_name:20s}: {value}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def compute_metrics_by_jurisdiction(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        doc_jurisdictions: Dict[str, str],
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by jurisdiction (v3.0).

        Multi-dimensional reporting for international benchmark:
        - Overall: Combined performance across all jurisdictions
        - US: Performance on US patents subset
        - EP: Performance on EP patents subset
        - WO: Performance on WO/PCT patents subset

        Args:
            results: Retrieved results {query_id: {doc_id: score}}.
            qrels: Ground truth {query_id: {doc_id: relevance}}.
            doc_jurisdictions: Mapping of doc_id to jurisdiction ("US", "EP", "WO").
            k_values: Values of k for metrics.

        Returns:
            Dictionary with metrics for each subset:
            {"overall": {...}, "US": {...}, "EP": {...}, "WO": {...}}
        """
        # Overall metrics
        overall_metrics = self.compute_all_metrics(results, qrels, k_values)

        # Filter qrels by jurisdiction
        jurisdiction_qrels = {jur: {} for jur in ALL_JURISDICTIONS}

        for query_id, doc_rels in qrels.items():
            for jur in ALL_JURISDICTIONS:
                jur_docs = {
                    doc_id: rel for doc_id, rel in doc_rels.items()
                    if doc_jurisdictions.get(doc_id) == jur
                }
                if jur_docs:
                    jurisdiction_qrels[jur][query_id] = jur_docs

        # Compute metrics for each jurisdiction
        result = {"overall": overall_metrics}

        for jur in ALL_JURISDICTIONS:
            if jurisdiction_qrels[jur]:
                jur_metrics = self.compute_all_metrics(
                    results, jurisdiction_qrels[jur], k_values
                )
                jur_metrics["num_queries"] = len(jurisdiction_qrels[jur])
                jur_metrics["num_relevant_docs"] = sum(
                    len(docs) for docs in jurisdiction_qrels[jur].values()
                )
            else:
                jur_metrics = {"num_queries": 0, "num_relevant_docs": 0}

            result[jur] = jur_metrics

        # Add overall counts
        result["overall"]["num_queries"] = len(qrels)
        result["overall"]["num_relevant_docs"] = sum(
            len(docs) for docs in qrels.values()
        )

        return result

    def compute_cross_jurisdiction_analysis(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        doc_jurisdictions: Dict[str, str],
        query_jurisdictions: Optional[Dict[str, str]] = None,
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, Any]:
        """Compute detailed cross-jurisdiction analysis (v3.0).

        Analyzes retrieval performance across jurisdictions,
        including cross-jurisdiction transfer (e.g., US query retrieving EP docs).

        Args:
            results: Retrieved results {query_id: {doc_id: score}}.
            qrels: Ground truth {query_id: {doc_id: relevance}}.
            doc_jurisdictions: Mapping of doc_id to jurisdiction.
            query_jurisdictions: Optional mapping of query_id to jurisdiction.
            k_values: Values of k for metrics.

        Returns:
            Detailed analysis dictionary.
        """
        analysis = {
            "by_jurisdiction": self.compute_metrics_by_jurisdiction(
                results, qrels, doc_jurisdictions, k_values
            ),
            "retrieval_distribution": {},
            "cross_jurisdiction_matrix": {},
        }

        # Analyze retrieval distribution per k
        for k in k_values:
            jur_counts = {jur: {"retrieved": 0, "relevant": 0} for jur in ALL_JURISDICTIONS}
            jur_counts["other"] = {"retrieved": 0, "relevant": 0}

            for query_id, retrieved_scores in results.items():
                if query_id not in qrels:
                    continue

                query_qrels = qrels[query_id]

                # Count relevant by jurisdiction
                for doc_id, rel in query_qrels.items():
                    if rel > 0:
                        jur = doc_jurisdictions.get(doc_id, "other")
                        if jur in jur_counts:
                            jur_counts[jur]["relevant"] += 1
                        else:
                            jur_counts["other"]["relevant"] += 1

                # Count retrieved by jurisdiction at k
                sorted_docs = sorted(
                    retrieved_scores.keys(),
                    key=lambda x: retrieved_scores[x],
                    reverse=True
                )[:k]

                for doc_id in sorted_docs:
                    if doc_id in query_qrels and query_qrels[doc_id] > 0:
                        jur = doc_jurisdictions.get(doc_id, "other")
                        if jur in jur_counts:
                            jur_counts[jur]["retrieved"] += 1
                        else:
                            jur_counts["other"]["retrieved"] += 1

            # Calculate recall per jurisdiction
            analysis["retrieval_distribution"][f"@{k}"] = {}
            for jur, counts in jur_counts.items():
                if counts["relevant"] > 0:
                    recall = counts["retrieved"] / counts["relevant"]
                else:
                    recall = 0.0

                analysis["retrieval_distribution"][f"@{k}"][jur] = {
                    "recall": recall,
                    "retrieved": counts["retrieved"],
                    "relevant": counts["relevant"],
                }

        # Cross-jurisdiction matrix (if query jurisdictions provided)
        if query_jurisdictions:
            matrix = {
                q_jur: {d_jur: {"hits": 0, "total": 0}
                        for d_jur in ALL_JURISDICTIONS + ["other"]}
                for q_jur in ALL_JURISDICTIONS + ["other"]
            }

            for query_id, retrieved_scores in results.items():
                if query_id not in qrels:
                    continue

                q_jur = query_jurisdictions.get(query_id, "other")
                query_qrels = qrels[query_id]

                # Count relevant docs by jurisdiction for this query
                for doc_id, rel in query_qrels.items():
                    if rel > 0:
                        d_jur = doc_jurisdictions.get(doc_id, "other")
                        matrix[q_jur][d_jur]["total"] += 1

                # Check top-100 retrievals
                sorted_docs = sorted(
                    retrieved_scores.keys(),
                    key=lambda x: retrieved_scores[x],
                    reverse=True
                )[:100]

                for doc_id in sorted_docs:
                    if doc_id in query_qrels and query_qrels[doc_id] > 0:
                        d_jur = doc_jurisdictions.get(doc_id, "other")
                        matrix[q_jur][d_jur]["hits"] += 1

            analysis["cross_jurisdiction_matrix"] = matrix

        return analysis

    def format_jurisdiction_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        name: str = "Model",
    ) -> str:
        """Format per-jurisdiction metrics as a detailed report (v3.0).

        Args:
            metrics: Metrics dictionary from compute_metrics_by_jurisdiction.
            name: Model/run name.

        Returns:
            Formatted report string.
        """
        lines = [
            f"\n{'=' * 70}",
            f"INTERNATIONAL BENCHMARK EVALUATION REPORT: {name}",
            "=" * 70,
        ]

        # Overall metrics
        lines.append("\n## OVERALL METRICS")
        lines.append("-" * 50)
        for metric_name, value in sorted(metrics.get("overall", {}).items()):
            if isinstance(value, float):
                lines.append(f"  {metric_name:20s}: {value:.4f}")
            else:
                lines.append(f"  {metric_name:20s}: {value}")

        # Per-jurisdiction metrics
        for jur in ALL_JURISDICTIONS:
            jur_metrics = metrics.get(jur, {})
            if not jur_metrics or jur_metrics.get("num_queries", 0) == 0:
                continue

            lines.append(f"\n## {jur} PATENTS")
            lines.append("-" * 50)
            for metric_name, value in sorted(jur_metrics.items()):
                if isinstance(value, float):
                    lines.append(f"  {metric_name:20s}: {value:.4f}")
                else:
                    lines.append(f"  {metric_name:20s}: {value}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def format_full_international_report(
        self,
        jurisdiction_metrics: Dict[str, Dict[str, float]],
        doc_type_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        cross_analysis: Optional[Dict[str, Any]] = None,
        name: str = "Model",
    ) -> str:
        """Format comprehensive international benchmark report (v3.0).

        Combines jurisdiction breakdown, doc type breakdown, and cross-analysis
        into a single detailed report.

        Args:
            jurisdiction_metrics: From compute_metrics_by_jurisdiction.
            doc_type_metrics: Optional from compute_metrics_by_doc_type.
            cross_analysis: Optional from compute_cross_jurisdiction_analysis.
            name: Model/run name.

        Returns:
            Formatted comprehensive report string.
        """
        lines = [
            f"\n{'=' * 80}",
            f"BIOPAT v3.0 INTERNATIONAL BENCHMARK REPORT",
            f"Model: {name}",
            "=" * 80,
        ]

        # Section 1: Overall Performance
        lines.append("\n" + "#" * 80)
        lines.append("# SECTION 1: OVERALL PERFORMANCE")
        lines.append("#" * 80)

        overall = jurisdiction_metrics.get("overall", {})
        lines.append(f"\n  Total Queries:         {overall.get('num_queries', 0)}")
        lines.append(f"  Total Relevant Docs:   {overall.get('num_relevant_docs', 0)}")
        lines.append("")

        key_metrics = ["MAP", "MRR", "NDCG@10", "NDCG@100", "Recall@100"]
        for m in key_metrics:
            if m in overall:
                lines.append(f"  {m:20s}: {overall[m]:.4f}")

        # Section 2: Performance by Jurisdiction
        lines.append("\n" + "#" * 80)
        lines.append("# SECTION 2: PERFORMANCE BY JURISDICTION")
        lines.append("#" * 80)

        # Table header
        lines.append(f"\n  {'Jurisdiction':<12} {'Queries':>10} {'Rel Docs':>10} {'MAP':>10} {'NDCG@10':>10} {'Recall@100':>12}")
        lines.append("  " + "-" * 66)

        for jur in ALL_JURISDICTIONS:
            jur_m = jurisdiction_metrics.get(jur, {})
            if jur_m.get("num_queries", 0) > 0:
                lines.append(
                    f"  {jur:<12} "
                    f"{jur_m.get('num_queries', 0):>10} "
                    f"{jur_m.get('num_relevant_docs', 0):>10} "
                    f"{jur_m.get('MAP', 0):>10.4f} "
                    f"{jur_m.get('NDCG@10', 0):>10.4f} "
                    f"{jur_m.get('Recall@100', 0):>12.4f}"
                )

        # Section 3: Performance by Document Type (if available)
        if doc_type_metrics:
            lines.append("\n" + "#" * 80)
            lines.append("# SECTION 3: PERFORMANCE BY DOCUMENT TYPE")
            lines.append("#" * 80)

            lines.append(f"\n  {'Doc Type':<12} {'Queries':>10} {'Rel Docs':>10} {'MAP':>10} {'NDCG@10':>10} {'Recall@100':>12}")
            lines.append("  " + "-" * 66)

            for dtype, label in [("papers", "Papers"), ("patents", "Patents")]:
                dt_m = doc_type_metrics.get(dtype, {})
                if dt_m.get("num_queries", 0) > 0:
                    lines.append(
                        f"  {label:<12} "
                        f"{dt_m.get('num_queries', 0):>10} "
                        f"{dt_m.get('num_relevant_docs', 0):>10} "
                        f"{dt_m.get('MAP', 0):>10.4f} "
                        f"{dt_m.get('NDCG@10', 0):>10.4f} "
                        f"{dt_m.get('Recall@100', 0):>12.4f}"
                    )

        # Section 4: Cross-Jurisdiction Analysis (if available)
        if cross_analysis and "retrieval_distribution" in cross_analysis:
            lines.append("\n" + "#" * 80)
            lines.append("# SECTION 4: CROSS-JURISDICTION RETRIEVAL ANALYSIS")
            lines.append("#" * 80)

            dist = cross_analysis.get("retrieval_distribution", {})
            if "@100" in dist:
                lines.append("\n  Recall@100 by Document Jurisdiction:")
                for jur in ALL_JURISDICTIONS:
                    jur_data = dist["@100"].get(jur, {})
                    if jur_data.get("relevant", 0) > 0:
                        lines.append(
                            f"    {jur}: {jur_data.get('recall', 0):.4f} "
                            f"({jur_data.get('retrieved', 0)}/{jur_data.get('relevant', 0)} docs)"
                        )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
