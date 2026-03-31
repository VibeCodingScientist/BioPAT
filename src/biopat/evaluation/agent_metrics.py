"""Agent-specific metrics for dual-retrieval evaluation (Experiment 7).

Extends standard IR metrics with agent-specific measures:
- Per-doc-type breakdown (paper vs patent recall/precision)
- Coverage balance (does the agent find both types?)
- Search efficiency (recall per search call)
- Refinement gain and curve (does iterative search help?)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_agent_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    doc_types: Dict[str, str],
    traces: Optional[List[Any]] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute comprehensive agent retrieval metrics.

    Args:
        results: {query_id: {doc_id: score}} from agent.
        qrels: {query_id: {doc_id: relevance}} ground truth.
        doc_types: {doc_id: "paper"|"patent"} mapping.
        traces: Optional list of AgentTrace objects for efficiency metrics.
        k_values: Cutoff values for P@k, R@k, NDCG@k.

    Returns:
        Flat dict of metric_name: value.
    """
    k_values = k_values or [10, 50, 100]
    metrics: Dict[str, float] = {}

    # --- Standard IR metrics (overall) ---
    overall = _compute_ir_metrics(results, qrels, k_values)
    metrics.update(overall)

    # --- Per-doc-type metrics ---
    paper_results, patent_results = _split_by_doc_type(results, doc_types)
    paper_qrels, patent_qrels = _split_qrels_by_doc_type(qrels, doc_types)

    if paper_qrels:
        paper_metrics = _compute_ir_metrics(paper_results, paper_qrels, k_values)
        for k, v in paper_metrics.items():
            metrics[f"paper_{k}"] = v

    if patent_qrels:
        patent_metrics = _compute_ir_metrics(patent_results, patent_qrels, k_values)
        for k, v in patent_metrics.items():
            metrics[f"patent_{k}"] = v

    # --- Coverage balance ---
    for k in k_values:
        paper_r = metrics.get(f"paper_recall@{k}", 0.0)
        patent_r = metrics.get(f"patent_recall@{k}", 0.0)
        max_r = max(paper_r, patent_r)
        min_r = min(paper_r, patent_r)
        balance = min_r / max_r if max_r > 0 else 0.0
        metrics[f"coverage_balance@{k}"] = balance

    # --- Agent-specific metrics (require traces) ---
    if traces:
        efficiency = _compute_efficiency_metrics(traces, results, qrels, k_values)
        metrics.update(efficiency)

    return metrics


def compute_refinement_curve(
    traces: List[Any],
    qrels: Dict[str, Dict[str, int]],
    search_tool: Any,
    k: int = 100,
) -> List[Dict[str, float]]:
    """Compute recall@k after each successive search call.

    Shows whether iterative refinement improves retrieval quality.

    Args:
        traces: List of AgentTrace objects.
        qrels: Ground truth qrels.
        search_tool: DualCorpusSearchTool for re-executing searches.
        k: Cutoff for recall computation.

    Returns:
        List of dicts, one per search step: {step, avg_recall, avg_docs_found}.
    """
    from biopat.evaluation.agent_retrieval import AgentTrace

    max_steps = max((len(t.search_calls) for t in traces), default=0)
    curve: List[Dict[str, float]] = []

    for step_idx in range(max_steps):
        recalls: List[float] = []
        doc_counts: List[int] = []

        for trace in traces:
            if step_idx >= len(trace.search_calls):
                continue

            qid = trace.query_id
            if qid not in qrels:
                continue

            # Accumulate results up to this step
            accumulated_docs: Dict[str, float] = {}
            for sc in trace.search_calls[:step_idx + 1]:
                # Re-execute search to get full results
                results = search_tool.search(sc.query, top_k=k)
                for r in results:
                    if r["doc_id"] not in accumulated_docs:
                        accumulated_docs[r["doc_id"]] = r["score"]

            # Compute recall at this step
            relevant = set(qrels[qid].keys())
            retrieved = set(list(accumulated_docs.keys())[:k])
            found = relevant & retrieved
            recall = len(found) / len(relevant) if relevant else 0.0

            recalls.append(recall)
            doc_counts.append(len(accumulated_docs))

        if recalls:
            curve.append({
                "step": step_idx + 1,
                "avg_recall": sum(recalls) / len(recalls),
                "avg_docs_found": sum(doc_counts) / len(doc_counts),
                "num_queries": len(recalls),
            })

    return curve


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_ir_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute NDCG@k, MAP@k, P@k, R@k for given results and qrels."""
    import numpy as np

    metrics: Dict[str, float] = {}
    ndcg_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ap_scores: List[float] = []

    for qid in qrels:
        if qid not in results or not results[qid]:
            for k in k_values:
                ndcg_scores[k].append(0.0)
                precision_scores[k].append(0.0)
                recall_scores[k].append(0.0)
            ap_scores.append(0.0)
            continue

        # Sort retrieved docs by score descending
        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        relevant = qrels[qid]
        num_relevant = sum(1 for v in relevant.values() if v > 0)

        if num_relevant == 0:
            continue

        for k in k_values:
            top_k_ids = ranked_ids[:k]

            # Precision@k
            rel_in_k = sum(1 for did in top_k_ids if relevant.get(did, 0) > 0)
            precision_scores[k].append(rel_in_k / k)

            # Recall@k
            recall_scores[k].append(rel_in_k / num_relevant)

            # NDCG@k
            dcg = 0.0
            for i, did in enumerate(top_k_ids):
                rel = relevant.get(did, 0)
                dcg += rel / np.log2(i + 2)

            ideal_rels = sorted(relevant.values(), reverse=True)[:k]
            idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
            ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

        # MAP (average precision)
        ap = 0.0
        num_found = 0
        for i, did in enumerate(ranked_ids):
            if relevant.get(did, 0) > 0:
                num_found += 1
                ap += num_found / (i + 1)
        ap_scores.append(ap / num_relevant)

    for k in k_values:
        if ndcg_scores[k]:
            metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))
        if precision_scores[k]:
            metrics[f"precision@{k}"] = float(np.mean(precision_scores[k]))
        if recall_scores[k]:
            metrics[f"recall@{k}"] = float(np.mean(recall_scores[k]))

    if ap_scores:
        metrics["map"] = float(np.mean(ap_scores))

    return metrics


def _split_by_doc_type(
    results: Dict[str, Dict[str, float]],
    doc_types: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Split results into paper-only and patent-only subsets."""
    paper_results: Dict[str, Dict[str, float]] = {}
    patent_results: Dict[str, Dict[str, float]] = {}

    for qid, doc_scores in results.items():
        paper_results[qid] = {}
        patent_results[qid] = {}
        for did, score in doc_scores.items():
            dtype = doc_types.get(did, "paper")
            if dtype == "paper":
                paper_results[qid][did] = score
            else:
                patent_results[qid][did] = score

    return paper_results, patent_results


def _split_qrels_by_doc_type(
    qrels: Dict[str, Dict[str, int]],
    doc_types: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Split qrels into paper-only and patent-only subsets."""
    paper_qrels: Dict[str, Dict[str, int]] = {}
    patent_qrels: Dict[str, Dict[str, int]] = {}

    for qid, rels in qrels.items():
        p_rels: Dict[str, int] = {}
        pat_rels: Dict[str, int] = {}
        for did, score in rels.items():
            dtype = doc_types.get(did, "paper")
            if dtype == "paper":
                p_rels[did] = score
            else:
                pat_rels[did] = score
        if p_rels:
            paper_qrels[qid] = p_rels
        if pat_rels:
            patent_qrels[qid] = pat_rels

    return paper_qrels, patent_qrels


def _compute_efficiency_metrics(
    traces: List[Any],
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute agent-specific efficiency metrics from traces."""
    metrics: Dict[str, float] = {}

    search_counts: List[int] = []
    for trace in traces:
        search_counts.append(len(trace.search_calls))

    if search_counts:
        import numpy as np
        metrics["avg_search_calls"] = float(np.mean(search_counts))
        metrics["median_search_calls"] = float(np.median(search_counts))

    # Search efficiency: recall@100 / num_search_calls
    efficiencies: List[float] = []
    for trace in traces:
        qid = trace.query_id
        if qid not in qrels or qid not in results:
            continue
        relevant = set(qrels[qid].keys())
        retrieved = set(list(results[qid].keys())[:100])
        found = relevant & retrieved
        recall = len(found) / len(relevant) if relevant else 0.0
        n_calls = len(trace.search_calls) or 1
        efficiencies.append(recall / n_calls)

    if efficiencies:
        import numpy as np
        metrics["search_efficiency"] = float(np.mean(efficiencies))

    # Refinement gain: compare recall after first search vs final
    gains: List[float] = []
    for trace in traces:
        if len(trace.search_calls) < 2:
            continue
        qid = trace.query_id
        if qid not in qrels:
            continue

        relevant = set(qrels[qid].keys())
        if not relevant:
            continue

        # Final recall
        if qid in results:
            final_retrieved = set(list(results[qid].keys())[:100])
            final_recall = len(relevant & final_retrieved) / len(relevant)
        else:
            final_recall = 0.0

        # We approximate first-search recall from the trace
        # (first search call's top results only)
        first_docs = set(trace.search_calls[0].top_doc_ids) if trace.search_calls else set()
        first_recall = len(relevant & first_docs) / len(relevant)

        gain = final_recall - first_recall
        gains.append(gain)

    if gains:
        import numpy as np
        metrics["avg_refinement_gain"] = float(np.mean(gains))

    # Per-query cost stats
    costs = [trace.total_cost_usd for trace in traces]
    if costs:
        import numpy as np
        metrics["avg_cost_per_query"] = float(np.mean(costs))
        metrics["total_agent_cost"] = float(np.sum(costs))

    return metrics
