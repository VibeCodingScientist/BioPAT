"""NovEx Evaluation Harness — Tier 1/2/3 for all retrieval methods and LLMs."""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from biopat.llm import CostTracker, create_provider
from biopat.novex._util import CheckpointMixin, parse_llm_json
from biopat.novex.benchmark import NovExBenchmark

logger = logging.getLogger(__name__)


@dataclass
class TierResult:
    tier: int
    method: str
    model: str
    metrics: Dict[str, float]
    per_query: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NovExEvaluator(CheckpointMixin):
    """Evaluation harness for 3 NovEx tiers.

    Usage:
        evaluator = NovExEvaluator(benchmark, results_dir="data/novex/results")
        result = evaluator.run_tier1_bm25()
        result = evaluator.run_tier2("openai", "gpt-4o")
    """

    def __init__(self, benchmark: NovExBenchmark, results_dir: str = "data/novex/results",
                 checkpoint_dir: Optional[str] = None, budget_usd: float = 200.0, seed: int = 42):
        self.benchmark = benchmark
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.results_dir / "checkpoints")
        self.seed = seed
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = CostTracker(max_budget_usd=budget_usd)

    def _cached_or_run(self, cp_name: str, fn):
        """Check checkpoint; if miss, run fn() and cache result."""
        cached = self._load_checkpoint(cp_name)
        if cached is not None:
            return TierResult(**cached)
        result = fn()
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    # ===================== TIER 1: Retrieval =====================

    def run_tier1_bm25(self, top_k=100, k_values=None) -> TierResult:
        k_values = k_values or [10, 50, 100]
        def _run():
            t0 = time.time()
            from biopat.evaluation.bm25 import BM25Evaluator
            ev = BM25Evaluator(benchmark_dir=str(self.benchmark.corpus_dir), results_dir=str(self.results_dir))
            ev.build_index(self.benchmark.corpus)
            raw = ev.retrieve(queries=self.benchmark.queries, top_k=top_k)
            results = _normalize_results(raw)
            m = self._tier1_metrics(results, k_values)
            return TierResult(tier=1, method="bm25", model="N/A", metrics=m,
                              per_query=self._tier1_per_query(results, k_values), elapsed_seconds=time.time()-t0)
        return self._cached_or_run("t1_bm25", _run)

    def run_tier1_dense(self, model_name="BAAI/bge-base-en-v1.5", top_k=100, k_values=None, batch_size=32) -> TierResult:
        k_values = k_values or [10, 50, 100]
        safe = model_name.replace("/", "_")
        def _run():
            t0 = time.time()
            from biopat.evaluation.dense import DenseRetriever, DenseRetrieverConfig
            r = DenseRetriever(DenseRetrieverConfig(model_name=model_name, batch_size=batch_size))
            r.build_index(self.benchmark.corpus)
            results = _normalize_results(r.retrieve(self.benchmark.queries, top_k=top_k))
            m = self._tier1_metrics(results, k_values)
            return TierResult(tier=1, method=f"dense_{safe}", model="N/A", metrics=m,
                              per_query=self._tier1_per_query(results, k_values), elapsed_seconds=time.time()-t0)
        return self._cached_or_run(f"t1_dense_{safe}", _run)

    def run_tier1_hybrid(self, dense_model="BAAI/bge-base-en-v1.5", fusion="rrf", top_k=100, k_values=None) -> TierResult:
        k_values = k_values or [10, 50, 100]
        def _run():
            t0 = time.time()
            from biopat.evaluation.hybrid import BM25DenseHybrid, FusionConfig
            h = BM25DenseHybrid(dense_model=dense_model, fusion_config=FusionConfig(fusion_method=fusion))
            h.build_indices(self.benchmark.corpus)
            results = h.retrieve(self.benchmark.queries, top_k=top_k)
            m = self._tier1_metrics(results, k_values)
            return TierResult(tier=1, method=f"hybrid_{fusion}", model="N/A", metrics=m,
                              per_query=self._tier1_per_query(results, k_values), elapsed_seconds=time.time()-t0)
        return self._cached_or_run(f"t1_hybrid_{fusion}", _run)

    def run_tier1_hyde(self, provider_name="openai", model_id="gpt-4o", embedding_model="BAAI/bge-base-en-v1.5", top_k=100, k_values=None) -> TierResult:
        k_values = k_values or [10, 50, 100]
        safe = f"{provider_name}_{model_id}".replace("-", "_")
        def _run():
            t0 = time.time()
            from biopat.retrieval.hyde import HyDEQueryExpander, HyDEConfig
            from biopat.evaluation.dense import DenseRetriever, DenseRetrieverConfig
            provider = create_provider(provider_name, model=model_id)
            expanded = HyDEQueryExpander(provider=provider, config=HyDEConfig(domain="patent")).expand_batch(self.benchmark.queries)
            r = DenseRetriever(DenseRetrieverConfig(model_name=embedding_model))
            r.build_index(self.benchmark.corpus)
            results = r.retrieve(expanded, top_k=top_k)
            m = self._tier1_metrics(results, k_values)
            return TierResult(tier=1, method="hyde", model=model_id, metrics=m,
                              per_query=self._tier1_per_query(results, k_values), elapsed_seconds=time.time()-t0)
        return self._cached_or_run(f"t1_hyde_{safe}", _run)

    def run_tier1_rerank(self, provider_name="openai", model_id="gpt-4o", bm25_top_k=100, top_k=100, k_values=None) -> TierResult:
        k_values = k_values or [10, 50, 100]
        safe = f"{provider_name}_{model_id}".replace("-", "_")
        def _run():
            t0 = time.time()
            from biopat.evaluation.bm25 import BM25Evaluator
            from biopat.retrieval.reranker import LLMReranker
            ev = BM25Evaluator(benchmark_dir=str(self.benchmark.corpus_dir), results_dir=str(self.results_dir))
            ev.build_index(self.benchmark.corpus)
            bm25_raw = ev.retrieve(queries=self.benchmark.queries, top_k=bm25_top_k)
            bm25 = _normalize_results(bm25_raw)
            provider = create_provider(provider_name, model=model_id)
            reranker = LLMReranker(llm_provider=provider)
            results = {}
            for qid, cands in bm25.items():
                results[qid] = reranker.rerank(self.benchmark.queries[qid], cands, self.benchmark.corpus, top_k)
            m = self._tier1_metrics(results, k_values)
            return TierResult(tier=1, method="bm25_rerank", model=model_id, metrics=m,
                              per_query=self._tier1_per_query(results, k_values), elapsed_seconds=time.time()-t0)
        return self._cached_or_run(f"t1_rerank_{safe}", _run)

    def run_tier1_agent(self, provider_name="openai", model_id="gpt-4o", top_k=100, k_values=None) -> TierResult:
        k_values = k_values or [10, 50, 100]
        safe = f"{provider_name}_{model_id}".replace("-", "_")
        def _run():
            t0 = time.time()
            from biopat.evaluation.agent_retrieval import AgentConfig, DualCorpusSearchTool, RetrievalAgent, results_to_qrels_format
            provider = create_provider(provider_name, model=model_id)
            tool = DualCorpusSearchTool(corpus=self.benchmark.corpus)
            agent = RetrievalAgent(provider=provider, search_tool=tool, config=AgentConfig(final_list_size=top_k))
            traces = [agent.run(query_id=qid, query_text=q) for qid, q in self.benchmark.queries.items()]
            raw = results_to_qrels_format(traces)
            results = _normalize_results(raw)
            m = self._tier1_metrics(results, k_values)
            return TierResult(tier=1, method="agent", model=model_id, metrics=m,
                              per_query=self._tier1_per_query(results, k_values), elapsed_seconds=time.time()-t0)
        return self._cached_or_run(f"t1_agent_{safe}", _run)

    # --- Tier 1 metrics ---

    def _tier1_metrics(self, results: Dict[str, List[Tuple[str, float]]], k_values: List[int]) -> Dict[str, float]:
        qrels = self.benchmark.tier1_qrels
        m: Dict[str, float] = {}
        recall, ndcg, paper_r, patent_r, aps = ({k: [] for k in k_values} for _ in range(4)), {k: [] for k in k_values}, [], [], []
        # Unpack properly
        all_recall = {k: [] for k in k_values}
        all_ndcg = {k: [] for k in k_values}
        all_paper = {k: [] for k in k_values}
        all_patent = {k: [] for k in k_values}
        all_ap = []

        for qid, rel in qrels.items():
            ranked = results.get(qid, [])
            n_rel = len(rel)
            if n_rel == 0:
                continue
            paper_rel = {d for d in rel if self.benchmark.doc_types.get(d, "paper") == "paper"}
            patent_rel = {d for d in rel if self.benchmark.doc_types.get(d, "paper") == "patent"}

            for k in k_values:
                top = set(d for d, _ in ranked[:k])
                all_recall[k].append(len(top & set(rel)) / n_rel)
                if paper_rel:
                    all_paper[k].append(len(top & paper_rel) / len(paper_rel))
                if patent_rel:
                    all_patent[k].append(len(top & patent_rel) / len(patent_rel))
                # NDCG
                dcg = sum(rel[d] / math.log2(r+2) for r, (d, _) in enumerate(ranked[:k]) if d in rel)
                idcg = sum(s / math.log2(r+2) for r, s in enumerate(sorted(rel.values(), reverse=True)[:k]))
                all_ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

            # AP
            hits = 0
            ap = 0.0
            for r, (d, _) in enumerate(ranked):
                if d in rel:
                    hits += 1
                    ap += hits / (r + 1)
            all_ap.append(ap / n_rel)

        for k in k_values:
            m[f"recall@{k}"] = _mean(all_recall[k])
            m[f"ndcg@{k}"] = _mean(all_ndcg[k])
            m[f"paper_recall@{k}"] = _mean(all_paper[k])
            m[f"patent_recall@{k}"] = _mean(all_patent[k])
        m["map"] = _mean(all_ap)
        return m

    def _tier1_per_query(self, results, k_values) -> Dict[str, Dict[str, float]]:
        pq = {}
        for qid, rel in self.benchmark.tier1_qrels.items():
            if not rel or qid not in results:
                continue
            ranked = results[qid]
            pq[qid] = {}
            for k in k_values:
                top = set(d for d, _ in ranked[:k])
                pq[qid][f"recall@{k}"] = len(top & set(rel)) / len(rel)
        return pq

    # ===================== TIER 2: Relevance =====================

    def run_tier2(self, provider_name="openai", model_id="gpt-4o", max_pairs=None) -> TierResult:
        safe = f"{provider_name}_{model_id}".replace("-", "_")
        def _run():
            t0 = time.time()
            provider = create_provider(provider_name, model=model_id)
            preds: Dict[str, Dict[str, int]] = {}
            n = 0
            for qid, docs in self.benchmark.tier2_qrels.items():
                if qid not in self.benchmark.queries:
                    continue
                text = self.benchmark.queries[qid]
                preds[qid] = {}
                for did in docs:
                    if max_pairs and n >= max_pairs:
                        break
                    doc = self.benchmark.corpus.get(did, {})
                    prompt = f"CLAIM: {text}\nDOCUMENT: {doc.get('title','')}\n{doc.get('text','')[:2000]}\nRate 0-3."
                    try:
                        parsed = parse_llm_json(provider, prompt,
                            "Patent relevance expert. Output JSON: {\"score\": <0-3>}",
                            self.cost_tracker, f"{qid}_{did}", "tier2_eval",
                            thinking=True)
                        preds[qid][did] = min(3, max(0, int(parsed.get("score", 0))))
                    except Exception:
                        pass
                    n += 1
                if max_pairs and n >= max_pairs:
                    break
            m = self._tier2_metrics(preds, self.benchmark.tier2_qrels)
            pq = self._tier2_per_query(preds, self.benchmark.tier2_qrels)
            return TierResult(tier=2, method="relevance", model=model_id, metrics=m,
                              per_query=pq, cost_usd=self.cost_tracker.total_cost,
                              elapsed_seconds=time.time()-t0)
        return self._cached_or_run(f"t2_{safe}", _run)

    @staticmethod
    def _tier2_metrics(pred, gt) -> Dict[str, float]:
        ps, gs = [], []
        for qid in pred:
            if qid not in gt:
                continue
            for did in pred[qid]:
                if did in gt[qid]:
                    ps.append(pred[qid][did])
                    gs.append(gt[qid][did])
        n = len(ps)
        if n == 0:
            return {"accuracy": 0, "mae": 0, "weighted_kappa": 0, "num_pairs": 0}
        acc = sum(p == g for p, g in zip(ps, gs)) / n
        mae = sum(abs(p - g) for p, g in zip(ps, gs)) / n
        # Quadratic weighted kappa
        k = 4
        conf = [[0]*k for _ in range(k)]
        for p, g in zip(ps, gs):
            conf[min(k-1,max(0,p))][min(k-1,max(0,g))] += 1
        w = [[(i-j)**2/(k-1)**2 for j in range(k)] for i in range(k)]
        rs = [sum(r) for r in conf]
        cs = [sum(conf[i][j] for i in range(k)) for j in range(k)]
        obs = sum(w[i][j]*conf[i][j] for i in range(k) for j in range(k)) / n
        exp = sum(w[i][j]*rs[i]*cs[j]/(n*n) for i in range(k) for j in range(k))
        kappa = 1 - obs/exp if abs(exp) > 1e-10 else 1.0
        return {"accuracy": acc, "mae": mae, "weighted_kappa": kappa, "num_pairs": float(n)}

    @staticmethod
    def _tier2_per_query(pred, gt) -> Dict[str, Dict[str, float]]:
        pq: Dict[str, Dict[str, float]] = {}
        for qid in pred:
            if qid not in gt:
                continue
            ps, gs = [], []
            for did in pred[qid]:
                if did in gt[qid]:
                    ps.append(pred[qid][did])
                    gs.append(gt[qid][did])
            if not ps:
                continue
            acc = sum(p == g for p, g in zip(ps, gs)) / len(ps)
            mae = sum(abs(p - g) for p, g in zip(ps, gs)) / len(ps)
            pq[qid] = {"accuracy": acc, "mae": mae}
        return pq

    # ===================== TIER 3: Novelty =====================

    def run_tier3(self, provider_name="openai", model_id="gpt-4o", with_context=True, context_k=10) -> TierResult:
        safe = f"{provider_name}_{model_id}".replace("-", "_")
        ctx = "ctx" if with_context else "zs"
        cp_suffix = f"_k{context_k}" if with_context else ""
        def _run():
            t0 = time.time()
            provider = create_provider(provider_name, model=model_id)
            preds = {}
            for qid, gt_label in self.benchmark.tier3_labels.items():
                if qid not in self.benchmark.queries:
                    continue
                text = self.benchmark.queries[qid]
                context = ""
                if with_context and qid in self.benchmark.tier1_qrels:
                    docs = sorted(self.benchmark.tier1_qrels[qid].items(), key=lambda x: -x[1])[:context_k]
                    context = "\n---\n".join(
                        f"[{d}] {self.benchmark.corpus.get(d,{}).get('title','')}\n{self.benchmark.corpus.get(d,{}).get('text','')[:300]}"
                        for d, _ in docs
                    )
                prompt = f"CLAIM: {text}\n\nPRIOR ART:\n{context or 'None.'}\n\nClassify: NOVEL / ANTICIPATED / PARTIALLY_ANTICIPATED"
                try:
                    parsed = parse_llm_json(provider, prompt,
                        'Novelty expert. Output JSON: {"label": "<>", "confidence": <0-1>}',
                        self.cost_tracker, qid, "tier3_eval",
                        thinking=True)
                    raw_label = parsed.get("label", "NOVEL")
                    preds[qid] = raw_label.upper().replace(" ", "_")
                except Exception as exc:
                    logger.warning("Tier3 parse failed for %s (%s): %s", qid, model_id, exc)
                    pass
            m = self._tier3_metrics(preds, self.benchmark.tier3_labels)
            pq: Dict[str, Dict[str, Any]] = {}
            for qid in preds:
                if qid in self.benchmark.tier3_labels:
                    pq[qid] = {
                        "correct": float(preds[qid] == self.benchmark.tier3_labels[qid]),
                        "predicted": preds[qid],
                    }
            return TierResult(tier=3, method=f"novelty_{'ctx' if with_context else 'zs'}", model=model_id,
                              metrics=m, per_query=pq, cost_usd=self.cost_tracker.total_cost,
                              elapsed_seconds=time.time()-t0,
                              metadata={"with_context": with_context, "context_k": context_k})
        return self._cached_or_run(f"t3_{safe}_{ctx}{cp_suffix}", _run)

    @staticmethod
    def _tier3_metrics(pred, gt) -> Dict[str, float]:
        labels = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
        ps = [(pred[q], gt[q]) for q in pred if q in gt]
        if not ps:
            return {"accuracy": 0, "macro_f1": 0}
        acc = sum(p == g for p, g in ps) / len(ps)
        f1s = {}
        for lab in labels:
            tp = sum(p == lab and g == lab for p, g in ps)
            fp = sum(p == lab and g != lab for p, g in ps)
            fn = sum(p != lab and g == lab for p, g in ps)
            prec = tp / (tp+fp) if tp+fp else 0
            rec = tp / (tp+fn) if tp+fn else 0
            f1s[f"f1_{lab.lower()}"] = 2*prec*rec/(prec+rec) if prec+rec else 0
        return {"accuracy": acc, "macro_f1": _mean(list(f1s.values())), **f1s}

    # ===================== Run All =====================

    def run_all(self, config: Dict) -> List[TierResult]:
        results = []
        t1 = config.get("tier1")
        t2 = config.get("tier2")
        t3 = config.get("tier3")

        if t1 is not None:
            kv = t1.get("k_values", [10, 50, 100])
            if t1.get("bm25", {}).get("enabled", True):
                results.append(self.run_tier1_bm25(k_values=kv))
            if t1.get("dense", {}).get("enabled", True):
                for m in t1.get("dense", {}).get("models", ["BAAI/bge-base-en-v1.5"]):
                    results.append(self.run_tier1_dense(m, k_values=kv))
            if t1.get("hybrid", {}).get("enabled", True):
                results.append(self.run_tier1_hybrid(k_values=kv))
            for mc in t1.get("llm_models", []):
                p, mid = mc["provider"], mc["model_id"]
                if t1.get("hyde", {}).get("enabled", True):
                    results.append(self.run_tier1_hyde(p, mid, k_values=kv))
                if t1.get("rerank", {}).get("enabled", True):
                    results.append(self.run_tier1_rerank(p, mid, k_values=kv))
                if t1.get("agent", {}).get("enabled", True):
                    results.append(self.run_tier1_agent(p, mid, k_values=kv))

        if t2 is not None:
            for mc in t2.get("models", []):
                results.append(self.run_tier2(mc["provider"], mc["model_id"], t2.get("max_pairs")))

        if t3 is not None:
            context_k = t3.get("context_k", 10)
            context_k_list = t3.get("context_k_ablation", [])
            for mc in t3.get("models", []):
                results.append(self.run_tier3(mc["provider"], mc["model_id"],
                                              with_context=True, context_k=context_k))
                if t3.get("run_zero_shot", True):
                    results.append(self.run_tier3(mc["provider"], mc["model_id"], with_context=False))
                for k in context_k_list:
                    results.append(self.run_tier3(mc["provider"], mc["model_id"],
                                                  with_context=True, context_k=k))

        with open(self.results_dir / "all_results.json", "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2, default=str)
        self.cost_tracker.save(str(self.results_dir / "costs.json"))
        return results


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _normalize_results(results: Dict) -> Dict[str, List[Tuple[str, float]]]:
    """Convert retrieval results to sorted list-of-tuples format.

    Handles both Dict[str, Dict[str, float]] and Dict[str, List[Tuple[str, float]]].
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, val in results.items():
        if isinstance(val, dict):
            out[qid] = sorted(val.items(), key=lambda x: -x[1])
        else:
            out[qid] = val
    return out
