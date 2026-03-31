"""LLM Benchmark Runner — orchestrates all 7 experiment types.

Evaluates frontier LLMs on the BioPAT benchmark across:
  1. BM25 baseline (no LLM)
  2. Dense retrieval baselines (no LLM)
  3. HyDE query expansion (per LLM)
  4. Listwise LLM reranking (per LLM)
  5. LLM relevance judgment (per LLM)
  6. Full novelty assessment (per LLM)
  7. Agent-based dual retrieval (per LLM)
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification for a model under evaluation."""

    name: str
    provider: str
    model_id: str
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    experiment: str
    model: str
    metrics: Dict[str, float]
    cost_usd: float = 0.0
    num_queries: int = 0
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMBenchmarkRunner:
    """Orchestrates all LLM evaluation experiments on BioPAT.

    Example:
        runner = LLMBenchmarkRunner(
            benchmark_dir="data/benchmark",
            results_dir="data/results",
        )
        runner.load_benchmark()
        results = runner.run_all(experiments_config)
    """

    def __init__(
        self,
        benchmark_dir: str = "data/benchmark",
        results_dir: str = "data/results",
        checkpoint_dir: Optional[str] = None,
        budget_usd: Optional[float] = 500.0,
        seed: int = 42,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.results_dir / "checkpoints")
        self.seed = seed

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Cost tracker
        from biopat.llm import CostTracker

        self.cost_tracker = CostTracker(max_budget_usd=budget_usd)

        # Data (loaded lazily)
        self.corpus: Dict[str, dict] = {}
        self.queries: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}
        self._loaded = False

    # --- Data loading ---

    def load_benchmark(self, split: str = "test") -> None:
        """Load BEIR-format benchmark data."""
        logger.info("Loading benchmark from %s", self.benchmark_dir)

        # Corpus
        corpus_path = self.benchmark_dir / "corpus.jsonl"
        if corpus_path.exists():
            with open(corpus_path) as f:
                for line in f:
                    doc = json.loads(line)
                    self.corpus[doc["_id"]] = {
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                    }

        # Queries
        queries_path = self.benchmark_dir / "queries.jsonl"
        if queries_path.exists():
            with open(queries_path) as f:
                for line in f:
                    q = json.loads(line)
                    self.queries[q["_id"]] = q["text"]

        # Qrels
        qrels_path = self.benchmark_dir / "qrels" / f"{split}.tsv"
        if qrels_path.exists():
            with open(qrels_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, did, score = parts[0], parts[1], int(parts[2])
                        self.qrels.setdefault(qid, {})[did] = score

        # Filter queries to those with qrels
        self.queries = {qid: text for qid, text in self.queries.items() if qid in self.qrels}

        self._loaded = True
        logger.info(
            "Loaded %d queries, %d docs, %d qrels",
            len(self.queries), len(self.corpus), sum(len(v) for v in self.qrels.values()),
        )

    # --- Checkpoint helpers ---

    def _checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.json"

    def _has_checkpoint(self, name: str) -> bool:
        return self._checkpoint_path(name).exists()

    def _save_checkpoint(self, name: str, data: Any) -> None:
        with open(self._checkpoint_path(name), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_checkpoint(self, name: str) -> Any:
        with open(self._checkpoint_path(name)) as f:
            return json.load(f)

    def _subsample_queries(self, max_queries: Optional[int]) -> Dict[str, str]:
        """Subsample queries for expensive experiments."""
        if max_queries is None or max_queries >= len(self.queries):
            return dict(self.queries)
        rng = random.Random(self.seed)
        sampled_ids = rng.sample(list(self.queries.keys()), max_queries)
        return {qid: self.queries[qid] for qid in sampled_ids}

    # --- Experiment 1: BM25 Baseline ---

    def run_bm25_baseline(
        self,
        top_k: int = 100,
        k_values: Optional[List[int]] = None,
    ) -> ExperimentResult:
        """Run BM25 baseline evaluation (no LLM, CPU only)."""
        cp_name = "exp1_bm25"
        if self._has_checkpoint(cp_name):
            data = self._load_checkpoint(cp_name)
            return ExperimentResult(**data)

        logger.info("Experiment 1: BM25 baseline")
        k_values = k_values or [10, 50, 100]
        t0 = time.time()

        from biopat.evaluation.bm25 import BM25Evaluator

        evaluator = BM25Evaluator(
            benchmark_dir=self.benchmark_dir,
            results_dir=self.results_dir,
        )
        metrics = evaluator.run_evaluation(
            split="test", top_k=top_k, k_values=k_values, save_results=True,
        )

        result = ExperimentResult(
            experiment="bm25_baseline",
            model="BM25",
            metrics=metrics,
            num_queries=len(self.queries),
            elapsed_seconds=time.time() - t0,
        )
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    # --- Experiment 2: Dense Baselines ---

    def run_dense_baselines(
        self,
        models: Optional[List[str]] = None,
        batch_size: int = 16,
        top_k: int = 100,
        k_values: Optional[List[int]] = None,
    ) -> List[ExperimentResult]:
        """Run dense retrieval baselines (no LLM, CPU only)."""
        models = models or ["all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"]
        k_values = k_values or [10, 50, 100]
        results = []

        for model_name in models:
            safe_name = model_name.replace("/", "_")
            cp_name = f"exp2_dense_{safe_name}"
            if self._has_checkpoint(cp_name):
                data = self._load_checkpoint(cp_name)
                results.append(ExperimentResult(**data))
                continue

            logger.info("Experiment 2: Dense baseline with %s", model_name)
            t0 = time.time()

            from biopat.evaluation.dense import DenseEvaluator

            evaluator = DenseEvaluator(
                benchmark_dir=self.benchmark_dir,
                results_dir=self.results_dir,
            )
            result_data = evaluator.run_baseline(
                model_name=model_name,
                split="test",
                top_k=top_k,
                k_values=k_values,
            )

            result = ExperimentResult(
                experiment="dense_baseline",
                model=model_name,
                metrics=result_data["metrics"],
                num_queries=len(self.queries),
                elapsed_seconds=time.time() - t0,
            )
            self._save_checkpoint(cp_name, result.__dict__)
            results.append(result)

        return results

    # --- Experiment 3: HyDE Query Expansion ---

    def run_hyde_experiment(
        self,
        model_spec: ModelSpec,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        max_queries: Optional[int] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 100,
        k_values: Optional[List[int]] = None,
    ) -> ExperimentResult:
        """Run HyDE experiment: LLM generates hypothetical doc, then dense retrieve."""
        safe = model_spec.name
        cp_name = f"exp3_hyde_{safe}"
        if self._has_checkpoint(cp_name):
            return ExperimentResult(**self._load_checkpoint(cp_name))

        logger.info("Experiment 3: HyDE with %s", model_spec.display_name)
        k_values = k_values or [10, 50, 100]
        t0 = time.time()
        queries = self._subsample_queries(max_queries)

        from biopat.llm import create_provider
        from biopat.retrieval.hyde import HyDEQueryExpander, HyDEConfig

        provider = create_provider(model_spec.provider, model=model_spec.model_id)
        hyde_config = HyDEConfig(
            max_tokens=max_tokens, temperature=temperature, domain="patent",
        )
        expander = HyDEQueryExpander(config=hyde_config, provider=provider)

        # Generate hypothetical docs for all queries
        hypothetical_queries: Dict[str, str] = {}
        for qid, text in queries.items():
            try:
                hyp = expander.expand_query(text)
                hypothetical_queries[qid] = hyp
                if expander._last_response:
                    self.cost_tracker.record_response(
                        expander._last_response, task="hyde", query_id=qid,
                    )
            except Exception as e:
                logger.warning("HyDE failed for %s: %s", qid, e)
                hypothetical_queries[qid] = text  # Fallback to original

        # Dense retrieval using hypothetical docs
        from biopat.evaluation.dense import DenseRetriever, DenseRetrieverConfig

        config = DenseRetrieverConfig(
            model_name=embedding_model, cache_dir=str(self.results_dir / "cache"),
        )
        retriever = DenseRetriever(config=config)
        retriever.build_index(self.corpus)
        results = retriever.retrieve(hypothetical_queries, top_k)

        # Evaluate
        from biopat.evaluation.metrics import MetricsComputer

        mc = MetricsComputer()
        filtered_qrels = {qid: self.qrels[qid] for qid in queries if qid in self.qrels}
        metrics = mc.compute_all_metrics(results, filtered_qrels, k_values)

        result = ExperimentResult(
            experiment="hyde",
            model=model_spec.display_name,
            metrics=metrics,
            cost_usd=self.cost_tracker.total_cost,
            num_queries=len(queries),
            elapsed_seconds=time.time() - t0,
        )
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    # --- Experiment 4: Listwise LLM Reranking ---

    def run_reranking_experiment(
        self,
        model_spec: ModelSpec,
        bm25_top_k: int = 100,
        rerank_window: int = 20,
        max_queries: Optional[int] = 500,
        k_values: Optional[List[int]] = None,
    ) -> ExperimentResult:
        """Run listwise reranking: BM25 top-100, then LLM reranks top-20."""
        safe = model_spec.name
        cp_name = f"exp4_rerank_{safe}"
        if self._has_checkpoint(cp_name):
            return ExperimentResult(**self._load_checkpoint(cp_name))

        logger.info("Experiment 4: Listwise reranking with %s", model_spec.display_name)
        k_values = k_values or [10, 20]
        t0 = time.time()
        queries = self._subsample_queries(max_queries)

        # Step 1: BM25 first-stage retrieval
        from rank_bm25 import BM25Okapi

        doc_ids = list(self.corpus.keys())
        doc_texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in self.corpus.values()]
        tokenized = [t.lower().split() for t in doc_texts]
        bm25 = BM25Okapi(tokenized)

        bm25_results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, text in queries.items():
            scores = bm25.get_scores(text.lower().split())
            top_indices = scores.argsort()[-bm25_top_k:][::-1]
            bm25_results[qid] = [(doc_ids[i], float(scores[i])) for i in top_indices]

        # Step 2: LLM listwise reranking
        from biopat.llm import create_provider
        from biopat.retrieval.reranker import ListwiseLLMReranker

        provider = create_provider(model_spec.provider, model=model_spec.model_id)
        reranker = ListwiseLLMReranker(provider, window_size=rerank_window)

        reranked_results: Dict[str, Dict[str, float]] = {}
        for qid in queries:
            candidates = bm25_results.get(qid, [])
            try:
                reranked = reranker.rerank(
                    queries[qid], candidates, self.corpus, top_k=max(k_values),
                )
                reranked_results[qid] = {did: score for did, score in reranked}
                if reranker._last_response:
                    self.cost_tracker.record_response(
                        reranker._last_response, task="reranking", query_id=qid,
                    )
            except Exception as e:
                logger.warning("Reranking failed for %s: %s", qid, e)
                reranked_results[qid] = {did: score for did, score in candidates[:max(k_values)]}

        # Evaluate
        from biopat.evaluation.metrics import MetricsComputer

        mc = MetricsComputer()
        filtered_qrels = {qid: self.qrels[qid] for qid in queries if qid in self.qrels}
        metrics = mc.compute_all_metrics(reranked_results, filtered_qrels, k_values)

        result = ExperimentResult(
            experiment="reranking",
            model=model_spec.display_name,
            metrics=metrics,
            cost_usd=self.cost_tracker.total_cost,
            num_queries=len(queries),
            elapsed_seconds=time.time() - t0,
        )
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    # --- Experiment 5: LLM Relevance Judgment ---

    def run_relevance_judgment_experiment(
        self,
        model_spec: ModelSpec,
        num_pairs: int = 500,
    ) -> ExperimentResult:
        """LLM assigns graded relevance to query-doc pairs; compare with silver standard."""
        safe = model_spec.name
        cp_name = f"exp5_judgment_{safe}"
        if self._has_checkpoint(cp_name):
            return ExperimentResult(**self._load_checkpoint(cp_name))

        logger.info("Experiment 5: Relevance judgment with %s", model_spec.display_name)
        t0 = time.time()

        # Sample stratified query-doc pairs (mix relevant + non-relevant)
        rng = random.Random(self.seed)
        pairs: List[Tuple[str, str, int]] = []  # (qid, did, gold_score)

        for qid, rels in self.qrels.items():
            for did, score in rels.items():
                pairs.append((qid, did, score))

        rng.shuffle(pairs)
        pairs = pairs[:num_pairs]

        # LLM judges each pair
        from biopat.llm import create_provider

        provider = create_provider(model_spec.provider, model=model_spec.model_id)

        gold_scores = []
        predicted_scores = []

        for qid, did, gold in pairs:
            query_text = self.queries.get(qid, "")
            doc = self.corpus.get(did, {})
            doc_text = f"{doc.get('title', '')} {doc.get('text', '')}" if isinstance(doc, dict) else str(doc)

            prompt = f"""Rate the relevance of this scientific article as prior art for the patent claim.

Patent Claim:
{query_text[:800]}

Scientific Article:
{doc_text[:1200]}

Rate on a scale of 0-3:
- 0: Not relevant
- 1: Marginally relevant
- 2: Highly relevant
- 3: Directly anticipates the claimed invention (novelty-destroying)

Respond with ONLY a JSON object: {{"score": <0-3>, "reasoning": "..."}}"""

            try:
                response = provider.generate(
                    prompt=prompt,
                    system_prompt="You are an expert patent examiner. Respond only with valid JSON.",
                    max_tokens=200,
                    temperature=0,
                )
                self.cost_tracker.record_response(response, task="relevance_judgment", query_id=qid)

                text = response.text.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                data = json.loads(text)
                predicted = int(data.get("score", 1))
            except Exception:
                predicted = 1  # Default

            gold_scores.append(gold)
            predicted_scores.append(predicted)

        # Compute agreement metrics
        metrics = self._compute_agreement(gold_scores, predicted_scores)

        result = ExperimentResult(
            experiment="relevance_judgment",
            model=model_spec.display_name,
            metrics=metrics,
            cost_usd=self.cost_tracker.total_cost,
            num_queries=num_pairs,
            elapsed_seconds=time.time() - t0,
            metadata={"num_pairs": num_pairs},
        )
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    def _compute_agreement(
        self, gold: List[int], predicted: List[int],
    ) -> Dict[str, float]:
        """Compute agreement between gold and predicted scores."""
        import numpy as np

        gold_arr = np.array(gold)
        pred_arr = np.array(predicted)

        # Accuracy
        accuracy = float(np.mean(gold_arr == pred_arr))

        # Mean absolute error
        mae = float(np.mean(np.abs(gold_arr - pred_arr)))

        # Cohen's kappa (linear)
        try:
            from scipy.stats import kendalltau

            tau, p_value = kendalltau(gold_arr, pred_arr)
            kendall_tau = float(tau)
        except ImportError:
            kendall_tau = 0.0
            p_value = 1.0

        # Cohen's kappa
        try:
            n = len(gold)
            labels = sorted(set(gold) | set(predicted))
            k = len(labels)
            label_idx = {l: i for i, l in enumerate(labels)}
            confusion = np.zeros((k, k), dtype=int)
            for g, p in zip(gold, predicted):
                confusion[label_idx[g]][label_idx[p]] += 1
            po = np.trace(confusion) / n
            pe = sum(confusion[i, :].sum() * confusion[:, i].sum() for i in range(k)) / (n * n)
            kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        except Exception:
            kappa = 0.0

        return {
            "accuracy": accuracy,
            "mae": mae,
            "cohens_kappa": float(kappa),
            "kendall_tau": kendall_tau,
            "kendall_p_value": float(p_value),
        }

    # --- Experiment 6: Full Novelty Assessment ---

    def run_novelty_experiment(
        self,
        model_spec: ModelSpec,
        num_patents: int = 100,
        retrieval_top_k: int = 50,
        max_refs: int = 10,
    ) -> ExperimentResult:
        """Full novelty assessment pipeline per LLM."""
        safe = model_spec.name
        cp_name = f"exp6_novelty_{safe}"
        if self._has_checkpoint(cp_name):
            return ExperimentResult(**self._load_checkpoint(cp_name))

        logger.info("Experiment 6: Novelty assessment with %s", model_spec.display_name)
        t0 = time.time()

        # Select patents (use unique patent IDs from queries)
        rng = random.Random(self.seed)
        query_ids = list(self.queries.keys())
        rng.shuffle(query_ids)
        selected = query_ids[:num_patents]

        from biopat.llm import create_provider
        from biopat.reasoning.claim_parser import LLMClaimParser
        from biopat.reasoning.novelty_reasoner import LLMNoveltyReasoner, NoveltyStatus
        from rank_bm25 import BM25Okapi

        provider = create_provider(model_spec.provider, model=model_spec.model_id)

        # Initialize components with unified provider
        claim_parser = LLMClaimParser(llm_provider=provider)
        novelty_reasoner = LLMNoveltyReasoner(llm_provider=provider)

        # Build BM25 index
        doc_ids = list(self.corpus.keys())
        doc_texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in self.corpus.values()]
        tokenized = [t.lower().split() for t in doc_texts]
        bm25 = BM25Okapi(tokenized)

        status_counts: Dict[str, int] = {}
        assessments_completed = 0

        for qid in selected:
            query_text = self.queries[qid]
            try:
                # Parse claim
                parsed = claim_parser.parse_claim(query_text, claim_number=1)
                if claim_parser._last_response:
                    self.cost_tracker.record_response(
                        claim_parser._last_response, task="novelty_parse", query_id=qid,
                    )

                # Retrieve
                scores = bm25.get_scores(query_text.lower().split())
                top_idx = scores.argsort()[-retrieval_top_k:][::-1]
                prior_art = [(doc_ids[i], float(scores[i])) for i in top_idx]

                # Assess novelty
                assessment = novelty_reasoner.assess_novelty(
                    parsed_claim=parsed,
                    prior_art_docs=prior_art[:max_refs],
                    corpus=self.corpus,
                    max_refs_to_analyze=max_refs,
                )
                if novelty_reasoner._last_response:
                    self.cost_tracker.record_response(
                        novelty_reasoner._last_response, task="novelty_assess", query_id=qid,
                    )

                status = assessment.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                assessments_completed += 1

            except Exception as e:
                logger.warning("Novelty failed for %s: %s", qid, e)

        metrics = {
            "patents_assessed": assessments_completed,
            "patents_attempted": len(selected),
            **{f"status_{k}": v for k, v in status_counts.items()},
        }

        result = ExperimentResult(
            experiment="novelty_assessment",
            model=model_spec.display_name,
            metrics=metrics,
            cost_usd=self.cost_tracker.total_cost,
            num_queries=len(selected),
            elapsed_seconds=time.time() - t0,
        )
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    # --- Experiment 7: Agent-based Dual Retrieval ---

    def run_agent_experiment(
        self,
        model_spec: ModelSpec,
        max_queries: Optional[int] = 200,
        query_type: str = "A",
        max_search_calls: int = 5,
        search_top_k: int = 20,
        final_list_size: int = 100,
        k_values: Optional[List[int]] = None,
    ) -> ExperimentResult:
        """Run agent-based dual retrieval: LLM iteratively searches papers + patents.

        Args:
            model_spec: Model to evaluate.
            max_queries: Max queries to evaluate (subsampled).
            query_type: "A" (patent→papers) or "B" (paper→patents).
            max_search_calls: Max BM25 searches per query.
            search_top_k: Results per search call.
            final_list_size: Size of final ranked list.
            k_values: Cutoff values for metrics.
        """
        safe = model_spec.name
        cp_name = f"exp7_agent_{safe}_type{query_type}"
        if self._has_checkpoint(cp_name):
            return ExperimentResult(**self._load_checkpoint(cp_name))

        logger.info(
            "Experiment 7: Agent retrieval with %s (Type %s)",
            model_spec.display_name, query_type,
        )
        k_values = k_values or [10, 50, 100]
        t0 = time.time()

        from biopat.llm import create_provider
        from biopat.evaluation.agent_retrieval import (
            AgentConfig, DualCorpusSearchTool, RetrievalAgent,
            results_to_qrels_format,
        )
        from biopat.evaluation.agent_metrics import compute_agent_metrics
        from biopat.evaluation.dual_qrels import build_dual_corpus, invert_qrels

        # Build dual corpus (papers + patents)
        dual_corpus = build_dual_corpus(str(self.benchmark_dir))
        doc_types = {did: doc["doc_type"] for did, doc in dual_corpus.items()}

        # Select queries and qrels based on type
        if query_type == "B":
            inverted = invert_qrels(str(self.benchmark_dir))
            from biopat.evaluation.dual_qrels import select_type_b_queries
            available_queries = select_type_b_queries(
                str(self.benchmark_dir), inverted, max_queries=max_queries or 200,
                seed=self.seed,
            )
            eval_qrels = inverted
        else:
            # Type A: patent claims as queries (default)
            available_queries = dict(self.queries)
            eval_qrels = dict(self.qrels)

        queries = self._subsample_queries_from(available_queries, max_queries)

        # Build search tool (BM25 over dual corpus)
        search_tool = DualCorpusSearchTool(dual_corpus)

        # Create agent
        provider = create_provider(model_spec.provider, model=model_spec.model_id)
        agent_config = AgentConfig(
            max_search_calls=max_search_calls,
            search_top_k=search_top_k,
            final_list_size=final_list_size,
        )
        agent = RetrievalAgent(
            provider=provider,
            search_tool=search_tool,
            config=agent_config,
            cost_tracker=self.cost_tracker,
        )

        # Run agent on each query
        traces = []
        for i, (qid, text) in enumerate(queries.items()):
            logger.info(
                "  Agent query %d/%d: %s", i + 1, len(queries), qid,
            )
            trace = agent.run(qid, text)
            traces.append(trace)

            # Periodic checkpoint
            if (i + 1) % 25 == 0:
                self._save_checkpoint(
                    f"{cp_name}_partial_{i+1}",
                    {"traces": [t.to_dict() for t in traces]},
                )

        # Convert traces to standard results format
        results = results_to_qrels_format(traces)

        # Filter qrels to evaluated queries
        filtered_qrels = {qid: eval_qrels[qid] for qid in queries if qid in eval_qrels}

        # Compute metrics
        metrics = compute_agent_metrics(
            results=results,
            qrels=filtered_qrels,
            doc_types=doc_types,
            traces=traces,
            k_values=k_values,
        )

        # Save traces
        traces_path = self.results_dir / f"agent_traces_{safe}_type{query_type}.json"
        with open(traces_path, "w") as f:
            json.dump([t.to_dict() for t in traces], f, indent=2, default=str)

        result = ExperimentResult(
            experiment=f"agent_retrieval_type{query_type}",
            model=model_spec.display_name,
            metrics=metrics,
            cost_usd=sum(t.total_cost_usd for t in traces),
            num_queries=len(queries),
            elapsed_seconds=time.time() - t0,
            metadata={
                "query_type": query_type,
                "max_search_calls": max_search_calls,
                "avg_search_calls": metrics.get("avg_search_calls", 0),
                "traces_path": str(traces_path),
            },
        )
        self._save_checkpoint(cp_name, result.__dict__)
        return result

    def _subsample_queries_from(
        self, queries: Dict[str, str], max_queries: Optional[int],
    ) -> Dict[str, str]:
        """Subsample from an arbitrary query dict."""
        if max_queries is None or max_queries >= len(queries):
            return dict(queries)
        rng = random.Random(self.seed)
        sampled_ids = rng.sample(sorted(queries.keys()), max_queries)
        return {qid: queries[qid] for qid in sampled_ids}

    # --- Run all experiments ---

    def run_all(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
    ) -> List[ExperimentResult]:
        """Run all configured experiments.

        Args:
            config: Parsed experiments.yaml config dict.
            dry_run: If True, estimate costs without API calls.

        Returns:
            List of ExperimentResult objects.
        """
        if not self._loaded:
            self.load_benchmark()

        results: List[ExperimentResult] = []
        experiments = config.get("experiments", {})
        model_defs = config.get("models", {})

        # Parse model specs
        models: Dict[str, ModelSpec] = {}
        for name, spec in model_defs.items():
            models[name] = ModelSpec(
                name=name,
                provider=spec["provider"],
                model_id=spec["model_id"],
                display_name=spec.get("display_name", name),
            )

        if dry_run:
            return self._estimate_costs(config, models)

        # Exp 1: BM25
        exp = experiments.get("bm25_baseline", {})
        if exp.get("enabled", False):
            results.append(self.run_bm25_baseline(
                top_k=exp.get("top_k", 100),
                k_values=exp.get("k_values"),
            ))

        # Exp 2: Dense
        exp = experiments.get("dense_baseline", {})
        if exp.get("enabled", False):
            results.extend(self.run_dense_baselines(
                models=exp.get("models"),
                batch_size=exp.get("batch_size", 16),
                top_k=exp.get("top_k", 100),
                k_values=exp.get("k_values"),
            ))

        # Exp 3: HyDE
        exp = experiments.get("hyde", {})
        if exp.get("enabled", False):
            for model_key in exp.get("llm_models", []):
                if model_key in models:
                    results.append(self.run_hyde_experiment(
                        model_spec=models[model_key],
                        embedding_model=exp.get("embedding_model", "BAAI/bge-base-en-v1.5"),
                        max_queries=exp.get("max_queries"),
                        max_tokens=exp.get("max_tokens", 256),
                        temperature=exp.get("temperature", 0.7),
                        top_k=exp.get("top_k", 100),
                        k_values=exp.get("k_values"),
                    ))

        # Exp 4: Reranking
        exp = experiments.get("reranking", {})
        if exp.get("enabled", False):
            for model_key in exp.get("llm_models", []):
                if model_key in models:
                    results.append(self.run_reranking_experiment(
                        model_spec=models[model_key],
                        bm25_top_k=exp.get("bm25_top_k", 100),
                        rerank_window=exp.get("rerank_window", 20),
                        max_queries=exp.get("max_queries", 500),
                        k_values=exp.get("k_values"),
                    ))

        # Exp 5: Relevance judgment
        exp = experiments.get("relevance_judgment", {})
        if exp.get("enabled", False):
            for model_key in exp.get("llm_models", []):
                if model_key in models:
                    results.append(self.run_relevance_judgment_experiment(
                        model_spec=models[model_key],
                        num_pairs=exp.get("num_pairs", 500),
                    ))

        # Exp 6: Novelty
        exp = experiments.get("novelty_assessment", {})
        if exp.get("enabled", False):
            for model_key in exp.get("llm_models", []):
                if model_key in models:
                    results.append(self.run_novelty_experiment(
                        model_spec=models[model_key],
                        num_patents=exp.get("num_patents", 100),
                        retrieval_top_k=exp.get("retrieval_top_k", 50),
                        max_refs=exp.get("max_refs_per_patent", 10),
                    ))

        # Exp 7: Agent-based dual retrieval
        exp = experiments.get("agent_retrieval", {})
        if exp.get("enabled", False):
            agent_params = exp.get("agent_params", {})
            for query_type in exp.get("query_types", ["A"]):
                for model_key in exp.get("llm_models", []):
                    if model_key in models:
                        results.append(self.run_agent_experiment(
                            model_spec=models[model_key],
                            max_queries=exp.get("max_queries", 200),
                            query_type=query_type,
                            max_search_calls=agent_params.get("max_search_calls", 5),
                            search_top_k=agent_params.get("search_top_k", 20),
                            final_list_size=agent_params.get("final_list_size", 100),
                            k_values=exp.get("k_values"),
                        ))

        # Save summary
        self._save_summary(results)
        self.cost_tracker.save(str(self.results_dir / "cost_tracker.json"))

        return results

    def _estimate_costs(
        self, config: Dict[str, Any], models: Dict[str, ModelSpec],
    ) -> List[ExperimentResult]:
        """Dry-run: estimate API costs without making calls."""
        from biopat.llm.providers import PRICING

        estimates = []
        experiments = config.get("experiments", {})
        num_queries = len(self.queries)

        # HyDE: ~500 tokens per call
        exp = experiments.get("hyde", {})
        if exp.get("enabled"):
            n = exp.get("max_queries") or num_queries
            for mk in exp.get("llm_models", []):
                if mk in models:
                    m = models[mk]
                    pricing = PRICING.get(m.model_id, (5.0, 15.0))
                    cost = n * (500 * pricing[0] + 256 * pricing[1]) / 1_000_000
                    estimates.append(ExperimentResult(
                        experiment="hyde", model=m.display_name,
                        metrics={}, cost_usd=cost, num_queries=n,
                        metadata={"dry_run": True},
                    ))

        # Reranking: ~2000 tokens per call
        exp = experiments.get("reranking", {})
        if exp.get("enabled"):
            n = exp.get("max_queries", 500)
            for mk in exp.get("llm_models", []):
                if mk in models:
                    m = models[mk]
                    pricing = PRICING.get(m.model_id, (5.0, 15.0))
                    cost = n * (2000 * pricing[0] + 500 * pricing[1]) / 1_000_000
                    estimates.append(ExperimentResult(
                        experiment="reranking", model=m.display_name,
                        metrics={}, cost_usd=cost, num_queries=n,
                        metadata={"dry_run": True},
                    ))

        # Relevance judgment: ~800 tokens per call
        exp = experiments.get("relevance_judgment", {})
        if exp.get("enabled"):
            n = exp.get("num_pairs", 500)
            for mk in exp.get("llm_models", []):
                if mk in models:
                    m = models[mk]
                    pricing = PRICING.get(m.model_id, (5.0, 15.0))
                    cost = n * (800 * pricing[0] + 200 * pricing[1]) / 1_000_000
                    estimates.append(ExperimentResult(
                        experiment="relevance_judgment", model=m.display_name,
                        metrics={}, cost_usd=cost, num_queries=n,
                        metadata={"dry_run": True},
                    ))

        # Novelty: ~36 calls per patent, ~2000 tokens each
        exp = experiments.get("novelty_assessment", {})
        if exp.get("enabled"):
            n = exp.get("num_patents", 100)
            calls_per = 36
            for mk in exp.get("llm_models", []):
                if mk in models:
                    m = models[mk]
                    pricing = PRICING.get(m.model_id, (5.0, 15.0))
                    cost = n * calls_per * (2000 * pricing[0] + 1000 * pricing[1]) / 1_000_000
                    estimates.append(ExperimentResult(
                        experiment="novelty_assessment", model=m.display_name,
                        metrics={}, cost_usd=cost, num_queries=n,
                        metadata={"dry_run": True},
                    ))

        # Agent retrieval: ~4 LLM calls per query, ~1500 tokens avg each
        exp = experiments.get("agent_retrieval", {})
        if exp.get("enabled"):
            n = exp.get("max_queries", 200)
            calls_per_query = 4  # plan + refine + rank (+ optional refine)
            for mk in exp.get("llm_models", []):
                if mk in models:
                    m = models[mk]
                    pricing = PRICING.get(m.model_id, (5.0, 15.0))
                    cost = n * calls_per_query * (1500 * pricing[0] + 800 * pricing[1]) / 1_000_000
                    for qt in exp.get("query_types", ["A"]):
                        estimates.append(ExperimentResult(
                            experiment=f"agent_retrieval_type{qt}",
                            model=m.display_name,
                            metrics={}, cost_usd=cost, num_queries=n,
                            metadata={"dry_run": True, "query_type": qt},
                        ))

        total = sum(e.cost_usd for e in estimates)
        logger.info("Estimated total cost: $%.2f", total)
        for e in estimates:
            logger.info("  %s / %s: $%.2f", e.experiment, e.model, e.cost_usd)

        return estimates

    def _save_summary(self, results: List[ExperimentResult]) -> None:
        """Save experiment summary."""
        summary = {
            "total_results": len(results),
            "total_cost_usd": sum(r.cost_usd for r in results),
            "results": [r.__dict__ for r in results],
        }
        with open(self.results_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
