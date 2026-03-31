"""Hybrid retrieval methods.

Implements fusion methods for combining multiple retrieval systems.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for result fusion."""

    method: str = "rrf"  # rrf, linear, weighted
    rrf_k: int = 60  # RRF constant
    normalize_scores: bool = True
    weights: Optional[Dict[str, float]] = None  # For weighted fusion


class ResultFusion:
    """Fuse results from multiple retrieval systems."""

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize result fusion.

        Args:
            config: Fusion configuration.
        """
        self.config = config or FusionConfig()

    def reciprocal_rank_fusion(
        self,
        results_list: List[Dict[str, Dict[str, float]]],
        k: int = 60,
    ) -> Dict[str, Dict[str, float]]:
        """Apply Reciprocal Rank Fusion to multiple result sets.

        RRF is a simple but effective rank-based fusion method.
        Score = sum(1 / (k + rank_i)) across all systems.

        Args:
            results_list: List of result dicts from different systems.
            k: RRF constant (typically 60).

        Returns:
            Fused results.
        """
        fused = {}

        # Get all query IDs
        all_query_ids = set()
        for results in results_list:
            all_query_ids.update(results.keys())

        for qid in all_query_ids:
            scores = defaultdict(float)

            for results in results_list:
                if qid not in results:
                    continue

                # Sort by score and get ranks
                sorted_docs = sorted(
                    results[qid].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for rank, (doc_id, _) in enumerate(sorted_docs, 1):
                    scores[doc_id] += 1.0 / (k + rank)

            fused[qid] = dict(scores)

        return fused

    def linear_fusion(
        self,
        results_list: List[Dict[str, Dict[str, float]]],
        normalize: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Apply linear score fusion.

        Combines scores from multiple systems by simple addition.

        Args:
            results_list: List of result dicts from different systems.
            normalize: Whether to normalize scores before fusion.

        Returns:
            Fused results.
        """
        fused = {}

        # Get all query IDs
        all_query_ids = set()
        for results in results_list:
            all_query_ids.update(results.keys())

        for qid in all_query_ids:
            scores = defaultdict(float)

            for results in results_list:
                if qid not in results:
                    continue

                query_results = results[qid]

                if normalize and query_results:
                    # Min-max normalization
                    min_score = min(query_results.values())
                    max_score = max(query_results.values())
                    score_range = max_score - min_score

                    if score_range > 0:
                        for doc_id, score in query_results.items():
                            normalized = (score - min_score) / score_range
                            scores[doc_id] += normalized
                    else:
                        for doc_id in query_results:
                            scores[doc_id] += 1.0
                else:
                    for doc_id, score in query_results.items():
                        scores[doc_id] += score

            fused[qid] = dict(scores)

        return fused

    def weighted_fusion(
        self,
        results_list: List[Dict[str, Dict[str, float]]],
        weights: List[float],
        normalize: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Apply weighted score fusion.

        Combines scores with system-specific weights.

        Args:
            results_list: List of result dicts from different systems.
            weights: Weight for each system.
            normalize: Whether to normalize scores before fusion.

        Returns:
            Fused results.
        """
        if len(weights) != len(results_list):
            raise ValueError("Number of weights must match number of result lists")

        fused = {}

        # Get all query IDs
        all_query_ids = set()
        for results in results_list:
            all_query_ids.update(results.keys())

        for qid in all_query_ids:
            scores = defaultdict(float)

            for results, weight in zip(results_list, weights):
                if qid not in results:
                    continue

                query_results = results[qid]

                if normalize and query_results:
                    # Min-max normalization
                    min_score = min(query_results.values())
                    max_score = max(query_results.values())
                    score_range = max_score - min_score

                    if score_range > 0:
                        for doc_id, score in query_results.items():
                            normalized = (score - min_score) / score_range
                            scores[doc_id] += weight * normalized
                    else:
                        for doc_id in query_results:
                            scores[doc_id] += weight
                else:
                    for doc_id, score in query_results.items():
                        scores[doc_id] += weight * score

            fused[qid] = dict(scores)

        return fused

    def fuse(
        self,
        results_list: List[Dict[str, Dict[str, float]]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Fuse results using configured method.

        Args:
            results_list: List of result dicts from different systems.
            weights: Optional weights for weighted fusion.

        Returns:
            Fused results.
        """
        if self.config.method == "rrf":
            return self.reciprocal_rank_fusion(results_list, k=self.config.rrf_k)
        elif self.config.method == "linear":
            return self.linear_fusion(results_list, normalize=self.config.normalize_scores)
        elif self.config.method == "weighted":
            if weights is None:
                raise ValueError("Weights required for weighted fusion")
            return self.weighted_fusion(
                results_list, weights, normalize=self.config.normalize_scores
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.config.method}")


class HybridRetriever:
    """Hybrid retrieval combining sparse and dense methods."""

    def __init__(
        self,
        sparse_retriever: Any = None,
        dense_retriever: Any = None,
        fusion_config: Optional[FusionConfig] = None,
    ):
        """Initialize hybrid retriever.

        Args:
            sparse_retriever: BM25 or other sparse retriever.
            dense_retriever: Dense retriever instance.
            fusion_config: Configuration for fusion.
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.fusion = ResultFusion(fusion_config or FusionConfig())

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        sparse_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve documents using hybrid approach.

        Args:
            queries: Dictionary of query_id to query text.
            top_k: Final number of results to return.
            sparse_top_k: Number of sparse results (default: top_k).
            dense_top_k: Number of dense results (default: top_k).

        Returns:
            Fused results.
        """
        sparse_top_k = sparse_top_k or top_k
        dense_top_k = dense_top_k or top_k

        results_list = []

        # Get sparse results
        if self.sparse_retriever is not None:
            logger.info("Running sparse retrieval")
            sparse_results = self.sparse_retriever.retrieve(queries, sparse_top_k)
            results_list.append(sparse_results)

        # Get dense results
        if self.dense_retriever is not None:
            logger.info("Running dense retrieval")
            dense_results = self.dense_retriever.retrieve(queries, dense_top_k)
            results_list.append(dense_results)

        if not results_list:
            raise RuntimeError("No retrievers configured")

        # Fuse results
        logger.info(f"Fusing results from {len(results_list)} systems")
        fused = self.fusion.fuse(results_list)

        # Trim to top_k
        trimmed = {}
        for qid, docs in fused.items():
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            trimmed[qid] = dict(sorted_docs)

        return trimmed


class BM25DenseHybrid:
    """Convenience class for BM25 + dense hybrid retrieval."""

    def __init__(
        self,
        benchmark_dir: str,
        dense_model: str = "contriever",
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        cache_dir: Optional[str] = None,
    ):
        """Initialize hybrid retriever.

        Args:
            benchmark_dir: Path to benchmark directory.
            dense_model: Dense model to use.
            fusion_method: Fusion method (rrf, linear, weighted).
            rrf_k: RRF constant.
            cache_dir: Cache directory for embeddings.
        """
        from pathlib import Path
        from .bm25 import BM25Evaluator
        from .dense import DenseRetriever, DenseRetrieverConfig

        self.benchmark_dir = Path(benchmark_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.benchmark_dir / "cache"

        # Initialize BM25
        self.bm25 = BM25Evaluator(self.benchmark_dir)

        # Initialize dense retriever
        config = DenseRetrieverConfig(
            model_name=dense_model,
            cache_dir=str(self.cache_dir),
        )
        self.dense = DenseRetriever(config=config)

        # Initialize fusion
        self.fusion = ResultFusion(
            FusionConfig(method=fusion_method, rrf_k=rrf_k)
        )

        self.corpus = None
        self.indexed = False

    def load_and_index(self, corpus: Optional[Dict[str, dict]] = None) -> None:
        """Load corpus and build indices.

        Args:
            corpus: Optional pre-loaded corpus.
        """
        if corpus is None:
            corpus = self.bm25.load_corpus()

        self.corpus = corpus

        # Build BM25 index
        logger.info("Building BM25 index")
        self.bm25.build_index(corpus)

        # Build dense index
        logger.info("Building dense index")
        self.dense.build_index(corpus)

        self.indexed = True

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve documents using hybrid approach.

        Args:
            queries: Dictionary of query_id to query text.
            top_k: Number of results to return.

        Returns:
            Fused results.
        """
        if not self.indexed:
            self.load_and_index()

        # Get BM25 results
        logger.info("Running BM25 retrieval")
        bm25_results = self.bm25.retrieve(queries, top_k)

        # Get dense results
        logger.info("Running dense retrieval")
        dense_results = self.dense.retrieve(queries, top_k)

        # Fuse
        logger.info("Fusing results")
        fused = self.fusion.fuse([bm25_results, dense_results])

        # Trim to top_k
        trimmed = {}
        for qid, docs in fused.items():
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            trimmed[qid] = dict(sorted_docs)

        return trimmed

    def run_evaluation(
        self,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Run hybrid evaluation.

        Args:
            split: Evaluation split.
            top_k: Number of documents to retrieve.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        from .metrics import MetricsComputer

        # Load data
        corpus = self.bm25.load_corpus()
        queries = self.bm25.load_queries()
        qrels = self.bm25.load_qrels(split)

        # Filter queries
        queries = {qid: text for qid, text in queries.items() if qid in qrels}

        # Load and index
        self.load_and_index(corpus)

        # Retrieve
        results = self.retrieve(queries, top_k)

        # Evaluate
        metrics_computer = MetricsComputer()
        metrics = metrics_computer.compute_all_metrics(results, qrels, k_values)

        logger.info(f"Hybrid evaluation results: {metrics}")
        return metrics
