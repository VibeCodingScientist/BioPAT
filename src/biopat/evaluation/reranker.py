"""Cross-encoder reranking module.

Implements cross-encoder reranking for improving retrieval results.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Cross-encoder model registry
CROSS_ENCODER_REGISTRY = {
    "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-minilm-12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "ms-marco-electra": "cross-encoder/ms-marco-electra-base",
    "ms-marco-roberta": "cross-encoder/stsb-roberta-base",
    "ce-biobert": "cross-encoder/nli-deberta-v3-base",  # General, but works
}


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""

    model_name: str = "ms-marco-minilm"
    batch_size: int = 32
    max_length: int = 512
    top_k: int = 100  # How many docs to rerank
    use_gpu: bool = True


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval results.

    Cross-encoders jointly encode query and document, producing
    more accurate relevance scores at the cost of efficiency.
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model_name: str = "ms-marco-minilm",
    ):
        """Initialize reranker.

        Args:
            config: Reranker configuration.
            model_name: Model name (short name or HuggingFace ID).
        """
        if config:
            self.config = config
        else:
            self.config = RerankerConfig(model_name=model_name)

        self.model = None

        # Resolve model name
        self.model_id = CROSS_ENCODER_REGISTRY.get(
            self.config.model_name, self.config.model_name
        )

    def _get_device(self) -> str:
        """Get device to use for inference."""
        if not self.config.use_gpu:
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        device = self._get_device()
        logger.info(f"Loading cross-encoder {self.model_id} on {device}")

        self.model = CrossEncoder(self.model_id, max_length=self.config.max_length, device=device)
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank documents for a single query.

        Args:
            query: Query text.
            documents: List of (doc_id, doc_text) tuples.
            top_k: Number of documents to return (default: all).

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if self.model is None:
            self.load_model()

        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [(query, doc_text) for _, doc_text in documents]
        doc_ids = [doc_id for doc_id, _ in documents]

        # Get scores
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Combine with doc IDs
        results = list(zip(doc_ids, scores))

        # Sort by score descending
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Apply top_k
        if top_k is not None:
            results = results[:top_k]

        return [(doc_id, float(score)) for doc_id, score in results]

    def rerank_results(
        self,
        queries: Dict[str, str],
        initial_results: Dict[str, Dict[str, float]],
        corpus: Dict[str, dict],
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Rerank initial retrieval results.

        Args:
            queries: Dictionary of query_id to query text.
            initial_results: Initial retrieval results {qid: {doc_id: score}}.
            corpus: Corpus dictionary {doc_id: {title, text}}.
            top_k: Final number of results to return.
            rerank_top_k: How many initial results to rerank.

        Returns:
            Reranked results.
        """
        if self.model is None:
            self.load_model()

        from tqdm import tqdm

        top_k = top_k or self.config.top_k
        rerank_top_k = rerank_top_k or self.config.top_k

        reranked = {}

        for qid, query_text in tqdm(queries.items(), desc="Reranking"):
            if qid not in initial_results:
                continue

            # Get top documents to rerank
            sorted_docs = sorted(
                initial_results[qid].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:rerank_top_k]

            # Prepare documents
            documents = []
            for doc_id, _ in sorted_docs:
                if doc_id in corpus:
                    doc = corpus[doc_id]
                    doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                    documents.append((doc_id, doc_text))

            # Rerank
            reranked_list = self.rerank(query_text, documents, top_k)
            reranked[qid] = {doc_id: score for doc_id, score in reranked_list}

        return reranked


class BM25CrossEncoderPipeline:
    """Pipeline combining BM25 retrieval with cross-encoder reranking."""

    def __init__(
        self,
        benchmark_dir: str,
        cross_encoder_model: str = "ms-marco-minilm",
        bm25_top_k: int = 100,
        rerank_top_k: int = 100,
    ):
        """Initialize pipeline.

        Args:
            benchmark_dir: Path to benchmark directory.
            cross_encoder_model: Cross-encoder model to use.
            bm25_top_k: Number of BM25 results to retrieve.
            rerank_top_k: Number of results to rerank.
        """
        from .bm25 import BM25Evaluator

        self.benchmark_dir = Path(benchmark_dir)
        self.bm25 = BM25Evaluator(self.benchmark_dir)
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k

        self.corpus = None
        self.indexed = False

    def load_and_index(self) -> None:
        """Load corpus and build BM25 index."""
        self.corpus = self.bm25.load_corpus()
        self.bm25.build_index(self.corpus)
        self.indexed = True

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve and rerank documents.

        Args:
            queries: Dictionary of query_id to query text.
            top_k: Final number of results to return.

        Returns:
            Reranked results.
        """
        if not self.indexed:
            self.load_and_index()

        # BM25 retrieval
        logger.info("Running BM25 retrieval")
        bm25_results = self.bm25.retrieve(queries, self.bm25_top_k)

        # Rerank
        logger.info("Running cross-encoder reranking")
        reranked = self.reranker.rerank_results(
            queries=queries,
            initial_results=bm25_results,
            corpus=self.corpus,
            top_k=top_k,
            rerank_top_k=self.rerank_top_k,
        )

        return reranked

    def run_evaluation(
        self,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Run BM25 + cross-encoder evaluation.

        Args:
            split: Evaluation split.
            top_k: Final number of results.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        from .metrics import MetricsComputer

        # Load data
        queries = self.bm25.load_queries()
        qrels = self.bm25.load_qrels(split)

        # Filter queries
        queries = {qid: text for qid, text in queries.items() if qid in qrels}

        # Load and index
        self.load_and_index()

        # Retrieve and rerank
        results = self.retrieve(queries, top_k)

        # Evaluate
        metrics_computer = MetricsComputer()
        metrics = metrics_computer.compute_all_metrics(results, qrels, k_values)

        logger.info(f"BM25 + CE evaluation results: {metrics}")
        return metrics


class DenseCrossEncoderPipeline:
    """Pipeline combining dense retrieval with cross-encoder reranking."""

    def __init__(
        self,
        benchmark_dir: str,
        dense_model: str = "contriever",
        cross_encoder_model: str = "ms-marco-minilm",
        dense_top_k: int = 100,
        rerank_top_k: int = 100,
        cache_dir: Optional[str] = None,
    ):
        """Initialize pipeline.

        Args:
            benchmark_dir: Path to benchmark directory.
            dense_model: Dense retrieval model.
            cross_encoder_model: Cross-encoder model.
            dense_top_k: Number of dense results to retrieve.
            rerank_top_k: Number of results to rerank.
            cache_dir: Cache directory for embeddings.
        """
        from .dense import DenseRetriever, DenseRetrieverConfig

        self.benchmark_dir = Path(benchmark_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.benchmark_dir / "cache"

        config = DenseRetrieverConfig(
            model_name=dense_model,
            cache_dir=str(self.cache_dir),
        )
        self.dense = DenseRetriever(config=config)
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)

        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k

        self.corpus = None
        self.indexed = False

    def load_and_index(self) -> None:
        """Load corpus and build dense index."""
        import json

        corpus_path = self.benchmark_dir / "corpus.jsonl"
        self.corpus = {}

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                self.corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }

        self.dense.build_index(self.corpus)
        self.indexed = True

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve and rerank documents.

        Args:
            queries: Dictionary of query_id to query text.
            top_k: Final number of results.

        Returns:
            Reranked results.
        """
        if not self.indexed:
            self.load_and_index()

        # Dense retrieval
        logger.info("Running dense retrieval")
        dense_results = self.dense.retrieve(queries, self.dense_top_k)

        # Rerank
        logger.info("Running cross-encoder reranking")
        reranked = self.reranker.rerank_results(
            queries=queries,
            initial_results=dense_results,
            corpus=self.corpus,
            top_k=top_k,
            rerank_top_k=self.rerank_top_k,
        )

        return reranked

    def run_evaluation(
        self,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Run dense + cross-encoder evaluation.

        Args:
            split: Evaluation split.
            top_k: Final number of results.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        import json
        from .metrics import MetricsComputer

        # Load queries
        queries_path = self.benchmark_dir / "queries.jsonl"
        queries = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                queries[query["_id"]] = query["text"]

        # Load qrels
        qrels_path = self.benchmark_dir / "qrels" / f"{split}.tsv"
        qrels = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], int(parts[2])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = score

        # Filter queries
        queries = {qid: text for qid, text in queries.items() if qid in qrels}

        # Load and index
        self.load_and_index()

        # Retrieve and rerank
        results = self.retrieve(queries, top_k)

        # Evaluate
        metrics_computer = MetricsComputer()
        metrics = metrics_computer.compute_all_metrics(results, qrels, k_values)

        logger.info(f"Dense + CE evaluation results: {metrics}")
        return metrics
