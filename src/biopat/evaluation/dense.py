"""Dense retrieval baseline module.

Implements dense retrieval baselines using sentence-transformers and FAISS.
"""

import json
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Model registry: maps short names to HuggingFace model IDs
MODEL_REGISTRY = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "specter2": "allenai/specter2",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "gtr-t5-base": "sentence-transformers/gtr-t5-base",
    "gtr-t5-large": "sentence-transformers/gtr-t5-large",
    "gtr-t5-xl": "sentence-transformers/gtr-t5-xl",
    "all-mpnet-base": "sentence-transformers/all-mpnet-base-v2",
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-base": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


@dataclass
class DenseRetrieverConfig:
    """Configuration for dense retriever."""

    model_name: str = "contriever"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    use_gpu: bool = True
    cache_embeddings: bool = True
    cache_dir: Optional[str] = None


class DenseRetriever:
    """Dense retrieval using sentence embeddings and FAISS.

    Supports multiple embedding models from sentence-transformers.
    """

    def __init__(
        self,
        config: Optional[DenseRetrieverConfig] = None,
        model_name: str = "contriever",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize dense retriever.

        Args:
            config: Configuration object.
            model_name: Short name or HuggingFace model ID.
            cache_dir: Directory for caching embeddings.
        """
        if config:
            self.config = config
        else:
            self.config = DenseRetrieverConfig(
                model_name=model_name,
                cache_dir=str(cache_dir) if cache_dir else None,
            )

        self.model = None
        self.index = None
        self.doc_ids = None
        self.doc_embeddings = None

        # Resolve model name
        self.model_id = MODEL_REGISTRY.get(
            self.config.model_name, self.config.model_name
        )

        # Set up cache directory
        if self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _get_device(self) -> str:
        """Get the device to use for encoding."""
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
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        device = self._get_device()
        logger.info(f"Loading model {self.model_id} on {device}")

        self.model = SentenceTransformer(self.model_id, device=device)
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def _get_cache_path(self, cache_key: str, prefix: str = "embeddings") -> Optional[Path]:
        """Get cache file path for embeddings."""
        if not self.cache_dir:
            return None

        # Create unique filename based on model and cache key
        model_hash = hashlib.md5(self.model_id.encode()).hexdigest()[:8]
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{prefix}_{model_hash}_{key_hash}.npy"

    def _load_cached_embeddings(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        return None

    def _save_cached_embeddings(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path:
            np.save(cache_path, embeddings)
            logger.info(f"Saved embeddings to {cache_path}")

    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode documents to embeddings.

        Args:
            documents: List of document texts.
            doc_ids: List of document IDs (for caching).
            show_progress: Show progress bar.

        Returns:
            Numpy array of embeddings.
        """
        if self.model is None:
            self.load_model()

        # Try to load from cache
        cache_key = f"docs_{len(documents)}_{doc_ids[0] if doc_ids else 'empty'}"
        if self.config.cache_embeddings:
            cached = self._load_cached_embeddings(cache_key)
            if cached is not None:
                return cached

        logger.info(f"Encoding {len(documents)} documents")
        embeddings = self.model.encode(
            documents,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        # Cache embeddings
        if self.config.cache_embeddings:
            self._save_cached_embeddings(cache_key, embeddings)

        return embeddings

    def encode_queries(
        self,
        queries: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode queries to embeddings.

        Args:
            queries: List of query texts.
            show_progress: Show progress bar.

        Returns:
            Numpy array of embeddings.
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Encoding {len(queries)} queries")
        embeddings = self.model.encode(
            queries,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings

    def build_index(
        self,
        corpus: Dict[str, dict],
        text_field: str = "text",
        title_field: str = "title",
        include_title: bool = True,
    ) -> None:
        """Build FAISS index from corpus.

        Args:
            corpus: Corpus dictionary {doc_id: {title, text}}.
            text_field: Field containing main text.
            title_field: Field containing title.
            include_title: Whether to include title in encoding.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. Run: pip install faiss-cpu (or faiss-gpu)"
            )

        self.doc_ids = list(corpus.keys())

        # Prepare texts for encoding
        texts = []
        for doc_id in self.doc_ids:
            doc = corpus[doc_id]
            if include_title:
                text = f"{doc.get(title_field, '')} {doc.get(text_field, '')}".strip()
            else:
                text = doc.get(text_field, "")
            texts.append(text)

        # Encode documents
        self.doc_embeddings = self.encode_documents(texts, self.doc_ids)

        # Build FAISS index
        d = self.doc_embeddings.shape[1]

        if self.config.normalize_embeddings:
            # Use inner product for normalized vectors (equivalent to cosine)
            self.index = faiss.IndexFlatIP(d)
        else:
            # Use L2 distance
            self.index = faiss.IndexFlatL2(d)

        self.index.add(self.doc_embeddings)
        logger.info(f"Built FAISS index with {len(self.doc_ids)} documents")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search for documents matching query embedding.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, score) tuples.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Convert to list of (doc_id, score)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(score)))

        return results

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve documents for all queries.

        Args:
            queries: Dictionary of query_id to query text.
            top_k: Number of results per query.

        Returns:
            Nested dict: {query_id: {doc_id: score}}.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Encode all queries
        query_embeddings = self.encode_queries(query_texts)

        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)

        # Build results
        results = {}
        for i, qid in enumerate(query_ids):
            results[qid] = {}
            for j, (idx, score) in enumerate(zip(indices[i], scores[i])):
                if idx < len(self.doc_ids):
                    results[qid][self.doc_ids[idx]] = float(score)

        return results

    def save_results_trec(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Path,
        run_name: Optional[str] = None,
    ) -> None:
        """Save results in TREC format.

        Args:
            results: Retrieved results.
            output_path: Output file path.
            run_name: Name for the run (defaults to model name).
        """
        run_name = run_name or self.config.model_name

        with open(output_path, "w") as f:
            for qid, docs in results.items():
                sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")

        logger.info(f"Saved TREC results to {output_path}")


class DenseEvaluator:
    """Evaluator for running dense retrieval baselines."""

    def __init__(
        self,
        benchmark_dir: Path,
        results_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize evaluator.

        Args:
            benchmark_dir: Path to BEIR-format benchmark.
            results_dir: Directory for saving results.
            cache_dir: Directory for caching embeddings.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir) if results_dir else benchmark_dir / "results"
        self.cache_dir = Path(cache_dir) if cache_dir else benchmark_dir / "cache"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_corpus(self) -> Dict[str, dict]:
        """Load corpus from BEIR format."""
        corpus_path = self.benchmark_dir / "corpus.jsonl"
        corpus = {}

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }

        logger.info(f"Loaded {len(corpus)} documents from corpus")
        return corpus

    def load_queries(self) -> Dict[str, str]:
        """Load queries from BEIR format."""
        queries_path = self.benchmark_dir / "queries.jsonl"
        queries = {}

        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                queries[query["_id"]] = query["text"]

        logger.info(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, int]]:
        """Load qrels for a split."""
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

        logger.info(f"Loaded qrels for {len(qrels)} queries")
        return qrels

    def run_baseline(
        self,
        model_name: str,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 50, 100],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Run a single dense retrieval baseline.

        Args:
            model_name: Model name (short name or HuggingFace ID).
            split: Evaluation split.
            top_k: Number of documents to retrieve.
            k_values: Values of k for metrics.
            save_results: Whether to save results.

        Returns:
            Dictionary with metrics and config.
        """
        logger.info(f"Running {model_name} baseline on {split} split")

        # Load data
        corpus = self.load_corpus()
        queries = self.load_queries()
        qrels = self.load_qrels(split)

        # Filter queries to those in qrels
        queries = {qid: text for qid, text in queries.items() if qid in qrels}
        logger.info(f"Evaluating {len(queries)} queries with qrels")

        # Create retriever
        config = DenseRetrieverConfig(
            model_name=model_name,
            cache_dir=str(self.cache_dir),
        )
        retriever = DenseRetriever(config=config)

        # Build index
        retriever.build_index(corpus)

        # Retrieve
        results = retriever.retrieve(queries, top_k)

        # Evaluate
        from .metrics import MetricsComputer
        metrics_computer = MetricsComputer()
        metrics = metrics_computer.compute_all_metrics(results, qrels, k_values)

        logger.info(f"{model_name} results: {metrics}")

        # Save results
        if save_results:
            # Save metrics
            results_path = self.results_dir / f"{model_name}_{split}_results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {"model": model_name, "split": split, "metrics": metrics},
                    f,
                    indent=2,
                )

            # Save TREC run
            run_path = self.results_dir / f"{model_name}_{split}.run"
            retriever.save_results_trec(results, run_path)

        return {"model": model_name, "metrics": metrics, "results": results}

    def run_all_baselines(
        self,
        models: Optional[List[str]] = None,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, Dict[str, float]]:
        """Run all dense retrieval baselines.

        Args:
            models: List of model names (defaults to standard baselines).
            split: Evaluation split.
            top_k: Number of documents to retrieve.
            k_values: Values of k for metrics.

        Returns:
            Dictionary mapping model name to metrics.
        """
        if models is None:
            models = ["contriever", "specter2", "all-mpnet-base", "e5-base"]

        all_metrics = {}

        for model_name in models:
            try:
                result = self.run_baseline(
                    model_name=model_name,
                    split=split,
                    top_k=top_k,
                    k_values=k_values,
                )
                all_metrics[model_name] = result["metrics"]
            except Exception as e:
                logger.error(f"Failed to run {model_name}: {e}")
                all_metrics[model_name] = {"error": str(e)}

        # Save combined results
        combined_path = self.results_dir / f"all_dense_{split}_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics

    def format_results_table(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        metrics_to_show: Optional[List[str]] = None,
    ) -> str:
        """Format results as a markdown table.

        Args:
            all_metrics: Dictionary of model to metrics.
            metrics_to_show: Which metrics to include in table.

        Returns:
            Markdown formatted table.
        """
        if metrics_to_show is None:
            metrics_to_show = ["NDCG@10", "NDCG@100", "Recall@100", "MAP"]

        # Header
        header = "| Model | " + " | ".join(metrics_to_show) + " |"
        separator = "|" + "|".join(["---"] * (len(metrics_to_show) + 1)) + "|"

        # Rows
        rows = []
        for model, metrics in all_metrics.items():
            if "error" in metrics:
                row = f"| {model} | " + " | ".join(["ERROR"] * len(metrics_to_show)) + " |"
            else:
                values = [f"{metrics.get(m, 0):.4f}" for m in metrics_to_show]
                row = f"| {model} | " + " | ".join(values) + " |"
            rows.append(row)

        return "\n".join([header, separator] + rows)
