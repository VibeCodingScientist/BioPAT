"""BM25 baseline evaluation module.

Implements BM25 retrieval for baseline evaluation of the benchmark.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Optional
import json
import polars as pl

logger = logging.getLogger(__name__)


class BM25Evaluator:
    """BM25 baseline evaluator for BioPAT benchmark."""

    def __init__(
        self,
        benchmark_dir: Path,
        results_dir: Optional[Path] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir) if results_dir else benchmark_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self.index = None
        self.corpus = None
        self.doc_ids = None

    def load_corpus(self) -> Dict[str, dict]:
        """Load corpus from BEIR format.

        Returns:
            Dictionary mapping doc_id to document dict.
        """
        corpus_path = self.benchmark_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")

        corpus = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["_id"]
                corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }

        logger.info(f"Loaded {len(corpus)} documents from corpus")
        return corpus

    def load_queries(self) -> Dict[str, str]:
        """Load queries from BEIR format.

        Returns:
            Dictionary mapping query_id to query text.
        """
        queries_path = self.benchmark_dir / "queries.jsonl"
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries not found at {queries_path}")

        queries = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                queries[query["_id"]] = query["text"]

        logger.info(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, int]]:
        """Load qrels for a split.

        Args:
            split: Split name ('train', 'dev', 'test').

        Returns:
            Nested dict: {query_id: {doc_id: relevance}}.
        """
        qrels_path = self.benchmark_dir / "qrels" / f"{split}.tsv"
        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels not found at {qrels_path}")

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

    def build_index(self, corpus: Dict[str, dict]) -> None:
        """Build BM25 index from corpus.

        Args:
            corpus: Corpus dictionary.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        logger.info("Building BM25 index")

        self.corpus = corpus
        self.doc_ids = list(corpus.keys())

        # Tokenize documents (title + text)
        tokenized_docs = []
        for doc_id in self.doc_ids:
            doc = corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            tokens = text.split()
            tokenized_docs.append(tokens)

        self.index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"Built BM25 index with {len(self.doc_ids)} documents")

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search for documents matching query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, score) tuples.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")

        # Tokenize query
        query_tokens = query.lower().split()

        # Get scores
        scores = self.index.get_scores(query_tokens)

        # Get top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [(self.doc_ids[i], float(scores[i])) for i in top_indices]
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
        from tqdm import tqdm

        results = {}
        for qid, query_text in tqdm(queries.items(), desc="Retrieving"):
            search_results = self.search(query_text, top_k)
            results[qid] = {doc_id: score for doc_id, score in search_results}

        return results

    def evaluate(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """Evaluate retrieval results.

        Args:
            results: Retrieved results.
            qrels: Ground truth relevance judgments.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metric_name to value.
        """
        try:
            import pytrec_eval
        except ImportError:
            # Fall back to simple metrics
            logger.warning("pytrec_eval not available, using simple metrics")
            return self._simple_evaluate(results, qrels, k_values)

        # Convert qrels to pytrec_eval format
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {f"ndcg_cut_{k}" for k in k_values} |
            {f"recall_{k}" for k in k_values} |
            {"map"}
        )

        # Evaluate
        eval_results = evaluator.evaluate(results)

        # Aggregate
        metrics = {}
        for k in k_values:
            ndcg_key = f"ndcg_cut_{k}"
            recall_key = f"recall_{k}"

            ndcg_scores = [r[ndcg_key] for r in eval_results.values() if ndcg_key in r]
            recall_scores = [r[recall_key] for r in eval_results.values() if recall_key in r]

            if ndcg_scores:
                metrics[f"NDCG@{k}"] = sum(ndcg_scores) / len(ndcg_scores)
            if recall_scores:
                metrics[f"Recall@{k}"] = sum(recall_scores) / len(recall_scores)

        # MAP
        map_scores = [r.get("map", 0) for r in eval_results.values()]
        if map_scores:
            metrics["MAP"] = sum(map_scores) / len(map_scores)

        return metrics

    def _simple_evaluate(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int],
    ) -> Dict[str, float]:
        """Simple evaluation without pytrec_eval.

        Args:
            results: Retrieved results.
            qrels: Ground truth.
            k_values: Values of k for metrics.

        Returns:
            Dictionary of metrics.
        """
        metrics = {}

        for k in k_values:
            recalls = []
            precisions = []

            for qid, retrieved in results.items():
                if qid not in qrels:
                    continue

                relevant = set(qrels[qid].keys())
                if not relevant:
                    continue

                # Get top-k retrieved
                sorted_docs = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)[:k]
                retrieved_set = {doc_id for doc_id, _ in sorted_docs}

                # Recall@k
                hits = len(retrieved_set & relevant)
                recalls.append(hits / len(relevant))

                # Precision@k
                precisions.append(hits / k)

            if recalls:
                metrics[f"Recall@{k}"] = sum(recalls) / len(recalls)
            if precisions:
                metrics[f"Precision@{k}"] = sum(precisions) / len(precisions)

        return metrics

    def run_evaluation(
        self,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 50, 100],
        save_results: bool = True,
    ) -> Dict[str, float]:
        """Run complete BM25 evaluation pipeline.

        Args:
            split: Split to evaluate on.
            top_k: Number of documents to retrieve per query.
            k_values: Values of k for metrics.
            save_results: Whether to save results to disk.

        Returns:
            Dictionary of metrics.
        """
        logger.info(f"Running BM25 evaluation on {split} split")

        # Load data
        corpus = self.load_corpus()
        queries = self.load_queries()
        qrels = self.load_qrels(split)

        # Filter queries to those in qrels
        queries = {qid: text for qid, text in queries.items() if qid in qrels}
        logger.info(f"Evaluating {len(queries)} queries with qrels")

        # Build index
        self.build_index(corpus)

        # Retrieve
        results = self.retrieve(queries, top_k)

        # Evaluate
        metrics = self.evaluate(results, qrels, k_values)

        logger.info(f"BM25 evaluation results: {metrics}")

        # Save results
        if save_results:
            results_path = self.results_dir / f"bm25_{split}_results.json"
            with open(results_path, "w") as f:
                json.dump({"metrics": metrics, "config": {"k1": self.k1, "b": self.b}}, f, indent=2)
            logger.info(f"Saved results to {results_path}")

            # Save run file in TREC format
            run_path = self.results_dir / f"bm25_{split}.run"
            with open(run_path, "w") as f:
                for qid, docs in results.items():
                    sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
                    for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                        f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\tBM25\n")
            logger.info(f"Saved run file to {run_path}")

        return metrics
