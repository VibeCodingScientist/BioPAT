"""Hybrid Retrieval combining Sparse (BM25) and Dense methods.

Implements SOTA hybrid search strategies:
- Linear combination of BM25 + dense scores
- Reciprocal Rank Fusion (RRF)
- Convex combination with learned weights
- Disjunctive (union) and conjunctive (intersection) fusion
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""

    # Fusion method: "linear", "rrf", "convex"
    fusion_method: str = "rrf"

    # Linear/convex weights
    sparse_weight: float = 0.5
    dense_weight: float = 0.5

    # RRF parameter (higher = more weight to top ranks)
    rrf_k: int = 60

    # Whether to normalize scores before fusion
    normalize_scores: bool = True

    # Minimum score threshold for inclusion
    min_score: float = 0.0

    # How to combine candidate sets: "union" or "intersection"
    candidate_fusion: str = "union"


class SparseRetriever:
    """BM25 sparse retriever using rank-bm25."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_lens: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.doc_ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9-]+\b', text)
        return tokens

    def index_corpus(self, corpus: Dict[str, Any]) -> None:
        """Build BM25 index from corpus."""
        self.doc_ids = list(corpus.keys())
        self.corpus_size = len(corpus)

        total_len = 0

        for doc_id, doc in corpus.items():
            # Get document text
            if isinstance(doc, str):
                text = doc
            else:
                text = f"{doc.get('title', '')} {doc.get('text', '')}"

            tokens = self._tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)

            # Build inverted index with term frequencies
            seen = set()
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}

                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = 0

                self.inverted_index[token][doc_id] += 1

                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0

        logger.info(f"BM25 index built: {self.corpus_size} docs, {len(self.doc_freqs)} terms")

    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_lens.get(doc_id, 0)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            if doc_id not in self.inverted_index[token]:
                continue

            tf = self.inverted_index[token][doc_id]
            df = self.doc_freqs[token]

            # IDF with smoothing
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )

            score += idf * tf_norm

        return score

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Search using BM25."""
        query_tokens = self._tokenize(query)

        # Find candidate documents (docs containing at least one query term)
        candidates: Set[str] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidates.update(self.inverted_index[token].keys())

        # Score candidates
        scores = []
        for doc_id in candidates:
            score = self._bm25_score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


class HybridRetriever:
    """SOTA Hybrid Retriever combining sparse and dense methods.

    Fusion strategies:
    - Linear: α * sparse_score + (1-α) * dense_score
    - RRF: Reciprocal Rank Fusion (robust, no tuning needed)
    - Convex: Learned weighted combination

    Example:
        ```python
        from biopat.retrieval import DenseRetriever, HybridRetriever

        dense = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        hybrid = HybridRetriever(dense_retriever=dense)

        hybrid.index_corpus(corpus)
        results = hybrid.search("anti-PD-1 antibody", top_k=100)
        ```
    """

    def __init__(
        self,
        dense_retriever: Optional[Any] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        config: Optional[HybridConfig] = None,
    ):
        self.config = config or HybridConfig()
        self.dense = dense_retriever
        self.sparse = sparse_retriever or SparseRetriever()
        self.corpus: Dict[str, Any] = {}

    def index_corpus(self, corpus: Dict[str, Any]) -> None:
        """Index corpus for both sparse and dense retrieval."""
        self.corpus = corpus

        # Index sparse
        logger.info("Building sparse (BM25) index...")
        self.sparse.index_corpus(corpus)

        # Index dense
        if self.dense is not None:
            logger.info("Building dense index...")
            self.dense.index_corpus(corpus)

    def _normalize_scores(
        self,
        results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return results

        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return [(doc_id, 1.0) for doc_id, _ in results]

        return [
            (doc_id, (score - min_s) / (max_s - min_s))
            for doc_id, score in results
        ]

    def _linear_fusion(
        self,
        sparse_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Linear combination of scores."""
        # Normalize if configured
        if self.config.normalize_scores:
            sparse_results = self._normalize_scores(sparse_results)
            dense_results = self._normalize_scores(dense_results)

        # Convert to dicts
        sparse_scores = dict(sparse_results)
        dense_scores = dict(dense_results)

        # Get all candidates
        if self.config.candidate_fusion == "union":
            all_docs = set(sparse_scores.keys()) | set(dense_scores.keys())
        else:  # intersection
            all_docs = set(sparse_scores.keys()) & set(dense_scores.keys())

        # Combine scores
        combined = []
        for doc_id in all_docs:
            s_score = sparse_scores.get(doc_id, 0.0)
            d_score = dense_scores.get(doc_id, 0.0)

            combined_score = (
                self.config.sparse_weight * s_score +
                self.config.dense_weight * d_score
            )

            if combined_score >= self.config.min_score:
                combined.append((doc_id, combined_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def _rrf_fusion(
        self,
        sparse_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion - robust, hyperparameter-free."""
        k = self.config.rrf_k

        rrf_scores: Dict[str, float] = {}

        # Process sparse results
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # Process dense results
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # Sort by RRF score
        combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return combined

    def search(
        self,
        query: str,
        top_k: int = 100,
        sparse_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Hybrid search combining sparse and dense retrieval.

        Args:
            query: Query text
            top_k: Final number of results to return
            sparse_top_k: Number of sparse candidates (default: 2*top_k)
            dense_top_k: Number of dense candidates (default: 2*top_k)

        Returns:
            List of (doc_id, score) tuples
        """
        sparse_top_k = sparse_top_k or (top_k * 2)
        dense_top_k = dense_top_k or (top_k * 2)

        # Get sparse results
        sparse_results = self.sparse.search(query, top_k=sparse_top_k)

        # Get dense results
        if self.dense is not None:
            dense_results = self.dense.search(query, top_k=dense_top_k)
        else:
            dense_results = []

        # Fuse results
        if self.config.fusion_method == "linear":
            combined = self._linear_fusion(sparse_results, dense_results)
        elif self.config.fusion_method == "rrf":
            combined = self._rrf_fusion(sparse_results, dense_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

        return combined[:top_k]

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch hybrid search."""
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k=top_k)
        return results


def create_hybrid_retriever(
    model_name: str = "BAAI/bge-base-en-v1.5",
    fusion_method: str = "rrf",
    use_gpu: bool = False,
) -> HybridRetriever:
    """Factory function for hybrid retriever.

    Args:
        model_name: Dense embedding model
        fusion_method: "rrf" (recommended) or "linear"
        use_gpu: Use GPU for dense encoding

    Returns:
        Configured HybridRetriever
    """
    from biopat.retrieval.dense import DenseRetriever, DenseRetrieverConfig

    dense_config = DenseRetrieverConfig(
        model_name=model_name,
        use_gpu=use_gpu,
    )
    dense = DenseRetriever(model_name=model_name, config=dense_config)

    hybrid_config = HybridConfig(fusion_method=fusion_method)

    return HybridRetriever(
        dense_retriever=dense,
        config=hybrid_config,
    )
