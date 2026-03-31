"""SPLADE: Sparse Lexical and Expansion Model.

Implements learned sparse retrieval that combines:
- Lexical matching (like BM25)
- Query/document expansion via MLM
- Learned term importance weights

SPLADE provides SOTA sparse retrieval performance, often matching or
exceeding dense models while maintaining interpretability.

References:
- Formal et al. (2021): "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
- Formal et al. (2022): "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval"
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_torch = None
_scipy_sparse = None


def _import_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
    return _torch


def _import_scipy_sparse():
    """Lazy import scipy.sparse."""
    global _scipy_sparse
    if _scipy_sparse is None:
        try:
            from scipy import sparse
            _scipy_sparse = sparse
        except ImportError:
            _scipy_sparse = None
    return _scipy_sparse


# Available SPLADE models
SPLADE_MODELS = {
    # Official SPLADE models
    "splade-cocondenser": "naver/splade-cocondenser-ensembledistil",
    "splade-v2": "naver/splade_v2_distil",
    "splade-max": "naver/splade_v2_max",

    # Efficient variants
    "splade-efficient": "naver/efficient-splade-VI-BT-large-query",
    "splade-doc": "naver/efficient-splade-VI-BT-large-doc",

    # Domain-specific
    "splade-biomedical": "naver/splade-cocondenser-ensembledistil",  # Fine-tune candidate
}


@dataclass
class SPLADEConfig:
    """Configuration for SPLADE retriever."""

    # Model settings
    model_name: str = "naver/splade-cocondenser-ensembledistil"
    max_length: int = 256

    # Encoding settings
    agg: str = "max"  # "max" or "sum" for token aggregation

    # Sparsity settings
    top_k_tokens: Optional[int] = None  # Limit expansion terms
    threshold: float = 0.0  # Minimum weight threshold

    # Index settings
    use_gpu: bool = False
    batch_size: int = 32

    # Quantization (for efficiency)
    quantize_8bit: bool = False


class SPLADEEncoder:
    """SPLADE encoder using transformers.

    SPLADE uses MLM heads to predict term importance:
    - Sparse output (most terms have zero weight)
    - Interpretable (can see which terms are important)
    - Efficient indexing with inverted indices
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        max_length: int = 256,
        agg: str = "max",
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.agg = agg

        torch = _import_torch()
        if torch is None:
            raise ImportError("PyTorch required for SPLADE")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Load model
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            logger.info(f"Loading SPLADE model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Get vocab size
            self.vocab_size = self.tokenizer.vocab_size

        except Exception as e:
            logger.error(f"Failed to load SPLADE model {model_name}: {e}")
            raise

    def encode(
        self,
        text: str,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> Dict[int, float]:
        """Encode text to sparse SPLADE representation.

        Args:
            text: Input text
            threshold: Minimum weight threshold
            top_k: Maximum number of terms (None = unlimited)

        Returns:
            Dictionary mapping token IDs to weights
        """
        torch = _import_torch()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

            # Apply log(1 + ReLU(x)) transformation
            # This creates sparse, positive weights
            weights = torch.log1p(torch.relu(logits))

            # Aggregate over sequence (max or sum)
            if self.agg == "max":
                weights = weights.max(dim=1).values  # (batch, vocab_size)
            else:
                weights = weights.sum(dim=1)

            # Move to CPU and convert to dict
            weights = weights.squeeze(0).cpu().numpy()

        # Build sparse representation
        sparse_rep = {}
        for token_id, weight in enumerate(weights):
            if weight > threshold:
                sparse_rep[token_id] = float(weight)

        # Apply top-k filtering
        if top_k is not None and len(sparse_rep) > top_k:
            sorted_items = sorted(sparse_rep.items(), key=lambda x: x[1], reverse=True)
            sparse_rep = dict(sorted_items[:top_k])

        return sparse_rep

    def encode_batch(
        self,
        texts: List[str],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        batch_size: int = 32,
    ) -> List[Dict[int, float]]:
        """Batch encode texts to sparse representations."""
        torch = _import_torch()
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Apply SPLADE transformation
                weights = torch.log1p(torch.relu(logits))

                if self.agg == "max":
                    weights = weights.max(dim=1).values
                else:
                    weights = weights.sum(dim=1)

                weights = weights.cpu().numpy()

            # Convert each to sparse dict
            for w in weights:
                sparse_rep = {}
                for token_id, weight in enumerate(w):
                    if weight > threshold:
                        sparse_rep[token_id] = float(weight)

                if top_k is not None and len(sparse_rep) > top_k:
                    sorted_items = sorted(sparse_rep.items(), key=lambda x: x[1], reverse=True)
                    sparse_rep = dict(sorted_items[:top_k])

                results.append(sparse_rep)

        return results

    def decode_representation(self, sparse_rep: Dict[int, float], top_k: int = 20) -> List[Tuple[str, float]]:
        """Decode sparse representation back to tokens (for interpretability).

        Args:
            sparse_rep: Token ID to weight mapping
            top_k: Number of top terms to return

        Returns:
            List of (token, weight) tuples
        """
        sorted_items = sorted(sparse_rep.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.tokenizer.decode([token_id]).strip(), weight) for token_id, weight in sorted_items]


class SPLADERetriever:
    """SPLADE-based sparse retriever.

    Uses learned sparse representations for efficient and effective retrieval.
    Supports both exact and approximate search.

    Example:
        ```python
        retriever = SPLADERetriever()

        # Index corpus
        corpus = {"D1": "CRISPR gene editing...", "D2": "CAR-T therapy..."}
        retriever.index_corpus(corpus)

        # Search
        results = retriever.search("gene editing methods", top_k=10)
        for doc_id, score in results:
            print(f"{doc_id}: {score:.3f}")

        # Interpret query expansion
        expansion = retriever.get_query_expansion("CRISPR")
        print("Query expanded to:", expansion)
        ```
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        config: Optional[SPLADEConfig] = None,
    ):
        self.config = config or SPLADEConfig(model_name=model_name)

        # Initialize encoder
        self.encoder = SPLADEEncoder(
            model_name=self.config.model_name,
            max_length=self.config.max_length,
            agg=self.config.agg,
            use_gpu=self.config.use_gpu,
        )

        # Index storage
        self.doc_ids: List[str] = []
        self.doc_vectors: List[Dict[int, float]] = []
        self.inverted_index: Dict[int, List[Tuple[int, float]]] = {}  # token_id -> [(doc_idx, weight)]

    def _build_inverted_index(self) -> None:
        """Build inverted index for efficient search."""
        self.inverted_index = {}

        for doc_idx, sparse_vec in enumerate(self.doc_vectors):
            for token_id, weight in sparse_vec.items():
                if token_id not in self.inverted_index:
                    self.inverted_index[token_id] = []
                self.inverted_index[token_id].append((doc_idx, weight))

        logger.info(f"Built inverted index with {len(self.inverted_index)} unique terms")

    def index_corpus(
        self,
        corpus: Dict[str, Union[str, Dict[str, Any]]],
        text_field: str = "text",
    ) -> None:
        """Index corpus for retrieval.

        Args:
            corpus: Dictionary mapping doc_id to text or dict with text_field
            text_field: Field name for text if corpus values are dicts
        """
        self.doc_ids = []
        texts = []

        for doc_id, doc in corpus.items():
            self.doc_ids.append(doc_id)
            if isinstance(doc, dict):
                text = doc.get(text_field, "")
                if "title" in doc:
                    text = doc["title"] + " " + text
            else:
                text = doc
            texts.append(text)

        logger.info(f"Encoding {len(texts)} documents with SPLADE...")
        self.doc_vectors = self.encoder.encode_batch(
            texts,
            threshold=self.config.threshold,
            top_k=self.config.top_k_tokens,
            batch_size=self.config.batch_size,
        )

        logger.info("Building inverted index...")
        self._build_inverted_index()

        logger.info(f"Indexed {len(self.doc_ids)} documents")

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search for relevant documents.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        # Encode query
        query_vec = self.encoder.encode(
            query,
            threshold=self.config.threshold,
            top_k=self.config.top_k_tokens,
        )

        # Compute scores using inverted index (efficient sparse dot product)
        scores = np.zeros(len(self.doc_ids))

        for token_id, query_weight in query_vec.items():
            if token_id in self.inverted_index:
                for doc_idx, doc_weight in self.inverted_index[token_id]:
                    scores[doc_idx] += query_weight * doc_weight

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch search for multiple queries."""
        results = {}

        # Encode all queries
        query_vecs = self.encoder.encode_batch(
            queries,
            threshold=self.config.threshold,
            top_k=self.config.top_k_tokens,
            batch_size=self.config.batch_size,
        )

        for query, query_vec in zip(queries, query_vecs):
            scores = np.zeros(len(self.doc_ids))

            for token_id, query_weight in query_vec.items():
                if token_id in self.inverted_index:
                    for doc_idx, doc_weight in self.inverted_index[token_id]:
                        scores[doc_idx] += query_weight * doc_weight

            top_indices = np.argsort(scores)[::-1][:top_k]

            query_results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    query_results.append((self.doc_ids[idx], float(scores[idx])))

            results[query] = query_results

        return results

    def get_query_expansion(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get interpretable query expansion.

        Shows which terms SPLADE expands the query to, useful for
        understanding and debugging retrieval behavior.

        Args:
            query: Query text
            top_k: Number of expansion terms to return

        Returns:
            List of (term, weight) tuples
        """
        sparse_rep = self.encoder.encode(query, threshold=0.0)
        return self.encoder.decode_representation(sparse_rep, top_k=top_k)

    def get_document_terms(self, doc_id: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get top terms for a document.

        Args:
            doc_id: Document ID
            top_k: Number of terms to return

        Returns:
            List of (term, weight) tuples
        """
        if doc_id not in self.doc_ids:
            return []

        doc_idx = self.doc_ids.index(doc_id)
        sparse_rep = self.doc_vectors[doc_idx]
        return self.encoder.decode_representation(sparse_rep, top_k=top_k)

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        import pickle

        data = {
            "doc_ids": self.doc_ids,
            "doc_vectors": self.doc_vectors,
            "config": self.config,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved SPLADE index to {path}")

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.doc_ids = data["doc_ids"]
        self.doc_vectors = data["doc_vectors"]

        # Rebuild inverted index
        self._build_inverted_index()

        logger.info(f"Loaded SPLADE index with {len(self.doc_ids)} documents")


class HybridSPLADERetriever:
    """Hybrid retriever combining SPLADE with dense retrieval.

    Combines the interpretability and lexical matching of SPLADE
    with the semantic understanding of dense retrieval.
    """

    def __init__(
        self,
        splade_retriever: SPLADERetriever,
        dense_retriever: Any,  # DenseRetriever
        splade_weight: float = 0.5,
        fusion_method: str = "rrf",  # "rrf" or "linear"
        rrf_k: int = 60,
    ):
        self.splade = splade_retriever
        self.dense = dense_retriever
        self.splade_weight = splade_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search using both SPLADE and dense, then fuse results."""

        # Get results from both
        splade_results = self.splade.search(query, top_k=top_k * 2)
        dense_results = self.dense.search(query, top_k=top_k * 2)

        if self.fusion_method == "rrf":
            return self._rrf_fusion(splade_results, dense_results, top_k)
        else:
            return self._linear_fusion(splade_results, dense_results, top_k)

    def _rrf_fusion(
        self,
        results1: List[Tuple[str, float]],
        results2: List[Tuple[str, float]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion."""
        scores = {}

        for rank, (doc_id, _) in enumerate(results1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (doc_id, _) in enumerate(results2):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _linear_fusion(
        self,
        results1: List[Tuple[str, float]],
        results2: List[Tuple[str, float]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Linear combination of normalized scores."""
        # Normalize scores
        def normalize(results):
            if not results:
                return {}
            max_score = max(s for _, s in results)
            min_score = min(s for _, s in results)
            if max_score == min_score:
                return {doc_id: 1.0 for doc_id, _ in results}
            return {doc_id: (s - min_score) / (max_score - min_score) for doc_id, s in results}

        scores1 = normalize(results1)
        scores2 = normalize(results2)

        # Combine
        all_docs = set(scores1.keys()) | set(scores2.keys())
        combined = {}

        for doc_id in all_docs:
            s1 = scores1.get(doc_id, 0)
            s2 = scores2.get(doc_id, 0)
            combined[doc_id] = self.splade_weight * s1 + (1 - self.splade_weight) * s2

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


def create_splade_retriever(
    model_name: str = "naver/splade-cocondenser-ensembledistil",
    use_gpu: bool = False,
) -> SPLADERetriever:
    """Factory function for SPLADE retriever.

    Args:
        model_name: SPLADE model name or shorthand
        use_gpu: Use GPU for encoding

    Returns:
        Configured SPLADERetriever
    """
    # Resolve model shorthand
    if model_name in SPLADE_MODELS:
        model_name = SPLADE_MODELS[model_name]

    config = SPLADEConfig(model_name=model_name, use_gpu=use_gpu)
    return SPLADERetriever(model_name=model_name, config=config)
