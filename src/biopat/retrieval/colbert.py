"""ColBERT: Contextualized Late Interaction over BERT.

Implements ColBERT for efficient and effective neural retrieval:
- Token-level representations (unlike single-vector dense retrieval)
- MaxSim late interaction (query token to document token matching)
- Efficient retrieval with approximate nearest neighbors

ColBERT provides superior retrieval quality by preserving fine-grained
token interactions while remaining efficient for large-scale search.

References:
- Khattab & Zaharia (2020): "ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT"
- Santhanam et al. (2022): "ColBERTv2: Effective and Efficient Retrieval via
  Lightweight Late Interaction"
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_torch = None
_faiss = None


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


def _import_faiss():
    """Lazy import FAISS."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            _faiss = None
    return _faiss


# Available ColBERT models
COLBERT_MODELS = {
    # Official ColBERT models
    "colbert-v2": "colbert-ir/colbertv2.0",
    "colbert-v1": "colbert-ir/colbertv1.9",

    # Domain-specific
    "colbert-biomedical": "colbert-ir/colbertv2.0",  # Fine-tune candidate
    "colbert-scientific": "colbert-ir/colbertv2.0",

    # Multilingual
    "colbert-multilingual": "colbert-ir/colbertv2.0_msmarco_mL14",
}


@dataclass
class ColBERTConfig:
    """Configuration for ColBERT retriever."""

    # Model settings
    model_name: str = "colbert-ir/colbertv2.0"
    dim: int = 128  # ColBERT embedding dimension
    max_query_length: int = 32
    max_doc_length: int = 180

    # Index settings
    use_gpu: bool = False
    batch_size: int = 32

    # Compression
    compression_dim: Optional[int] = None  # Reduce dimension
    quantize: bool = False  # Use quantization

    # Search settings
    nprobe: int = 10  # FAISS IVF nprobe
    ncells: int = 100  # Number of IVF cells


class ColBERTEncoder:
    """ColBERT encoder producing token-level representations.

    Unlike dense retrieval which produces a single vector per text,
    ColBERT produces one vector per token, enabling fine-grained matching.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        dim: int = 128,
        max_query_length: int = 32,
        max_doc_length: int = 180,
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.dim = dim
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

        torch = _import_torch()
        if torch is None:
            raise ImportError("PyTorch required for ColBERT")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Load model
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading ColBERT model: {model_name}")

            # ColBERT uses BERT with a linear projection
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)

            # Check if model has ColBERT-specific linear layer
            if hasattr(self.bert, "linear"):
                self.linear = self.bert.linear
            else:
                # Create projection layer
                self.linear = torch.nn.Linear(self.bert.config.hidden_size, dim)
                self.linear.to(self.device)

            self.bert.to(self.device)
            self.bert.eval()

            # Special tokens for ColBERT
            self.query_token = "[Q]"
            self.doc_token = "[D]"
            self.mask_token = self.tokenizer.mask_token

        except Exception as e:
            logger.error(f"Failed to load ColBERT model {model_name}: {e}")
            # Fallback to standard BERT
            logger.info("Falling back to bert-base-uncased with projection")
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            self.linear = torch.nn.Linear(self.bert.config.hidden_size, dim)

            self.bert.to(self.device)
            self.linear.to(self.device)
            self.bert.eval()

            self.query_token = "[unused0]"
            self.doc_token = "[unused1]"
            self.mask_token = self.tokenizer.mask_token

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to token embeddings.

        Args:
            query: Query text

        Returns:
            Token embeddings (num_tokens, dim)
        """
        torch = _import_torch()

        # Add query marker
        query = f"{self.query_token} {query}"

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.max_query_length,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)

            # Project to ColBERT dimension
            embeddings = self.linear(embeddings)

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            # Get attention mask to know valid tokens
            mask = inputs["attention_mask"]

        return embeddings[0].cpu().numpy(), mask[0].cpu().numpy()

    def encode_document(self, document: str) -> np.ndarray:
        """Encode document to token embeddings.

        Args:
            document: Document text

        Returns:
            Token embeddings (num_tokens, dim)
        """
        torch = _import_torch()

        # Add document marker
        document = f"{self.doc_token} {document}"

        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            max_length=self.max_doc_length,
            truncation=True,
            padding=True,  # Variable length for docs
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state

            # Project
            embeddings = self.linear(embeddings)

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            # Get valid tokens (non-padding)
            mask = inputs["attention_mask"][0]
            valid_length = mask.sum().item()

        # Return only valid tokens
        return embeddings[0, :valid_length].cpu().numpy()

    def encode_documents_batch(
        self,
        documents: List[str],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Batch encode documents.

        Args:
            documents: List of document texts
            batch_size: Batch size

        Returns:
            List of token embeddings per document
        """
        torch = _import_torch()
        results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Add document markers
            batch = [f"{self.doc_token} {doc}" for doc in batch]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.max_doc_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.bert(**inputs)
                embeddings = self.linear(outputs.last_hidden_state)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

                masks = inputs["attention_mask"]

            # Extract valid tokens for each document
            for j, (emb, mask) in enumerate(zip(embeddings, masks)):
                valid_length = mask.sum().item()
                results.append(emb[:valid_length].cpu().numpy())

        return results


class ColBERTRetriever:
    """ColBERT retriever with late interaction scoring.

    Uses MaxSim: for each query token, find max similarity to any
    document token, then sum over query tokens.

    Example:
        ```python
        retriever = ColBERTRetriever()

        # Index corpus
        corpus = {"D1": "CRISPR gene editing...", "D2": "CAR-T therapy..."}
        retriever.index_corpus(corpus)

        # Search
        results = retriever.search("gene editing methods", top_k=10)
        for doc_id, score in results:
            print(f"{doc_id}: {score:.3f}")
        ```
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        config: Optional[ColBERTConfig] = None,
    ):
        self.config = config or ColBERTConfig(model_name=model_name)

        # Resolve model name
        if model_name in COLBERT_MODELS:
            model_name = COLBERT_MODELS[model_name]

        # Initialize encoder
        self.encoder = ColBERTEncoder(
            model_name=model_name,
            dim=self.config.dim,
            max_query_length=self.config.max_query_length,
            max_doc_length=self.config.max_doc_length,
            use_gpu=self.config.use_gpu,
        )

        # Index storage
        self.doc_ids: List[str] = []
        self.doc_embeddings: List[np.ndarray] = []  # List of (tokens, dim) arrays
        self.doc_offsets: List[int] = []  # Start offset for each doc in flattened index

        # FAISS index for approximate search
        self.faiss_index = None

    def _maxsim(
        self,
        query_emb: np.ndarray,
        query_mask: np.ndarray,
        doc_emb: np.ndarray,
    ) -> float:
        """Compute MaxSim score between query and document.

        MaxSim: sum over query tokens of max similarity to any doc token.

        Args:
            query_emb: Query embeddings (query_len, dim)
            query_mask: Query attention mask
            doc_emb: Document embeddings (doc_len, dim)

        Returns:
            MaxSim score
        """
        # Compute all pairwise similarities
        # (query_len, doc_len)
        similarities = np.dot(query_emb, doc_emb.T)

        # Max over document tokens for each query token
        max_sims = similarities.max(axis=1)

        # Sum over valid query tokens
        valid_tokens = query_mask.astype(bool)
        score = max_sims[valid_tokens].sum()

        return float(score)

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

        logger.info(f"Encoding {len(texts)} documents with ColBERT...")
        self.doc_embeddings = self.encoder.encode_documents_batch(
            texts,
            batch_size=self.config.batch_size,
        )

        # Build FAISS index for approximate candidate generation
        self._build_faiss_index()

        logger.info(f"Indexed {len(self.doc_ids)} documents")

    def _build_faiss_index(self) -> None:
        """Build FAISS index over all document tokens for approximate search."""
        faiss = _import_faiss()
        if faiss is None:
            logger.warning("FAISS not available, using exact search")
            return

        # Flatten all document embeddings
        all_embeddings = []
        self.doc_offsets = [0]

        for emb in self.doc_embeddings:
            all_embeddings.append(emb)
            self.doc_offsets.append(self.doc_offsets[-1] + len(emb))

        all_embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Build index
        dim = all_embeddings.shape[1]
        n_tokens = all_embeddings.shape[0]

        if n_tokens < 1000:
            # Small corpus: use flat index
            self.faiss_index = faiss.IndexFlatIP(dim)
        else:
            # Larger corpus: use IVF index
            n_cells = min(self.config.ncells, n_tokens // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, n_cells)
            self.faiss_index.train(all_embeddings)
            self.faiss_index.nprobe = self.config.nprobe

        self.faiss_index.add(all_embeddings)

        logger.info(f"Built FAISS index with {n_tokens} token embeddings")

    def _get_candidate_docs(
        self,
        query_emb: np.ndarray,
        n_candidates: int = 100,
    ) -> List[int]:
        """Get candidate documents using FAISS.

        For each query token, find nearest document tokens,
        then aggregate to find candidate documents.
        """
        faiss = _import_faiss()
        if faiss is None or self.faiss_index is None:
            return list(range(len(self.doc_ids)))

        # Search for each query token
        k = min(n_candidates * 4, self.faiss_index.ntotal)
        _, indices = self.faiss_index.search(query_emb.astype(np.float32), k)

        # Map token indices to document indices
        doc_hits = {}
        for token_results in indices:
            for token_idx in token_results:
                if token_idx >= 0:
                    # Binary search to find document
                    doc_idx = np.searchsorted(self.doc_offsets[1:], token_idx, side='right')
                    doc_hits[doc_idx] = doc_hits.get(doc_idx, 0) + 1

        # Return top candidates by hit count
        sorted_docs = sorted(doc_hits.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in sorted_docs[:n_candidates]]

    def search(
        self,
        query: str,
        top_k: int = 100,
        n_candidates: int = 1000,
    ) -> List[Tuple[str, float]]:
        """Search for relevant documents.

        Uses two-stage retrieval:
        1. Approximate candidate generation with FAISS
        2. Exact MaxSim scoring for candidates

        Args:
            query: Query text
            top_k: Number of results to return
            n_candidates: Number of candidates for reranking

        Returns:
            List of (doc_id, score) tuples
        """
        # Encode query
        query_emb, query_mask = self.encoder.encode_query(query)

        # Get candidates (approximate)
        candidate_indices = self._get_candidate_docs(query_emb, n_candidates)

        # Score candidates with exact MaxSim
        scores = []
        for doc_idx in candidate_indices:
            score = self._maxsim(query_emb, query_mask, self.doc_embeddings[doc_idx])
            scores.append((self.doc_ids[doc_idx], score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def search_exact(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Exact search over all documents (slower but more accurate).

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        # Encode query
        query_emb, query_mask = self.encoder.encode_query(query)

        # Score all documents
        scores = []
        for doc_idx, doc_emb in enumerate(self.doc_embeddings):
            score = self._maxsim(query_emb, query_mask, doc_emb)
            scores.append((self.doc_ids[doc_idx], score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def explain_match(
        self,
        query: str,
        doc_id: str,
        top_k_tokens: int = 10,
    ) -> Dict[str, Any]:
        """Explain why a document matches a query.

        Shows which query tokens match which document tokens,
        useful for debugging and interpretability.

        Args:
            query: Query text
            doc_id: Document ID
            top_k_tokens: Number of top matching token pairs to show

        Returns:
            Explanation dictionary with token alignments
        """
        if doc_id not in self.doc_ids:
            return {"error": f"Document {doc_id} not found"}

        doc_idx = self.doc_ids.index(doc_id)

        # Encode query
        query_emb, query_mask = self.encoder.encode_query(query)
        doc_emb = self.doc_embeddings[doc_idx]

        # Get query tokens
        query_tokens = self.encoder.tokenizer.tokenize(f"[Q] {query}")[:self.config.max_query_length]

        # Compute similarities
        similarities = np.dot(query_emb, doc_emb.T)

        # Find best matches
        matches = []
        for q_idx, q_token in enumerate(query_tokens):
            if q_idx >= len(query_emb):
                break

            best_d_idx = similarities[q_idx].argmax()
            best_score = similarities[q_idx, best_d_idx]
            matches.append({
                "query_token": q_token,
                "doc_token_idx": int(best_d_idx),
                "score": float(best_score),
            })

        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "doc_id": doc_id,
            "total_score": float(self._maxsim(query_emb, query_mask, doc_emb)),
            "top_matches": matches[:top_k_tokens],
        }

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        import pickle

        data = {
            "doc_ids": self.doc_ids,
            "doc_embeddings": self.doc_embeddings,
            "doc_offsets": self.doc_offsets,
            "config": self.config,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved ColBERT index to {path}")

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.doc_ids = data["doc_ids"]
        self.doc_embeddings = data["doc_embeddings"]
        self.doc_offsets = data.get("doc_offsets", [])

        # Rebuild FAISS index
        if not self.doc_offsets:
            self._build_faiss_index()

        logger.info(f"Loaded ColBERT index with {len(self.doc_ids)} documents")


def create_colbert_retriever(
    model_name: str = "colbert-v2",
    use_gpu: bool = False,
) -> ColBERTRetriever:
    """Factory function for ColBERT retriever.

    Args:
        model_name: ColBERT model name or shorthand
        use_gpu: Use GPU for encoding

    Returns:
        Configured ColBERTRetriever
    """
    # Resolve model shorthand
    if model_name in COLBERT_MODELS:
        model_name = COLBERT_MODELS[model_name]

    config = ColBERTConfig(model_name=model_name, use_gpu=use_gpu)
    return ColBERTRetriever(model_name=model_name, config=config)
