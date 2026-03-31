"""Dense Retrieval with Domain-Specific Embeddings.

Implements SOTA dense retrieval using:
- Sentence transformers (E5, BGE, Contriever)
- Domain-specific models (PubMedBERT, SciBERT)
- Efficient FAISS indexing with HNSW
- Batch encoding with GPU acceleration
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_sentence_transformers = None
_faiss = None
_torch = None


def _import_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            import sentence_transformers
            _sentence_transformers = sentence_transformers
        except ImportError:
            raise ImportError(
                "sentence-transformers required for dense retrieval. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformers


def _import_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss required for dense retrieval. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
    return _faiss


def _import_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("torch required for dense retrieval")
    return _torch


# Model configurations for different domains
DOMAIN_MODELS = {
    # General scientific
    "general": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-large-v2",
    "contriever": "facebook/contriever-msmarco",

    # Biomedical text
    "pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",
    "scibert": "allenai/scibert_scivocab_uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",

    # Patent-specific
    "patent": "AI-Growth-Lab/PatentSBERTa",

    # Instruction-tuned (SOTA)
    "e5-instruct": "intfloat/e5-mistral-7b-instruct",
    "bge-m3": "BAAI/bge-m3",
}


@dataclass
class DenseRetrieverConfig:
    """Configuration for dense retriever."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    index_type: str = "hnsw"  # flat, ivf, hnsw
    dimension: int = 768
    use_gpu: bool = False
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True

    # HNSW parameters
    hnsw_m: int = 32  # Number of connections
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128

    # IVF parameters
    ivf_nlist: int = 100
    ivf_nprobe: int = 10

    # Query instruction for instruction-tuned models
    query_instruction: str = "Represent this patent claim for retrieving relevant scientific prior art: "
    doc_instruction: str = ""


class DenseRetriever:
    """SOTA Dense Retriever using sentence embeddings and FAISS.

    Features:
    - Multiple embedding model support (BGE, E5, domain-specific)
    - Efficient FAISS indexing (flat, IVF, HNSW)
    - GPU acceleration
    - Instruction-tuning support
    - Batch processing

    Example:
        ```python
        retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.index_corpus(corpus)
        results = retriever.search("anti-PD-1 antibody for melanoma", top_k=100)
        ```
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        config: Optional[DenseRetrieverConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DenseRetrieverConfig(model_name=model_name)
        self.model_name = model_name

        # Import dependencies
        self.st = _import_sentence_transformers()
        self.faiss = _import_faiss()
        self.torch = _import_torch()

        # Determine device
        if device:
            self.device = device
        elif self.config.use_gpu and self.torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = self.st.SentenceTransformer(model_name, device=self.device)

        # Update dimension from model
        self.config.dimension = self.model.get_sentence_embedding_dimension()

        # Index and document storage
        self.index = None
        self.doc_ids: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

    def _get_query_text(self, query: str) -> str:
        """Format query with instruction if needed."""
        if self.config.query_instruction and "e5" in self.model_name.lower():
            return f"query: {query}"
        elif self.config.query_instruction and "bge" in self.model_name.lower():
            return self.config.query_instruction + query
        return query

    def _get_doc_text(self, doc: Union[str, Dict]) -> str:
        """Extract and format document text."""
        if isinstance(doc, str):
            text = doc
        elif isinstance(doc, dict):
            title = doc.get("title", "")
            body = doc.get("text", doc.get("abstract", doc.get("content", "")))
            text = f"{title} {body}".strip()
        else:
            text = str(doc)

        if self.config.doc_instruction:
            return self.config.doc_instruction + text
        return text

    def encode_queries(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode queries to embeddings."""
        batch_size = batch_size or self.config.batch_size

        # Format queries
        formatted = [self._get_query_text(q) for q in queries]

        embeddings = self.model.encode(
            formatted,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_documents(
        self,
        documents: List[Union[str, Dict]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode documents to embeddings."""
        batch_size = batch_size or self.config.batch_size

        # Format documents
        texts = [self._get_doc_text(doc) for doc in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings

    def _create_index(self, dimension: int) -> Any:
        """Create FAISS index based on config."""
        if self.config.index_type == "flat":
            # Exact search - best quality, slower for large corpora
            if self.config.normalize_embeddings:
                index = self.faiss.IndexFlatIP(dimension)  # Inner product for normalized
            else:
                index = self.faiss.IndexFlatL2(dimension)

        elif self.config.index_type == "hnsw":
            # HNSW - good balance of speed and quality
            index = self.faiss.IndexHNSWFlat(dimension, self.config.hnsw_m)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search

        elif self.config.index_type == "ivf":
            # IVF - fastest for very large corpora
            quantizer = self.faiss.IndexFlatIP(dimension)
            index = self.faiss.IndexIVFFlat(
                quantizer, dimension, self.config.ivf_nlist
            )

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        # Move to GPU if available
        if self.config.use_gpu and self.torch.cuda.is_available():
            try:
                res = self.faiss.StandardGpuResources()
                index = self.faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")

        return index

    def index_corpus(
        self,
        corpus: Dict[str, Union[str, Dict]],
        show_progress: bool = True,
    ) -> None:
        """Index a corpus of documents.

        Args:
            corpus: Dictionary mapping doc_id to document (str or dict with title/text)
            show_progress: Show progress bar during encoding
        """
        self.doc_ids = list(corpus.keys())
        documents = list(corpus.values())

        logger.info(f"Encoding {len(documents)} documents...")
        self.doc_embeddings = self.encode_documents(documents, show_progress=show_progress)

        logger.info(f"Building {self.config.index_type.upper()} index...")
        self.index = self._create_index(self.config.dimension)

        # Train IVF index if needed
        if self.config.index_type == "ivf":
            self.index.train(self.doc_embeddings)

        self.index.add(self.doc_embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")

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
        if self.index is None:
            raise ValueError("Index not built. Call index_corpus first.")

        # Encode query
        query_embedding = self.encode_queries([query], show_progress=False)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                doc_id = self.doc_ids[idx]
                results.append((doc_id, float(score)))

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch search for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query

        Returns:
            Dictionary mapping query to list of (doc_id, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call index_corpus first.")

        # Encode all queries
        query_embeddings = self.encode_queries(queries, show_progress=True)

        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)

        # Convert to results
        results = {}
        for i, query in enumerate(queries):
            query_results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    query_results.append((doc_id, float(score)))
            results[query] = query_results

        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Save index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.config.use_gpu:
            index_cpu = self.faiss.index_gpu_to_cpu(self.index)
            self.faiss.write_index(index_cpu, str(path / "index.faiss"))
        else:
            self.faiss.write_index(self.index, str(path / "index.faiss"))

        # Save document IDs
        import json
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)

        # Save embeddings for potential re-use
        np.save(path / "embeddings.npy", self.doc_embeddings)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: Union[str, Path]) -> None:
        """Load index and metadata from disk."""
        path = Path(path)

        # Load FAISS index
        self.index = self.faiss.read_index(str(path / "index.faiss"))

        if self.config.use_gpu and self.torch.cuda.is_available():
            res = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(res, 0, self.index)

        # Load document IDs
        import json
        with open(path / "doc_ids.json") as f:
            self.doc_ids = json.load(f)

        # Load embeddings
        self.doc_embeddings = np.load(path / "embeddings.npy")

        logger.info(f"Index loaded from {path}")


def create_domain_retriever(
    domain: str = "biomedical",
    use_gpu: bool = False,
) -> DenseRetriever:
    """Factory function to create domain-specific retrievers.

    Args:
        domain: One of "general", "biomedical", "patent", "chemical"
        use_gpu: Whether to use GPU acceleration

    Returns:
        Configured DenseRetriever instance
    """
    if domain == "biomedical":
        model = DOMAIN_MODELS["pubmedbert"]
    elif domain == "patent":
        model = DOMAIN_MODELS["patent"]
    elif domain == "general":
        model = DOMAIN_MODELS["bge-m3"]
    else:
        model = DOMAIN_MODELS.get(domain, DOMAIN_MODELS["general"])

    config = DenseRetrieverConfig(
        model_name=model,
        use_gpu=use_gpu,
        index_type="hnsw" if not use_gpu else "flat",
    )

    return DenseRetriever(model_name=model, config=config)
