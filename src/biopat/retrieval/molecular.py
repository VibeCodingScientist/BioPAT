"""Molecular Embeddings for Chemical Structure Search.

Implements SOTA molecular representation learning for:
- SMILES-based similarity search
- Morgan fingerprint computation
- Learned molecular embeddings (MoLFormer, ChemBERTa)
- Tanimoto similarity with FAISS

This enables finding prior art with similar chemical structures,
critical for small molecule drug patents.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_rdkit = None
_faiss = None
_torch = None


def _import_rdkit():
    """Lazy import RDKit."""
    global _rdkit
    if _rdkit is None:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
            _rdkit = {"Chem": Chem, "AllChem": AllChem, "DataStructs": DataStructs}
        except ImportError:
            logger.warning("RDKit not available. Install with: pip install rdkit")
            _rdkit = {}
    return _rdkit


def _import_faiss():
    """Lazy import FAISS."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            logger.warning("FAISS not available. Install with: pip install faiss-cpu")
            _faiss = None
    return _faiss


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


# Available molecular embedding models
MOLECULAR_MODELS = {
    # Fingerprint-based (no deep learning)
    "morgan": "morgan_fingerprint",
    "rdkit": "rdkit_fingerprint",
    "maccs": "maccs_keys",

    # Transformer-based (SOTA)
    "molformer": "ibm/MoLFormer-XL-both-10pct",
    "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
    "chemgpt": "ncfrey/ChemGPT-4.7M",

    # Contrastive learning
    "molclr": "yikuan8/MolCLR",
}


@dataclass
class MolecularConfig:
    """Configuration for molecular retriever."""

    # Embedding method
    method: str = "morgan"  # morgan, chemberta, molformer

    # Morgan fingerprint settings
    morgan_radius: int = 2
    morgan_bits: int = 2048

    # Model settings (for transformer-based)
    model_name: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32
    use_gpu: bool = False

    # FAISS index settings
    index_type: str = "flat"  # flat, ivf
    use_binary: bool = True   # Use binary index for fingerprints


class MorganFingerprintEncoder:
    """Morgan (circular) fingerprint encoder using RDKit.

    Morgan fingerprints are the gold standard for molecular similarity.
    ECFP4 (radius=2) is commonly used in drug discovery.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.rdkit = _import_rdkit()

        if not self.rdkit:
            raise ImportError("RDKit required for Morgan fingerprints")

    def encode(self, smiles: str) -> Optional[np.ndarray]:
        """Encode SMILES to Morgan fingerprint.

        Args:
            smiles: SMILES string

        Returns:
            Binary fingerprint as numpy array, or None if invalid
        """
        try:
            mol = self.rdkit["Chem"].MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = self.rdkit["AllChem"].GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )

            # Convert to numpy array
            arr = np.zeros(self.n_bits, dtype=np.uint8)
            self.rdkit["DataStructs"].ConvertToNumpyArray(fp, arr)

            return arr

        except Exception as e:
            logger.warning(f"Failed to encode SMILES '{smiles[:50]}...': {e}")
            return None

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode multiple SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of fingerprints (N x n_bits)
        """
        fingerprints = []
        for smiles in smiles_list:
            fp = self.encode(smiles)
            if fp is not None:
                fingerprints.append(fp)
            else:
                # Use zero vector for invalid SMILES
                fingerprints.append(np.zeros(self.n_bits, dtype=np.uint8))

        return np.array(fingerprints)

    def tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute Tanimoto similarity between fingerprints."""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0


class TransformerMolecularEncoder:
    """Transformer-based molecular encoder (MoLFormer, ChemBERTa).

    Uses pre-trained language models on SMILES strings to generate
    continuous embeddings that capture chemical semantics.
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: int = 512,
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length

        torch = _import_torch()
        if torch is None:
            raise ImportError("PyTorch required for transformer encoders")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Load model
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading molecular model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def encode(self, smiles: str) -> np.ndarray:
        """Encode SMILES to embedding vector."""
        torch = _import_torch()

        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy().flatten()

    def encode_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode multiple SMILES strings."""
        torch = _import_torch()
        embeddings = []

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]

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
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_emb = outputs.pooler_output
                else:
                    batch_emb = outputs.last_hidden_state.mean(dim=1)

                embeddings.append(batch_emb.cpu().numpy())

        return np.vstack(embeddings)


class MolecularRetriever:
    """SOTA Molecular Retriever for chemical structure search.

    Supports multiple encoding methods:
    - Morgan fingerprints (fast, interpretable)
    - ChemBERTa/MoLFormer (learned representations)

    Example:
        ```python
        retriever = MolecularRetriever(method="morgan")

        # Index chemicals from corpus
        chemicals = [
            {"doc_id": "D1", "smiles": "CCO"},
            {"doc_id": "D2", "smiles": "CC(=O)O"},
        ]
        retriever.index_chemicals(chemicals)

        # Search for similar structures
        results = retriever.search("CC(=O)OC", top_k=10)
        for doc_id, similarity in results:
            print(f"{doc_id}: {similarity:.3f}")
        ```
    """

    def __init__(
        self,
        method: str = "morgan",
        config: Optional[MolecularConfig] = None,
    ):
        self.config = config or MolecularConfig(method=method)
        self.method = method

        # Initialize encoder
        if method in ["morgan", "rdkit", "maccs"]:
            self.encoder = MorganFingerprintEncoder(
                radius=self.config.morgan_radius,
                n_bits=self.config.morgan_bits,
            )
            self.use_binary = True
            self.embedding_dim = self.config.morgan_bits
        else:
            model_name = MOLECULAR_MODELS.get(method, method)
            self.encoder = TransformerMolecularEncoder(
                model_name=model_name,
                max_length=self.config.max_length,
                use_gpu=self.config.use_gpu,
            )
            self.use_binary = False
            self.embedding_dim = self.encoder.embedding_dim

        # Index storage
        self.index = None
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def _build_index(self, embeddings: np.ndarray) -> Any:
        """Build FAISS index for similarity search."""
        faiss = _import_faiss()
        if faiss is None:
            logger.warning("FAISS not available, using brute force search")
            return None

        n, d = embeddings.shape

        if self.use_binary:
            # Binary index for fingerprints (Hamming distance)
            # Convert to packed bits
            n_bytes = (d + 7) // 8
            packed = np.packbits(embeddings, axis=1)
            index = faiss.IndexBinaryFlat(d)
            index.add(packed)
        else:
            # Float index for embeddings (cosine similarity)
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / (norms + 1e-8)

            index = faiss.IndexFlatIP(d)  # Inner product = cosine for normalized
            index.add(embeddings_norm.astype(np.float32))

        return index

    def index_chemicals(
        self,
        chemicals: List[Dict[str, str]],
        doc_id_field: str = "doc_id",
        smiles_field: str = "smiles",
    ) -> None:
        """Index chemicals for similarity search.

        Args:
            chemicals: List of dicts with doc_id and smiles
            doc_id_field: Field name for document ID
            smiles_field: Field name for SMILES string
        """
        self.doc_ids = []
        smiles_list = []

        for chem in chemicals:
            doc_id = chem.get(doc_id_field)
            smiles = chem.get(smiles_field)

            if doc_id and smiles:
                self.doc_ids.append(doc_id)
                smiles_list.append(smiles)

        logger.info(f"Encoding {len(smiles_list)} chemical structures...")
        self.embeddings = self.encoder.encode_batch(smiles_list)

        logger.info("Building search index...")
        self.index = self._build_index(self.embeddings)

        logger.info(f"Indexed {len(self.doc_ids)} chemicals")

    def search(
        self,
        query_smiles: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search for similar chemical structures.

        Args:
            query_smiles: Query SMILES string
            top_k: Number of results to return

        Returns:
            List of (doc_id, similarity) tuples
        """
        # Encode query
        query_emb = self.encoder.encode(query_smiles)

        if query_emb is None:
            logger.warning(f"Invalid query SMILES: {query_smiles}")
            return []

        faiss = _import_faiss()

        if self.index is not None and faiss is not None:
            # FAISS search
            if self.use_binary:
                query_packed = np.packbits(query_emb.reshape(1, -1), axis=1)
                distances, indices = self.index.search(query_packed, top_k)
                # Convert Hamming distance to Tanimoto-like similarity
                similarities = 1 - distances[0] / self.embedding_dim
            else:
                query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                similarities, indices = self.index.search(
                    query_norm.reshape(1, -1).astype(np.float32), top_k
                )
                similarities = similarities[0]
                indices = indices[0]

            results = []
            for sim, idx in zip(similarities, indices):
                if idx >= 0:
                    results.append((self.doc_ids[idx], float(sim)))
            return results

        else:
            # Brute force search
            if self.use_binary:
                similarities = []
                for i, emb in enumerate(self.embeddings):
                    sim = self.encoder.tanimoto_similarity(query_emb, emb)
                    similarities.append((self.doc_ids[i], sim))
            else:
                query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
                sims = np.dot(emb_norms, query_norm)
                similarities = [(self.doc_ids[i], float(sims[i])) for i in range(len(self.doc_ids))]

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def search_batch(
        self,
        query_smiles_list: List[str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch search for multiple queries."""
        results = {}
        for smiles in query_smiles_list:
            results[smiles] = self.search(smiles, top_k=top_k)
        return results


def create_molecular_retriever(
    method: str = "morgan",
    use_gpu: bool = False,
) -> MolecularRetriever:
    """Factory function for molecular retriever.

    Args:
        method: "morgan" (fast) or "chemberta"/"molformer" (learned)
        use_gpu: Use GPU for transformer models

    Returns:
        Configured MolecularRetriever
    """
    config = MolecularConfig(method=method, use_gpu=use_gpu)
    return MolecularRetriever(method=method, config=config)
