"""FAISS-based Morgan Fingerprint Index for Chemical Similarity.

Phase 4.0 (Advanced): Manages chemical structure similarity search using
Morgan fingerprints and FAISS for efficient approximate nearest neighbor search.

This module handles:
- Computing Morgan fingerprints from SMILES
- Building FAISS indices for fast similarity search
- Approximating Tanimoto similarity via L2-normalized inner product
- Managing chemical structure collections for patents and publications
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import RDKit and FAISS - these are optional dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Chemical fingerprinting disabled.")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Fast similarity search disabled.")


@dataclass
class ChemicalRecord:
    """A chemical structure for indexing."""

    chemical_id: str  # Unique identifier (e.g., InChIKey)
    smiles: str  # Canonical SMILES
    source_id: str  # Patent number or paper ID
    source_type: str  # "patent" or "publication"
    inchi_key: Optional[str] = None
    name: Optional[str] = None
    mol_formula: Optional[str] = None

    def __post_init__(self):
        """Compute InChIKey if not provided."""
        if not self.inchi_key and RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol:
                self.inchi_key = Chem.MolToInchiKey(mol)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chemical_id": self.chemical_id,
            "smiles": self.smiles,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "inchi_key": self.inchi_key,
            "name": self.name,
            "mol_formula": self.mol_formula,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChemicalRecord":
        """Create from dictionary."""
        return cls(
            chemical_id=data["chemical_id"],
            smiles=data["smiles"],
            source_id=data["source_id"],
            source_type=data["source_type"],
            inchi_key=data.get("inchi_key"),
            name=data.get("name"),
            mol_formula=data.get("mol_formula"),
        )


@dataclass
class ChemicalSearchHit:
    """A hit from chemical similarity search."""

    query_id: str
    chemical_id: str
    smiles: str
    source_id: str
    source_type: str
    similarity: float  # Tanimoto similarity (0-1)
    inchi_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "chemical_id": self.chemical_id,
            "smiles": self.smiles,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "similarity": self.similarity,
            "inchi_key": self.inchi_key,
        }


class MorganFingerprintCalculator:
    """Computes Morgan fingerprints from molecular structures."""

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        """Initialize fingerprint calculator.

        Args:
            radius: Morgan fingerprint radius (default 2 = ECFP4 equivalent).
            n_bits: Number of bits in the fingerprint vector.
        """
        self.radius = radius
        self.n_bits = n_bits

    def compute_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Compute Morgan fingerprint from SMILES.

        Args:
            smiles: SMILES string.

        Returns:
            Numpy array of fingerprint bits, or None if invalid.
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for fingerprint computation")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )

        # Convert to numpy array
        arr = np.zeros(self.n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    def compute_fingerprint_normalized(self, smiles: str) -> Optional[np.ndarray]:
        """Compute L2-normalized Morgan fingerprint.

        L2 normalization allows using inner product for Tanimoto approximation.

        Args:
            smiles: SMILES string.

        Returns:
            L2-normalized numpy array, or None if invalid.
        """
        fp = self.compute_fingerprint(smiles)
        if fp is None:
            return None

        # L2 normalize for inner product similarity
        norm = np.linalg.norm(fp)
        if norm > 0:
            fp = fp / norm

        return fp

    def compute_batch(
        self, smiles_list: List[str], normalize: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """Compute fingerprints for a batch of SMILES.

        Args:
            smiles_list: List of SMILES strings.
            normalize: Whether to L2-normalize fingerprints.

        Returns:
            Tuple of (fingerprint matrix, list of valid indices).
        """
        fingerprints = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            if normalize:
                fp = self.compute_fingerprint_normalized(smiles)
            else:
                fp = self.compute_fingerprint(smiles)

            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(i)

        if fingerprints:
            return np.vstack(fingerprints).astype(np.float32), valid_indices
        return np.array([]).reshape(0, self.n_bits).astype(np.float32), []


def compute_tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints.

    Args:
        fp1: First fingerprint (binary or count vector).
        fp2: Second fingerprint (binary or count vector).

    Returns:
        Tanimoto similarity (0-1).
    """
    # For binary fingerprints: Tanimoto = |A & B| / |A | B|
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))

    if union == 0:
        return 0.0
    return float(intersection / union)


def tanimoto_from_inner_product(inner_product: float) -> float:
    """Convert inner product of L2-normalized fingerprints to Tanimoto.

    This is an approximation that works well for sparse fingerprints.

    Args:
        inner_product: Inner product of L2-normalized fingerprints.

    Returns:
        Approximate Tanimoto similarity (0-1).
    """
    # Clamp to valid range
    inner_product = max(0.0, min(1.0, inner_product))
    return inner_product


class FaissChemicalIndex:
    """FAISS-based index for chemical similarity search.

    Uses L2-normalized Morgan fingerprints with inner product similarity.
    """

    def __init__(
        self,
        fingerprint_dim: int = 2048,
        use_gpu: bool = False,
    ):
        """Initialize FAISS index.

        Args:
            fingerprint_dim: Dimension of fingerprint vectors.
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu).
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is required for FaissChemicalIndex")

        self.fingerprint_dim = fingerprint_dim
        self.use_gpu = use_gpu

        # Create index for inner product similarity
        self.index = faiss.IndexFlatIP(fingerprint_dim)

        if use_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU-accelerated FAISS index")
            except Exception as e:
                logger.warning(f"Could not initialize GPU index: {e}")
                self.use_gpu = False

        # Metadata storage
        self._records: List[ChemicalRecord] = []

    @property
    def size(self) -> int:
        """Number of indexed chemicals."""
        return self.index.ntotal

    def add(self, fingerprints: np.ndarray, records: List[ChemicalRecord]) -> int:
        """Add fingerprints to the index.

        Args:
            fingerprints: L2-normalized fingerprint matrix (N x dim).
            records: Corresponding ChemicalRecord objects.

        Returns:
            Number of vectors added.
        """
        if len(fingerprints) != len(records):
            raise ValueError("Fingerprints and records must have same length")

        if len(fingerprints) == 0:
            return 0

        # Ensure proper shape and type
        fingerprints = np.ascontiguousarray(fingerprints, dtype=np.float32)

        self.index.add(fingerprints)
        self._records.extend(records)

        return len(fingerprints)

    def search(
        self,
        query_fp: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[ChemicalRecord, float]]:
        """Search for similar chemicals.

        Args:
            query_fp: L2-normalized query fingerprint.
            k: Number of results to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of (ChemicalRecord, similarity) tuples.
        """
        if self.size == 0:
            return []

        # Reshape for FAISS
        query_fp = np.ascontiguousarray(query_fp.reshape(1, -1), dtype=np.float32)

        # Search
        k = min(k, self.size)
        similarities, indices = self.index.search(query_fp, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and sim >= min_similarity:
                tanimoto = tanimoto_from_inner_product(sim)
                results.append((self._records[idx], tanimoto))

        return results

    def batch_search(
        self,
        query_fps: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[List[Tuple[ChemicalRecord, float]]]:
        """Search for similar chemicals for multiple queries.

        Args:
            query_fps: L2-normalized query fingerprints (N x dim).
            k: Number of results per query.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of result lists.
        """
        if self.size == 0:
            return [[] for _ in range(len(query_fps))]

        query_fps = np.ascontiguousarray(query_fps, dtype=np.float32)
        k = min(k, self.size)

        similarities, indices = self.index.search(query_fps, k)

        all_results = []
        for sim_row, idx_row in zip(similarities, indices):
            results = []
            for sim, idx in zip(sim_row, idx_row):
                if idx >= 0 and sim >= min_similarity:
                    tanimoto = tanimoto_from_inner_product(sim)
                    results.append((self._records[idx], tanimoto))
            all_results.append(results)

        return all_results

    def save(self, path: Path) -> None:
        """Save index to disk.

        Args:
            path: Path to save index (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = str(path) + ".faiss"
        if self.use_gpu:
            # Convert back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)

        # Save metadata
        metadata_path = str(path) + ".json"
        metadata = {
            "fingerprint_dim": self.fingerprint_dim,
            "records": [r.to_dict() for r in self._records],
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved chemical index with {self.size} records to {path}")

    @classmethod
    def load(cls, path: Path, use_gpu: bool = False) -> "FaissChemicalIndex":
        """Load index from disk.

        Args:
            path: Path to saved index (without extension).
            use_gpu: Whether to use GPU acceleration.

        Returns:
            Loaded FaissChemicalIndex.
        """
        path = Path(path)

        # Load metadata
        metadata_path = str(path) + ".json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            fingerprint_dim=metadata["fingerprint_dim"],
            use_gpu=use_gpu,
        )

        # Load FAISS index
        index_path = str(path) + ".faiss"
        instance.index = faiss.read_index(index_path)

        if use_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                res = faiss.StandardGpuResources()
                instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
            except Exception as e:
                logger.warning(f"Could not load to GPU: {e}")
                instance.use_gpu = False

        # Load records
        instance._records = [
            ChemicalRecord.from_dict(r) for r in metadata["records"]
        ]

        logger.info(f"Loaded chemical index with {instance.size} records from {path}")
        return instance


class ChemicalIndex:
    """High-level chemical index for BioPAT benchmark.

    Manages separate indices for patents and publications,
    supporting chemical structure-based prior art search.
    """

    def __init__(
        self,
        index_dir: Path,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        use_gpu: bool = False,
    ):
        """Initialize chemical index.

        Args:
            index_dir: Directory for storing indices.
            fingerprint_radius: Morgan fingerprint radius.
            fingerprint_bits: Number of fingerprint bits.
            use_gpu: Whether to use GPU acceleration.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.fingerprint_calculator = MorganFingerprintCalculator(
            radius=fingerprint_radius,
            n_bits=fingerprint_bits,
        )
        self.fingerprint_bits = fingerprint_bits
        self.use_gpu = use_gpu

        # Lazy-loaded indices
        self._patent_index: Optional[FaissChemicalIndex] = None
        self._publication_index: Optional[FaissChemicalIndex] = None

    @property
    def patent_index(self) -> FaissChemicalIndex:
        """Get or create patent chemical index."""
        if self._patent_index is None:
            index_path = self.index_dir / "patent_chemicals"
            if (Path(str(index_path) + ".faiss")).exists():
                self._patent_index = FaissChemicalIndex.load(
                    index_path, self.use_gpu
                )
            else:
                self._patent_index = FaissChemicalIndex(
                    self.fingerprint_bits, self.use_gpu
                )
        return self._patent_index

    @property
    def publication_index(self) -> FaissChemicalIndex:
        """Get or create publication chemical index."""
        if self._publication_index is None:
            index_path = self.index_dir / "publication_chemicals"
            if (Path(str(index_path) + ".faiss")).exists():
                self._publication_index = FaissChemicalIndex.load(
                    index_path, self.use_gpu
                )
            else:
                self._publication_index = FaissChemicalIndex(
                    self.fingerprint_bits, self.use_gpu
                )
        return self._publication_index

    def index_chemicals(
        self,
        chemicals: List[ChemicalRecord],
        source_type: str = "patent",
    ) -> int:
        """Index chemical structures.

        Args:
            chemicals: Chemical records to index.
            source_type: "patent" or "publication".

        Returns:
            Number of chemicals indexed.
        """
        if not RDKIT_AVAILABLE or not FAISS_AVAILABLE:
            logger.error("RDKit and FAISS are required for chemical indexing")
            return 0

        # Select target index
        index = self.patent_index if source_type == "patent" else self.publication_index

        # Compute fingerprints
        smiles_list = [c.smiles for c in chemicals]
        fingerprints, valid_indices = self.fingerprint_calculator.compute_batch(
            smiles_list, normalize=True
        )

        if len(valid_indices) == 0:
            return 0

        # Filter to valid records
        valid_records = [chemicals[i] for i in valid_indices]

        # Add to index
        return index.add(fingerprints, valid_records)

    def search_prior_art(
        self,
        query_smiles: str,
        k: int = 50,
        min_similarity: float = 0.5,
    ) -> List[ChemicalSearchHit]:
        """Search for chemically similar prior art.

        Args:
            query_smiles: Query SMILES string.
            k: Maximum number of results.
            min_similarity: Minimum Tanimoto similarity threshold.

        Returns:
            List of ChemicalSearchHit objects.
        """
        if not RDKIT_AVAILABLE or not FAISS_AVAILABLE:
            logger.error("RDKit and FAISS are required for chemical search")
            return []

        # Compute query fingerprint
        query_fp = self.fingerprint_calculator.compute_fingerprint_normalized(query_smiles)
        if query_fp is None:
            logger.warning(f"Could not compute fingerprint for: {query_smiles}")
            return []

        # Search publication index for prior art
        results = self.publication_index.search(query_fp, k, min_similarity)

        # Convert to search hits
        hits = []
        for record, similarity in results:
            hit = ChemicalSearchHit(
                query_id=query_smiles[:50],  # Truncated SMILES as ID
                chemical_id=record.chemical_id,
                smiles=record.smiles,
                source_id=record.source_id,
                source_type=record.source_type,
                similarity=similarity,
                inchi_key=record.inchi_key,
            )
            hits.append(hit)

        return sorted(hits, key=lambda x: x.similarity, reverse=True)

    def search_similar(
        self,
        query_smiles: str,
        source_type: str = "publication",
        k: int = 50,
        min_similarity: float = 0.5,
    ) -> List[ChemicalSearchHit]:
        """Search for similar chemicals in specified index.

        Args:
            query_smiles: Query SMILES string.
            source_type: "patent" or "publication".
            k: Maximum number of results.
            min_similarity: Minimum Tanimoto similarity threshold.

        Returns:
            List of ChemicalSearchHit objects.
        """
        if not RDKIT_AVAILABLE or not FAISS_AVAILABLE:
            logger.error("RDKit and FAISS are required for chemical search")
            return []

        # Compute query fingerprint
        query_fp = self.fingerprint_calculator.compute_fingerprint_normalized(query_smiles)
        if query_fp is None:
            return []

        # Select index
        index = self.patent_index if source_type == "patent" else self.publication_index

        # Search
        results = index.search(query_fp, k, min_similarity)

        # Convert to search hits
        hits = []
        for record, similarity in results:
            hit = ChemicalSearchHit(
                query_id=query_smiles[:50],
                chemical_id=record.chemical_id,
                smiles=record.smiles,
                source_id=record.source_id,
                source_type=record.source_type,
                similarity=similarity,
                inchi_key=record.inchi_key,
            )
            hits.append(hit)

        return sorted(hits, key=lambda x: x.similarity, reverse=True)

    def save(self) -> None:
        """Save all indices to disk."""
        if self._patent_index is not None:
            self._patent_index.save(self.index_dir / "patent_chemicals")
        if self._publication_index is not None:
            self._publication_index.save(self.index_dir / "publication_chemicals")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "patent_chemicals": self.patent_index.size,
            "publication_chemicals": self.publication_index.size,
            "fingerprint_bits": self.fingerprint_bits,
            "use_gpu": self.use_gpu,
        }


def tanimoto_to_relevance_tier(tanimoto: float) -> int:
    """Convert Tanimoto similarity to relevance tier.

    Based on medicinal chemistry similarity thresholds:
    - >= 0.85: Highly relevant (tier 3) - likely same scaffold
    - >= 0.70: Relevant (tier 2) - similar activity profile likely
    - >= 0.50: Marginally relevant (tier 1) - weak structural similarity
    - < 0.50: Not relevant (tier 0)
    """
    if tanimoto >= 0.85:
        return 3
    elif tanimoto >= 0.70:
        return 2
    elif tanimoto >= 0.50:
        return 1
    return 0


def compute_chemical_id(smiles: str) -> str:
    """Compute a unique chemical ID from SMILES.

    Uses canonical SMILES -> InChIKey if RDKit available,
    otherwise falls back to hash of original SMILES.
    """
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            inchi_key = Chem.MolToInchiKey(mol)
            if inchi_key:
                return inchi_key
            # Fall back to hash of canonical SMILES
            return hashlib.sha256(canonical.encode()).hexdigest()[:27]

    # Non-RDKit fallback: hash of original SMILES
    return hashlib.sha256(smiles.encode()).hexdigest()[:27]
