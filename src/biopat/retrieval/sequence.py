"""Protein/Nucleotide Sequence Embeddings for Biological Sequence Search.

Implements SOTA sequence representation learning for:
- Protein sequence similarity (ESM-2, ProtBERT)
- Nucleotide sequence similarity
- BLAST-based alignment
- Learned sequence embeddings

This enables finding prior art with similar biological sequences,
critical for antibody, protein therapeutic, and gene therapy patents.
"""

import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_torch = None
_faiss = None


def _import_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
    return _torch


def _import_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            _faiss = None
    return _faiss


# Available sequence embedding models
SEQUENCE_MODELS = {
    # Protein language models (SOTA)
    "esm2": "facebook/esm2_t33_650M_UR50D",
    "esm2-small": "facebook/esm2_t6_8M_UR50D",
    "protbert": "Rostlab/prot_bert",
    "prott5": "Rostlab/prot_t5_xl_uniref50",

    # Antibody-specific
    "ablang": "ablang/ablang-heavy",
    "antiberta": "alchemab/antiberta",

    # Nucleotide models
    "dnabert": "zhihan1996/DNA_bert_6",
    "nucleotide-transformer": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
}


@dataclass
class SequenceConfig:
    """Configuration for sequence retriever."""

    # Sequence type
    sequence_type: str = "protein"  # protein, nucleotide, antibody

    # Embedding method
    method: str = "esm2"  # esm2, protbert, blast

    # Model settings
    model_name: Optional[str] = None
    max_length: int = 1024
    batch_size: int = 8
    use_gpu: bool = False

    # BLAST settings
    blast_db_path: Optional[str] = None
    blast_evalue: float = 1e-5
    blast_threads: int = 4

    # Index settings
    index_type: str = "flat"


class BLASTSearcher:
    """BLAST-based sequence alignment search.

    Uses NCBI BLAST+ for traditional sequence alignment.
    Gold standard for finding homologous sequences.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        sequence_type: str = "protein",
        evalue: float = 1e-5,
        threads: int = 4,
    ):
        self.db_path = db_path
        self.sequence_type = sequence_type
        self.evalue = evalue
        self.threads = threads

        # Determine BLAST program
        self.blast_program = "blastp" if sequence_type == "protein" else "blastn"

        # Check if BLAST is available
        self._check_blast_available()

    def _check_blast_available(self) -> bool:
        """Check if BLAST+ is installed."""
        try:
            result = subprocess.run(
                [self.blast_program, "-version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning(f"{self.blast_program} not found. Install BLAST+.")
            return False

    def create_database(
        self,
        sequences: List[Dict[str, str]],
        db_path: str,
        doc_id_field: str = "doc_id",
        sequence_field: str = "sequence",
    ) -> None:
        """Create BLAST database from sequences.

        Args:
            sequences: List of dicts with doc_id and sequence
            db_path: Path to create database
            doc_id_field: Field name for document ID
            sequence_field: Field name for sequence
        """
        self.db_path = db_path
        db_type = "prot" if self.sequence_type == "protein" else "nucl"

        # Write FASTA file
        fasta_path = f"{db_path}.fasta"
        with open(fasta_path, "w") as f:
            for seq_data in sequences:
                doc_id = seq_data.get(doc_id_field, "unknown")
                sequence = seq_data.get(sequence_field, "")
                if sequence:
                    f.write(f">{doc_id}\n{sequence}\n")

        # Create BLAST database
        cmd = [
            "makeblastdb",
            "-in", fasta_path,
            "-dbtype", db_type,
            "-out", db_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"BLAST database created at {db_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create BLAST database: {e}")
            raise

    def search(
        self,
        query_sequence: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search for similar sequences using BLAST.

        Args:
            query_sequence: Query sequence string
            top_k: Maximum number of results

        Returns:
            List of (doc_id, identity_score) tuples
        """
        if not self.db_path:
            logger.warning("No BLAST database configured")
            return []

        # Write query to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(f">query\n{query_sequence}\n")
            query_path = f.name

        # Run BLAST
        cmd = [
            self.blast_program,
            "-query", query_path,
            "-db", self.db_path,
            "-outfmt", "6 sseqid pident evalue bitscore",
            "-evalue", str(self.evalue),
            "-max_target_seqs", str(top_k),
            "-num_threads", str(self.threads),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"BLAST search failed: {e}")
            return []
        finally:
            Path(query_path).unlink(missing_ok=True)

        # Parse results
        results = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    doc_id = parts[0]
                    identity = float(parts[1]) / 100.0  # Convert percentage to 0-1
                    results.append((doc_id, identity))

        return results


class TransformerSequenceEncoder:
    """Transformer-based sequence encoder (ESM-2, ProtBERT).

    Uses protein language models to generate embeddings that
    capture evolutionary and structural information.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        max_length: int = 1024,
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length

        torch = _import_torch()
        if torch is None:
            raise ImportError("PyTorch required for transformer sequence encoder")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Determine model type and load
        if "esm" in model_name.lower():
            self._load_esm_model(model_name)
        else:
            self._load_hf_model(model_name)

    def _load_esm_model(self, model_name: str) -> None:
        """Load ESM model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading ESM model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self.model_type = "esm"

        except Exception as e:
            logger.error(f"Failed to load ESM model: {e}")
            raise

    def _load_hf_model(self, model_name: str) -> None:
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading sequence model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.hidden_size
            self.model_type = "hf"

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess sequence for model input."""
        # Remove whitespace and convert to uppercase
        sequence = "".join(sequence.split()).upper()

        # For ProtBERT, add spaces between amino acids
        if "prot_bert" in self.model_name.lower():
            sequence = " ".join(list(sequence))

        return sequence

    def encode(self, sequence: str) -> np.ndarray:
        """Encode sequence to embedding vector."""
        torch = _import_torch()

        sequence = self._preprocess_sequence(sequence)

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Use mean pooling over sequence positions
            if hasattr(outputs, "last_hidden_state"):
                # Mask padding tokens
                attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                embedding = sum_embeddings / sum_mask
            else:
                embedding = outputs[0].mean(dim=1)

        return embedding.cpu().numpy().flatten()

    def encode_batch(
        self,
        sequences: List[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Encode multiple sequences."""
        torch = _import_torch()
        embeddings = []

        processed_seqs = [self._preprocess_sequence(s) for s in sequences]

        for i in range(0, len(processed_seqs), batch_size):
            batch = processed_seqs[i:i + batch_size]

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

                if hasattr(outputs, "last_hidden_state"):
                    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
                    mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1)
                    batch_emb = sum_embeddings / sum_mask
                else:
                    batch_emb = outputs[0].mean(dim=1)

                embeddings.append(batch_emb.cpu().numpy())

        return np.vstack(embeddings)


class SequenceRetriever:
    """SOTA Sequence Retriever for biological sequence search.

    Supports multiple methods:
    - BLAST alignment (gold standard)
    - ESM-2/ProtBERT embeddings (learned representations)

    Example:
        ```python
        # Protein search with ESM-2
        retriever = SequenceRetriever(method="esm2", sequence_type="protein")

        sequences = [
            {"doc_id": "D1", "sequence": "MVLSPADKTNV..."},
            {"doc_id": "D2", "sequence": "MWLLLLLLLL..."},
        ]
        retriever.index_sequences(sequences)

        results = retriever.search("MVLSPADKTNVKAAWGKVGAHAG...", top_k=10)
        ```
    """

    def __init__(
        self,
        method: str = "esm2-small",
        sequence_type: str = "protein",
        config: Optional[SequenceConfig] = None,
    ):
        self.config = config or SequenceConfig(method=method, sequence_type=sequence_type)
        self.method = method
        self.sequence_type = sequence_type

        # Initialize encoder based on method
        if method == "blast":
            self.encoder = None
            self.blast_searcher = BLASTSearcher(
                sequence_type=sequence_type,
                evalue=self.config.blast_evalue,
                threads=self.config.blast_threads,
            )
            self.use_blast = True
        else:
            model_name = SEQUENCE_MODELS.get(method, method)
            self.encoder = TransformerSequenceEncoder(
                model_name=model_name,
                max_length=self.config.max_length,
                use_gpu=self.config.use_gpu,
            )
            self.blast_searcher = None
            self.use_blast = False
            self.embedding_dim = self.encoder.embedding_dim

        # Index storage
        self.index = None
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def _build_faiss_index(self, embeddings: np.ndarray) -> Any:
        """Build FAISS index for embedding search."""
        faiss = _import_faiss()
        if faiss is None:
            return None

        n, d = embeddings.shape

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)

        index = faiss.IndexFlatIP(d)
        index.add(embeddings_norm.astype(np.float32))

        return index

    def index_sequences(
        self,
        sequences: List[Dict[str, str]],
        doc_id_field: str = "doc_id",
        sequence_field: str = "sequence",
        blast_db_path: Optional[str] = None,
    ) -> None:
        """Index sequences for similarity search.

        Args:
            sequences: List of dicts with doc_id and sequence
            doc_id_field: Field name for document ID
            sequence_field: Field name for sequence
            blast_db_path: Path for BLAST database (if using BLAST)
        """
        self.doc_ids = []
        seq_list = []

        for seq_data in sequences:
            doc_id = seq_data.get(doc_id_field)
            sequence = seq_data.get(sequence_field)

            if doc_id and sequence:
                self.doc_ids.append(doc_id)
                seq_list.append(sequence)

        logger.info(f"Indexing {len(seq_list)} sequences...")

        if self.use_blast:
            # Create BLAST database
            if blast_db_path:
                self.blast_searcher.create_database(
                    sequences, blast_db_path, doc_id_field, sequence_field
                )
        else:
            # Encode sequences
            self.embeddings = self.encoder.encode_batch(seq_list, batch_size=self.config.batch_size)

            # Build FAISS index
            self.index = self._build_faiss_index(self.embeddings)

        logger.info(f"Indexed {len(self.doc_ids)} sequences")

    def search(
        self,
        query_sequence: str,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Search for similar sequences.

        Args:
            query_sequence: Query sequence string
            top_k: Number of results to return

        Returns:
            List of (doc_id, similarity) tuples
        """
        if self.use_blast:
            return self.blast_searcher.search(query_sequence, top_k=top_k)

        # Encode query
        query_emb = self.encoder.encode(query_sequence)

        faiss = _import_faiss()

        if self.index is not None and faiss is not None:
            # FAISS search
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            similarities, indices = self.index.search(
                query_norm.reshape(1, -1).astype(np.float32), top_k
            )

            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0:
                    results.append((self.doc_ids[idx], float(sim)))
            return results

        else:
            # Brute force search
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            sims = np.dot(emb_norms, query_norm)

            indices = np.argsort(sims)[::-1][:top_k]
            return [(self.doc_ids[i], float(sims[i])) for i in indices]

    def search_batch(
        self,
        query_sequences: List[str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch search for multiple queries."""
        results = {}
        for seq in query_sequences:
            # Use hash as key for sequences
            seq_hash = hashlib.md5(seq.encode()).hexdigest()[:8]
            results[seq_hash] = self.search(seq, top_k=top_k)
        return results


def create_sequence_retriever(
    method: str = "esm2-small",
    sequence_type: str = "protein",
    use_gpu: bool = False,
) -> SequenceRetriever:
    """Factory function for sequence retriever.

    Args:
        method: "esm2", "protbert", "blast"
        sequence_type: "protein" or "nucleotide"
        use_gpu: Use GPU for transformer models

    Returns:
        Configured SequenceRetriever
    """
    config = SequenceConfig(
        method=method,
        sequence_type=sequence_type,
        use_gpu=use_gpu,
    )
    return SequenceRetriever(method=method, sequence_type=sequence_type, config=config)
