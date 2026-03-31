"""Weighted Trimodal Retrieval for Multi-Modal Prior Art Discovery.

Phase 4.0 (Advanced): Combines text, chemical structure, and biological
sequence similarity into a unified prior art discovery engine.

This module implements:
- Weighted combination of three modality scores
- Configurable thresholds for each modality
- Score normalization and fusion strategies
- Support for partial modality availability (graceful degradation)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Type of evidence supporting a prior art match."""

    TEXT = "text"
    CHEMICAL = "chemical"
    SEQUENCE = "sequence"
    TEXT_CHEMICAL = "text_chemical"
    TEXT_SEQUENCE = "text_sequence"
    CHEMICAL_SEQUENCE = "chemical_sequence"
    ALL = "all"


@dataclass
class ModalityScore:
    """Score from a single retrieval modality."""

    modality: str  # "text", "chemical", or "sequence"
    score: float  # Raw score (0-1 normalized)
    rank: Optional[int] = None  # Rank in this modality
    raw_score: Optional[float] = None  # Original unnormalized score
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if this score is above a minimal threshold."""
        thresholds = {"text": 0.1, "chemical": 0.3, "sequence": 0.3}
        return self.score >= thresholds.get(self.modality, 0.1)


@dataclass
class TrimodalHit:
    """A hit from trimodal retrieval with evidence from multiple modalities."""

    doc_id: str  # Document identifier
    doc_type: str  # "paper" or "patent"
    combined_score: float  # Weighted combined score
    text_score: Optional[ModalityScore] = None
    chemical_score: Optional[ModalityScore] = None
    sequence_score: Optional[ModalityScore] = None
    match_type: MatchType = MatchType.TEXT

    def __post_init__(self):
        """Determine match type based on available evidence."""
        has_text = self.text_score is not None and self.text_score.is_significant
        has_chemical = (
            self.chemical_score is not None and self.chemical_score.is_significant
        )
        has_sequence = (
            self.sequence_score is not None and self.sequence_score.is_significant
        )

        if has_text and has_chemical and has_sequence:
            self.match_type = MatchType.ALL
        elif has_text and has_chemical:
            self.match_type = MatchType.TEXT_CHEMICAL
        elif has_text and has_sequence:
            self.match_type = MatchType.TEXT_SEQUENCE
        elif has_chemical and has_sequence:
            self.match_type = MatchType.CHEMICAL_SEQUENCE
        elif has_chemical:
            self.match_type = MatchType.CHEMICAL
        elif has_sequence:
            self.match_type = MatchType.SEQUENCE
        else:
            self.match_type = MatchType.TEXT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "combined_score": self.combined_score,
            "match_type": self.match_type.value,
        }

        if self.text_score:
            result["text_score"] = self.text_score.score
        if self.chemical_score:
            result["chemical_score"] = self.chemical_score.score
        if self.sequence_score:
            result["sequence_score"] = self.sequence_score.score

        return result


@dataclass
class TrimodalConfig:
    """Configuration for trimodal retrieval."""

    # Modality weights (must sum to 1.0)
    text_weight: float = 0.5
    chemical_weight: float = 0.25
    sequence_weight: float = 0.25

    # Score thresholds for each modality
    text_threshold: float = 0.0  # Minimum text similarity
    chemical_threshold: float = 0.5  # Tanimoto >= 0.5
    sequence_threshold: float = 0.5  # Identity >= 50%

    # Fusion parameters
    normalize_scores: bool = True
    use_rank_fusion: bool = False
    rank_fusion_k: int = 60  # RRF parameter

    # Result limits
    max_results: int = 100
    min_combined_score: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        total = self.text_weight + self.chemical_weight + self.sequence_weight
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.text_weight /= total
            self.chemical_weight /= total
            self.sequence_weight /= total
            logger.warning(
                f"Normalized modality weights to sum to 1.0: "
                f"text={self.text_weight:.3f}, chemical={self.chemical_weight:.3f}, "
                f"sequence={self.sequence_weight:.3f}"
            )


class ScoreNormalizer:
    """Normalizes scores from different modalities to comparable ranges."""

    def __init__(self):
        """Initialize normalizer."""
        # Track score distributions per modality
        self._score_history: Dict[str, List[float]] = {
            "text": [],
            "chemical": [],
            "sequence": [],
        }
        self._max_history = 1000

    def record_score(self, modality: str, score: float) -> None:
        """Record a score for distribution tracking."""
        if modality in self._score_history:
            history = self._score_history[modality]
            history.append(score)
            if len(history) > self._max_history:
                self._score_history[modality] = history[-self._max_history :]

    def normalize(self, modality: str, score: float) -> float:
        """Normalize score to 0-1 range based on observed distribution."""
        history = self._score_history.get(modality, [])

        if len(history) < 10:
            # Not enough data, use default normalization
            return self._default_normalize(modality, score)

        # Use percentile-based normalization
        arr = np.array(history)
        min_score = np.percentile(arr, 5)
        max_score = np.percentile(arr, 95)

        if max_score <= min_score:
            return 0.5

        normalized = (score - min_score) / (max_score - min_score)
        return float(np.clip(normalized, 0.0, 1.0))

    def _default_normalize(self, modality: str, score: float) -> float:
        """Default normalization for each modality."""
        if modality == "text":
            # BM25 scores typically range 0-30+, dense 0-1
            if score > 1.0:  # Likely BM25
                return min(1.0, score / 25.0)
            return score
        elif modality == "chemical":
            # Tanimoto is already 0-1
            return score
        elif modality == "sequence":
            # Identity percentage to 0-1
            if score > 1.0:
                return min(1.0, score / 100.0)
            return score
        return score


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Combine ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked results [(doc_id, score), ...] for each modality.
        k: RRF parameter (higher = more weight to lower ranks).

    Returns:
        Fused ranked list.
    """
    rrf_scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class TrimodalRetriever:
    """Combines text, chemical, and sequence retrieval into unified ranking.

    Supports graceful degradation when modalities are unavailable.
    """

    def __init__(
        self,
        config: Optional[TrimodalConfig] = None,
        text_retriever: Optional[Any] = None,
        chemical_index: Optional[Any] = None,
        sequence_index: Optional[Any] = None,
    ):
        """Initialize trimodal retriever.

        Args:
            config: Retrieval configuration.
            text_retriever: Text retrieval component (BM25, dense, or hybrid).
            chemical_index: Chemical similarity index (ChemicalIndex).
            sequence_index: Sequence similarity index (SequenceIndex).
        """
        self.config = config or TrimodalConfig()
        self.text_retriever = text_retriever
        self.chemical_index = chemical_index
        self.sequence_index = sequence_index
        self.normalizer = ScoreNormalizer()

    @property
    def available_modalities(self) -> List[str]:
        """List of available modalities."""
        modalities = []
        if self.text_retriever is not None:
            modalities.append("text")
        if self.chemical_index is not None:
            modalities.append("chemical")
        if self.sequence_index is not None:
            modalities.append("sequence")
        return modalities

    def _adjust_weights(self) -> Tuple[float, float, float]:
        """Adjust weights based on available modalities."""
        text_w = self.config.text_weight if self.text_retriever else 0.0
        chem_w = self.config.chemical_weight if self.chemical_index else 0.0
        seq_w = self.config.sequence_weight if self.sequence_index else 0.0

        total = text_w + chem_w + seq_w
        if total == 0:
            return 1.0, 0.0, 0.0  # Fallback to text-only

        return text_w / total, chem_w / total, seq_w / total

    async def retrieve(
        self,
        query_text: str,
        query_smiles: Optional[List[str]] = None,
        query_sequences: Optional[List[Tuple[str, str]]] = None,  # [(seq, type), ...]
        k: int = 100,
    ) -> List[TrimodalHit]:
        """Perform trimodal retrieval.

        Args:
            query_text: Text query (patent claims, abstract, etc.).
            query_smiles: Optional list of SMILES strings from the query.
            query_sequences: Optional list of (sequence, type) tuples.
            k: Number of results to return.

        Returns:
            Ranked list of TrimodalHit objects.
        """
        # Collect results from each modality
        text_results: Dict[str, ModalityScore] = {}
        chemical_results: Dict[str, ModalityScore] = {}
        sequence_results: Dict[str, ModalityScore] = {}

        # Text retrieval
        if self.text_retriever and query_text:
            text_results = await self._text_search(query_text, k * 2)

        # Chemical retrieval
        if self.chemical_index and query_smiles:
            chemical_results = await self._chemical_search(query_smiles, k * 2)

        # Sequence retrieval
        if self.sequence_index and query_sequences:
            sequence_results = await self._sequence_search(query_sequences, k * 2)

        # Combine results
        if self.config.use_rank_fusion:
            return self._combine_with_rrf(
                text_results, chemical_results, sequence_results, k
            )
        else:
            return self._combine_weighted(
                text_results, chemical_results, sequence_results, k
            )

    async def _text_search(
        self, query: str, k: int
    ) -> Dict[str, ModalityScore]:
        """Perform text-based retrieval."""
        results = {}

        try:
            # Handle different retriever types
            if hasattr(self.text_retriever, "search"):
                # Generic search interface
                hits = self.text_retriever.search(query, k=k)
            elif hasattr(self.text_retriever, "retrieve"):
                # Dense retriever interface
                hits = await self.text_retriever.retrieve(query, k=k)
            else:
                logger.warning("Unknown text retriever interface")
                return results

            for rank, hit in enumerate(hits, start=1):
                doc_id = hit.get("doc_id") or hit.get("id")
                score = hit.get("score", 0.0)

                # Normalize and record
                norm_score = self.normalizer.normalize("text", score)
                self.normalizer.record_score("text", score)

                if norm_score >= self.config.text_threshold:
                    results[doc_id] = ModalityScore(
                        modality="text",
                        score=norm_score,
                        rank=rank,
                        raw_score=score,
                        details={"query_match": query[:100]},
                    )

        except Exception as e:
            logger.error(f"Text search failed: {e}")

        return results

    async def _chemical_search(
        self, smiles_list: List[str], k: int
    ) -> Dict[str, ModalityScore]:
        """Perform chemical similarity search."""
        results: Dict[str, ModalityScore] = {}

        try:
            for smiles in smiles_list:
                hits = self.chemical_index.search_prior_art(
                    smiles,
                    k=k,
                    min_similarity=self.config.chemical_threshold,
                )

                for rank, hit in enumerate(hits, start=1):
                    doc_id = hit.source_id
                    similarity = hit.similarity

                    # Record for normalization
                    self.normalizer.record_score("chemical", similarity)

                    # Keep best score for each document
                    if doc_id not in results or similarity > results[doc_id].score:
                        results[doc_id] = ModalityScore(
                            modality="chemical",
                            score=similarity,
                            rank=rank,
                            raw_score=similarity,
                            details={
                                "query_smiles": smiles[:50],
                                "hit_smiles": hit.smiles[:50],
                                "inchi_key": hit.inchi_key,
                            },
                        )

        except Exception as e:
            logger.error(f"Chemical search failed: {e}")

        return results

    async def _sequence_search(
        self, sequences: List[Tuple[str, str]], k: int
    ) -> Dict[str, ModalityScore]:
        """Perform sequence similarity search."""
        results: Dict[str, ModalityScore] = {}

        try:
            for seq, seq_type in sequences:
                hits = await self.sequence_index.search_prior_art(
                    query_sequence=seq,
                    sequence_type=seq_type,
                    min_identity=self.config.sequence_threshold * 100,
                    max_hits=k,
                )

                for rank, hit in enumerate(hits, start=1):
                    doc_id = hit["source_id"]
                    identity = hit["identity"] / 100.0  # Normalize to 0-1

                    # Record for normalization
                    self.normalizer.record_score("sequence", identity)

                    # Keep best score for each document
                    if doc_id not in results or identity > results[doc_id].score:
                        results[doc_id] = ModalityScore(
                            modality="sequence",
                            score=identity,
                            rank=rank,
                            raw_score=hit["identity"],
                            details={
                                "alignment_length": hit.get("alignment_length"),
                                "evalue": hit.get("evalue"),
                                "bit_score": hit.get("bit_score"),
                            },
                        )

        except Exception as e:
            logger.error(f"Sequence search failed: {e}")

        return results

    def _combine_weighted(
        self,
        text_results: Dict[str, ModalityScore],
        chemical_results: Dict[str, ModalityScore],
        sequence_results: Dict[str, ModalityScore],
        k: int,
    ) -> List[TrimodalHit]:
        """Combine results using weighted scoring."""
        # Get adjusted weights
        text_w, chem_w, seq_w = self._adjust_weights()

        # Collect all document IDs
        all_docs = set(text_results.keys()) | set(chemical_results.keys()) | set(
            sequence_results.keys()
        )

        hits = []
        for doc_id in all_docs:
            text_score = text_results.get(doc_id)
            chem_score = chemical_results.get(doc_id)
            seq_score = sequence_results.get(doc_id)

            # Compute weighted sum
            combined = 0.0
            if text_score:
                combined += text_w * text_score.score
            if chem_score:
                combined += chem_w * chem_score.score
            if seq_score:
                combined += seq_w * seq_score.score

            if combined >= self.config.min_combined_score:
                # Determine doc type from available info
                doc_type = "paper"  # Default
                if chem_score and chem_score.details.get("source_type") == "patent":
                    doc_type = "patent"

                hit = TrimodalHit(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    combined_score=combined,
                    text_score=text_score,
                    chemical_score=chem_score,
                    sequence_score=seq_score,
                )
                hits.append(hit)

        # Sort by combined score and limit
        hits.sort(key=lambda x: x.combined_score, reverse=True)
        return hits[:k]

    def _combine_with_rrf(
        self,
        text_results: Dict[str, ModalityScore],
        chemical_results: Dict[str, ModalityScore],
        sequence_results: Dict[str, ModalityScore],
        k: int,
    ) -> List[TrimodalHit]:
        """Combine results using Reciprocal Rank Fusion."""
        # Convert to ranked lists
        ranked_lists = []

        if text_results:
            text_ranked = sorted(
                [(doc_id, score.score) for doc_id, score in text_results.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            ranked_lists.append(text_ranked)

        if chemical_results:
            chem_ranked = sorted(
                [(doc_id, score.score) for doc_id, score in chemical_results.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            ranked_lists.append(chem_ranked)

        if sequence_results:
            seq_ranked = sorted(
                [(doc_id, score.score) for doc_id, score in sequence_results.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            ranked_lists.append(seq_ranked)

        if not ranked_lists:
            return []

        # Apply RRF
        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rank_fusion_k)

        # Convert to TrimodalHit objects
        hits = []
        for doc_id, rrf_score in fused[:k]:
            text_score = text_results.get(doc_id)
            chem_score = chemical_results.get(doc_id)
            seq_score = sequence_results.get(doc_id)

            doc_type = "paper"
            if chem_score and chem_score.details.get("source_type") == "patent":
                doc_type = "patent"

            hit = TrimodalHit(
                doc_id=doc_id,
                doc_type=doc_type,
                combined_score=rrf_score,
                text_score=text_score,
                chemical_score=chem_score,
                sequence_score=seq_score,
            )
            hits.append(hit)

        return hits


@dataclass
class TrimodalEvaluationResult:
    """Results from trimodal retrieval evaluation."""

    # Overall metrics
    ndcg_at_10: float = 0.0
    map_score: float = 0.0
    recall_at_100: float = 0.0

    # Per-modality contribution
    text_only_ndcg: float = 0.0
    chemical_boost: float = 0.0  # NDCG improvement from chemical
    sequence_boost: float = 0.0  # NDCG improvement from sequence

    # Match type distribution
    match_type_counts: Dict[str, int] = field(default_factory=dict)

    # Coverage
    queries_with_chemical: int = 0
    queries_with_sequence: int = 0
    unique_docs_found: int = 0


class TrimodalEvaluator:
    """Evaluates trimodal retrieval performance."""

    def __init__(
        self,
        retriever: TrimodalRetriever,
        qrels: Dict[str, Dict[str, int]],  # query_id -> {doc_id: relevance}
    ):
        """Initialize evaluator.

        Args:
            retriever: Trimodal retriever to evaluate.
            qrels: Ground truth relevance judgments.
        """
        self.retriever = retriever
        self.qrels = qrels

    async def evaluate(
        self,
        queries: List[Dict[str, Any]],
    ) -> TrimodalEvaluationResult:
        """Evaluate retrieval on a set of queries.

        Args:
            queries: List of query dicts with keys:
                - query_id: Unique identifier
                - text: Query text
                - smiles: Optional list of SMILES
                - sequences: Optional list of (seq, type) tuples

        Returns:
            TrimodalEvaluationResult with metrics.
        """
        result = TrimodalEvaluationResult()
        match_counts: Dict[str, int] = {}
        all_docs = set()

        ndcg_scores = []
        recall_scores = []

        for query in queries:
            query_id = query["query_id"]
            if query_id not in self.qrels:
                continue

            relevant = self.qrels[query_id]

            # Retrieve
            hits = await self.retriever.retrieve(
                query_text=query.get("text", ""),
                query_smiles=query.get("smiles"),
                query_sequences=query.get("sequences"),
                k=100,
            )

            # Track modality usage
            if query.get("smiles"):
                result.queries_with_chemical += 1
            if query.get("sequences"):
                result.queries_with_sequence += 1

            # Count match types
            for hit in hits:
                match_type = hit.match_type.value
                match_counts[match_type] = match_counts.get(match_type, 0) + 1
                all_docs.add(hit.doc_id)

            # Compute NDCG@10
            ndcg = self._compute_ndcg(hits[:10], relevant)
            ndcg_scores.append(ndcg)

            # Compute Recall@100
            retrieved_relevant = sum(
                1 for hit in hits if hit.doc_id in relevant and relevant[hit.doc_id] > 0
            )
            total_relevant = sum(1 for r in relevant.values() if r > 0)
            if total_relevant > 0:
                recall_scores.append(retrieved_relevant / total_relevant)

        # Aggregate metrics
        if ndcg_scores:
            result.ndcg_at_10 = sum(ndcg_scores) / len(ndcg_scores)
        if recall_scores:
            result.recall_at_100 = sum(recall_scores) / len(recall_scores)

        result.match_type_counts = match_counts
        result.unique_docs_found = len(all_docs)

        return result

    def _compute_ndcg(
        self, hits: List[TrimodalHit], relevant: Dict[str, int], k: int = 10
    ) -> float:
        """Compute NDCG@k."""
        dcg = 0.0
        for i, hit in enumerate(hits[:k]):
            rel = relevant.get(hit.doc_id, 0)
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        if idcg == 0:
            return 0.0
        return dcg / idcg


def create_trimodal_retriever(
    text_retriever: Any = None,
    chemical_index_dir: Optional[str] = None,
    sequence_index_dir: Optional[str] = None,
    config: Optional[TrimodalConfig] = None,
) -> TrimodalRetriever:
    """Factory function to create a trimodal retriever.

    Args:
        text_retriever: Pre-configured text retriever.
        chemical_index_dir: Path to chemical index directory.
        sequence_index_dir: Path to sequence index directory.
        config: Retrieval configuration.

    Returns:
        Configured TrimodalRetriever.
    """
    from pathlib import Path

    chemical_index = None
    sequence_index = None

    # Load chemical index if path provided
    if chemical_index_dir:
        try:
            from ..processing.chemical_index import ChemicalIndex

            chemical_index = ChemicalIndex(index_dir=Path(chemical_index_dir))
            logger.info(f"Loaded chemical index from {chemical_index_dir}")
        except Exception as e:
            logger.warning(f"Could not load chemical index: {e}")

    # Load sequence index if path provided
    if sequence_index_dir:
        try:
            from ..processing.sequence_index import SequenceIndex

            sequence_index = SequenceIndex(index_dir=Path(sequence_index_dir))
            logger.info(f"Loaded sequence index from {sequence_index_dir}")
        except Exception as e:
            logger.warning(f"Could not load sequence index: {e}")

    return TrimodalRetriever(
        config=config or TrimodalConfig(),
        text_retriever=text_retriever,
        chemical_index=chemical_index,
        sequence_index=sequence_index,
    )
