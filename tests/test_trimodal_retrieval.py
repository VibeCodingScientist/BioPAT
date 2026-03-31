"""Tests for trimodal retrieval combining text, chemical, and sequence search."""

import sys
import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# AsyncMock is only available in Python 3.8+
if sys.version_info >= (3, 8):
    from unittest.mock import AsyncMock
else:
    # Provide a simple fallback for Python < 3.8
    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

from biopat.evaluation.trimodal_retrieval import (
    MatchType,
    ModalityScore,
    TrimodalHit,
    TrimodalConfig,
    ScoreNormalizer,
    TrimodalRetriever,
    reciprocal_rank_fusion,
)


class TestMatchType:
    """Tests for MatchType enum."""

    def test_match_type_values(self):
        """Test match type enum values."""
        assert MatchType.TEXT.value == "text"
        assert MatchType.CHEMICAL.value == "chemical"
        assert MatchType.SEQUENCE.value == "sequence"
        assert MatchType.ALL.value == "all"


class TestModalityScore:
    """Tests for ModalityScore dataclass."""

    def test_modality_score_creation(self):
        """Test creating a modality score."""
        score = ModalityScore(
            modality="text",
            score=0.75,
            rank=1,
            raw_score=18.5,
        )
        assert score.modality == "text"
        assert score.score == 0.75
        assert score.rank == 1

    def test_is_significant_text(self):
        """Test significance threshold for text."""
        high_score = ModalityScore(modality="text", score=0.5)
        low_score = ModalityScore(modality="text", score=0.05)

        assert high_score.is_significant is True
        assert low_score.is_significant is False

    def test_is_significant_chemical(self):
        """Test significance threshold for chemical."""
        high_score = ModalityScore(modality="chemical", score=0.7)
        low_score = ModalityScore(modality="chemical", score=0.2)

        assert high_score.is_significant is True
        assert low_score.is_significant is False

    def test_is_significant_sequence(self):
        """Test significance threshold for sequence."""
        high_score = ModalityScore(modality="sequence", score=0.8)
        low_score = ModalityScore(modality="sequence", score=0.1)

        assert high_score.is_significant is True
        assert low_score.is_significant is False


class TestTrimodalHit:
    """Tests for TrimodalHit dataclass."""

    def test_trimodal_hit_creation(self):
        """Test creating a trimodal hit."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.85,
        )
        assert hit.doc_id == "DOC001"
        assert hit.combined_score == 0.85
        assert hit.match_type == MatchType.TEXT

    def test_match_type_detection_text_only(self):
        """Test match type for text-only hit."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.5,
            text_score=ModalityScore(modality="text", score=0.5),
        )
        assert hit.match_type == MatchType.TEXT

    def test_match_type_detection_chemical_only(self):
        """Test match type for chemical-only hit."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.7,
            chemical_score=ModalityScore(modality="chemical", score=0.7),
        )
        assert hit.match_type == MatchType.CHEMICAL

    def test_match_type_detection_text_chemical(self):
        """Test match type for text+chemical hit."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.8,
            text_score=ModalityScore(modality="text", score=0.6),
            chemical_score=ModalityScore(modality="chemical", score=0.8),
        )
        assert hit.match_type == MatchType.TEXT_CHEMICAL

    def test_match_type_detection_all(self):
        """Test match type for all modalities."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.9,
            text_score=ModalityScore(modality="text", score=0.5),
            chemical_score=ModalityScore(modality="chemical", score=0.7),
            sequence_score=ModalityScore(modality="sequence", score=0.8),
        )
        assert hit.match_type == MatchType.ALL

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="patent",
            combined_score=0.75,
            text_score=ModalityScore(modality="text", score=0.6),
            chemical_score=ModalityScore(modality="chemical", score=0.8),
        )
        d = hit.to_dict()
        assert d["doc_id"] == "DOC001"
        assert d["combined_score"] == 0.75
        assert d["text_score"] == 0.6
        assert d["chemical_score"] == 0.8


class TestTrimodalConfig:
    """Tests for TrimodalConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrimodalConfig()
        assert config.text_weight == 0.5
        assert config.chemical_weight == 0.25
        assert config.sequence_weight == 0.25
        assert config.normalize_scores is True

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        config = TrimodalConfig(
            text_weight=1.0,
            chemical_weight=1.0,
            sequence_weight=1.0,
        )
        total = config.text_weight + config.chemical_weight + config.sequence_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = TrimodalConfig(
            text_threshold=0.2,
            chemical_threshold=0.7,
            sequence_threshold=0.6,
        )
        assert config.text_threshold == 0.2
        assert config.chemical_threshold == 0.7
        assert config.sequence_threshold == 0.6


class TestScoreNormalizer:
    """Tests for ScoreNormalizer."""

    def test_normalizer_initialization(self):
        """Test normalizer initialization."""
        normalizer = ScoreNormalizer()
        assert len(normalizer._score_history["text"]) == 0

    def test_record_score(self):
        """Test recording scores."""
        normalizer = ScoreNormalizer()
        normalizer.record_score("text", 0.5)
        normalizer.record_score("text", 0.7)
        assert len(normalizer._score_history["text"]) == 2

    def test_default_normalize_text_bm25(self):
        """Test default normalization for BM25-like scores."""
        normalizer = ScoreNormalizer()
        # BM25 scores are typically > 1
        norm = normalizer._default_normalize("text", 20.0)
        assert 0.0 <= norm <= 1.0

    def test_default_normalize_chemical(self):
        """Test default normalization for chemical (Tanimoto)."""
        normalizer = ScoreNormalizer()
        # Tanimoto is already 0-1
        norm = normalizer._default_normalize("chemical", 0.75)
        assert norm == 0.75

    def test_default_normalize_sequence(self):
        """Test default normalization for sequence identity."""
        normalizer = ScoreNormalizer()
        # Identity as percentage
        norm = normalizer._default_normalize("sequence", 85.0)
        assert norm == 0.85


class TestReciprocalRankFusion:
    """Tests for RRF function."""

    def test_rrf_single_list(self):
        """Test RRF with single ranked list."""
        ranked_lists = [
            [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=60)
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"
        assert fused[2][0] == "doc3"

    def test_rrf_multiple_lists(self):
        """Test RRF with multiple ranked lists."""
        ranked_lists = [
            [("doc1", 0.9), ("doc2", 0.8)],
            [("doc2", 0.95), ("doc3", 0.7)],
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=60)

        # doc2 appears in both lists, should be ranked higher
        doc_ids = [doc_id for doc_id, _ in fused]
        assert "doc2" in doc_ids
        # doc2 should have highest combined RRF score
        assert fused[0][0] == "doc2"

    def test_rrf_disjoint_lists(self):
        """Test RRF with disjoint document sets."""
        ranked_lists = [
            [("doc1", 0.9)],
            [("doc2", 0.95)],
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=60)
        doc_ids = {doc_id for doc_id, _ in fused}
        assert doc_ids == {"doc1", "doc2"}


class TestTrimodalRetriever:
    """Tests for TrimodalRetriever."""

    def test_retriever_initialization(self):
        """Test retriever initialization."""
        retriever = TrimodalRetriever()
        assert retriever.config is not None
        assert retriever.text_retriever is None
        assert retriever.chemical_index is None
        assert retriever.sequence_index is None

    def test_available_modalities_none(self):
        """Test available modalities when none configured."""
        retriever = TrimodalRetriever()
        assert retriever.available_modalities == []

    def test_available_modalities_text_only(self):
        """Test available modalities with text only."""
        mock_text = Mock()
        retriever = TrimodalRetriever(text_retriever=mock_text)
        assert "text" in retriever.available_modalities

    def test_available_modalities_all(self):
        """Test available modalities with all configured."""
        retriever = TrimodalRetriever(
            text_retriever=Mock(),
            chemical_index=Mock(),
            sequence_index=Mock(),
        )
        assert set(retriever.available_modalities) == {"text", "chemical", "sequence"}

    def test_adjust_weights_text_only(self):
        """Test weight adjustment with text only."""
        config = TrimodalConfig(
            text_weight=0.5,
            chemical_weight=0.25,
            sequence_weight=0.25,
        )
        retriever = TrimodalRetriever(
            config=config,
            text_retriever=Mock(),
        )
        text_w, chem_w, seq_w = retriever._adjust_weights()
        assert text_w == 1.0
        assert chem_w == 0.0
        assert seq_w == 0.0

    def test_adjust_weights_text_chemical(self):
        """Test weight adjustment with text and chemical."""
        config = TrimodalConfig(
            text_weight=0.5,
            chemical_weight=0.25,
            sequence_weight=0.25,
        )
        retriever = TrimodalRetriever(
            config=config,
            text_retriever=Mock(),
            chemical_index=Mock(),
        )
        text_w, chem_w, seq_w = retriever._adjust_weights()
        # 0.5 + 0.25 = 0.75, normalized: 0.5/0.75, 0.25/0.75
        assert abs(text_w - 2/3) < 0.01
        assert abs(chem_w - 1/3) < 0.01
        assert seq_w == 0.0

    @pytest.mark.asyncio
    async def test_combine_weighted_empty(self):
        """Test weighted combination with empty results."""
        retriever = TrimodalRetriever()
        hits = retriever._combine_weighted({}, {}, {}, k=10)
        assert hits == []

    @pytest.mark.asyncio
    async def test_combine_weighted_text_only(self):
        """Test weighted combination with text results only."""
        config = TrimodalConfig(text_weight=1.0, chemical_weight=0.0, sequence_weight=0.0)
        retriever = TrimodalRetriever(config=config, text_retriever=Mock())

        text_results = {
            "doc1": ModalityScore(modality="text", score=0.8),
            "doc2": ModalityScore(modality="text", score=0.6),
        }

        hits = retriever._combine_weighted(text_results, {}, {}, k=10)
        assert len(hits) == 2
        assert hits[0].doc_id == "doc1"
        assert hits[0].combined_score == 0.8

    @pytest.mark.asyncio
    async def test_combine_weighted_multimodal(self):
        """Test weighted combination with multiple modalities."""
        config = TrimodalConfig(
            text_weight=0.5,
            chemical_weight=0.5,
            sequence_weight=0.0,
        )
        retriever = TrimodalRetriever(
            config=config,
            text_retriever=Mock(),
            chemical_index=Mock(),
        )

        text_results = {
            "doc1": ModalityScore(modality="text", score=0.6),
            "doc2": ModalityScore(modality="text", score=0.4),
        }
        chemical_results = {
            "doc1": ModalityScore(modality="chemical", score=0.8),
            "doc3": ModalityScore(modality="chemical", score=0.9),
        }

        hits = retriever._combine_weighted(text_results, chemical_results, {}, k=10)

        # doc1: 0.5*0.6 + 0.5*0.8 = 0.7
        # doc2: 0.5*0.4 = 0.2
        # doc3: 0.5*0.9 = 0.45
        doc_scores = {h.doc_id: h.combined_score for h in hits}
        assert abs(doc_scores["doc1"] - 0.7) < 0.01
        assert abs(doc_scores["doc2"] - 0.2) < 0.01
        assert abs(doc_scores["doc3"] - 0.45) < 0.01

    @pytest.mark.asyncio
    async def test_combine_with_rrf(self):
        """Test RRF combination."""
        config = TrimodalConfig(use_rank_fusion=True)
        retriever = TrimodalRetriever(
            config=config,
            text_retriever=Mock(),
            chemical_index=Mock(),
        )

        text_results = {
            "doc1": ModalityScore(modality="text", score=0.9, rank=1),
            "doc2": ModalityScore(modality="text", score=0.7, rank=2),
        }
        chemical_results = {
            "doc2": ModalityScore(modality="chemical", score=0.95, rank=1),
            "doc3": ModalityScore(modality="chemical", score=0.6, rank=2),
        }

        hits = retriever._combine_with_rrf(text_results, chemical_results, {}, k=10)

        # doc2 appears in both, should be ranked first
        assert hits[0].doc_id == "doc2"


class TestTrimodalHitMatchTypes:
    """Additional tests for match type determination."""

    def test_insignificant_scores_ignored(self):
        """Test that low scores don't count as significant."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.3,
            text_score=ModalityScore(modality="text", score=0.3),
            chemical_score=ModalityScore(modality="chemical", score=0.1),  # Below threshold
        )
        # Chemical score is below threshold, so only text counts
        assert hit.match_type == MatchType.TEXT

    def test_text_sequence_combination(self):
        """Test text + sequence match type."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.7,
            text_score=ModalityScore(modality="text", score=0.5),
            sequence_score=ModalityScore(modality="sequence", score=0.8),
        )
        assert hit.match_type == MatchType.TEXT_SEQUENCE

    def test_chemical_sequence_combination(self):
        """Test chemical + sequence match type."""
        hit = TrimodalHit(
            doc_id="DOC001",
            doc_type="paper",
            combined_score=0.85,
            chemical_score=ModalityScore(modality="chemical", score=0.8),
            sequence_score=ModalityScore(modality="sequence", score=0.9),
        )
        assert hit.match_type == MatchType.CHEMICAL_SEQUENCE
