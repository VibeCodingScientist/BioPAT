"""Comprehensive tests for trimodal retrieval (v4.0).

Tests end-to-end retrieval scenarios including:
- Text-only retrieval (BM25 fallback)
- Chemical structure similarity search
- Biological sequence similarity search
- Multimodal fusion with different weighting schemes
- Score normalization and combination
- Edge cases and error handling
"""

import asyncio
import sys
import numpy as np
import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch
import polars as pl

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
    TrimodalConfig,
    TrimodalHit,
    TrimodalRetriever,
    ScoreNormalizer,
    reciprocal_rank_fusion,
)


# Test fixtures
@pytest.fixture
def sample_corpus():
    """Sample corpus with documents containing text, chemical, and sequence info."""
    return pl.DataFrame({
        "doc_id": ["D001", "D002", "D003", "D004", "D005"],
        "title": [
            "Novel kinase inhibitor for cancer treatment",
            "CRISPR-based gene editing in human cells",
            "Monoclonal antibody targeting PD-1 receptor",
            "Small molecule JAK2 inhibitor with improved selectivity",
            "Protein engineering for enhanced stability",
        ],
        "abstract": [
            "We describe a novel kinase inhibitor compound with IC50 of 5nM against EGFR...",
            "This study presents a modified CRISPR-Cas9 system for efficient editing...",
            "A humanized monoclonal antibody was developed that binds PD-1 with high affinity...",
            "Structure-activity relationship studies led to identification of a selective JAK2 inhibitor...",
            "Directed evolution was used to improve the thermal stability of industrial enzymes...",
        ],
    })


@pytest.fixture
def trimodal_config():
    """Standard trimodal configuration."""
    return TrimodalConfig(
        text_weight=0.5,
        chemical_weight=0.3,
        sequence_weight=0.2,
        text_threshold=0.1,
        chemical_threshold=0.5,
        sequence_threshold=0.3,
    )


class TestTrimodalConfigValidation:
    """Test configuration validation and normalization."""

    def test_weights_sum_to_one(self):
        """Weights should be normalized to sum to 1."""
        config = TrimodalConfig(
            text_weight=0.6,
            chemical_weight=0.3,
            sequence_weight=0.3,  # Sum = 1.2
        )
        total = config.text_weight + config.chemical_weight + config.sequence_weight
        assert abs(total - 1.0) < 0.01  # Should normalize

    def test_zero_weights_handled(self):
        """Zero weights should be handled gracefully."""
        config = TrimodalConfig(
            text_weight=1.0,
            chemical_weight=0.0,
            sequence_weight=0.0,
        )
        assert config.text_weight == 1.0
        assert config.chemical_weight == 0.0

    def test_threshold_bounds(self):
        """Thresholds should be between 0 and 1."""
        config = TrimodalConfig(
            text_threshold=0.1,
            chemical_threshold=0.85,
            sequence_threshold=0.7,
        )
        assert 0 <= config.text_threshold <= 1
        assert 0 <= config.chemical_threshold <= 1
        assert 0 <= config.sequence_threshold <= 1


class TestScoreNormalizerComprehensive:
    """Comprehensive tests for score normalization."""

    def test_normalize_bm25_scores_with_history(self):
        """BM25 scores should be normalized based on history when available."""
        normalizer = ScoreNormalizer()

        # Record enough BM25 scores for percentile normalization
        bm25_scores = [5.2, 12.8, 3.1, 8.7, 15.0, 7.0, 9.0, 11.0, 6.0, 10.0, 4.0, 13.0]
        for score in bm25_scores:
            normalizer.record_score("text", score)

        # Normalize
        normalized = [normalizer.normalize("text", s) for s in bm25_scores]

        # Check bounds
        assert all(0 <= n <= 1 for n in normalized)

    def test_default_normalize_text(self):
        """Test default normalization for text (BM25-like) scores."""
        normalizer = ScoreNormalizer()

        # Without history, uses default normalization
        # BM25 scores > 1.0 are divided by 25
        result = normalizer.normalize("text", 25.0)
        assert result == 1.0  # 25/25 = 1.0

        result = normalizer.normalize("text", 12.5)
        assert result == 0.5  # 12.5/25 = 0.5

    def test_normalize_tanimoto_scores(self):
        """Tanimoto scores are already 0-1, should be passed through."""
        normalizer = ScoreNormalizer()

        # Without history, Tanimoto scores pass through
        tanimoto_scores = [0.95, 0.72, 0.55, 0.88, 0.31]
        normalized = [normalizer.normalize("chemical", s) for s in tanimoto_scores]

        # Chemical scores pass through as-is in default normalization
        for orig, norm in zip(tanimoto_scores, normalized):
            assert orig == norm

    def test_normalize_sequence_identity(self):
        """Sequence identity scores (0-1) should pass through."""
        normalizer = ScoreNormalizer()

        # Identity percentages already 0-1
        identity_scores = [0.95, 0.725, 0.50, 0.883]
        normalized = [normalizer.normalize("sequence", s) for s in identity_scores]

        for orig, norm in zip(identity_scores, normalized):
            assert orig == norm

    def test_empty_normalizer_defaults(self):
        """Normalizer should return sensible defaults with no recorded scores."""
        normalizer = ScoreNormalizer()

        # Text (BM25-like scores) - divided by 25 if > 1
        assert normalizer.normalize("text", 10.0) == 0.4  # 10/25

        # Chemical (Tanimoto) - pass through
        assert normalizer.normalize("chemical", 0.75) == 0.75

        # Sequence (identity) - pass through
        assert normalizer.normalize("sequence", 0.85) == 0.85


class TestTrimodalHitComprehensive:
    """Comprehensive tests for TrimodalHit score combination."""

    def test_text_only_hit(self):
        """Hit with only text match."""
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="paper",
            combined_score=0.75,
            text_score=ModalityScore(modality="text", score=0.75, rank=1),
            chemical_score=None,
            sequence_score=None,
        )

        assert hit.match_type == MatchType.TEXT
        assert hit.combined_score == 0.75

    def test_chemical_only_hit(self):
        """Hit with only chemical match."""
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="paper",
            combined_score=0.88,
            text_score=ModalityScore(modality="text", score=0.05, rank=50),  # Below threshold
            chemical_score=ModalityScore(modality="chemical", score=0.88, rank=1),
            sequence_score=None,
        )

        assert hit.match_type == MatchType.CHEMICAL

    def test_sequence_only_hit(self):
        """Hit with only sequence match."""
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="patent",
            combined_score=0.92,
            text_score=None,
            chemical_score=None,
            sequence_score=ModalityScore(modality="sequence", score=0.92, rank=1),
        )

        assert hit.match_type == MatchType.SEQUENCE

    def test_text_chemical_hit(self):
        """Hit with text and chemical matches."""
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="paper",
            combined_score=0.80,
            text_score=ModalityScore(modality="text", score=0.65, rank=3),
            chemical_score=ModalityScore(modality="chemical", score=0.78, rank=2),
            sequence_score=None,
        )

        assert hit.match_type == MatchType.TEXT_CHEMICAL

    def test_all_modalities_hit(self):
        """Hit with all three modality matches."""
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="patent",
            combined_score=0.85,
            text_score=ModalityScore(modality="text", score=0.70, rank=2),
            chemical_score=ModalityScore(modality="chemical", score=0.82, rank=1),
            sequence_score=ModalityScore(modality="sequence", score=0.65, rank=3),
        )

        assert hit.match_type == MatchType.ALL

    def test_to_dict_complete(self):
        """Test complete serialization to dictionary."""
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="paper",
            combined_score=0.85,
            text_score=ModalityScore(modality="text", score=0.70, rank=2),
            chemical_score=ModalityScore(modality="chemical", score=0.82, rank=1),
            sequence_score=None,
        )

        d = hit.to_dict()

        assert d["doc_id"] == "D001"
        assert d["combined_score"] == 0.85
        assert d["text_score"] == 0.70
        assert d["chemical_score"] == 0.82
        assert "sequence_score" not in d  # None scores not included
        assert "match_type" in d


class TestReciprocalRankFusionComprehensive:
    """Comprehensive tests for RRF score combination."""

    def test_single_ranking(self):
        """RRF with single ranking should preserve order."""
        rankings = [
            [("D001", 10.0), ("D002", 8.0), ("D003", 6.0)],
        ]

        fused = reciprocal_rank_fusion(rankings, k=60)

        # Returns list of tuples
        doc_ids = [d for d, s in fused]
        assert doc_ids == ["D001", "D002", "D003"]

    def test_two_identical_rankings(self):
        """RRF with identical rankings should boost scores."""
        rankings = [
            [("D001", 10.0), ("D002", 8.0), ("D003", 6.0)],
            [("D001", 10.0), ("D002", 8.0), ("D003", 6.0)],
        ]

        fused = reciprocal_rank_fusion(rankings, k=60)
        fused_dict = dict(fused)

        # D001 should have highest score
        assert fused_dict["D001"] > fused_dict["D002"] > fused_dict["D003"]

    def test_conflicting_rankings(self):
        """RRF with conflicting rankings should average."""
        rankings = [
            [("D001", 10.0), ("D002", 8.0), ("D003", 6.0)],  # D001 first
            [("D003", 10.0), ("D002", 8.0), ("D001", 6.0)],  # D003 first
        ]

        fused = reciprocal_rank_fusion(rankings, k=60)
        fused_dict = dict(fused)

        # D002 is rank 2 in both, should have consistent score
        # D001 and D003 should be close since they swap positions
        assert abs(fused_dict["D001"] - fused_dict["D003"]) < fused_dict["D002"] * 0.5

    def test_disjoint_rankings(self):
        """RRF with non-overlapping documents."""
        rankings = [
            [("D001", 10.0), ("D002", 8.0)],
            [("D003", 10.0), ("D004", 8.0)],
        ]

        fused = reciprocal_rank_fusion(rankings, k=60)

        assert len(fused) == 4
        doc_ids = [d for d, s in fused]
        assert "D001" in doc_ids
        assert "D003" in doc_ids

    def test_empty_ranking(self):
        """RRF should handle empty rankings gracefully."""
        rankings = [
            [("D001", 10.0)],
            [],  # Empty ranking
        ]

        fused = reciprocal_rank_fusion(rankings, k=60)
        doc_ids = [d for d, s in fused]

        assert "D001" in doc_ids

    def test_k_parameter_effect(self):
        """Different k values should affect score distribution."""
        rankings = [
            [("D001", 10.0), ("D002", 8.0), ("D003", 6.0)],
        ]

        fused_k60 = dict(reciprocal_rank_fusion(rankings, k=60))
        fused_k1 = dict(reciprocal_rank_fusion(rankings, k=1))

        # With k=1, rank differences matter more
        ratio_k60 = fused_k60["D001"] / fused_k60["D003"]
        ratio_k1 = fused_k1["D001"] / fused_k1["D003"]

        assert ratio_k1 > ratio_k60  # k=1 exaggerates differences


class TestTrimodalRetrieverComprehensive:
    """Comprehensive tests for the full TrimodalRetriever."""

    def test_text_only_mode(self, trimodal_config):
        """Test retriever with only text search available."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=None,
            sequence_index=None,
        )

        assert retriever.available_modalities == ["text"]

    def test_all_modalities_mode(self, trimodal_config):
        """Test retriever with all modalities available."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=MagicMock(),
            sequence_index=MagicMock(),
        )

        assert set(retriever.available_modalities) == {"text", "chemical", "sequence"}

    def test_weight_adjustment_text_only(self, trimodal_config):
        """When only text available, all weight should go to text."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=None,
            sequence_index=None,
        )

        text_w, chem_w, seq_w = retriever._adjust_weights()

        assert text_w == 1.0
        assert chem_w == 0.0
        assert seq_w == 0.0

    def test_weight_adjustment_text_chemical(self, trimodal_config):
        """When text and chemical available, weights should be renormalized."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=MagicMock(),
            sequence_index=None,
        )

        text_w, chem_w, seq_w = retriever._adjust_weights()

        # Should renormalize to sum to 1
        assert abs(text_w + chem_w + seq_w - 1.0) < 0.01
        # Ratio should be preserved: 0.5/0.3 = 1.67
        expected_ratio = 0.5 / 0.3
        actual_ratio = text_w / chem_w
        assert abs(expected_ratio - actual_ratio) < 0.01

    def test_combine_weighted_internal(self, trimodal_config):
        """Test internal weighted combination method."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=MagicMock(),
            sequence_index=MagicMock(),
        )

        # Create mock ModalityScore objects
        text_results = {
            "D001": ModalityScore(modality="text", score=0.8, rank=1),
            "D002": ModalityScore(modality="text", score=0.6, rank=2),
        }
        chemical_results = {
            "D001": ModalityScore(modality="chemical", score=0.9, rank=1),
        }
        sequence_results = {
            "D001": ModalityScore(modality="sequence", score=0.7, rank=1),
        }

        hits = retriever._combine_weighted(
            text_results=text_results,
            chemical_results=chemical_results,
            sequence_results=sequence_results,
            k=10,
        )

        # D001 should be highest (present in all modalities)
        assert len(hits) > 0
        assert hits[0].doc_id == "D001"

    def test_empty_results_handling(self, trimodal_config):
        """Test handling when no results are found."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=None,
            sequence_index=None,
        )

        hits = retriever._combine_weighted(
            text_results={},
            chemical_results={},
            sequence_results={},
            k=10,
        )

        assert len(hits) == 0


class TestTrimodalEvaluationMetrics:
    """Test evaluation metrics for trimodal retrieval."""

    def test_modality_contribution_tracking(self):
        """Track which modalities contributed to each hit."""
        hits = [
            TrimodalHit(
                doc_id="D001",
                doc_type="paper",
                combined_score=0.9,
                text_score=ModalityScore(modality="text", score=0.8, rank=1),
                chemical_score=ModalityScore(modality="chemical", score=0.95, rank=1),
                sequence_score=None,
            ),
            TrimodalHit(
                doc_id="D002",
                doc_type="paper",
                combined_score=0.7,
                text_score=ModalityScore(modality="text", score=0.7, rank=2),
                chemical_score=None,
                sequence_score=None,
            ),
            TrimodalHit(
                doc_id="D003",
                doc_type="patent",
                combined_score=0.65,
                text_score=None,
                chemical_score=None,
                sequence_score=ModalityScore(modality="sequence", score=0.65, rank=1),
            ),
        ]

        # Count match types
        type_counts = {}
        for hit in hits:
            mt = hit.match_type
            type_counts[mt] = type_counts.get(mt, 0) + 1

        assert type_counts[MatchType.TEXT_CHEMICAL] == 1
        assert type_counts[MatchType.TEXT] == 1
        assert type_counts[MatchType.SEQUENCE] == 1

    def test_precision_by_modality(self):
        """Calculate precision per modality type."""
        # Ground truth: D001, D002, D003 are relevant
        ground_truth = {"D001", "D002", "D003"}

        # Retrieved hits
        hits = [
            TrimodalHit(
                doc_id="D001",
                doc_type="paper",
                combined_score=0.9,
                text_score=ModalityScore(modality="text", score=0.8, rank=1),
                chemical_score=ModalityScore(modality="chemical", score=0.95, rank=1),
                sequence_score=None,
            ),
            TrimodalHit(
                doc_id="D002",
                doc_type="paper",
                combined_score=0.7,
                text_score=ModalityScore(modality="text", score=0.7, rank=2),
                chemical_score=None,
                sequence_score=None,
            ),
            TrimodalHit(
                doc_id="D004",
                doc_type="paper",
                combined_score=0.65,
                text_score=None,
                chemical_score=ModalityScore(modality="chemical", score=0.6, rank=2),
                sequence_score=None,
            ),
            TrimodalHit(
                doc_id="D003",
                doc_type="patent",
                combined_score=0.6,
                text_score=None,
                chemical_score=None,
                sequence_score=ModalityScore(modality="sequence", score=0.6, rank=1),
            ),
        ]

        # Overall P@4
        relevant_in_top_4 = sum(1 for h in hits if h.doc_id in ground_truth)
        precision_at_4 = relevant_in_top_4 / len(hits)

        assert precision_at_4 == 0.75  # 3 out of 4

        # Precision by modality
        text_hits = [h for h in hits if h.text_score and h.text_score.score > 0.1]
        text_relevant = sum(1 for h in text_hits if h.doc_id in ground_truth)
        text_precision = text_relevant / len(text_hits) if text_hits else 0

        assert text_precision == 1.0  # Both text hits are relevant


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_all_modalities(self):
        """Handle case where all modalities are unavailable."""
        config = TrimodalConfig()
        retriever = TrimodalRetriever(
            config=config,
            text_retriever=None,
            chemical_index=None,
            sequence_index=None,
        )

        assert retriever.available_modalities == []

    def test_modality_score_significance(self):
        """Test ModalityScore significance thresholds."""
        # Text threshold is 0.1
        text_high = ModalityScore(modality="text", score=0.5, rank=1)
        text_low = ModalityScore(modality="text", score=0.05, rank=10)

        assert text_high.is_significant == True
        assert text_low.is_significant == False

        # Chemical threshold is 0.3
        chem_high = ModalityScore(modality="chemical", score=0.5, rank=1)
        chem_low = ModalityScore(modality="chemical", score=0.2, rank=10)

        assert chem_high.is_significant == True
        assert chem_low.is_significant == False

    def test_very_large_bm25_scores(self):
        """Handle large BM25 scores in normalization."""
        normalizer = ScoreNormalizer()

        # Default normalize caps at 1.0
        result = normalizer.normalize("text", 100.0)
        assert result <= 1.0

    def test_trimodal_hit_post_init(self):
        """Test TrimodalHit __post_init__ determines match type correctly."""
        # Text + sequence but no chemical
        hit = TrimodalHit(
            doc_id="D001",
            doc_type="paper",
            combined_score=0.8,
            text_score=ModalityScore(modality="text", score=0.7, rank=1),
            chemical_score=None,
            sequence_score=ModalityScore(modality="sequence", score=0.6, rank=2),
        )

        assert hit.match_type == MatchType.TEXT_SEQUENCE

        # Chemical + sequence but text below threshold
        hit2 = TrimodalHit(
            doc_id="D002",
            doc_type="patent",
            combined_score=0.75,
            text_score=ModalityScore(modality="text", score=0.05, rank=50),  # Below threshold
            chemical_score=ModalityScore(modality="chemical", score=0.8, rank=1),
            sequence_score=ModalityScore(modality="sequence", score=0.7, rank=2),
        )

        assert hit2.match_type == MatchType.CHEMICAL_SEQUENCE


class TestTrimodalRetrieverIntegration:
    """Integration tests simulating full retrieval pipeline."""

    def test_retriever_initialization(self, trimodal_config):
        """Test retriever can be initialized with various configurations."""
        # Text only
        r1 = TrimodalRetriever(config=trimodal_config, text_retriever=MagicMock())
        assert "text" in r1.available_modalities

        # All modalities
        r2 = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=MagicMock(),
            sequence_index=MagicMock(),
        )
        assert len(r2.available_modalities) == 3

    def test_config_normalization(self):
        """Test that configs with unnormalized weights get normalized."""
        config = TrimodalConfig(
            text_weight=2.0,
            chemical_weight=1.0,
            sequence_weight=1.0,
        )

        total = config.text_weight + config.chemical_weight + config.sequence_weight
        assert abs(total - 1.0) < 0.01

    def test_combine_with_rrf_method(self, trimodal_config):
        """Test RRF combination method."""
        retriever = TrimodalRetriever(
            config=trimodal_config,
            text_retriever=MagicMock(),
            chemical_index=MagicMock(),
            sequence_index=None,
        )

        text_results = {
            "D001": ModalityScore(modality="text", score=0.9, rank=1),
            "D002": ModalityScore(modality="text", score=0.7, rank=2),
        }
        chemical_results = {
            "D001": ModalityScore(modality="chemical", score=0.85, rank=1),
            "D003": ModalityScore(modality="chemical", score=0.6, rank=2),
        }

        hits = retriever._combine_with_rrf(
            text_results=text_results,
            chemical_results=chemical_results,
            sequence_results={},
            k=10,
        )

        # Should have results from both modalities
        doc_ids = {h.doc_id for h in hits}
        assert "D001" in doc_ids
        assert "D002" in doc_ids
        assert "D003" in doc_ids
