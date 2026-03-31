"""Tests for statistical significance testing module."""

import pytest
import numpy as np

from biopat.evaluation.statistical_tests import (
    paired_t_test,
    bootstrap_confidence_interval,
    bootstrap_paired_test,
    bonferroni_correction,
    significance_matrix,
)


class TestPairedTTest:
    def test_identical_scores(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        t_stat, p_val = paired_t_test(scores, scores)
        assert t_stat == 0.0 or np.isnan(t_stat)

    def test_significantly_different(self):
        a = [0.1, 0.2, 0.15, 0.18, 0.12, 0.19, 0.14, 0.16, 0.11, 0.17]
        b = [0.5, 0.6, 0.55, 0.58, 0.52, 0.59, 0.54, 0.56, 0.51, 0.57]
        t_stat, p_val = paired_t_test(a, b)
        assert p_val < 0.01

    def test_not_significantly_different(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0.5, 0.1, 100).tolist()
        b = (np.array(a) + rng.normal(0, 0.01, 100)).tolist()
        t_stat, p_val = paired_t_test(a, b)
        assert p_val > 0.01


class TestBootstrapCI:
    def test_basic_ci(self):
        scores = [0.5, 0.6, 0.55, 0.58, 0.52, 0.57, 0.53, 0.56, 0.54, 0.59]
        mean, lower, upper = bootstrap_confidence_interval(scores)

        assert lower < mean < upper
        assert lower > 0
        assert upper < 1

    def test_narrow_ci_for_identical(self):
        scores = [0.5] * 100
        mean, lower, upper = bootstrap_confidence_interval(scores)
        assert mean == pytest.approx(0.5)
        assert upper - lower < 0.001

    def test_wider_ci_for_variance(self):
        scores = list(np.linspace(0.1, 0.9, 50))
        mean, lower, upper = bootstrap_confidence_interval(scores)
        assert upper - lower > 0.05


class TestBootstrapPairedTest:
    def test_clear_difference(self):
        a = [0.8, 0.85, 0.82, 0.88, 0.83]
        b = [0.5, 0.55, 0.52, 0.58, 0.53]
        mean_diff, lower, upper = bootstrap_paired_test(a, b)
        # CI should not contain 0 (significant difference)
        assert lower > 0
        assert mean_diff > 0

    def test_no_difference(self):
        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        mean_diff, lower, upper = bootstrap_paired_test(a, b)
        assert mean_diff == pytest.approx(0, abs=0.01)


class TestBonferroniCorrection:
    def test_basic_correction(self):
        p_values = [0.01, 0.03, 0.05]
        corrected = bonferroni_correction(p_values)
        assert corrected == [0.03, 0.09, 0.15]

    def test_capped_at_one(self):
        p_values = [0.5, 0.8]
        corrected = bonferroni_correction(p_values)
        assert corrected == [1.0, 1.0]

    def test_single_value(self):
        corrected = bonferroni_correction([0.04])
        assert corrected == [0.04]


class TestSignificanceMatrix:
    def test_basic_matrix(self):
        scores = {
            "BM25": [0.3, 0.4, 0.35, 0.38, 0.32, 0.37, 0.34, 0.36, 0.33, 0.39],
            "Dense": [0.5, 0.6, 0.55, 0.58, 0.52, 0.57, 0.54, 0.56, 0.53, 0.59],
        }
        matrix = significance_matrix(scores, metric_name="NDCG@10")

        assert "BM25" in matrix
        assert "Dense" in matrix
        assert matrix["BM25"]["BM25"]["p_value"] == 1.0
        assert matrix["BM25"]["Dense"]["p_value"] < 0.05

    def test_symmetric(self):
        scores = {
            "A": [0.5, 0.6, 0.55],
            "B": [0.3, 0.4, 0.35],
        }
        matrix = significance_matrix(scores)
        assert matrix["A"]["B"]["p_value"] == matrix["B"]["A"]["p_value"]
