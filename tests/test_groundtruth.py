"""Tests for ground truth modules."""

import pytest
import polars as pl
from pathlib import Path
import tempfile
from datetime import date

from biopat.groundtruth.relevance import RelevanceAssigner, RelevanceLevel
from biopat.groundtruth.temporal import TemporalValidator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestRelevanceAssigner:
    """Tests for RelevanceAssigner."""

    def test_assign_binary_relevance(self, temp_dir):
        """Test binary relevance assignment."""
        assigner = RelevanceAssigner(temp_dir)

        links = pl.DataFrame({
            "query_id": ["Q1", "Q1", "Q2"],
            "paper_id": ["W1", "W2", "W3"],
            "confidence": [9, 7, 8],
            "source": ["examiner", "applicant", "examiner"],
        })

        qrels = assigner.assign_binary_relevance(links, confidence_threshold=8)

        # Should only include confidence >= 8
        assert len(qrels) == 2
        assert "W1" in qrels["doc_id"].to_list()
        assert "W3" in qrels["doc_id"].to_list()
        assert "W2" not in qrels["doc_id"].to_list()

    def test_binary_relevance_values(self, temp_dir):
        """Test that binary relevance is always 1."""
        assigner = RelevanceAssigner(temp_dir)

        links = pl.DataFrame({
            "query_id": ["Q1"],
            "paper_id": ["W1"],
            "confidence": [9],
            "source": ["examiner"],
        })

        qrels = assigner.assign_binary_relevance(links)

        assert qrels["relevance"].to_list() == [1]


class TestTemporalValidator:
    """Tests for TemporalValidator."""

    def test_validate_single_valid(self):
        """Test validation of valid prior art."""
        validator = TemporalValidator()

        # Paper published before patent
        assert validator.validate_single("2015-06-01", "2016-01-01") is True

    def test_validate_single_invalid_same_date(self):
        """Test validation rejects same-date publication."""
        validator = TemporalValidator()

        # Paper published same day as patent priority
        assert validator.validate_single("2016-01-01", "2016-01-01") is False

    def test_validate_single_invalid_after(self):
        """Test validation rejects publication after priority date."""
        validator = TemporalValidator()

        # Paper published after patent
        assert validator.validate_single("2017-01-01", "2016-01-01") is False

    def test_validate_single_missing_date(self):
        """Test validation handles missing dates."""
        validator = TemporalValidator()

        assert validator.validate_single(None, "2016-01-01") is False
        assert validator.validate_single("2015-01-01", None) is False

    def test_validate_dataframe(self, temp_dir):
        """Test DataFrame validation."""
        validator = TemporalValidator(temp_dir)

        df = pl.DataFrame({
            "query_id": ["Q1", "Q2", "Q3"],
            "publication_date": ["2015-01-01", "2016-06-01", "2014-01-01"],
            "priority_date": ["2016-01-01", "2016-01-01", "2016-01-01"],
        })

        valid, invalid = validator.validate_dataframe(df)

        # Q1 and Q3 are valid (published before priority)
        # Q2 is invalid (published after priority)
        assert len(valid) == 2
        assert len(invalid) == 1

    def test_parse_date_formats(self):
        """Test parsing of various date formats."""
        validator = TemporalValidator()

        assert validator.parse_date("2020-01-15") == date(2020, 1, 15)
        assert validator.parse_date("2020/01/15") == date(2020, 1, 15)
        assert validator.parse_date("2020") == date(2020, 1, 1)
        assert validator.parse_date(date(2020, 1, 15)) == date(2020, 1, 15)

    def test_analyze_violations(self, temp_dir):
        """Test violation analysis."""
        validator = TemporalValidator(temp_dir)

        invalid_df = pl.DataFrame({
            "query_id": ["Q1", "Q2"],
            "publication_date": ["2017-01-01", None],
            "priority_date": ["2016-01-01", "2016-01-01"],
        })

        analysis = validator.analyze_violations(invalid_df)

        assert analysis["total_violations"] == 2
        assert analysis["missing_paper_date"] == 1
        assert analysis["date_violations"] == 1
