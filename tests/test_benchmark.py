"""Tests for benchmark assembly modules."""

import pytest
import polars as pl
from pathlib import Path
import tempfile
import json

from biopat.benchmark.sampling import BenchmarkSampler
from biopat.benchmark.splits import DatasetSplitter
from biopat.benchmark.beir_format import BEIRFormatter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBenchmarkSampler:
    """Tests for BenchmarkSampler."""

    def test_sample_queries_stratified(self):
        """Test stratified query sampling."""
        sampler = BenchmarkSampler(seed=42)

        queries_df = pl.DataFrame({
            "query_id": [f"Q{i}" for i in range(100)],
            "domain": ["A61K"] * 60 + ["C07D"] * 30 + ["C12N"] * 10,
        })

        sampled = sampler.sample_queries_stratified(queries_df, target_count=20)

        assert len(sampled) <= 20

        # Check proportions are roughly maintained
        domain_counts = sampled.group_by("domain").agg(pl.len().alias("count"))
        counts_dict = {row["domain"]: row["count"] for row in domain_counts.iter_rows(named=True)}

        # A61K should have more than C12N
        assert counts_dict.get("A61K", 0) >= counts_dict.get("C12N", 0)

    def test_sample_no_reduction_needed(self):
        """Test sampling when target exceeds available."""
        sampler = BenchmarkSampler(seed=42)

        queries_df = pl.DataFrame({
            "query_id": ["Q1", "Q2", "Q3"],
            "domain": ["A61K", "A61K", "C07D"],
        })

        sampled = sampler.sample_queries_stratified(queries_df, target_count=100)

        # Should return all queries
        assert len(sampled) == 3


class TestDatasetSplitter:
    """Tests for DatasetSplitter."""

    def test_split_ratios(self):
        """Test that splits maintain specified ratios."""
        splitter = DatasetSplitter(
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        queries_df = pl.DataFrame({
            "query_id": [f"Q{i}" for i in range(100)],
            "domain": ["A61K"] * 50 + ["C07D"] * 50,
        })

        train, dev, test = splitter.split_queries_stratified(queries_df)

        total = len(train) + len(dev) + len(test)
        assert total == 100

        # Check approximate ratios (allow some variance due to stratification)
        assert 60 <= len(train) <= 80
        assert 10 <= len(dev) <= 25
        assert 10 <= len(test) <= 25

    def test_no_overlap(self):
        """Test that splits have no overlap."""
        splitter = DatasetSplitter(seed=42)

        queries_df = pl.DataFrame({
            "query_id": [f"Q{i}" for i in range(50)],
            "domain": ["A61K"] * 25 + ["C07D"] * 25,
        })

        train, dev, test = splitter.split_queries_stratified(queries_df)

        train_ids = set(train["query_id"].to_list())
        dev_ids = set(dev["query_id"].to_list())
        test_ids = set(test["query_id"].to_list())

        assert len(train_ids & dev_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(dev_ids & test_ids) == 0

    def test_split_by_patent(self):
        """Test splitting by patent keeps claims together."""
        splitter = DatasetSplitter(seed=42)

        queries_df = pl.DataFrame({
            "query_id": ["P1-c1", "P1-c2", "P2-c1", "P2-c2", "P3-c1"],
            "patent_id": ["P1", "P1", "P2", "P2", "P3"],
            "domain": ["A61K", "A61K", "C07D", "C07D", "A61K"],
        })

        train, dev, test = splitter.split_by_patent(queries_df)

        # All claims from same patent should be in same split
        for split_df in [train, dev, test]:
            patents = split_df["patent_id"].unique().to_list()
            for patent_id in patents:
                claims_in_split = split_df.filter(pl.col("patent_id") == patent_id)
                all_claims = queries_df.filter(pl.col("patent_id") == patent_id)
                assert len(claims_in_split) == len(all_claims)


class TestBEIRFormatter:
    """Tests for BEIRFormatter."""

    def test_format_corpus(self, temp_dir):
        """Test corpus formatting."""
        formatter = BEIRFormatter(temp_dir)

        papers_df = pl.DataFrame({
            "paper_id": ["W1", "W2"],
            "title": ["Title 1", "Title 2"],
            "abstract": ["Abstract 1", "Abstract 2"],
        })

        formatter.format_corpus(papers_df)

        # Check file exists and content
        assert formatter.corpus_path.exists()

        with open(formatter.corpus_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        doc1 = json.loads(lines[0])
        assert doc1["_id"] == "W1"
        assert doc1["title"] == "Title 1"
        assert doc1["text"] == "Abstract 1"

    def test_format_queries(self, temp_dir):
        """Test queries formatting."""
        formatter = BEIRFormatter(temp_dir)

        queries_df = pl.DataFrame({
            "query_id": ["Q1", "Q2"],
            "claim_text": ["Claim 1 text", "Claim 2 text"],
        })

        formatter.format_queries(queries_df)

        assert formatter.queries_path.exists()

        with open(formatter.queries_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        q1 = json.loads(lines[0])
        assert q1["_id"] == "Q1"
        assert q1["text"] == "Claim 1 text"

    def test_format_qrels(self, temp_dir):
        """Test qrels formatting."""
        formatter = BEIRFormatter(temp_dir)

        train_qrels = pl.DataFrame({
            "query_id": ["Q1"],
            "doc_id": ["W1"],
            "relevance": [1],
        })

        dev_qrels = pl.DataFrame({
            "query_id": ["Q2"],
            "doc_id": ["W2"],
            "relevance": [2],
        })

        test_qrels = pl.DataFrame({
            "query_id": ["Q3"],
            "doc_id": ["W3"],
            "relevance": [3],
        })

        formatter.format_qrels(train_qrels, dev_qrels, test_qrels)

        # Check all files exist
        assert (formatter.qrels_dir / "train.tsv").exists()
        assert (formatter.qrels_dir / "dev.tsv").exists()
        assert (formatter.qrels_dir / "test.tsv").exists()

        # Check content
        with open(formatter.qrels_dir / "test.tsv") as f:
            line = f.readline().strip()

        parts = line.split("\t")
        assert parts[0] == "Q3"
        assert parts[1] == "W3"
        assert parts[2] == "3"

    def test_validate_output(self, temp_dir):
        """Test output validation."""
        formatter = BEIRFormatter(temp_dir)

        # Create minimal valid output
        papers_df = pl.DataFrame({
            "paper_id": ["W1"],
            "title": ["Title"],
            "abstract": ["Abstract"],
        })
        formatter.format_corpus(papers_df)

        queries_df = pl.DataFrame({
            "query_id": ["Q1"],
            "claim_text": ["Claim"],
        })
        formatter.format_queries(queries_df)

        validation = formatter.validate_output()

        assert validation["corpus_exists"] is True
        assert validation["queries_exists"] is True
        assert validation["corpus_count"] == 1
        assert validation["queries_count"] == 1
        assert validation["corpus_valid_json"] is True
        assert validation["queries_valid_json"] is True
