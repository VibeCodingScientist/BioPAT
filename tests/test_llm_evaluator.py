"""Tests for LLM benchmark evaluator."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biopat.evaluation.llm_evaluator import (
    LLMBenchmarkRunner,
    ModelSpec,
    ExperimentResult,
)


@pytest.fixture
def mock_benchmark(tmp_path):
    """Create a minimal BEIR-format benchmark for testing."""
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    # Corpus
    corpus = [
        {"_id": "D1", "title": "Anti-PD-1 therapy", "text": "Pembrolizumab is a humanized antibody..."},
        {"_id": "D2", "title": "CRISPR gene editing", "text": "Cas9 nuclease enables precise genome editing..."},
        {"_id": "D3", "title": "mRNA vaccines", "text": "Lipid nanoparticle delivery of modified mRNA..."},
        {"_id": "D4", "title": "CAR-T cell therapy", "text": "Chimeric antigen receptor T cells target CD19..."},
        {"_id": "D5", "title": "Protein folding", "text": "AlphaFold predicts 3D structures from sequence..."},
    ]
    with open(benchmark_dir / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    # Queries
    queries = [
        {"_id": "Q1", "text": "Method for treating melanoma with anti-PD-1 antibodies"},
        {"_id": "Q2", "text": "CRISPR-based gene therapy for sickle cell disease"},
        {"_id": "Q3", "text": "mRNA vaccine formulation with lipid nanoparticles"},
    ]
    with open(benchmark_dir / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # Qrels
    qrels_dir = benchmark_dir / "qrels"
    qrels_dir.mkdir()
    with open(qrels_dir / "test.tsv", "w") as f:
        f.write("Q1\tD1\t2\n")
        f.write("Q1\tD4\t1\n")
        f.write("Q2\tD2\t3\n")
        f.write("Q3\tD3\t2\n")

    return benchmark_dir


class TestModelSpec:
    def test_defaults(self):
        spec = ModelSpec(name="test", provider="openai", model_id="gpt-4o")
        assert spec.display_name == "test"

    def test_display_name(self):
        spec = ModelSpec(name="x", provider="openai", model_id="gpt-4o", display_name="GPT-4o")
        assert spec.display_name == "GPT-4o"


class TestExperimentResult:
    def test_basic(self):
        r = ExperimentResult(
            experiment="bm25", model="BM25",
            metrics={"NDCG@10": 0.45}, cost_usd=0.0,
        )
        assert r.experiment == "bm25"
        assert r.metrics["NDCG@10"] == 0.45


class TestLLMBenchmarkRunner:
    def test_load_benchmark(self, mock_benchmark):
        runner = LLMBenchmarkRunner(
            benchmark_dir=str(mock_benchmark),
            results_dir=str(mock_benchmark / "results"),
        )
        runner.load_benchmark()

        assert len(runner.corpus) == 5
        assert len(runner.queries) == 3
        assert "Q1" in runner.qrels
        assert runner.qrels["Q1"]["D1"] == 2

    def test_subsample_queries(self, mock_benchmark):
        runner = LLMBenchmarkRunner(
            benchmark_dir=str(mock_benchmark),
            results_dir=str(mock_benchmark / "results"),
        )
        runner.load_benchmark()

        sub = runner._subsample_queries(2)
        assert len(sub) == 2

        all_q = runner._subsample_queries(None)
        assert len(all_q) == 3

    def test_checkpoint_roundtrip(self, mock_benchmark):
        runner = LLMBenchmarkRunner(
            benchmark_dir=str(mock_benchmark),
            results_dir=str(mock_benchmark / "results"),
        )

        data = {"foo": "bar", "count": 42}
        runner._save_checkpoint("test_cp", data)
        assert runner._has_checkpoint("test_cp")

        loaded = runner._load_checkpoint("test_cp")
        assert loaded == data

    def test_dry_run(self, mock_benchmark):
        """Test dry run produces cost estimates."""
        runner = LLMBenchmarkRunner(
            benchmark_dir=str(mock_benchmark),
            results_dir=str(mock_benchmark / "results"),
        )
        runner.load_benchmark()

        config = {
            "models": {
                "test_model": {
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "display_name": "Test GPT",
                },
            },
            "experiments": {
                "bm25_baseline": {"enabled": False},
                "dense_baseline": {"enabled": False},
                "hyde": {
                    "enabled": True,
                    "llm_models": ["test_model"],
                    "max_queries": 10,
                },
                "reranking": {"enabled": False},
                "relevance_judgment": {"enabled": False},
                "novelty_assessment": {"enabled": False},
            },
        }

        results = runner.run_all(config, dry_run=True)
        assert len(results) >= 1
        assert all(r.metadata.get("dry_run") for r in results)
        assert all(r.cost_usd > 0 for r in results)

    def test_agreement_metrics(self, mock_benchmark):
        """Test agreement metric computation."""
        runner = LLMBenchmarkRunner(
            benchmark_dir=str(mock_benchmark),
            results_dir=str(mock_benchmark / "results"),
        )

        gold = [0, 1, 2, 3, 1, 2]
        pred = [0, 1, 2, 3, 1, 2]
        metrics = runner._compute_agreement(gold, pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["mae"] == 0.0
        assert metrics["cohens_kappa"] == pytest.approx(1.0)

    def test_agreement_imperfect(self, mock_benchmark):
        """Test agreement with imperfect predictions."""
        runner = LLMBenchmarkRunner(
            benchmark_dir=str(mock_benchmark),
            results_dir=str(mock_benchmark / "results"),
        )

        gold = [0, 1, 2, 3]
        pred = [0, 2, 1, 3]
        metrics = runner._compute_agreement(gold, pred)

        assert 0 < metrics["accuracy"] < 1.0
        assert metrics["mae"] > 0
