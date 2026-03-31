#!/usr/bin/env python3
"""BioPAT Experiment Runner CLI.

Run all or individual LLM evaluation experiments on the BioPAT benchmark.

Usage:
    python scripts/run_experiments.py --config configs/experiments.yaml
    python scripts/run_experiments.py --config configs/experiments.yaml --experiment hyde
    python scripts/run_experiments.py --dry-run  # estimate costs only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run BioPAT benchmark experiments",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments.yaml",
        help="Path to experiments config YAML",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmark",
        help="Path to BEIR-format benchmark data",
    )
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory for experiment results",
    )
    parser.add_argument(
        "--experiment",
        choices=["bm25", "dense", "hyde", "reranking", "relevance", "novelty", "agent"],
        help="Run only this experiment (default: all enabled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs without calling APIs",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override max budget in USD",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    config = load_config(args.config)
    budget = args.budget or config.get("budget", {}).get("max_total_usd", 500.0)

    # If single experiment requested, disable all others
    if args.experiment:
        exp_map = {
            "bm25": "bm25_baseline",
            "dense": "dense_baseline",
            "hyde": "hyde",
            "reranking": "reranking",
            "relevance": "relevance_judgment",
            "novelty": "novelty_assessment",
            "agent": "agent_retrieval",
        }
        target = exp_map[args.experiment]
        for name in config.get("experiments", {}):
            config["experiments"][name]["enabled"] = (name == target)

    # Initialize runner
    from biopat.evaluation.llm_evaluator import LLMBenchmarkRunner

    runner = LLMBenchmarkRunner(
        benchmark_dir=args.benchmark_dir,
        results_dir=args.results_dir,
        budget_usd=budget,
    )
    runner.load_benchmark()

    logger.info("Benchmark loaded: %d queries, %d docs", len(runner.queries), len(runner.corpus))

    t0 = time.time()

    # Run experiments
    results = runner.run_all(config, dry_run=args.dry_run)

    elapsed = time.time() - t0
    total_cost = sum(r.cost_usd for r in results)

    # Print summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN — COST ESTIMATES (no API calls made)")
    else:
        print("EXPERIMENT RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n  {r.experiment} / {r.model}")
        if r.metrics:
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        print(f"    Cost: ${r.cost_usd:.2f}")
        if r.elapsed_seconds:
            print(f"    Time: {r.elapsed_seconds:.0f}s")

    print(f"\n{'=' * 70}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if budget:
        print(f"Budget remaining: ${budget - total_cost:.2f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
