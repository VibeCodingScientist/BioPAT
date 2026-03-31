#!/usr/bin/env python3
"""NovEx Analysis CLI — generate paper figures and tables.

Usage:
    PYTHONPATH=src python3.12 scripts/analyze_novex.py --analysis all
    PYTHONPATH=src python3.12 scripts/analyze_novex.py --analysis grade_dist
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

# Direct imports to avoid __init__.py pulling in polars/torch
_base = Path(__file__).resolve().parent.parent / "src" / "biopat" / "novex"

def _load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, _base / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_src = Path(__file__).resolve().parent.parent / "src"

def _load_mod_abs(name, filepath):
    """Load a module by absolute path, registering parent packages as needed."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            import types
            pkg = types.ModuleType(parent)
            pkg.__path__ = [str(_src / parent.replace(".", "/"))]
            pkg.__package__ = parent
            sys.modules[parent] = pkg
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_util = _load_mod("biopat.novex._util", "_util.py")
setup_logging = _util.setup_logging
load_yaml_config = _util.load_yaml_config

# Pre-register statistical_tests to skip heavy biopat.evaluation.__init__
_load_mod_abs(
    "biopat.evaluation.statistical_tests",
    _src / "biopat" / "evaluation" / "statistical_tests.py",
)


def main():
    p = argparse.ArgumentParser(description="NovEx paper analysis")
    p.add_argument("--config", default="configs/novex.yaml")
    p.add_argument("--data-dir", default="data/novex")
    p.add_argument("--results-dir", default="data/novex/results")
    p.add_argument("--analysis", choices=[
        "vocab_gap", "per_domain", "cross_domain", "tables",
        "agreement", "errors", "significance", "summary",
        "grade_dist", "tier_corr", "difficulty", "cost_perf",
        "confusion", "ablation", "all",
    ], default="all")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    try:
        config = load_yaml_config(args.config)
    except (ImportError, FileNotFoundError):
        config = {}

    benchmark_mod = _load_mod("biopat.novex.benchmark", "benchmark.py")

    # TierResult is a simple dataclass — define inline to avoid evaluator's heavy deps
    from dataclasses import dataclass, field
    from typing import Any, Dict

    @dataclass
    class TierResult:
        tier: int
        method: str
        model: str
        metrics: Dict[str, float]
        per_query: Dict[str, Dict[str, float]] = field(default_factory=dict)
        cost_usd: float = 0.0
        elapsed_seconds: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

    # Register so analysis.py's import resolves
    import types
    evaluator_stub = types.ModuleType("biopat.novex.evaluator")
    evaluator_stub.TierResult = TierResult
    sys.modules["biopat.novex.evaluator"] = evaluator_stub

    # Now load analysis (it imports TierResult from evaluator)
    analysis_mod = _load_mod("biopat.novex.analysis", "analysis.py")

    NovExBenchmark = benchmark_mod.NovExBenchmark
    NovExAnalyzer = analysis_mod.NovExAnalyzer

    b = NovExBenchmark(data_dir=args.data_dir)
    b.load()

    results = []
    rp = Path(args.results_dir) / "all_results.json"
    if rp.exists():
        with open(rp) as f:
            results = [TierResult(**r) for r in json.load(f)]

    out_dir = config.get("analysis", {}).get("output_dir", "data/novex/analysis")
    a = NovExAnalyzer(b, results, output_dir=out_dir)

    if args.analysis == "all":
        a.run_all()
    elif args.analysis == "vocab_gap":
        a.vocabulary_gap()
    elif args.analysis == "per_domain":
        a.per_domain()
    elif args.analysis == "cross_domain":
        a.cross_domain()
    elif args.analysis == "tables":
        a.tier1_table(); a.tier2_table(); a.tier3_table()
    elif args.analysis == "agreement":
        a.inter_model_agreement()
    elif args.analysis == "errors":
        a.error_analysis()
    elif args.analysis == "significance":
        a.significance_tests()
    elif args.analysis == "summary":
        a.summary()
    elif args.analysis == "grade_dist":
        a.tier2_grade_distribution()
    elif args.analysis == "tier_corr":
        a.tier_correlation()
    elif args.analysis == "difficulty":
        a.difficulty_stratification()
    elif args.analysis == "cost_perf":
        a.cost_performance()
    elif args.analysis == "confusion":
        a.confusion_matrices()
    elif args.analysis == "ablation":
        a.context_ablation()

    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
