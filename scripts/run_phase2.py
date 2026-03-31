#!/usr/bin/env python3
"""Phase 2: T3 re-runs with predicted labels + context ablation.

Creates fresh checkpoints alongside the old ones:
  Old:  t3_{provider}_{model}_ctx       (no predicted, k=10 implicit)
  New:  t3_{provider}_{model}_ctx_k10   (with predicted)
  New:  t3_{provider}_{model}_ctx_k1    (with predicted, 1 doc)
  ...etc

Usage:
    nohup PYTHONPATH=src python3.12 scripts/run_phase2.py 2>&1 | tee logs/phase2.log &

Estimated cost: ~$8-10 total
  - 3 ctx re-runs at k=10:  ~$1-2
  - 12 ablation runs (k=1,3,5,20 × 3 models): ~$6-8
"""

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase2")

# --- Direct imports (bypass heavy __init__.py) ---
_src = Path(__file__).resolve().parent.parent / "src"

def _load_mod(name, filepath):
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

_load_mod("biopat.novex._util", _src / "biopat/novex/_util.py")
_load_mod("biopat.novex.benchmark", _src / "biopat/novex/benchmark.py")
_load_mod("biopat.novex.evaluator", _src / "biopat/novex/evaluator.py")

from biopat.novex.benchmark import NovExBenchmark
from biopat.novex.evaluator import NovExEvaluator

# --- Config ---
MODELS = [
    ("openai", "gpt-5.2"),
    ("anthropic", "claude-sonnet-4-6"),
    ("google", "gemini-3-pro-preview"),
]
ABLATION_K = [1, 3, 5, 10, 20]  # 10 = standard re-run with fresh checkpoint
BUDGET = 200.0


def main():
    t0 = time.time()

    logger.info("Loading benchmark...")
    b = NovExBenchmark(data_dir="data/novex")
    b.load()
    logger.info("Loaded: %d statements, %d docs", len(b.statements), len(b.corpus))

    ev = NovExEvaluator(
        benchmark=b,
        results_dir="data/novex/results",
        budget_usd=BUDGET,
        seed=42,
    )

    results = []
    total_runs = len(MODELS) * len(ABLATION_K)
    completed = 0

    for provider, model_id in MODELS:
        # --- Context runs at each k ---
        for k in ABLATION_K:
            completed += 1
            logger.info("[%d/%d] T3 ctx k=%d — %s/%s",
                        completed, total_runs, k, provider, model_id)
            try:
                r = ev.run_tier3(provider, model_id, with_context=True, context_k=k)
                results.append(r)
                logger.info("  → acc=%.4f  cost=$%.2f  (checkpoint: t3_%s_%s_ctx_k%d)",
                            r.metrics.get("accuracy", 0), r.cost_usd,
                            provider, model_id.replace("-", "_"), k)
            except Exception as exc:
                logger.error("  FAILED: %s", exc)

    # --- Merge with existing all_results.json ---
    rp = Path("data/novex/results/all_results.json")
    existing = []
    if rp.exists():
        with open(rp) as f:
            existing = json.load(f)

    # Remove any old T3 entries that we're replacing (same model + ctx + context_k)
    new_keys = set()
    for r in results:
        key = (r.model, r.metadata.get("with_context"), r.metadata.get("context_k"))
        new_keys.add(key)

    merged = []
    for entry in existing:
        key = (entry["model"],
               entry.get("metadata", {}).get("with_context"),
               entry.get("metadata", {}).get("context_k"))
        if key not in new_keys:
            merged.append(entry)

    # Add new results
    for r in results:
        merged.append(r.__dict__)

    with open(rp, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    logger.info("Merged %d new results into all_results.json (%d total entries)",
                len(results), len(merged))

    elapsed = time.time() - t0
    total_cost = ev.cost_tracker.total_cost
    logger.info("Phase 2 complete: %d runs, $%.2f, %.0fs", len(results), total_cost, elapsed)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Runs:    {len(results)}/{total_runs}")
    print(f"Cost:    ${total_cost:.2f}")
    print(f"Time:    {elapsed:.0f}s")
    print(f"{'='*60}")
    for r in results:
        k = r.metadata.get("context_k", "?")
        print(f"  {r.model:30s} k={k:>2}  acc={r.metrics.get('accuracy',0):.4f}")
    print(f"{'='*60}")
    print(f"\nNext: PYTHONPATH=src python3.12 scripts/analyze_novex.py --analysis all")


if __name__ == "__main__":
    main()
