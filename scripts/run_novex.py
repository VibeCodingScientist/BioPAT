#!/usr/bin/env python3
"""NovEx Evaluation Runner CLI.

Usage:
    python scripts/run_novex.py --config configs/novex.yaml
    python scripts/run_novex.py --tier 1 --method bm25
    python scripts/run_novex.py --dry-run
"""

import argparse
import json
import logging
import sys
import time

from biopat.novex._util import setup_logging, load_yaml_config


def main():
    p = argparse.ArgumentParser(description="Run NovEx evaluations")
    p.add_argument("--config", default="configs/novex.yaml")
    p.add_argument("--data-dir", default="data/novex")
    p.add_argument("--corpus-dir", default=None)
    p.add_argument("--tier", type=int, choices=[1, 2, 3])
    p.add_argument("--method", choices=["bm25", "dense", "hybrid", "hyde", "rerank", "agent"])
    p.add_argument("--model", default=None)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--context-k", type=int, default=None,
                   help="Number of context documents for T3 (default: 10). "
                        "Pass comma-separated values for ablation: --context-k 1,3,5,10,20")
    p.add_argument("--context-k-ablation", type=str, default=None,
                   help="Comma-separated k values for context ablation (e.g., 1,3,5,20)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    config = load_yaml_config(args.config)
    eval_cfg = config.get("evaluation", {})
    budget = args.budget or config.get("budget", {}).get("max_total_usd", 350.0)

    from biopat.novex.benchmark import NovExBenchmark
    from biopat.novex.evaluator import NovExEvaluator

    b = NovExBenchmark(data_dir=args.data_dir, corpus_dir=args.corpus_dir)
    b.load()
    logging.getLogger(__name__).info("%d statements, %d docs", len(b.statements), len(b.corpus))

    if args.dry_run:
        n = len(b.statements)
        nm = len(eval_cfg.get("tier1", {}).get("llm_models", []))
        print(f"\nCost estimate: Tier1 HyDE ~${n*nm*0.005:.0f}, Agent ~${n*nm*0.05:.0f}, "
              f"Tier2 ~${n*50*3*0.003:.0f}, Tier3 ~${n*3*2*0.01:.0f}")
        return 0

    ev = NovExEvaluator(benchmark=b, results_dir=eval_cfg.get("results_dir", "data/novex/results"),
                        budget_usd=budget, seed=eval_cfg.get("seed", 42))

    # Filter config
    run_cfg = {}
    for tk in ["tier1", "tier2", "tier3"]:
        tn = int(tk[-1])
        if args.tier and args.tier != tn:
            continue
        tc = dict(eval_cfg.get(tk, {}))
        if args.method and tn == 1:
            for m in ["bm25", "dense", "hybrid", "hyde", "rerank", "agent"]:
                if m in tc:
                    tc[m] = dict(tc[m])
                    tc[m]["enabled"] = (m == args.method)
        if args.model:
            for key in ["models", "llm_models"]:
                if key in tc:
                    tc[key] = [m for m in tc[key] if m.get("model_id") == args.model]
        if tn == 3:
            if args.context_k is not None:
                tc["context_k"] = args.context_k
            if args.context_k_ablation:
                tc["context_k_ablation"] = [int(x) for x in args.context_k_ablation.split(",")]
        run_cfg[tk] = tc

    t0 = time.time()
    results = ev.run_all(run_cfg)
    cost = sum(r.cost_usd for r in results)

    print(f"\n{'='*60}\nNovEx RESULTS\n{'='*60}")
    for r in results:
        print(f"\n  T{r.tier} {r.method}/{r.model}")
        for k, v in sorted(r.metrics.items()):
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        if r.cost_usd:
            print(f"    cost: ${r.cost_usd:.2f}")
    print(f"\n{'='*60}\nTotal: ${cost:.2f} | {time.time()-t0:.0f}s | Budget left: ${budget-cost:.2f}\n{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
