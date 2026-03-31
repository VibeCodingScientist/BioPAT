#!/usr/bin/env python3
"""NovEx Statement Curation CLI.

Usage:
    python scripts/curate_statements.py --config configs/novex.yaml
    python scripts/curate_statements.py --step candidates
"""

import argparse
import asyncio
import sys
import time

from biopat.novex._util import setup_logging, load_yaml_config


async def run(args):
    config = load_yaml_config(args.config).get("curation", {})

    from biopat.novex.curate import StatementCurator
    curator = StatementCurator(
        qrels_path=config.get("qrels_path", "data/checkpoints/step5_qrels.parquet"),
        papers_path=config.get("papers_path", "data/checkpoints/step4_papers.parquet"),
        patents_path=config.get("patents_path", "data/checkpoints/step3_patents.parquet"),
        corpus_dir=config.get("corpus_dir", "data/benchmark"),
        output_dir=config.get("output_dir", "data/novex"),
        budget_usd=config.get("budget_usd", 50.0),
        seed=config.get("seed", 42),
    )

    if args.step == "candidates":
        print(f"Selected {len(curator.select_candidates())} candidates")
    elif args.step == "extract":
        c = curator.select_candidates()
        extracted = await curator.extract_statements(c)
        print(f"Extracted {len(extracted)} statements")
    else:
        stmts = await curator.run_pipeline()
        print(f"Done: {len(stmts)} statements")


def main():
    p = argparse.ArgumentParser(description="Curate NovEx statements")
    p.add_argument("--config", default="configs/novex.yaml")
    p.add_argument("--step", choices=["candidates", "extract", "full"], default="full")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)
    t0 = time.time()
    asyncio.run(run(args))
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
