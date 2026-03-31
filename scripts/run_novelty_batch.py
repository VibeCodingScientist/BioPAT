#!/usr/bin/env python3
"""Run novelty assessment for a batch of patents via 3-LLM consensus.

Reads a novelty queue JSON, calls AnnotationProtocol.annotate_tier3(),
merges results into step8_novelty.json, and regenerates output files.

Usage:
  python scripts/run_novelty_batch.py data/novex/reverse/novelty_queue_outliers.json
  python scripts/run_novelty_batch.py data/novex/reverse/novelty_queue_scale.json
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/novex")
REVERSE_DIR = DATA_DIR / "reverse"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)


def build_prior_art(queue, grades):
    """Build prior art dict from step5 grades for each queued patent."""
    prior_art = {}
    for e in queue:
        pid = e["patent_id"]
        sid = e["statement_id"]
        if pid in grades:
            docs = sorted(
                [(did, info) for did, info in grades[pid].items()
                 if info.get("consensus", 0) >= 1],
                key=lambda x: (-x[1]["consensus"], -x[1].get("bm25_score", 0)),
            )[:10]
            prior_art[sid] = [
                {
                    "_id": did,
                    "title": f"Patent {did}" if not did.startswith("W") else f"Paper {did}",
                    "text": "",
                }
                for did, _ in docs
            ]
        else:
            prior_art[sid] = []
    return prior_art


def merge_into_step8(consensus, step8_path):
    """Merge novelty consensus into step8_novelty.json."""
    step8 = load_json(step8_path)
    consensus_by_sid = {c["statement_id"]: c for c in consensus}
    updated = 0
    for entry in step8:
        sid = entry["statement_id"]
        if sid in consensus_by_sid:
            c = consensus_by_sid[sid]
            entry["novelty"] = {
                "label": c["label"],
                "agreement": c["agreement"],
                "individual": c.get("individual", {}),
            }
            updated += 1
            logger.info("  %s: %s (%s)", sid, c["label"], c["agreement"])
    save_json(step8_path, step8)
    logger.info("Merged novelty for %d entries", updated)
    return step8


def regenerate_outputs(step8, grades):
    """Regenerate statements.jsonl, queries.jsonl, tier1.tsv, tier3.tsv."""
    novelty_to_score = {"NOVEL": 0, "PARTIALLY_ANTICIPATED": 1, "ANTICIPATED": 2, "PENDING": 0}

    stmts_out = []
    for entry in step8:
        pid = entry["patent_id"]
        tier1_qrels = {}
        if pid in grades:
            tier1_qrels = {
                did: info["consensus"]
                for did, info in grades[pid].items()
                if info.get("consensus", 0) >= 1
            }
        novelty = entry.get("novelty", {})
        stmts_out.append({
            "statement_id": entry["statement_id"],
            "text": entry["claim"],
            "source_patent_id": pid,
            "source_paper_id": "",
            "source_paper_title": "",
            "domain": entry["domain"],
            "difficulty": "medium",
            "category": entry["category"],
            "num_citing_patents": 0,
            "patent_ids": [pid],
            "ground_truth": {
                "tier1_relevant_docs": tier1_qrels,
                "tier3_novelty_label": novelty.get("label", "PENDING"),
                "tier3_agreement": novelty.get("agreement", "pending"),
                "tier3_individual": novelty.get("individual", {}),
            },
        })

    with open(DATA_DIR / "statements.jsonl", "w") as f:
        for s in stmts_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Wrote statements.jsonl (%d)", len(stmts_out))

    with open(DATA_DIR / "queries.jsonl", "w") as f:
        for s in stmts_out:
            f.write(json.dumps({"_id": s["statement_id"], "text": s["text"]}) + "\n")
    logger.info("Wrote queries.jsonl (%d)", len(stmts_out))

    qrels_dir = DATA_DIR / "qrels"
    qrels_dir.mkdir(exist_ok=True)

    with open(qrels_dir / "tier1.tsv", "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for s in stmts_out:
            for did in sorted(s["ground_truth"]["tier1_relevant_docs"]):
                f.write(f"{s['statement_id']}\t{did}\t{s['ground_truth']['tier1_relevant_docs'][did]}\n")
    logger.info("Wrote tier1.tsv")

    with open(qrels_dir / "tier3.tsv", "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for s in stmts_out:
            label = s["ground_truth"]["tier3_novelty_label"]
            score = novelty_to_score.get(label, 0)
            f.write(f"{s['statement_id']}\t{s['statement_id']}\t{score}\n")
    logger.info("Wrote tier3.tsv")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <novelty_queue.json>")
        sys.exit(1)

    queue_path = sys.argv[1]
    tag = Path(queue_path).stem  # e.g. "novelty_queue_outliers"

    logger.info("Loading queue from %s", queue_path)
    queue = load_json(queue_path)
    logger.info("Queue: %d patents x 3 models = %d API calls", len(queue), len(queue) * 3)

    grades = load_json(REVERSE_DIR / "step5_grades.json")
    statements = [{"statement_id": e["statement_id"], "text": e["claim"]} for e in queue]
    prior_art = build_prior_art(queue, grades)

    # Budget: ~$0.03 per statement
    budget = max(5.0, len(queue) * 0.05)
    logger.info("Budget: $%.2f", budget)

    from biopat.novex.annotation import AnnotationProtocol

    protocol = AnnotationProtocol(
        output_dir=f"data/novex/annotation_{tag}",
        checkpoint_dir=f"data/novex/annotation_{tag}/checkpoints",
        budget_usd=budget,
    )

    t0 = time.time()

    async def run():
        return await protocol.annotate_tier3(statements, prior_art)

    judgments, consensus = asyncio.run(run())
    elapsed = time.time() - t0
    logger.info("Done: %d judgments, %d consensus in %.1f min ($%.2f)",
                len(judgments), len(consensus), elapsed / 60,
                protocol.cost_tracker.total_cost)

    # Save raw results
    result_path = REVERSE_DIR / f"{tag}_result.json"
    save_json(result_path, consensus)

    # Merge into step8 and regenerate outputs
    step8 = merge_into_step8(consensus, REVERSE_DIR / "step8_novelty.json")
    regenerate_outputs(step8, grades)

    # Save costs
    protocol.cost_tracker.save(f"data/novex/annotation_{tag}/costs.json")

    logger.info("=== BATCH COMPLETE: %s ===", tag)


if __name__ == "__main__":
    main()
