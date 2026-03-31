#!/usr/bin/env python3
"""Fix 3 NovEx outlier statements: remove off-domain / auto-labeled entries, replace from unused pool.

Removes:
  RN027 (patent 8930191)  — NLP/digital assistant, misclassified as A61
  RN038 (patent 10101822) — Touch keyboard, misclassified as C12
  RN087 (patent 11471510) — Novel category, 0 evidence docs, auto-labeled ("override")

Replaces with:
  RN027 -> patent 10058621 — Anti-HLA-DR antibodies + kinase inhibitors (both, A61)
  RN038 -> patent 11001863 — RNA-directed DNA modification / CRISPR (patents_only, C12)
  RN087 -> patent 11352426 — CD3 binding polypeptides (both, A61)

Usage:
  python scripts/fix_outliers.py                    # Dry run: show changes
  python scripts/fix_outliers.py --apply            # Apply changes to step7/output files
  python scripts/fix_outliers.py --apply --queue    # Also write novelty queue for VPS
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/novex")
REVERSE_DIR = DATA_DIR / "reverse"

# Patent IDs to remove and their statement IDs
REMOVE = {
    "RN027": "8930191",
    "RN038": "10101822",
    "RN087": "11471510",
}

# Replacement patent IDs mapped to the statement IDs they replace
REPLACE = {
    "RN027": "10058621",
    "RN038": "11001863",
    "RN087": "11352426",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)


def build_replacement_entry(patent_id, statement_id, categorized, grades):
    """Build a step7-format entry for a replacement patent."""
    cat_entry = next(e for e in categorized if e["patent_id"] == patent_id)
    entry = {
        "patent_id": cat_entry["patent_id"],
        "title": cat_entry["title"],
        "abstract": cat_entry["abstract"],
        "domain": cat_entry["domain"],
        "claim": cat_entry["claim"],
        "quality_scores": cat_entry["quality_scores"],
        "quality_avg": cat_entry["quality_avg"],
        "category": cat_entry["category"],
        "relevant_papers": cat_entry["relevant_papers"],
        "relevant_patents": cat_entry["relevant_patents"],
        "n_relevant": cat_entry["n_relevant"],
        "statement_id": statement_id,
    }
    return entry


def build_tier1_qrels(patent_id, statement_id, grades):
    """Build tier1 qrels from step5 grades (consensus >= 1 -> relevant)."""
    if patent_id not in grades:
        logger.warning("Patent %s not found in step5_grades!", patent_id)
        return {}
    qrels = {}
    for doc_id, info in grades[patent_id].items():
        consensus = info.get("consensus", 0)
        if consensus >= 1:
            qrels[doc_id] = consensus
    return qrels


def build_statement_entry(entry, tier1_qrels):
    """Build a statements.jsonl entry (without novelty — placeholder)."""
    return {
        "statement_id": entry["statement_id"],
        "text": entry["claim"],
        "source_patent_id": entry["patent_id"],
        "source_paper_id": "",
        "source_paper_title": "",
        "domain": entry["domain"],
        "difficulty": "medium",
        "category": entry["category"],
        "num_citing_patents": 0,
        "patent_ids": [entry["patent_id"]],
        "ground_truth": {
            "tier1_relevant_docs": tier1_qrels,
            "tier3_novelty_label": "PENDING",
            "tier3_agreement": "pending",
            "tier3_individual": {},
        },
    }


def update_novelty_in_outputs(statement_id, novelty_data, statements_path):
    """After novelty assessment completes, update the novelty fields."""
    lines = []
    with open(statements_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["statement_id"] == statement_id:
                entry["ground_truth"]["tier3_novelty_label"] = novelty_data["label"]
                entry["ground_truth"]["tier3_agreement"] = novelty_data["agreement"]
                entry["ground_truth"]["tier3_individual"] = novelty_data.get("individual", {})
            lines.append(json.dumps(entry, ensure_ascii=False))
    with open(statements_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_novelty_queue(replacements, output_path):
    """Write a JSON queue file for VPS novelty assessment (3 patents x 3 models)."""
    queue = []
    for sid, entry in replacements.items():
        queue.append({
            "statement_id": sid,
            "patent_id": entry["patent_id"],
            "claim": entry["claim"],
            "title": entry["title"],
            "abstract": entry["abstract"],
            "domain": entry["domain"],
        })
    save_json(output_path, queue)
    logger.info("Novelty queue: %d patents x 3 models = %d API calls", len(queue), len(queue) * 3)


def main():
    parser = argparse.ArgumentParser(description="Fix 3 NovEx outlier statements")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--queue", action="store_true", help="Write novelty queue for VPS")
    args = parser.parse_args()

    # Load data
    step7 = load_json(REVERSE_DIR / "step7_selected.json")
    step8 = load_json(REVERSE_DIR / "step8_novelty.json")
    grades = load_json(REVERSE_DIR / "step5_grades.json")
    categorized = load_json(REVERSE_DIR / "checkpoints" / "step6_categorized.json")

    # Validate outliers exist
    step7_by_sid = {e["statement_id"]: e for e in step7}
    step8_by_sid = {e["statement_id"]: e for e in step8}
    for sid, pid in REMOVE.items():
        assert sid in step7_by_sid, f"{sid} not found in step7"
        assert step7_by_sid[sid]["patent_id"] == pid, f"{sid} patent mismatch"
        logger.info("REMOVE %s: patent=%s title='%s' category=%s domain=%s",
                     sid, pid, step7_by_sid[sid]["title"][:60], step7_by_sid[sid]["category"],
                     step7_by_sid[sid]["domain"])

    # Build replacement entries
    replacements = {}
    for sid, new_pid in REPLACE.items():
        entry = build_replacement_entry(new_pid, sid, categorized, grades)
        replacements[sid] = entry
        tier1 = build_tier1_qrels(new_pid, sid, grades)
        logger.info("ADD    %s: patent=%s title='%s' category=%s domain=%s n_tier1=%d",
                     sid, new_pid, entry["title"][:60], entry["category"], entry["domain"], len(tier1))

    if not args.apply:
        logger.info("Dry run complete. Use --apply to write changes.")
        return

    # === Apply changes to step7_selected.json ===
    new_step7 = []
    for entry in step7:
        if entry["statement_id"] in REMOVE:
            new_step7.append(replacements[entry["statement_id"]])
        else:
            new_step7.append(entry)
    save_json(REVERSE_DIR / "step7_selected.json", new_step7)

    # === Apply changes to step8_novelty.json ===
    # New entries get placeholder novelty (to be filled after VPS run)
    new_step8 = []
    for entry in step8:
        if entry["statement_id"] in REMOVE:
            rep = dict(replacements[entry["statement_id"]])
            rep["novelty"] = {"label": "PENDING", "agreement": "pending", "individual": {}}
            new_step8.append(rep)
        else:
            new_step8.append(entry)
    save_json(REVERSE_DIR / "step8_novelty.json", new_step8)

    # === Regenerate statements.jsonl ===
    statements = []
    for entry in new_step8:
        sid = entry["statement_id"]
        tier1_qrels = build_tier1_qrels(entry["patent_id"], sid, grades)
        novelty = entry.get("novelty", {})
        stmt = {
            "statement_id": sid,
            "text": entry["claim"],
            "source_patent_id": entry["patent_id"],
            "source_paper_id": "",
            "source_paper_title": "",
            "domain": entry["domain"],
            "difficulty": "medium",
            "category": entry["category"],
            "num_citing_patents": 0,
            "patent_ids": [entry["patent_id"]],
            "ground_truth": {
                "tier1_relevant_docs": tier1_qrels,
                "tier3_novelty_label": novelty.get("label", "PENDING"),
                "tier3_agreement": novelty.get("agreement", "pending"),
                "tier3_individual": novelty.get("individual", {}),
            },
        }
        statements.append(stmt)

    with open(DATA_DIR / "statements.jsonl", "w") as f:
        for stmt in statements:
            f.write(json.dumps(stmt, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d statements)", DATA_DIR / "statements.jsonl", len(statements))

    # === Regenerate queries.jsonl ===
    with open(DATA_DIR / "queries.jsonl", "w") as f:
        for stmt in statements:
            f.write(json.dumps({"_id": stmt["statement_id"], "text": stmt["text"]}) + "\n")
    logger.info("Wrote %s (%d queries)", DATA_DIR / "queries.jsonl", len(statements))

    # === Regenerate tier1.tsv ===
    qrels_dir = DATA_DIR / "qrels"
    qrels_dir.mkdir(exist_ok=True)
    with open(qrels_dir / "tier1.tsv", "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for stmt in statements:
            rel = stmt["ground_truth"]["tier1_relevant_docs"]
            for doc_id in sorted(rel):
                f.write(f"{stmt['statement_id']}\t{doc_id}\t{rel[doc_id]}\n")
    n_tier1 = sum(len(s["ground_truth"]["tier1_relevant_docs"]) for s in statements)
    logger.info("Wrote %s (%d qrel entries)", qrels_dir / "tier1.tsv", n_tier1)

    # === Update tier3.tsv (swap 3 rows) ===
    novelty_to_score = {"NOVEL": 0, "PARTIALLY_ANTICIPATED": 1, "ANTICIPATED": 2, "PENDING": 0}
    with open(qrels_dir / "tier3.tsv", "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for stmt in statements:
            label = stmt["ground_truth"]["tier3_novelty_label"]
            score = novelty_to_score.get(label, 0)
            f.write(f"{stmt['statement_id']}\t{stmt['statement_id']}\t{score}\n")
    logger.info("Wrote %s (%d entries)", qrels_dir / "tier3.tsv", len(statements))

    # === Novelty queue ===
    if args.queue:
        write_novelty_queue(replacements, REVERSE_DIR / "novelty_queue_outliers.json")

    # === Summary ===
    from collections import Counter
    cats = Counter(e["category"] for e in new_step7)
    doms = Counter(e["domain"] for e in new_step7)
    logger.info("Category distribution: %s", dict(cats))
    logger.info("Domain distribution: %s", dict(doms))
    logger.info("Done! After VPS novelty run, update step8 with actual labels.")


if __name__ == "__main__":
    main()
