#!/usr/bin/env python3
"""Scale NovEx benchmark from 100 to 300 statements using unused pre-graded candidates.

All 213 unused candidates already have: claim extraction, quality filtering, BM25 retrieval,
and 3-LLM relevance grading completed. Only novelty assessment (step 8) is missing for
the new 200 entries.

Target distribution (300 total):
  both:          153 (51%)
  patents_only:   95 (32%)
  papers_only:    33 (11%)
  novel:          19  (6%)

Domain targets (proportional): A61 ~120, C07 ~93, C12 ~87

Usage:
  python scripts/scale_benchmark.py                        # Dry run: show selection
  python scripts/scale_benchmark.py --apply                # Apply: write step7, outputs, queue
  python scripts/scale_benchmark.py --merge-novelty FILE   # After VPS: merge novelty results
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/novex")
REVERSE_DIR = DATA_DIR / "reverse"

# Target sizes per category (total 300 = current 100 + new 200)
TARGETS = {
    "both": 153,
    "patents_only": 95,
    "papers_only": 33,
    "novel": 19,
}

# Approximate domain proportions (from original 100)
DOMAIN_WEIGHTS = {"A61": 0.41, "C07": 0.33, "C12": 0.26}

# Patent IDs handled by fix_outliers.py — removed originals and their replacements
# (replacements are in step7 post-fix, but guard against running before fix_outliers)
EXCLUDED_PATENTS = {
    "8930191", "10101822", "11471510",   # removed outliers
    "10058621", "11001863", "11352426",  # their replacements (already in step7 post-fix)
}

SEED = 42


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)


def select_new_entries(current, unused, targets):
    """Select entries from unused pool to reach target distribution, domain-balanced."""
    rng = random.Random(SEED)
    current_by_cat = Counter(e["category"] for e in current)

    # Group unused by category
    unused_by_cat = {}
    for e in unused:
        unused_by_cat.setdefault(e["category"], []).append(e)

    new_entries = []
    for cat, target in targets.items():
        need = target - current_by_cat.get(cat, 0)
        available = unused_by_cat.get(cat, [])
        if need <= 0:
            logger.info("Category %s: already at %d (target %d), need 0",
                        cat, current_by_cat.get(cat, 0), target)
            continue
        if need > len(available):
            logger.warning("Category %s: need %d but only %d available, taking all",
                           cat, need, len(available))
            need = len(available)

        # Domain-balanced selection within category
        selected = _domain_balanced_select(available, need, rng)
        new_entries.extend(selected)
        logger.info("Category %s: selected %d new (from %d available), total will be %d",
                     cat, len(selected), len(available),
                     current_by_cat.get(cat, 0) + len(selected))

    return new_entries


def _domain_balanced_select(pool, n, rng):
    """Select n entries from pool with proportional domain distribution."""
    by_domain = {}
    for e in pool:
        by_domain.setdefault(e["domain"][:3], []).append(e)

    # Shuffle each domain pool
    for entries in by_domain.values():
        rng.shuffle(entries)

    # Compute domain targets proportional to pool composition
    total_available = len(pool)
    domain_targets = {}
    for domain, entries in by_domain.items():
        # Proportional to the pool size in this domain
        domain_targets[domain] = round(n * len(entries) / total_available)

    # Adjust for rounding
    total_target = sum(domain_targets.values())
    if total_target < n:
        # Add extras to the largest domain
        largest = max(domain_targets, key=domain_targets.get)
        domain_targets[largest] += n - total_target
    elif total_target > n:
        largest = max(domain_targets, key=domain_targets.get)
        domain_targets[largest] -= total_target - n

    selected = []
    for domain, target in domain_targets.items():
        entries = by_domain.get(domain, [])
        take = min(target, len(entries))
        selected.extend(entries[:take])

    # If still short (some domain ran out), fill from remaining
    selected_pids = {e["patent_id"] for e in selected}
    remaining = [e for e in pool if e["patent_id"] not in selected_pids]
    rng.shuffle(remaining)
    while len(selected) < n and remaining:
        selected.append(remaining.pop(0))

    return selected[:n]


def build_tier1_qrels(patent_id, grades):
    """Build tier1 qrels from step5 grades (consensus >= 1)."""
    if patent_id not in grades:
        return {}
    return {
        doc_id: info["consensus"]
        for doc_id, info in grades[patent_id].items()
        if info.get("consensus", 0) >= 1
    }


def assign_ids(entries, start_id=101):
    """Assign RN IDs to new entries."""
    for i, entry in enumerate(entries):
        entry["statement_id"] = f"RN{start_id + i:03d}"
    return entries


def write_output_files(all_entries, grades, data_dir):
    """Write statements.jsonl, queries.jsonl, tier1.tsv, tier3.tsv."""
    novelty_to_score = {"NOVEL": 0, "PARTIALLY_ANTICIPATED": 1, "ANTICIPATED": 2, "PENDING": 0}

    statements = []
    for entry in all_entries:
        sid = entry["statement_id"]
        tier1_qrels = build_tier1_qrels(entry["patent_id"], grades)
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

    # statements.jsonl
    with open(data_dir / "statements.jsonl", "w") as f:
        for stmt in statements:
            f.write(json.dumps(stmt, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d statements)", data_dir / "statements.jsonl", len(statements))

    # queries.jsonl
    with open(data_dir / "queries.jsonl", "w") as f:
        for stmt in statements:
            f.write(json.dumps({"_id": stmt["statement_id"], "text": stmt["text"]}) + "\n")
    logger.info("Wrote %s (%d queries)", data_dir / "queries.jsonl", len(statements))

    # tier1.tsv
    qrels_dir = data_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)
    with open(qrels_dir / "tier1.tsv", "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for stmt in statements:
            rel = stmt["ground_truth"]["tier1_relevant_docs"]
            for doc_id in sorted(rel):
                f.write(f"{stmt['statement_id']}\t{doc_id}\t{rel[doc_id]}\n")
    n_tier1 = sum(len(s["ground_truth"]["tier1_relevant_docs"]) for s in statements)
    logger.info("Wrote %s (%d qrel entries)", qrels_dir / "tier1.tsv", n_tier1)

    # tier3.tsv
    with open(qrels_dir / "tier3.tsv", "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for stmt in statements:
            label = stmt["ground_truth"]["tier3_novelty_label"]
            score = novelty_to_score.get(label, 0)
            f.write(f"{stmt['statement_id']}\t{stmt['statement_id']}\t{score}\n")
    logger.info("Wrote %s (%d entries)", qrels_dir / "tier3.tsv", len(statements))


def write_novelty_queue(new_entries, output_path):
    """Write novelty queue for VPS (200 patents x 3 models = 600 API calls)."""
    queue = []
    for entry in new_entries:
        queue.append({
            "statement_id": entry["statement_id"],
            "patent_id": entry["patent_id"],
            "claim": entry["claim"],
            "title": entry["title"],
            "abstract": entry["abstract"],
            "domain": entry["domain"],
        })
    save_json(output_path, queue)
    logger.info("Novelty queue: %d patents x 3 models = %d API calls (~$%.2f)",
                len(queue), len(queue) * 3, len(queue) * 0.03)


def merge_novelty(step8_path, novelty_results_path):
    """Merge VPS novelty results back into step8_novelty.json and regenerate outputs."""
    step8 = load_json(step8_path)
    novelty_results = load_json(novelty_results_path)

    # Index novelty results by statement_id
    nov_by_sid = {r["statement_id"]: r["novelty"] for r in novelty_results}

    updated = 0
    for entry in step8:
        sid = entry["statement_id"]
        if sid in nov_by_sid:
            entry["novelty"] = nov_by_sid[sid]
            updated += 1

    save_json(step8_path, step8)
    logger.info("Merged novelty for %d entries into %s", updated, step8_path)
    return step8


def main():
    parser = argparse.ArgumentParser(description="Scale NovEx from 100 to 300 statements")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    parser.add_argument("--merge-novelty", type=str, help="Merge VPS novelty results from JSON file")
    args = parser.parse_args()

    # Load data
    step7 = load_json(REVERSE_DIR / "step7_selected.json")
    grades = load_json(REVERSE_DIR / "step5_grades.json")
    categorized = load_json(REVERSE_DIR / "checkpoints" / "step6_categorized.json")

    current_pids = {e["patent_id"] for e in step7} | EXCLUDED_PATENTS
    unused = [e for e in categorized if e["patent_id"] not in current_pids]
    logger.info("Current: %d, Excluded: %d, Unused pool: %d",
                len(step7), len(EXCLUDED_PATENTS), len(unused))

    # Handle merge mode separately
    if args.merge_novelty:
        step8 = merge_novelty(REVERSE_DIR / "step8_novelty.json", args.merge_novelty)
        write_output_files(step8, grades, DATA_DIR)
        # Verify
        cats = Counter(e["category"] for e in step8)
        doms = Counter(e["domain"] for e in step8)
        pending = sum(1 for e in step8 if e.get("novelty", {}).get("label") == "PENDING")
        logger.info("Final: %d statements, categories=%s, domains=%s, pending_novelty=%d",
                     len(step8), dict(cats), dict(doms), pending)
        return

    # Select new entries
    new_entries = select_new_entries(step7, unused, TARGETS)
    logger.info("Selected %d new entries", len(new_entries))

    # Assign IDs
    assign_ids(new_entries, start_id=101)

    # Validate no duplicates
    all_pids = {e["patent_id"] for e in step7} | {e["patent_id"] for e in new_entries}
    assert len(all_pids) == len(step7) + len(new_entries), "Duplicate patent IDs!"

    # Summary
    all_entries = step7 + new_entries
    cats = Counter(e["category"] for e in all_entries)
    doms = Counter(e["domain"] for e in all_entries)
    logger.info("Total: %d statements", len(all_entries))
    logger.info("Categories: %s", dict(cats))
    logger.info("Domains: %s", dict(doms))

    # Check domain balance
    for dom, weight in DOMAIN_WEIGHTS.items():
        actual = doms.get(dom, 0) / len(all_entries)
        logger.info("Domain %s: target=%.1f%%, actual=%.1f%% (%d)",
                     dom, weight * 100, actual * 100, doms.get(dom, 0))

    if not args.apply:
        logger.info("Dry run complete. Use --apply to write changes.")
        # Print new entry summary
        for e in new_entries:
            logger.info("  %s: patent=%s domain=%s cat=%s title='%s'",
                         e["statement_id"], e["patent_id"], e["domain"],
                         e["category"], e["title"][:50])
        return

    # === Apply ===

    # Update step7_selected.json (add new entries)
    save_json(REVERSE_DIR / "step7_selected.json", all_entries)

    # Update step8_novelty.json (new entries with PENDING novelty)
    step8 = load_json(REVERSE_DIR / "step8_novelty.json")
    for entry in new_entries:
        step8_entry = dict(entry)
        step8_entry["novelty"] = {"label": "PENDING", "agreement": "pending", "individual": {}}
        step8.append(step8_entry)
    save_json(REVERSE_DIR / "step8_novelty.json", step8)

    # Write output files
    write_output_files(step8, grades, DATA_DIR)

    # Write novelty queue
    write_novelty_queue(new_entries, REVERSE_DIR / "novelty_queue_scale.json")

    # Verification
    dup_check = [e["patent_id"] for e in step8]
    if len(dup_check) != len(set(dup_check)):
        logger.error("DUPLICATE patent IDs detected!")
    else:
        logger.info("No duplicate patent IDs. All %d unique.", len(dup_check))


if __name__ == "__main__":
    main()
