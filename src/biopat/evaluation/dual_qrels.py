"""Dual-corpus data preparation for agent-based retrieval evaluation.

Builds a unified corpus of papers + patents and supports bidirectional
qrel generation (patent→paper forward, paper→patent inverted).
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DOC_TYPE_PAPER = "paper"
DOC_TYPE_PATENT = "patent"


def build_dual_corpus(
    benchmark_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """Merge papers and patent claims into a single corpus with doc_type metadata.

    Reads the existing BEIR-format corpus.jsonl and queries.jsonl. Papers are
    the original corpus docs; patents are derived from queries (patent claims).

    Args:
        benchmark_dir: Path to BEIR-format benchmark directory containing
            corpus.jsonl and queries.jsonl.
        output_dir: If provided, writes dual_corpus.jsonl here.

    Returns:
        Dict mapping doc_id to {title, text, doc_type}.
    """
    benchmark_path = Path(benchmark_dir)
    dual_corpus: Dict[str, dict] = {}

    # Load existing corpus (papers)
    corpus_path = benchmark_path / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    paper_count = 0
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["_id"]
            dual_corpus[doc_id] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "doc_type": DOC_TYPE_PAPER,
            }
            paper_count += 1

    # Load queries (patent claims) and add as patent documents
    queries_path = benchmark_path / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries not found: {queries_path}")

    patent_count = 0
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            query_id = q["_id"]
            # Prefix patent docs to avoid ID collisions with paper IDs
            patent_doc_id = f"patent_{query_id}"
            dual_corpus[patent_doc_id] = {
                "title": f"Patent Claim {query_id}",
                "text": q["text"],
                "doc_type": DOC_TYPE_PATENT,
            }
            patent_count += 1

    logger.info(
        "Built dual corpus: %d papers + %d patents = %d total",
        paper_count, patent_count, len(dual_corpus),
    )

    # Write to disk if output_dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        corpus_out = out_path / "dual_corpus.jsonl"
        with open(corpus_out, "w") as f:
            for doc_id, doc in dual_corpus.items():
                f.write(json.dumps({"_id": doc_id, **doc}) + "\n")
        logger.info("Dual corpus written to %s", corpus_out)

    return dual_corpus


def invert_qrels(
    benchmark_dir: str,
    split: str = "test",
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """Flip patent→paper qrels to paper→patent direction.

    For each qrel (patent_query → paper_doc : relevance), creates an
    inverted qrel (paper_doc → patent_patent_query : relevance).

    Args:
        benchmark_dir: Path to BEIR-format benchmark directory.
        split: Which qrels split to invert ("train", "dev", "test").
        output_dir: If provided, writes inverted qrels TSV here.

    Returns:
        Inverted qrels: {paper_id: {patent_doc_id: relevance}}.
    """
    benchmark_path = Path(benchmark_dir)
    qrels_path = benchmark_path / "qrels" / f"{split}.tsv"

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")

    # Read forward qrels
    forward: List[Tuple[str, str, int]] = []
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(parts[2])
                forward.append((qid, did, score))

    # Invert: paper becomes query, patent becomes document
    inverted: Dict[str, Dict[str, int]] = {}
    for patent_qid, paper_did, relevance in forward:
        patent_doc_id = f"patent_{patent_qid}"
        if paper_did not in inverted:
            inverted[paper_did] = {}
        # Keep highest relevance if duplicated
        existing = inverted[paper_did].get(patent_doc_id, 0)
        inverted[paper_did][patent_doc_id] = max(existing, relevance)

    logger.info(
        "Inverted qrels: %d forward → %d paper queries, %d total rels",
        len(forward), len(inverted), sum(len(v) for v in inverted.values()),
    )

    # Write to disk
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        qrels_out = out_path / f"inverted_{split}.tsv"
        with open(qrels_out, "w") as f:
            for paper_qid, rels in sorted(inverted.items()):
                for patent_did, score in sorted(rels.items()):
                    f.write(f"{paper_qid}\t{patent_did}\t{score}\n")
        logger.info("Inverted qrels written to %s", qrels_out)

    return inverted


def select_type_b_queries(
    benchmark_dir: str,
    inverted_qrels: Dict[str, Dict[str, int]],
    min_patents: int = 2,
    min_abstract_len: int = 50,
    max_queries: int = 200,
    seed: int = 42,
) -> Dict[str, str]:
    """Select papers as Type B queries for paper→patent retrieval.

    Selects papers with sufficiently long abstracts that are cited by
    at least `min_patents` patents, providing a richer evaluation signal.

    Args:
        benchmark_dir: Path to BEIR-format benchmark directory.
        inverted_qrels: Output of invert_qrels().
        min_patents: Minimum number of citing patents per paper.
        min_abstract_len: Minimum abstract length in characters.
        max_queries: Maximum number of Type B queries to select.
        seed: Random seed for reproducible sampling.

    Returns:
        Dict mapping paper_id to abstract text.
    """
    benchmark_path = Path(benchmark_dir)

    # Load paper corpus for abstract text
    corpus_path = benchmark_path / "corpus.jsonl"
    corpus: Dict[str, str] = {}
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc.get("text", "")

    # Filter candidates: must have enough citing patents and long enough abstract
    candidates: Dict[str, str] = {}
    for paper_id, patent_rels in inverted_qrels.items():
        if len(patent_rels) < min_patents:
            continue
        abstract = corpus.get(paper_id, "")
        if len(abstract) < min_abstract_len:
            continue
        candidates[paper_id] = abstract

    logger.info(
        "Type B candidates: %d papers with %d+ citing patents and abstract >= %d chars",
        len(candidates), min_patents, min_abstract_len,
    )

    # Stratified sample if too many
    if len(candidates) > max_queries:
        rng = random.Random(seed)
        sampled_ids = rng.sample(sorted(candidates.keys()), max_queries)
        candidates = {pid: candidates[pid] for pid in sampled_ids}

    logger.info("Selected %d Type B queries", len(candidates))
    return candidates


def load_doc_types(dual_corpus_path: str) -> Dict[str, str]:
    """Load doc_type mapping from a dual_corpus.jsonl file.

    Args:
        dual_corpus_path: Path to dual_corpus.jsonl.

    Returns:
        Dict mapping doc_id to doc_type ("paper" or "patent").
    """
    doc_types: Dict[str, str] = {}
    with open(dual_corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            doc_types[doc["_id"]] = doc.get("doc_type", DOC_TYPE_PAPER)
    return doc_types
