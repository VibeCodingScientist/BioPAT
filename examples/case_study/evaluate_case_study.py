#!/usr/bin/env python3
"""Evaluate retrieval performance on the BioPAT case study dataset.

This script demonstrates how to:
1. Load the case study data (corpus, queries, qrels)
2. Run a simple BM25 baseline retrieval
3. Compute standard IR evaluation metrics
4. Display per-query analysis

Usage:
    python evaluate_case_study.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_corpus(path: Path) -> Dict[str, dict]:
    """Load corpus from JSONL file."""
    corpus = {}
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    return corpus


def load_queries(path: Path) -> Dict[str, dict]:
    """Load queries from JSONL file."""
    queries = {}
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q
    return queries


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Load relevance judgments from TSV file."""
    qrels = {}
    with open(path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = rel
    return qrels


class SimpleBM25:
    """Simple BM25 implementation for demonstration."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lens = {}
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.index = defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def index_corpus(self, corpus: Dict[str, dict]):
        """Build inverted index from corpus."""
        total_len = 0

        for doc_id, doc in corpus.items():
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = self._tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)

            # Build inverted index
            seen_terms = set()
            for token in tokens:
                if token not in seen_terms:
                    self.doc_freqs[token] += 1
                    seen_terms.add(token)
                self.index[token].append(doc_id)

        self.corpus_size = len(corpus)
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Search for documents matching query."""
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self.index:
                continue

            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            # Count term frequency in each doc
            tf_in_docs = defaultdict(int)
            for doc_id in self.index[token]:
                tf_in_docs[doc_id] += 1

            for doc_id, tf in tf_in_docs.items():
                doc_len = self.doc_lens[doc_id]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )
                scores[doc_id] += idf * tf_norm

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def ndcg_at_k(retrieved: List[str], relevances: Dict[str, int], k: int) -> float:
    """Compute NDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        rel = relevances.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    # Ideal DCG
    sorted_rels = sorted(relevances.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(sorted_rels, 1))

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Precision@k."""
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def main():
    """Run evaluation on case study dataset."""
    print("=" * 60)
    print("BioPAT Case Study Evaluation")
    print("=" * 60)
    print()

    # Paths
    case_study_dir = Path(__file__).parent
    corpus_path = case_study_dir / "corpus.jsonl"
    queries_path = case_study_dir / "queries.jsonl"
    qrels_path = case_study_dir / "qrels.tsv"

    # Load data
    print("Loading data...")
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} patents")
    print(f"  Judgments: {sum(len(v) for v in qrels.values())} total")
    print()

    # Build BM25 index
    print("Building BM25 index...")
    bm25 = SimpleBM25()
    bm25.index_corpus(corpus)
    print(f"  Vocabulary: {len(bm25.doc_freqs)} terms")
    print()

    # Evaluate
    print("Running retrieval and evaluation...")
    print()

    all_ndcg_10 = []
    all_p_10 = []
    all_r_100 = []
    all_mrr = []

    print("-" * 60)
    print(f"{'Query ID':<12} {'NDCG@10':>10} {'P@10':>10} {'R@100':>10} {'MRR':>10}")
    print("-" * 60)

    for qid, query in sorted(queries.items()):
        # Get query text
        query_text = query["text"]

        # Retrieve
        results = bm25.search(query_text, top_k=100)
        retrieved_ids = [doc_id for doc_id, _ in results]

        # Get relevance info
        query_qrels = qrels.get(qid, {})
        relevant_ids = set(doc_id for doc_id, rel in query_qrels.items() if rel > 0)

        # Compute metrics
        ndcg = ndcg_at_k(retrieved_ids, query_qrels, k=10)
        p10 = precision_at_k(retrieved_ids, relevant_ids, k=10)
        r100 = recall_at_k(retrieved_ids, relevant_ids, k=100)
        rr = mrr(retrieved_ids, relevant_ids)

        all_ndcg_10.append(ndcg)
        all_p_10.append(p10)
        all_r_100.append(r100)
        all_mrr.append(rr)

        print(f"{qid:<12} {ndcg:>10.4f} {p10:>10.4f} {r100:>10.4f} {rr:>10.4f}")

    print("-" * 60)

    # Compute averages
    avg_ndcg = sum(all_ndcg_10) / len(all_ndcg_10)
    avg_p10 = sum(all_p_10) / len(all_p_10)
    avg_r100 = sum(all_r_100) / len(all_r_100)
    avg_mrr = sum(all_mrr) / len(all_mrr)

    print(f"{'AVERAGE':<12} {avg_ndcg:>10.4f} {avg_p10:>10.4f} {avg_r100:>10.4f} {avg_mrr:>10.4f}")
    print("-" * 60)
    print()

    # Summary
    print("=" * 60)
    print("Summary Results (BM25 Baseline)")
    print("=" * 60)
    print(f"NDCG@10:  {avg_ndcg:.4f}")
    print(f"P@10:     {avg_p10:.4f}")
    print(f"R@100:    {avg_r100:.4f}")
    print(f"MRR:      {avg_mrr:.4f}")
    print()

    # Per-relevance analysis
    print("=" * 60)
    print("Retrieval Analysis by Relevance Level")
    print("=" * 60)

    relevance_found = {3: 0, 2: 0, 1: 0}
    relevance_total = {3: 0, 2: 0, 1: 0}

    for qid, query in queries.items():
        results = bm25.search(query["text"], top_k=100)
        retrieved_ids = set(doc_id for doc_id, _ in results)

        for doc_id, rel in qrels.get(qid, {}).items():
            if rel > 0:
                relevance_total[rel] += 1
                if doc_id in retrieved_ids:
                    relevance_found[rel] += 1

    print(f"{'Relevance':<15} {'Found@100':<12} {'Total':<10} {'Recall':<10}")
    print("-" * 47)
    for rel in [3, 2, 1]:
        found = relevance_found[rel]
        total = relevance_total[rel]
        recall = found / total if total > 0 else 0
        level = {3: "Novelty-dest.", 2: "Highly rel.", 1: "Relevant"}[rel]
        print(f"{level:<15} {found:<12} {total:<10} {recall:.4f}")
    print()

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
