"""NovEx Benchmark loader — BEIR-compatible, all 3 tiers."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from biopat.novex._util import read_qrels_tsv

logger = logging.getLogger(__name__)

NOVELTY_MAP = {0: "NOVEL", 1: "PARTIALLY_ANTICIPATED", 2: "ANTICIPATED"}


@dataclass
class NovExStatement:
    statement_id: str
    text: str
    source_paper_id: str
    source_paper_title: str
    domain: str
    difficulty: str
    category: str
    num_citing_patents: int
    ground_truth: Dict[str, Any] = field(default_factory=dict)


class NovExBenchmark:
    """BEIR-compatible benchmark for 3-tier NovEx evaluation.

    Usage:
        b = NovExBenchmark("data/novex")
        b.load()
        corpus, queries, qrels = b.get_beir_format(tier=1)
    """

    def __init__(self, data_dir: str = "data/novex", corpus_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.corpus_dir = Path(corpus_dir) if corpus_dir else self.data_dir
        self.statements: Dict[str, NovExStatement] = {}
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.queries: Dict[str, str] = {}
        self.tier1_qrels: Dict[str, Dict[str, int]] = {}
        self.tier2_qrels: Dict[str, Dict[str, int]] = {}
        self.tier3_qrels: Dict[str, Dict[str, int]] = {}
        self.tier3_labels: Dict[str, str] = {}
        self.doc_types: Dict[str, str] = {}
        self._loaded = False

    def load(self, load_corpus: bool = True) -> None:
        # Statements
        stmt_path = self.data_dir / "statements.jsonl"
        if stmt_path.exists():
            with open(stmt_path) as f:
                for line in f:
                    d = json.loads(line)
                    s = NovExStatement(
                        statement_id=d["statement_id"], text=d["text"],
                        source_paper_id=d["source_paper_id"],
                        source_paper_title=d.get("source_paper_title", ""),
                        domain=d["domain"], difficulty=d.get("difficulty", "medium"),
                        category=d["category"],
                        num_citing_patents=d.get("num_citing_patents", 0),
                        ground_truth=d.get("ground_truth", {}),
                    )
                    self.statements[s.statement_id] = s
                    self.queries[s.statement_id] = s.text

        # Corpus
        if load_corpus:
            for p in [
                self.corpus_dir / "dual_corpus.jsonl",
                self.corpus_dir / "corpus.jsonl",
                self.data_dir.parent / "benchmark" / "dual_corpus.jsonl",
                self.data_dir.parent / "benchmark" / "corpus.jsonl",
            ]:
                if p.exists():
                    with open(p) as f:
                        for line in f:
                            doc = json.loads(line)
                            self.corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc.get("text", "")}
                            if "doc_type" in doc:
                                self.doc_types[doc["_id"]] = doc["doc_type"]
                    break

        # Qrels — load from TSV files if they exist
        qd = self.data_dir / "qrels"
        if qd.exists():
            for name, attr in [("tier1", "tier1_qrels"), ("tier2", "tier2_qrels"), ("tier3", "tier3_qrels")]:
                p = qd / f"{name}.tsv"
                if p.exists():
                    setattr(self, attr, read_qrels_tsv(p))
            for qid, docs in self.tier3_qrels.items():
                for _, score in docs.items():
                    self.tier3_labels[qid] = NOVELTY_MAP.get(score, "NOVEL")

        # Fallback: populate from statement ground_truth when TSV files don't exist
        if not self.tier1_qrels and self.statements:
            for sid, stmt in self.statements.items():
                gt = stmt.ground_truth
                rel_docs = gt.get("tier1_relevant_docs", [])
                if rel_docs:
                    self.tier1_qrels[sid] = {d: 1 for d in rel_docs}
            logger.info("Populated tier1_qrels from statements: %d queries",
                        len(self.tier1_qrels))

        if not self.tier2_qrels and self.tier1_qrels:
            for qid, docs in self.tier1_qrels.items():
                relevant = {did: score for did, score in docs.items() if score >= 1}
                if relevant:
                    self.tier2_qrels[qid] = relevant
            logger.info("Populated tier2_qrels from tier1_qrels: %d queries, %d pairs",
                        len(self.tier2_qrels),
                        sum(len(v) for v in self.tier2_qrels.values()))

        if not self.tier3_labels and self.statements:
            label_to_score = {"NOVEL": 0, "PARTIALLY_ANTICIPATED": 1, "ANTICIPATED": 2}
            for sid, stmt in self.statements.items():
                gt = stmt.ground_truth
                label = gt.get("tier3_novelty_label")
                if label:
                    self.tier3_labels[sid] = label
                    self.tier3_qrels[sid] = {sid: label_to_score.get(label, 0)}
            logger.info("Populated tier3_labels from statements: %d labels",
                        len(self.tier3_labels))

        self._loaded = True
        logger.info("NovEx: %d statements, %d docs, T1=%d T2=%d T3=%d qrels",
                     len(self.statements), len(self.corpus),
                     sum(len(v) for v in self.tier1_qrels.values()),
                     sum(len(v) for v in self.tier2_qrels.values()),
                     len(self.tier3_labels))

    def get_beir_format(self, tier: int = 1):
        qmap = {1: self.tier1_qrels, 2: self.tier2_qrels, 3: self.tier3_qrels}
        return self.corpus, self.queries, qmap.get(tier, self.tier1_qrels)

    def filter(self, predicate: Callable[[NovExStatement], bool]) -> "NovExBenchmark":
        """Generic filtered view. Usage: b.filter(lambda s: s.domain.startswith('A61'))"""
        f = NovExBenchmark.__new__(NovExBenchmark)
        f.data_dir, f.corpus_dir = self.data_dir, self.corpus_dir
        f.corpus, f.doc_types = self.corpus, self.doc_types
        f.statements = {k: v for k, v in self.statements.items() if predicate(v)}
        ids = set(f.statements)
        f.queries = {k: v for k, v in self.queries.items() if k in ids}
        f.tier1_qrels = {k: v for k, v in self.tier1_qrels.items() if k in ids}
        f.tier2_qrels = {k: v for k, v in self.tier2_qrels.items() if k in ids}
        f.tier3_qrels = {k: v for k, v in self.tier3_qrels.items() if k in ids}
        f.tier3_labels = {k: v for k, v in self.tier3_labels.items() if k in ids}
        f._loaded = True
        return f

    def get_stats(self) -> Dict[str, Any]:
        from collections import Counter
        stmts = list(self.statements.values())
        return {
            "num_statements": len(stmts),
            "num_corpus_docs": len(self.corpus),
            "domains": dict(Counter(s.domain for s in stmts)),
            "categories": dict(Counter(s.category for s in stmts)),
            "difficulties": dict(Counter(s.difficulty for s in stmts)),
            "doc_types": dict(Counter(self.doc_types.values())),
            "tier1_qrels": sum(len(v) for v in self.tier1_qrels.values()),
            "tier2_qrels": sum(len(v) for v in self.tier2_qrels.values()),
            "tier3_labels": len(self.tier3_labels),
        }

    def write_beir_format(self, output_dir: Optional[str] = None) -> None:
        out = Path(output_dir) if output_dir else self.data_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "qrels").mkdir(exist_ok=True)
        with open(out / "queries.jsonl", "w") as f:
            for sid, text in self.queries.items():
                f.write(json.dumps({"_id": sid, "text": text}) + "\n")
        for name, qrels in [("tier1", self.tier1_qrels), ("tier2", self.tier2_qrels), ("tier3", self.tier3_qrels)]:
            with open(out / "qrels" / f"{name}.tsv", "w") as f:
                f.write("query_id\tdoc_id\tscore\n")
                for qid in sorted(qrels):
                    for did, score in sorted(qrels[qid].items()):
                        f.write(f"{qid}\t{did}\t{score}\n")
