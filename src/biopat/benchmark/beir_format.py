"""BEIR format output module.

Formats the benchmark data into BEIR-compatible JSON Lines format
for use with standard IR evaluation tools.

Phase 5 (v2.0): Extended to support dual-corpus (papers + patents) with
doc_type field for cross-type evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import polars as pl

logger = logging.getLogger(__name__)

# Document type constants for v2.0 dual-corpus
DOC_TYPE_PAPER = "paper"
DOC_TYPE_PATENT = "patent"


class BEIRFormatter:
    """Formatter for BEIR-compatible benchmark output."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def corpus_path(self) -> Path:
        return self.output_dir / "corpus.jsonl"

    @property
    def queries_path(self) -> Path:
        return self.output_dir / "queries.jsonl"

    @property
    def qrels_dir(self) -> Path:
        return self.output_dir / "qrels"

    def format_corpus(
        self,
        papers_df: pl.DataFrame,
        id_col: str = "paper_id",
        title_col: str = "title",
        text_col: str = "abstract",
        doc_type: str = DOC_TYPE_PAPER,
        date_col: Optional[str] = None,
    ) -> None:
        """Format and write corpus to BEIR JSON Lines format.

        BEIR corpus format (v2.0 extended):
        {"_id": "doc_id", "title": "...", "text": "...", "doc_type": "paper|patent", "date": "..."}

        Args:
            papers_df: Papers DataFrame.
            id_col: Column containing document IDs.
            title_col: Column containing titles.
            text_col: Column containing main text (abstract).
            doc_type: Document type for all entries ("paper" or "patent").
            date_col: Optional column containing publication/priority date.
        """
        logger.info(f"Writing {len(papers_df)} {doc_type} documents to {self.corpus_path}")

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for row in papers_df.iter_rows(named=True):
                doc = {
                    "_id": str(row[id_col]),
                    "title": row.get(title_col, "") or "",
                    "text": row.get(text_col, "") or "",
                    "doc_type": doc_type,
                }
                # Add date if available
                if date_col and date_col in row:
                    doc["date"] = str(row[date_col]) if row[date_col] else None
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info(f"Wrote corpus to {self.corpus_path}")

    def format_dual_corpus(
        self,
        papers_df: pl.DataFrame,
        patents_df: pl.DataFrame,
        paper_id_col: str = "paper_id",
        paper_title_col: str = "title",
        paper_text_col: str = "abstract",
        paper_date_col: str = "publication_date",
        patent_id_col: str = "patent_id",
        patent_title_col: str = "title",
        patent_text_col: str = "corpus_text",
        patent_date_col: str = "priority_date",
    ) -> Dict[str, int]:
        """Format and write dual corpus (papers + patents) to BEIR format.

        v2.0 feature: Combines scientific papers and prior patents into
        a single corpus with doc_type field for cross-type evaluation.

        Args:
            papers_df: Papers DataFrame.
            patents_df: Patents DataFrame (with corpus_text = abstract + claim).
            paper_id_col: Paper ID column.
            paper_title_col: Paper title column.
            paper_text_col: Paper text column.
            paper_date_col: Paper date column.
            patent_id_col: Patent ID column.
            patent_title_col: Patent title column.
            patent_text_col: Patent text column (abstract + first independent claim).
            patent_date_col: Patent date column.

        Returns:
            Dictionary with corpus statistics by doc_type.
        """
        logger.info(
            f"Writing dual corpus: {len(papers_df)} papers + {len(patents_df)} patents"
        )

        paper_count = 0
        patent_count = 0

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            # Write papers
            for row in papers_df.iter_rows(named=True):
                doc = {
                    "_id": str(row[paper_id_col]),
                    "title": row.get(paper_title_col, "") or "",
                    "text": row.get(paper_text_col, "") or "",
                    "doc_type": DOC_TYPE_PAPER,
                    "date": str(row.get(paper_date_col, "")) if row.get(paper_date_col) else None,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                paper_count += 1

            # Write patents
            for row in patents_df.iter_rows(named=True):
                doc = {
                    "_id": str(row[patent_id_col]),
                    "title": row.get(patent_title_col, "") or "",
                    "text": row.get(patent_text_col, "") or "",
                    "doc_type": DOC_TYPE_PATENT,
                    "date": str(row.get(patent_date_col, "")) if row.get(patent_date_col) else None,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                patent_count += 1

        stats = {
            "papers": paper_count,
            "patents": patent_count,
            "total": paper_count + patent_count,
        }
        logger.info(f"Wrote dual corpus: {stats}")
        return stats

    @staticmethod
    def create_patent_corpus_text(
        abstract: Optional[str],
        first_independent_claim: Optional[str],
    ) -> str:
        """Create corpus text for a patent by combining abstract and first claim.

        v2.0 feature: Patent corpus entries consist of Abstract + first
        independent claim to provide comparable textual length to papers.

        Args:
            abstract: Patent abstract text.
            first_independent_claim: Text of first independent claim.

        Returns:
            Combined corpus text.
        """
        parts = []
        if abstract:
            parts.append(abstract.strip())
        if first_independent_claim:
            # Add separator and claim
            if parts:
                parts.append("\n\n[CLAIM 1]\n")
            parts.append(first_independent_claim.strip())
        return "".join(parts)

    def format_queries(
        self,
        queries_df: pl.DataFrame,
        id_col: str = "query_id",
        text_col: str = "claim_text",
        metadata_cols: Optional[List[str]] = None,
    ) -> None:
        """Format and write queries to BEIR JSON Lines format.

        BEIR queries format:
        {"_id": "query_id", "text": "..."}

        Args:
            queries_df: Queries DataFrame.
            id_col: Column containing query IDs.
            text_col: Column containing query text (claim text).
            metadata_cols: Optional additional columns to include as metadata.
        """
        logger.info(f"Writing {len(queries_df)} queries to {self.queries_path}")

        with open(self.queries_path, "w", encoding="utf-8") as f:
            for row in queries_df.iter_rows(named=True):
                query = {
                    "_id": str(row[id_col]),
                    "text": row.get(text_col, "") or "",
                }

                # Add optional metadata
                if metadata_cols:
                    for col in metadata_cols:
                        if col in row:
                            query[col] = row[col]

                f.write(json.dumps(query, ensure_ascii=False) + "\n")

        logger.info(f"Wrote queries to {self.queries_path}")

    def format_qrels(
        self,
        train_qrels: pl.DataFrame,
        dev_qrels: pl.DataFrame,
        test_qrels: pl.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        relevance_col: str = "relevance",
    ) -> None:
        """Format and write qrels to BEIR TSV format.

        BEIR qrels format (TSV):
        query_id<tab>doc_id<tab>score

        Args:
            train_qrels: Train split qrels.
            dev_qrels: Dev split qrels.
            test_qrels: Test split qrels.
            query_col: Query ID column.
            doc_col: Document ID column.
            relevance_col: Relevance score column.
        """
        self.qrels_dir.mkdir(parents=True, exist_ok=True)

        splits = [
            ("train", train_qrels),
            ("dev", dev_qrels),
            ("test", test_qrels),
        ]

        for split_name, qrels_df in splits:
            if len(qrels_df) == 0:
                logger.warning(f"No qrels for {split_name} split")
                continue

            output_path = self.qrels_dir / f"{split_name}.tsv"
            logger.info(f"Writing {len(qrels_df)} qrels to {output_path}")

            with open(output_path, "w", encoding="utf-8") as f:
                for row in qrels_df.iter_rows(named=True):
                    query_id = str(row[query_col])
                    doc_id = str(row[doc_col])
                    score = int(row[relevance_col])
                    f.write(f"{query_id}\t{doc_id}\t{score}\n")

            logger.info(f"Wrote {split_name} qrels to {output_path}")

    def format_single_qrels(
        self,
        qrels_df: pl.DataFrame,
        split_name: str,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        relevance_col: str = "relevance",
    ) -> None:
        """Format and write qrels for a single split.

        Args:
            qrels_df: Qrels DataFrame.
            split_name: Name of split ('train', 'dev', 'test').
            query_col: Query ID column.
            doc_col: Document ID column.
            relevance_col: Relevance score column.
        """
        self.qrels_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.qrels_dir / f"{split_name}.tsv"
        logger.info(f"Writing {len(qrels_df)} qrels to {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            for row in qrels_df.iter_rows(named=True):
                query_id = str(row[query_col])
                doc_id = str(row[doc_col])
                score = int(row[relevance_col])
                f.write(f"{query_id}\t{doc_id}\t{score}\n")

    def validate_output(self) -> dict:
        """Validate the BEIR output files.

        Returns:
            Dictionary with validation results.
        """
        results = {
            "corpus_exists": self.corpus_path.exists(),
            "queries_exists": self.queries_path.exists(),
            "qrels_dir_exists": self.qrels_dir.exists(),
        }

        # Count records and doc_types
        if results["corpus_exists"]:
            paper_count = 0
            patent_count = 0
            total_count = 0
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    total_count += 1
                    try:
                        doc = json.loads(line)
                        doc_type = doc.get("doc_type", DOC_TYPE_PAPER)
                        if doc_type == DOC_TYPE_PAPER:
                            paper_count += 1
                        elif doc_type == DOC_TYPE_PATENT:
                            patent_count += 1
                    except json.JSONDecodeError:
                        pass
            results["corpus_count"] = total_count
            results["corpus_papers"] = paper_count
            results["corpus_patents"] = patent_count
            results["is_dual_corpus"] = patent_count > 0

        if results["queries_exists"]:
            with open(self.queries_path, "r", encoding="utf-8") as f:
                results["queries_count"] = sum(1 for _ in f)

        # Check qrels splits
        if results["qrels_dir_exists"]:
            results["qrels_splits"] = {}
            for split in ["train", "dev", "test"]:
                split_path = self.qrels_dir / f"{split}.tsv"
                if split_path.exists():
                    with open(split_path, "r", encoding="utf-8") as f:
                        results["qrels_splits"][split] = sum(1 for _ in f)

        # Validate JSONL format
        if results["corpus_exists"]:
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    doc = json.loads(first_line)
                    results["corpus_valid_json"] = "_id" in doc and "text" in doc
                    results["corpus_has_doc_type"] = "doc_type" in doc
            except (json.JSONDecodeError, KeyError):
                results["corpus_valid_json"] = False
                results["corpus_has_doc_type"] = False

        if results["queries_exists"]:
            try:
                with open(self.queries_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    query = json.loads(first_line)
                    results["queries_valid_json"] = "_id" in query and "text" in query
            except (json.JSONDecodeError, KeyError):
                results["queries_valid_json"] = False

        logger.info(f"Validation results: {results}")
        return results

    def get_benchmark_stats(self) -> dict:
        """Get statistics about the generated benchmark.

        Returns:
            Dictionary with benchmark statistics.
        """
        validation = self.validate_output()

        stats = {
            "corpus_size": validation.get("corpus_count", 0),
            "queries_size": validation.get("queries_count", 0),
        }

        # v2.0: Add doc_type breakdown
        if validation.get("is_dual_corpus"):
            stats["corpus_papers"] = validation.get("corpus_papers", 0)
            stats["corpus_patents"] = validation.get("corpus_patents", 0)
            stats["is_dual_corpus"] = True
        else:
            stats["is_dual_corpus"] = False

        # Qrels per split
        qrels_splits = validation.get("qrels_splits", {})
        for split, count in qrels_splits.items():
            stats[f"{split}_qrels"] = count

        stats["total_qrels"] = sum(qrels_splits.values())

        return stats

    def format_benchmark(
        self,
        corpus_df: pl.DataFrame,
        queries_df: pl.DataFrame,
        splits: Dict[str, Tuple[pl.DataFrame, pl.DataFrame]],
        corpus_id_col: str = "paper_id",
        corpus_title_col: str = "title",
        corpus_text_col: str = "abstract",
        query_id_col: str = "query_id",
        query_text_col: str = "claim_text",
        qrel_query_col: str = "query_id",
        qrel_doc_col: str = "doc_id",
        qrel_relevance_col: str = "relevance",
    ) -> dict:
        """Format complete benchmark in BEIR format.

        Args:
            corpus_df: Corpus DataFrame.
            queries_df: Queries DataFrame (all queries).
            splits: Dictionary with 'train', 'dev', 'test' keys containing
                    (queries_df, qrels_df) tuples.
            corpus_id_col: Corpus ID column.
            corpus_title_col: Corpus title column.
            corpus_text_col: Corpus text column.
            query_id_col: Query ID column.
            query_text_col: Query text column.
            qrel_query_col: Qrel query ID column.
            qrel_doc_col: Qrel document ID column.
            qrel_relevance_col: Qrel relevance column.

        Returns:
            Benchmark statistics.
        """
        logger.info("Formatting benchmark in BEIR format")

        # Write corpus
        self.format_corpus(
            corpus_df,
            id_col=corpus_id_col,
            title_col=corpus_title_col,
            text_col=corpus_text_col,
        )

        # Write queries
        self.format_queries(
            queries_df,
            id_col=query_id_col,
            text_col=query_text_col,
        )

        # Write qrels for each split
        train_qrels = splits.get("train", (None, pl.DataFrame()))[1]
        dev_qrels = splits.get("dev", (None, pl.DataFrame()))[1]
        test_qrels = splits.get("test", (None, pl.DataFrame()))[1]

        self.format_qrels(
            train_qrels,
            dev_qrels,
            test_qrels,
            query_col=qrel_query_col,
            doc_col=qrel_doc_col,
            relevance_col=qrel_relevance_col,
        )

        # Validate and return stats
        self.validate_output()
        stats = self.get_benchmark_stats()

        logger.info(f"Benchmark formatted: {stats}")
        return stats

    def load_corpus(self) -> List[dict]:
        """Load corpus from BEIR format.

        Returns:
            List of document dictionaries.
        """
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {self.corpus_path}")

        corpus = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                corpus.append(json.loads(line))
        return corpus

    def load_queries(self) -> List[dict]:
        """Load queries from BEIR format.

        Returns:
            List of query dictionaries.
        """
        if not self.queries_path.exists():
            raise FileNotFoundError(f"Queries not found at {self.queries_path}")

        queries = []
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                queries.append(json.loads(line))
        return queries

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, int]]:
        """Load qrels from BEIR format.

        Args:
            split: Split name ('train', 'dev', 'test').

        Returns:
            Nested dictionary: {query_id: {doc_id: relevance}}.
        """
        qrels_path = self.qrels_dir / f"{split}.tsv"
        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels not found at {qrels_path}")

        qrels = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = score

        return qrels
