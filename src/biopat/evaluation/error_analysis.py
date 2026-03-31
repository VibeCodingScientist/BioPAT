"""Error analysis module.

Implements error analysis for characterizing retrieval failures.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of retrieval failures."""

    VOCABULARY_MISMATCH = "vocabulary_mismatch"
    ABSTRACTION_LEVEL = "abstraction_level"
    CROSS_DOMAIN = "cross_domain"
    SEMANTIC_GAP = "semantic_gap"
    FALSE_NEGATIVE = "false_negative"
    LOW_RELEVANCE = "low_relevance"
    UNKNOWN = "unknown"


@dataclass
class FailureCase:
    """A single failure case for analysis."""

    query_id: str
    query_text: str
    relevant_doc_id: str
    relevant_doc_text: str
    relevance_score: int
    retrieved_rank: Optional[int]
    retrieved_score: Optional[float]
    category: FailureCategory = FailureCategory.UNKNOWN
    notes: str = ""


@dataclass
class ErrorAnalysisConfig:
    """Configuration for error analysis."""

    sample_size: int = 100  # Number of failures to sample
    min_relevance: int = 1  # Minimum relevance to consider as failure
    rank_threshold: int = 100  # Rank threshold for failure
    seed: int = 42


class ErrorAnalyzer:
    """Analyzer for retrieval errors."""

    def __init__(
        self,
        config: Optional[ErrorAnalysisConfig] = None,
    ):
        """Initialize analyzer.

        Args:
            config: Error analysis configuration.
        """
        self.config = config or ErrorAnalysisConfig()
        random.seed(self.config.seed)

    def identify_failures(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, str, int, Optional[int]]]:
        """Identify retrieval failures.

        A failure is when a relevant document is not retrieved in top-k
        or is ranked below the threshold.

        Args:
            results: Retrieved results {qid: {doc_id: score}}.
            qrels: Ground truth {qid: {doc_id: relevance}}.

        Returns:
            List of (query_id, doc_id, relevance, rank) tuples.
        """
        failures = []

        for qid, doc_rels in qrels.items():
            if qid not in results:
                # Query not in results - all relevant docs are failures
                for doc_id, rel in doc_rels.items():
                    if rel >= self.config.min_relevance:
                        failures.append((qid, doc_id, rel, None))
                continue

            # Get ranked results
            ranked_docs = sorted(
                results[qid].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            doc_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked_docs, 1)}

            # Check each relevant document
            for doc_id, rel in doc_rels.items():
                if rel < self.config.min_relevance:
                    continue

                rank = doc_ranks.get(doc_id)

                # It's a failure if:
                # 1. Not retrieved at all
                # 2. Ranked below threshold
                if rank is None or rank > self.config.rank_threshold:
                    failures.append((qid, doc_id, rel, rank))

        return failures

    def sample_failures(
        self,
        failures: List[Tuple[str, str, int, Optional[int]]],
        queries: Dict[str, str],
        corpus: Dict[str, dict],
    ) -> List[FailureCase]:
        """Sample failures for manual analysis.

        Args:
            failures: List of failure tuples.
            queries: Query texts.
            corpus: Document corpus.

        Returns:
            List of sampled failure cases.
        """
        # Sample if too many failures
        if len(failures) > self.config.sample_size:
            failures = random.sample(failures, self.config.sample_size)

        cases = []
        for qid, doc_id, rel, rank in failures:
            query_text = queries.get(qid, "")

            doc = corpus.get(doc_id, {})
            doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()

            case = FailureCase(
                query_id=qid,
                query_text=query_text,
                relevant_doc_id=doc_id,
                relevant_doc_text=doc_text[:1000],  # Truncate
                relevance_score=rel,
                retrieved_rank=rank,
                retrieved_score=None,
            )
            cases.append(case)

        return cases

    def compute_failure_statistics(
        self,
        failures: List[Tuple[str, str, int, Optional[int]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        """Compute statistics about failures.

        Args:
            failures: List of failure tuples.
            qrels: Ground truth relevance judgments.

        Returns:
            Dictionary of statistics.
        """
        # Count by relevance level
        by_relevance = {}
        for _, _, rel, _ in failures:
            by_relevance[rel] = by_relevance.get(rel, 0) + 1

        # Count not retrieved vs low ranked
        not_retrieved = sum(1 for _, _, _, rank in failures if rank is None)
        low_ranked = len(failures) - not_retrieved

        # Count total relevant documents
        total_relevant = sum(
            1 for docs in qrels.values()
            for rel in docs.values()
            if rel >= self.config.min_relevance
        )

        # Failure rate
        failure_rate = len(failures) / total_relevant if total_relevant > 0 else 0

        # Count failures by query
        failures_by_query = {}
        for qid, _, _, _ in failures:
            failures_by_query[qid] = failures_by_query.get(qid, 0) + 1

        # Queries with most failures
        worst_queries = sorted(
            failures_by_query.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "total_failures": len(failures),
            "total_relevant": total_relevant,
            "failure_rate": failure_rate,
            "not_retrieved": not_retrieved,
            "low_ranked": low_ranked,
            "by_relevance": by_relevance,
            "queries_with_failures": len(failures_by_query),
            "worst_queries": worst_queries,
        }


class VocabularyAnalyzer:
    """Analyze vocabulary mismatch between queries and documents."""

    def __init__(self, stopwords: Optional[Set[str]] = None):
        """Initialize analyzer.

        Args:
            stopwords: Set of stopwords to ignore.
        """
        self.stopwords = stopwords or self._default_stopwords()

    def _default_stopwords(self) -> Set[str]:
        """Get default stopwords."""
        return {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "that", "which", "who", "whom", "this", "these", "those", "it", "its",
            "comprising", "including", "wherein", "selected", "group", "consisting",
        }

    def tokenize(self, text: str) -> Set[str]:
        """Simple tokenization.

        Args:
            text: Input text.

        Returns:
            Set of tokens.
        """
        import re

        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z]+\b', text.lower())

        # Remove stopwords and short tokens
        tokens = {t for t in tokens if t not in self.stopwords and len(t) > 2}

        return tokens

    def compute_overlap(
        self,
        query_text: str,
        doc_text: str,
    ) -> Dict[str, Any]:
        """Compute vocabulary overlap between query and document.

        Args:
            query_text: Query text.
            doc_text: Document text.

        Returns:
            Overlap statistics.
        """
        query_tokens = self.tokenize(query_text)
        doc_tokens = self.tokenize(doc_text)

        if not query_tokens:
            return {"overlap": 0, "jaccard": 0, "coverage": 0}

        overlap = query_tokens & doc_tokens
        union = query_tokens | doc_tokens

        return {
            "query_tokens": len(query_tokens),
            "doc_tokens": len(doc_tokens),
            "overlap_count": len(overlap),
            "overlap_tokens": list(overlap)[:20],
            "jaccard": len(overlap) / len(union) if union else 0,
            "coverage": len(overlap) / len(query_tokens),
            "query_only": list(query_tokens - doc_tokens)[:20],
        }

    def classify_vocabulary_mismatch(
        self,
        case: FailureCase,
    ) -> Tuple[bool, str]:
        """Classify if failure is due to vocabulary mismatch.

        Args:
            case: Failure case.

        Returns:
            Tuple of (is_mismatch, explanation).
        """
        overlap = self.compute_overlap(case.query_text, case.relevant_doc_text)

        if overlap["coverage"] < 0.2:
            return True, f"Low coverage ({overlap['coverage']:.2f}). Missing terms: {overlap['query_only']}"
        elif overlap["jaccard"] < 0.1:
            return True, f"Low Jaccard ({overlap['jaccard']:.2f})"

        return False, ""


class DomainAnalyzer:
    """Analyze cross-domain failures."""

    def __init__(
        self,
        query_domains: Optional[Dict[str, str]] = None,
        doc_domains: Optional[Dict[str, str]] = None,
    ):
        """Initialize analyzer.

        Args:
            query_domains: Mapping of query_id to domain.
            doc_domains: Mapping of doc_id to domain.
        """
        self.query_domains = query_domains or {}
        self.doc_domains = doc_domains or {}

    def classify_cross_domain(
        self,
        case: FailureCase,
    ) -> Tuple[bool, str]:
        """Classify if failure is due to cross-domain issue.

        Args:
            case: Failure case.

        Returns:
            Tuple of (is_cross_domain, explanation).
        """
        query_domain = self.query_domains.get(case.query_id, "unknown")
        doc_domain = self.doc_domains.get(case.relevant_doc_id, "unknown")

        if query_domain != "unknown" and doc_domain != "unknown":
            if query_domain != doc_domain:
                return True, f"Query domain: {query_domain}, Doc domain: {doc_domain}"

        return False, ""


class ErrorReportGenerator:
    """Generate error analysis reports."""

    def __init__(
        self,
        output_dir: Path,
    ):
        """Initialize generator.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary_report(
        self,
        statistics: Dict[str, Any],
        categorized_cases: Dict[FailureCategory, List[FailureCase]],
    ) -> str:
        """Generate summary report.

        Args:
            statistics: Failure statistics.
            categorized_cases: Cases by category.

        Returns:
            Markdown report.
        """
        lines = [
            "# Error Analysis Report",
            "",
            "## Summary Statistics",
            "",
            f"- Total failures: {statistics['total_failures']}",
            f"- Total relevant documents: {statistics['total_relevant']}",
            f"- Failure rate: {statistics['failure_rate']:.2%}",
            f"- Not retrieved: {statistics['not_retrieved']}",
            f"- Low ranked: {statistics['low_ranked']}",
            "",
            "### Failures by Relevance Level",
            "",
        ]

        for rel, count in sorted(statistics.get("by_relevance", {}).items()):
            lines.append(f"- Relevance {rel}: {count}")

        lines.extend([
            "",
            "## Failure Categories",
            "",
        ])

        for category, cases in categorized_cases.items():
            lines.append(f"### {category.value.replace('_', ' ').title()}")
            lines.append(f"Count: {len(cases)}")
            lines.append("")

        # Worst queries
        lines.extend([
            "## Queries with Most Failures",
            "",
            "| Query ID | Failures |",
            "|----------|----------|",
        ])

        for qid, count in statistics.get("worst_queries", []):
            lines.append(f"| {qid} | {count} |")

        return "\n".join(lines)

    def generate_case_examples(
        self,
        categorized_cases: Dict[FailureCategory, List[FailureCase]],
        num_examples: int = 5,
    ) -> str:
        """Generate example cases for each category.

        Args:
            categorized_cases: Cases by category.
            num_examples: Number of examples per category.

        Returns:
            Markdown with examples.
        """
        lines = ["# Failure Examples", ""]

        for category, cases in categorized_cases.items():
            lines.append(f"## {category.value.replace('_', ' ').title()}")
            lines.append("")

            for case in cases[:num_examples]:
                lines.extend([
                    f"### Query: {case.query_id}",
                    "",
                    "**Query text:**",
                    f"```",
                    case.query_text[:500],
                    "```",
                    "",
                    f"**Relevant doc:** {case.relevant_doc_id} (relevance: {case.relevance_score})",
                    "",
                    "**Document text:**",
                    "```",
                    case.relevant_doc_text[:500],
                    "```",
                    "",
                    f"**Rank:** {case.retrieved_rank or 'Not retrieved'}",
                    "",
                    f"**Notes:** {case.notes}",
                    "",
                    "---",
                    "",
                ])

        return "\n".join(lines)

    def save_reports(
        self,
        statistics: Dict[str, Any],
        categorized_cases: Dict[FailureCategory, List[FailureCase]],
    ) -> None:
        """Save all reports to files.

        Args:
            statistics: Failure statistics.
            categorized_cases: Cases by category.
        """
        # Summary report
        summary = self.generate_summary_report(statistics, categorized_cases)
        with open(self.output_dir / "error_summary.md", "w") as f:
            f.write(summary)

        # Examples
        examples = self.generate_case_examples(categorized_cases)
        with open(self.output_dir / "error_examples.md", "w") as f:
            f.write(examples)

        # JSON data
        data = {
            "statistics": statistics,
            "categories": {
                cat.value: len(cases) for cat, cases in categorized_cases.items()
            },
        }
        with open(self.output_dir / "error_data.json", "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved error reports to {self.output_dir}")


def run_error_analysis(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    output_dir: Path,
    config: Optional[ErrorAnalysisConfig] = None,
) -> Dict[str, Any]:
    """Run complete error analysis.

    Args:
        results: Retrieved results.
        qrels: Ground truth relevance.
        queries: Query texts.
        corpus: Document corpus.
        output_dir: Output directory.
        config: Analysis configuration.

    Returns:
        Analysis results.
    """
    analyzer = ErrorAnalyzer(config)
    vocab_analyzer = VocabularyAnalyzer()
    report_generator = ErrorReportGenerator(output_dir)

    # Identify failures
    failures = analyzer.identify_failures(results, qrels)
    logger.info(f"Identified {len(failures)} failures")

    # Compute statistics
    statistics = analyzer.compute_failure_statistics(failures, qrels)

    # Sample for detailed analysis
    cases = analyzer.sample_failures(failures, queries, corpus)

    # Categorize failures
    categorized = {cat: [] for cat in FailureCategory}

    for case in cases:
        # Check vocabulary mismatch
        is_mismatch, explanation = vocab_analyzer.classify_vocabulary_mismatch(case)
        if is_mismatch:
            case.category = FailureCategory.VOCABULARY_MISMATCH
            case.notes = explanation
            categorized[FailureCategory.VOCABULARY_MISMATCH].append(case)
            continue

        # Default to unknown
        case.category = FailureCategory.UNKNOWN
        categorized[FailureCategory.UNKNOWN].append(case)

    # Generate reports
    report_generator.save_reports(statistics, categorized)

    return {
        "statistics": statistics,
        "num_sampled": len(cases),
        "categories": {cat.value: len(cases) for cat, cases in categorized.items()},
    }
