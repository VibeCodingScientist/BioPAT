"""Ablation studies module.

Implements ablation experiments for analyzing retrieval performance
across different conditions.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    benchmark_dir: str = "data/benchmark"
    results_dir: Optional[str] = None
    k_values: List[int] = field(default_factory=lambda: [10, 50, 100])


@dataclass
class AblationResult:
    """Result of an ablation experiment."""

    name: str
    description: str
    metrics: Dict[str, float]
    num_queries: int
    config: Dict[str, Any]


class QueryRepresentationAblation:
    """Ablation study on query representation.

    Tests different ways to represent patent claims as queries:
    - claim_only: Just the claim text
    - claim_title: Claim text + patent title
    - claim_abstract: Claim text + patent abstract
    - claim_title_abstract: Full context
    """

    def __init__(
        self,
        benchmark_dir: Path,
        patents_path: Optional[Path] = None,
    ):
        """Initialize ablation.

        Args:
            benchmark_dir: Path to benchmark directory.
            patents_path: Path to patents parquet file with full text.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.patents_path = patents_path

        # Load queries
        self.queries = self._load_queries()

        # Load patents if available
        self.patents = self._load_patents() if patents_path else {}

    def _load_queries(self) -> Dict[str, dict]:
        """Load queries from BEIR format."""
        queries_path = self.benchmark_dir / "queries.jsonl"
        queries = {}

        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                # Parse query ID to get patent ID
                qid = query["_id"]
                parts = qid.rsplit("-", 1)
                patent_id = parts[0] if len(parts) > 1 else qid

                queries[qid] = {
                    "text": query["text"],
                    "patent_id": patent_id,
                }

        return queries

    def _load_patents(self) -> Dict[str, dict]:
        """Load patents from parquet file."""
        if not self.patents_path or not self.patents_path.exists():
            return {}

        df = pl.read_parquet(self.patents_path)

        patents = {}
        for row in df.iter_rows(named=True):
            patents[row["patent_id"]] = {
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }

        return patents

    def create_query_variants(self) -> Dict[str, Dict[str, str]]:
        """Create different query representations.

        Returns:
            Dictionary mapping variant name to {qid: query_text}.
        """
        variants = {
            "claim_only": {},
            "claim_title": {},
            "claim_abstract": {},
            "claim_title_abstract": {},
        }

        for qid, query_info in self.queries.items():
            claim_text = query_info["text"]
            patent_id = query_info["patent_id"]

            # Get patent info
            patent = self.patents.get(patent_id, {})
            title = patent.get("title", "")
            abstract = patent.get("abstract", "")

            # Create variants
            variants["claim_only"][qid] = claim_text

            if title:
                variants["claim_title"][qid] = f"{title} {claim_text}"
            else:
                variants["claim_title"][qid] = claim_text

            if abstract:
                variants["claim_abstract"][qid] = f"{claim_text} {abstract}"
            else:
                variants["claim_abstract"][qid] = claim_text

            if title or abstract:
                variants["claim_title_abstract"][qid] = f"{title} {claim_text} {abstract}".strip()
            else:
                variants["claim_title_abstract"][qid] = claim_text

        return variants


class DocumentRepresentationAblation:
    """Ablation study on document representation.

    Tests different ways to represent papers:
    - title_abstract: Title + abstract (default)
    - title_only: Just the title
    - abstract_only: Just the abstract
    """

    def __init__(self, benchmark_dir: Path):
        """Initialize ablation.

        Args:
            benchmark_dir: Path to benchmark directory.
        """
        self.benchmark_dir = Path(benchmark_dir)

    def create_corpus_variants(self) -> Dict[str, Dict[str, dict]]:
        """Create different corpus representations.

        Returns:
            Dictionary mapping variant name to corpus dict.
        """
        corpus_path = self.benchmark_dir / "corpus.jsonl"

        variants = {
            "title_abstract": {},
            "title_only": {},
            "abstract_only": {},
        }

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["_id"]
                title = doc.get("title", "")
                text = doc.get("text", "")

                variants["title_abstract"][doc_id] = {"title": title, "text": text}
                variants["title_only"][doc_id] = {"title": title, "text": ""}
                variants["abstract_only"][doc_id] = {"title": "", "text": text}

        return variants


class DomainAblation:
    """Ablation study on domain (IN-domain vs OUT-domain).

    Analyzes performance difference between queries where relevant
    documents are from the same domain vs different domains.
    """

    def __init__(
        self,
        benchmark_dir: Path,
        queries_metadata_path: Optional[Path] = None,
    ):
        """Initialize ablation.

        Args:
            benchmark_dir: Path to benchmark directory.
            queries_metadata_path: Path to queries metadata with domain info.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.queries_metadata_path = queries_metadata_path

        # Load domain info
        self.query_domains = self._load_query_domains()

    def _load_query_domains(self) -> Dict[str, str]:
        """Load query domain information."""
        if not self.queries_metadata_path:
            return {}

        if not self.queries_metadata_path.exists():
            return {}

        df = pl.read_parquet(self.queries_metadata_path)

        domains = {}
        for row in df.iter_rows(named=True):
            qid = row.get("query_id", row.get("claim_id", ""))
            domain = row.get("domain", row.get("ipc3", "unknown"))
            domains[qid] = domain

        return domains

    def split_by_domain_type(
        self,
        qrels: Dict[str, Dict[str, int]],
        domain_classification: Dict[str, str],
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Split qrels by domain type.

        Args:
            qrels: Query relevance judgments.
            domain_classification: Mapping of qid to "IN" or "OUT".

        Returns:
            Dict with "IN" and "OUT" qrels.
        """
        splits = {"IN": {}, "OUT": {}}

        for qid, docs in qrels.items():
            domain_type = domain_classification.get(qid, "unknown")
            if domain_type in splits:
                splits[domain_type][qid] = docs

        return splits


class TemporalAblation:
    """Ablation study on temporal characteristics.

    Analyzes performance on recent vs older patents.
    """

    def __init__(
        self,
        benchmark_dir: Path,
        queries_metadata_path: Optional[Path] = None,
        recent_cutoff: str = "2015-01-01",
    ):
        """Initialize ablation.

        Args:
            benchmark_dir: Path to benchmark directory.
            queries_metadata_path: Path to queries metadata with dates.
            recent_cutoff: Date string for recent/older split.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.queries_metadata_path = queries_metadata_path
        self.recent_cutoff = recent_cutoff

        # Load date info
        self.query_dates = self._load_query_dates()

    def _load_query_dates(self) -> Dict[str, str]:
        """Load query date information."""
        if not self.queries_metadata_path:
            return {}

        if not self.queries_metadata_path.exists():
            return {}

        df = pl.read_parquet(self.queries_metadata_path)

        dates = {}
        for row in df.iter_rows(named=True):
            qid = row.get("query_id", row.get("claim_id", ""))
            date = row.get("priority_date", row.get("filing_date", ""))
            if date:
                dates[qid] = str(date)

        return dates

    def split_by_temporal(
        self,
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Split qrels by temporal period.

        Args:
            qrels: Query relevance judgments.

        Returns:
            Dict with "recent" and "older" qrels.
        """
        splits = {"recent": {}, "older": {}}

        for qid, docs in qrels.items():
            date = self.query_dates.get(qid, "")
            if date >= self.recent_cutoff:
                splits["recent"][qid] = docs
            else:
                splits["older"][qid] = docs

        return splits


class IPCAblation:
    """Ablation study on IPC subclass performance.

    Analyzes performance breakdown by IPC code.
    """

    def __init__(
        self,
        benchmark_dir: Path,
        queries_metadata_path: Optional[Path] = None,
    ):
        """Initialize ablation.

        Args:
            benchmark_dir: Path to benchmark directory.
            queries_metadata_path: Path to queries metadata with IPC codes.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.queries_metadata_path = queries_metadata_path

        # Load IPC info
        self.query_ipcs = self._load_query_ipcs()

    def _load_query_ipcs(self) -> Dict[str, List[str]]:
        """Load query IPC codes."""
        if not self.queries_metadata_path:
            return {}

        if not self.queries_metadata_path.exists():
            return {}

        df = pl.read_parquet(self.queries_metadata_path)

        ipcs = {}
        for row in df.iter_rows(named=True):
            qid = row.get("query_id", row.get("claim_id", ""))
            ipc_codes = row.get("ipc_codes", [])
            if isinstance(ipc_codes, str):
                ipc_codes = [ipc_codes]
            ipcs[qid] = ipc_codes

        return ipcs

    def get_ipc3(self, ipc_codes: List[str]) -> Set[str]:
        """Extract IPC3 (first 4 characters) from IPC codes.

        Args:
            ipc_codes: List of IPC codes.

        Returns:
            Set of IPC3 codes.
        """
        ipc3 = set()
        for code in ipc_codes:
            if len(code) >= 4:
                ipc3.add(code[:4])
        return ipc3

    def split_by_ipc(
        self,
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Split qrels by IPC3 code.

        Args:
            qrels: Query relevance judgments.

        Returns:
            Dict mapping IPC3 code to qrels.
        """
        splits = {}

        for qid, docs in qrels.items():
            ipc_codes = self.query_ipcs.get(qid, [])
            ipc3_set = self.get_ipc3(ipc_codes)

            # Add to each IPC3
            for ipc3 in ipc3_set:
                if ipc3 not in splits:
                    splits[ipc3] = {}
                splits[ipc3][qid] = docs

        return splits


class AblationRunner:
    """Runner for executing ablation studies."""

    def __init__(
        self,
        benchmark_dir: Path,
        results_dir: Optional[Path] = None,
        k_values: List[int] = [10, 50, 100],
    ):
        """Initialize runner.

        Args:
            benchmark_dir: Path to benchmark directory.
            results_dir: Path to save results.
            k_values: Values of k for metrics.
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir) if results_dir else benchmark_dir / "ablation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.k_values = k_values

    def run_domain_ablation(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        domain_classification: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Run domain ablation.

        Args:
            results: Retrieval results.
            qrels: Query relevance judgments.
            domain_classification: Mapping of qid to "IN" or "OUT".

        Returns:
            Metrics by domain type.
        """
        from .metrics import MetricsComputer

        domain_ablation = DomainAblation(self.benchmark_dir)
        splits = domain_ablation.split_by_domain_type(qrels, domain_classification)

        metrics_computer = MetricsComputer()
        domain_metrics = {}

        for domain_type, domain_qrels in splits.items():
            if not domain_qrels:
                continue

            # Filter results to this domain
            domain_results = {qid: r for qid, r in results.items() if qid in domain_qrels}

            metrics = metrics_computer.compute_all_metrics(
                domain_results, domain_qrels, self.k_values
            )
            metrics["num_queries"] = len(domain_qrels)
            domain_metrics[domain_type] = metrics

        return domain_metrics

    def run_ipc_ablation(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        query_ipcs: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, float]]:
        """Run IPC ablation.

        Args:
            results: Retrieval results.
            qrels: Query relevance judgments.
            query_ipcs: Mapping of qid to IPC codes.

        Returns:
            Metrics by IPC3 code.
        """
        from .metrics import MetricsComputer

        ipc_ablation = IPCAblation(self.benchmark_dir)
        ipc_ablation.query_ipcs = query_ipcs

        splits = ipc_ablation.split_by_ipc(qrels)

        metrics_computer = MetricsComputer()
        ipc_metrics = {}

        for ipc3, ipc_qrels in splits.items():
            if not ipc_qrels or len(ipc_qrels) < 10:  # Skip small groups
                continue

            # Filter results to this IPC
            ipc_results = {qid: r for qid, r in results.items() if qid in ipc_qrels}

            metrics = metrics_computer.compute_all_metrics(
                ipc_results, ipc_qrels, self.k_values
            )
            metrics["num_queries"] = len(ipc_qrels)
            ipc_metrics[ipc3] = metrics

        return ipc_metrics

    def save_ablation_results(
        self,
        results: Dict[str, Any],
        name: str,
    ) -> None:
        """Save ablation results to file.

        Args:
            results: Ablation results.
            name: Name for the ablation study.
        """
        output_path = self.results_dir / f"{name}_ablation.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved ablation results to {output_path}")

    def format_ablation_table(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        metrics_to_show: List[str] = ["NDCG@10", "NDCG@100", "Recall@100"],
    ) -> str:
        """Format ablation results as markdown table.

        Args:
            ablation_results: Ablation results.
            metrics_to_show: Metrics to include.

        Returns:
            Markdown table string.
        """
        # Header
        header = "| Split | " + " | ".join(metrics_to_show) + " | Queries |"
        separator = "|" + "|".join(["---"] * (len(metrics_to_show) + 2)) + "|"

        # Rows
        rows = []
        for split_name, metrics in ablation_results.items():
            values = [f"{metrics.get(m, 0):.4f}" for m in metrics_to_show]
            num_queries = metrics.get("num_queries", 0)
            row = f"| {split_name} | " + " | ".join(values) + f" | {num_queries} |"
            rows.append(row)

        return "\n".join([header, separator] + rows)
