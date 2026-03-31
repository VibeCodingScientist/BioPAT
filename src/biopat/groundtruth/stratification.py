"""Domain stratification module.

Implements IN-domain vs OUT-domain classification following
DAPFAM methodology for cross-domain retrieval evaluation.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

from biopat import compat

logger = logging.getLogger(__name__)

# Mapping from OpenAlex concept IDs to approximate IPC codes
CONCEPT_TO_IPC = {
    # Medicine / Medical
    "C71924100": ["A61"],  # Medicine
    "C126322002": ["A61B", "A61F"],  # Medical devices
    "C203014093": ["A61K"],  # Pharmacology
    "C502942594": ["A61P"],  # Drug therapy

    # Chemistry
    "C185592680": ["C07"],  # Chemistry
    "C178790620": ["C07D"],  # Organic chemistry - heterocyclic
    "C55493867": ["C07K"],  # Peptides

    # Biology
    "C86803240": ["C12"],  # Biology
    "C54355233": ["C12N"],  # Biochemistry/Genetic engineering
    "C104317684": ["C12Q"],  # Measuring/testing processes

    # Diagnostics
    "C76155785": ["G01N"],  # Clinical chemistry
}

# IPC code descriptions
IPC_DESCRIPTIONS = {
    "A61K": "Medical preparations",
    "A61B": "Diagnosis/Surgery",
    "A61F": "Prostheses",
    "A61P": "Therapeutic activity",
    "C07D": "Heterocyclic compounds",
    "C07K": "Peptides",
    "C12N": "Microorganisms/Enzymes",
    "C12Q": "Measuring/Testing",
    "G01N": "Material analysis",
}


class DomainStratifier:
    """Stratifies queries and documents by domain."""

    def __init__(self):
        self.concept_to_ipc = CONCEPT_TO_IPC
        self.ipc_descriptions = IPC_DESCRIPTIONS

    def get_ipc3(self, ipc_codes: List[str]) -> Set[str]:
        """Extract IPC3 codes (first 3-4 characters).

        Args:
            ipc_codes: List of full IPC codes.

        Returns:
            Set of IPC3 codes.
        """
        ipc3_set = set()
        for code in ipc_codes:
            if code and len(code) >= 3:
                # Extract class (letter + 2 digits) or subclass (letter + 2 digits + letter)
                if len(code) >= 4 and code[3].isalpha():
                    ipc3_set.add(code[:4])
                else:
                    ipc3_set.add(code[:3])
        return ipc3_set

    def map_concepts_to_ipc(self, concepts: List[Dict]) -> Set[str]:
        """Map OpenAlex concepts to approximate IPC codes.

        Args:
            concepts: List of concept dicts with 'id' and 'score'.

        Returns:
            Set of IPC codes.
        """
        ipc_codes = set()
        for concept in concepts:
            concept_id = concept.get("id", "")
            # Extract concept ID if it's a URL
            if "/" in concept_id:
                concept_id = concept_id.split("/")[-1]

            if concept_id in self.concept_to_ipc:
                ipc_codes.update(self.concept_to_ipc[concept_id])

        return ipc_codes

    def classify_domain_type(
        self,
        query_ipc3: Set[str],
        doc_ipc3: Set[str],
    ) -> str:
        """Classify if query-doc pair is IN-domain or OUT-domain.

        Args:
            query_ipc3: IPC3 codes for the query (patent claim).
            doc_ipc3: IPC3 codes for the document (paper).

        Returns:
            "IN" if domains overlap, "OUT" otherwise.
        """
        if query_ipc3 & doc_ipc3:
            return "IN"
        return "OUT"

    def add_domain_info_to_queries(
        self,
        queries_df: pl.DataFrame,
        ipc_col: str = "ipc_codes",
    ) -> pl.DataFrame:
        """Add domain classification info to queries.

        Args:
            queries_df: Queries DataFrame with IPC codes.
            ipc_col: Name of IPC codes column.

        Returns:
            DataFrame with domain info columns.
        """
        domains = []
        ipc3_sets = []

        for row in compat.iter_rows(queries_df, named=True):
            ipc_codes = row.get(ipc_col) or []
            if isinstance(ipc_codes, str):
                ipc_codes = [ipc_codes]

            ipc3 = self.get_ipc3(ipc_codes)
            ipc3_sets.append(list(ipc3))

            # Use primary IPC3 as domain
            primary_domain = list(ipc3)[0] if ipc3 else "unknown"
            domains.append(primary_domain)

        return queries_df.with_columns([
            pl.Series("domain", domains),
            pl.Series("ipc3_list", ipc3_sets),
        ])

    def add_domain_info_to_papers(
        self,
        papers_df: pl.DataFrame,
        concepts_col: str = "concepts",
    ) -> pl.DataFrame:
        """Add domain classification info to papers.

        Args:
            papers_df: Papers DataFrame with concepts.
            concepts_col: Name of concepts column.

        Returns:
            DataFrame with domain info columns.
        """
        domains = []
        ipc3_sets = []

        for row in compat.iter_rows(papers_df, named=True):
            concepts = row.get(concepts_col) or []

            ipc3 = self.map_concepts_to_ipc(concepts)
            ipc3_sets.append(list(ipc3))

            # Use primary IPC3 as domain
            primary_domain = list(ipc3)[0] if ipc3 else "unknown"
            domains.append(primary_domain)

        return papers_df.with_columns([
            pl.Series("domain", domains),
            pl.Series("ipc3_list", ipc3_sets),
        ])

    def classify_qrels_domain(
        self,
        qrels_df: pl.DataFrame,
        queries_df: pl.DataFrame,
        papers_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Add domain classification to qrels.

        Args:
            qrels_df: Qrels with query_id and doc_id.
            queries_df: Queries with ipc3_list.
            papers_df: Papers with ipc3_list.

        Returns:
            Qrels with domain_type column.
        """
        # Get query IPC3 mapping
        query_ipc3 = {}
        for row in queries_df.select(["query_id", "ipc3_list"]).iter_rows(named=True):
            query_ipc3[row["query_id"]] = set(row["ipc3_list"] or [])

        # Get paper IPC3 mapping
        paper_ipc3 = {}
        for row in papers_df.select(["paper_id", "ipc3_list"]).iter_rows(named=True):
            paper_ipc3[row["paper_id"]] = set(row["ipc3_list"] or [])

        # Classify each qrel
        domain_types = []
        for row in qrels_df.iter_rows(named=True):
            q_ipc3 = query_ipc3.get(row["query_id"], set())
            d_ipc3 = paper_ipc3.get(row["doc_id"], set())
            domain_type = self.classify_domain_type(q_ipc3, d_ipc3)
            domain_types.append(domain_type)

        return qrels_df.with_columns(
            pl.Series("domain_type", domain_types)
        )

    def split_qrels_by_domain(
        self,
        qrels_df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Split qrels into IN-domain and OUT-domain subsets.

        Args:
            qrels_df: Qrels with domain_type column.

        Returns:
            Tuple of (in_domain_qrels, out_domain_qrels).
        """
        if "domain_type" not in qrels_df.columns:
            raise ValueError("qrels_df must have domain_type column")

        in_domain = qrels_df.filter(pl.col("domain_type") == "IN")
        out_domain = qrels_df.filter(pl.col("domain_type") == "OUT")

        logger.info(
            f"Split qrels: {len(in_domain)} IN-domain, {len(out_domain)} OUT-domain "
            f"({len(in_domain) / len(qrels_df) * 100:.1f}% IN)"
        )

        return in_domain, out_domain

    def get_domain_stats(
        self,
        queries_df: pl.DataFrame,
        qrels_df: Optional[pl.DataFrame] = None,
    ) -> Dict:
        """Get statistics about domain distribution.

        Args:
            queries_df: Queries with domain column.
            qrels_df: Optional qrels with domain_type.

        Returns:
            Dict of domain statistics.
        """
        stats = {}

        # Query domain distribution
        if "domain" in queries_df.columns:
            domain_counts = (
                compat.group_by(queries_df, "domain")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .to_dicts()
            )
            stats["query_domains"] = {
                d["domain"]: d["count"] for d in domain_counts
            }

        # Qrel domain type distribution
        if qrels_df is not None and "domain_type" in qrels_df.columns:
            type_counts = (
                compat.group_by(qrels_df, "domain_type")
                .agg(pl.len().alias("count"))
                .to_dicts()
            )
            stats["qrel_domain_types"] = {
                d["domain_type"]: d["count"] for d in type_counts
            }

        return stats


def create_stratified_evaluation(
    qrels_df: pl.DataFrame,
    queries_df: pl.DataFrame,
    papers_df: pl.DataFrame,
    output_dir: Optional[str] = None,
) -> Dict[str, pl.DataFrame]:
    """Create stratified evaluation splits.

    Args:
        qrels_df: Full qrels DataFrame.
        queries_df: Queries with domain info.
        papers_df: Papers with domain info.
        output_dir: Optional directory to save splits.

    Returns:
        Dict with 'all', 'in_domain', 'out_domain' qrels.
    """
    stratifier = DomainStratifier()

    # Add domain info if not present
    if "ipc3_list" not in queries_df.columns:
        queries_df = stratifier.add_domain_info_to_queries(queries_df)
    if "ipc3_list" not in papers_df.columns:
        papers_df = stratifier.add_domain_info_to_papers(papers_df)

    # Classify qrels
    qrels_with_domain = stratifier.classify_qrels_domain(
        qrels_df, queries_df, papers_df
    )

    # Split
    in_domain, out_domain = stratifier.split_qrels_by_domain(qrels_with_domain)

    result = {
        "all": qrels_with_domain,
        "in_domain": in_domain,
        "out_domain": out_domain,
    }

    # Save if output_dir provided
    if output_dir:
        from pathlib import Path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for name, df in result.items():
            df.write_parquet(out_path / f"qrels_{name}.parquet")
            logger.info(f"Saved {name} qrels to {out_path / f'qrels_{name}.parquet'}")

    return result
