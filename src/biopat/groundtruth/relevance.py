"""Relevance assignment module.

Computes graded relevance scores for query-document pairs
based on citation source, confidence, and rejection type.

Phase 5 (v2.0): Extended to support dual-corpus relevance assignment
for both scientific papers (NPL) and prior patents.
"""

import logging
from pathlib import Path
from typing import Optional, Literal
from enum import IntEnum
import polars as pl

logger = logging.getLogger(__name__)

# Document type constants for v2.0
DOC_TYPE_PAPER = "paper"
DOC_TYPE_PATENT = "patent"


class RelevanceLevel(IntEnum):
    """Relevance tier definitions following BioPAT specification."""
    NOT_RELEVANT = 0
    RELEVANT = 1
    HIGHLY_RELEVANT = 2
    NOVELTY_DESTROYING = 3


RELEVANCE_LABELS = {
    RelevanceLevel.NOT_RELEVANT: "not_relevant",
    RelevanceLevel.RELEVANT: "relevant",
    RelevanceLevel.HIGHLY_RELEVANT: "highly_relevant",
    RelevanceLevel.NOVELTY_DESTROYING: "novelty_destroying",
}


class RelevanceAssigner:
    """Assigns relevance scores to query-document pairs."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def qrels_path(self) -> Path:
        return self.processed_dir / "qrels.parquet"

    def assign_binary_relevance(
        self,
        citation_links: pl.DataFrame,
        confidence_threshold: int = 8,
    ) -> pl.DataFrame:
        """Assign binary relevance (Phase 1).

        Phase 1 uses binary relevance:
        - 1 if high-confidence citation exists
        - 0 otherwise

        Args:
            citation_links: Validated citation links with temporal constraints applied.
            confidence_threshold: Minimum confidence for positive relevance.

        Returns:
            DataFrame with query_id, doc_id, relevance columns.
        """
        # Filter by confidence
        qrels = citation_links.filter(
            pl.col("confidence") >= confidence_threshold
        )

        # Create qrels format
        qrels = qrels.select([
            pl.col("query_id"),
            pl.col("paper_id").alias("doc_id"),
            pl.lit(1).alias("relevance"),
            pl.lit("citation").alias("evidence_source"),
            pl.col("confidence"),
            pl.col("source"),
        ])

        # Deduplicate (keep highest confidence if multiple citations)
        qrels = qrels.sort("confidence", descending=True).unique(
            subset=["query_id", "doc_id"],
            keep="first"
        )

        logger.info(f"Created {len(qrels)} binary relevance judgments")
        return qrels

    def assign_graded_relevance(
        self,
        citation_links: pl.DataFrame,
        office_action_data: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Assign graded relevance (Phase 2+).

        Relevance tiers:
        - 3 (novelty_destroying): Examiner cited under 102 (anticipation)
        - 2 (highly_relevant): Examiner cited under 103 OR high-confidence examiner citation
        - 1 (relevant): Medium-confidence examiner OR high-confidence applicant citation
        - 0 (not_relevant): Low confidence or no link

        Args:
            citation_links: Validated citation links.
            office_action_data: Optional Office Action rejection data (Phase 2+).

        Returns:
            DataFrame with graded relevance scores.
        """
        # Start with basic relevance from citation source and confidence
        qrels = citation_links.with_columns([
            pl.lit(0).alias("relevance"),
            pl.lit("unknown").alias("label"),
        ])

        # Apply relevance rules
        qrels = qrels.with_columns(
            pl.when(
                # High-confidence examiner citations
                (pl.col("source") == "examiner") &
                (pl.col("confidence") >= 9)
            )
            .then(pl.lit(RelevanceLevel.HIGHLY_RELEVANT))
            .when(
                # Medium-confidence examiner citations
                (pl.col("source") == "examiner") &
                (pl.col("confidence") >= 7)
            )
            .then(pl.lit(RelevanceLevel.RELEVANT))
            .when(
                # High-confidence applicant citations
                (pl.col("source") == "applicant") &
                (pl.col("confidence") >= 8)
            )
            .then(pl.lit(RelevanceLevel.RELEVANT))
            .otherwise(pl.lit(RelevanceLevel.NOT_RELEVANT))
            .alias("relevance")
        )

        # If Office Action data is provided, upgrade relevance for 102/103 rejections
        if office_action_data is not None and len(office_action_data) > 0:
            # Join with OA data
            qrels = qrels.join(
                office_action_data.select(["query_id", "doc_id", "rejection_type"]),
                on=["query_id", "doc_id"],
                how="left"
            )

            # Upgrade relevance based on rejection type
            qrels = qrels.with_columns(
                pl.when(pl.col("rejection_type") == "102")
                .then(pl.lit(RelevanceLevel.NOVELTY_DESTROYING))
                .when(pl.col("rejection_type") == "103")
                .then(pl.lit(RelevanceLevel.HIGHLY_RELEVANT))
                .otherwise(pl.col("relevance"))
                .alias("relevance")
            )

        # Add human-readable labels
        qrels = qrels.with_columns(
            pl.col("relevance").map_elements(
                lambda r: RELEVANCE_LABELS.get(r, "unknown"),
                return_dtype=pl.Utf8
            ).alias("label")
        )

        # Format output
        qrels = qrels.select([
            pl.col("query_id"),
            pl.col("paper_id").alias("doc_id"),
            pl.col("relevance"),
            pl.col("label"),
            pl.col("source").alias("evidence_source"),
            pl.col("confidence"),
        ])

        # Deduplicate (keep highest relevance if multiple entries)
        qrels = qrels.sort("relevance", descending=True).unique(
            subset=["query_id", "doc_id"],
            keep="first"
        )

        # Filter out not_relevant (keep only positive judgments for qrels)
        qrels = qrels.filter(pl.col("relevance") > 0)

        logger.info(f"Created {len(qrels)} graded relevance judgments")
        self._log_relevance_distribution(qrels)

        return qrels

    def _log_relevance_distribution(self, qrels: pl.DataFrame):
        """Log the distribution of relevance levels."""
        distribution = (
            qrels.group_by(["relevance", "label"])
            .agg(pl.len().alias("count"))
            .sort("relevance", descending=True)
        )

        logger.info("Relevance distribution:")
        for row in distribution.iter_rows(named=True):
            logger.info(f"  {row['label']} ({row['relevance']}): {row['count']}")

    def get_qrels_stats(self, qrels: pl.DataFrame) -> dict:
        """Get statistics about relevance judgments.

        Args:
            qrels: Qrels DataFrame.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "total_judgments": len(qrels),
            "unique_queries": qrels["query_id"].n_unique(),
            "unique_docs": qrels["doc_id"].n_unique(),
        }

        # Relevance distribution
        stats["relevance_distribution"] = (
            qrels.group_by("relevance")
            .agg(pl.len().alias("count"))
            .sort("relevance")
            .to_dicts()
        )

        # Judgments per query
        per_query = qrels.group_by("query_id").agg(pl.len().alias("count"))
        stats["mean_judgments_per_query"] = per_query["count"].mean()
        stats["median_judgments_per_query"] = per_query["count"].median()

        # Source distribution
        if "evidence_source" in qrels.columns:
            stats["source_distribution"] = (
                qrels.group_by("evidence_source")
                .agg(pl.len().alias("count"))
                .to_dicts()
            )

        return stats

    def create_qrels(
        self,
        citation_links: pl.DataFrame,
        graded: bool = False,
        office_action_data: Optional[pl.DataFrame] = None,
        confidence_threshold: int = 8,
        save: bool = True,
    ) -> pl.DataFrame:
        """Create relevance judgments (qrels) for the benchmark.

        Args:
            citation_links: Validated citation links.
            graded: If True, use graded relevance (Phase 2+). If False, binary (Phase 1).
            office_action_data: Optional Office Action data for Phase 2+.
            confidence_threshold: Minimum confidence for binary relevance.
            save: Whether to save qrels to disk.

        Returns:
            Qrels DataFrame.
        """
        if graded:
            qrels = self.assign_graded_relevance(citation_links, office_action_data)
        else:
            qrels = self.assign_binary_relevance(citation_links, confidence_threshold)

        # Log statistics
        stats = self.get_qrels_stats(qrels)
        logger.info(f"Qrels statistics: {stats}")

        if save:
            qrels.write_parquet(self.qrels_path)
            logger.info(f"Saved qrels to {self.qrels_path}")

        return qrels

    def load(self) -> pl.DataFrame:
        """Load qrels from disk.

        Returns:
            Qrels DataFrame.
        """
        if not self.qrels_path.exists():
            raise FileNotFoundError(f"Qrels not found at {self.qrels_path}")
        return pl.read_parquet(self.qrels_path)

    def merge_ros_and_oa_citations(
        self,
        ros_links: pl.DataFrame,
        oa_links: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge RoS citations with Office Action citations.

        Office Action citations take precedence as they are examiner-validated.

        Args:
            ros_links: RoS citation links with query_id, paper_id, confidence, source.
            oa_links: OA citation links with query_id, paper_id, rejection_type.

        Returns:
            Merged DataFrame with all citation evidence.
        """
        # Standardize column names
        if "doc_id" in ros_links.columns:
            ros_links = ros_links.rename({"doc_id": "paper_id"})
        if "doc_id" in oa_links.columns:
            oa_links = oa_links.rename({"doc_id": "paper_id"})

        # Add source to OA links if not present
        if "source" not in oa_links.columns:
            oa_links = oa_links.with_columns(pl.lit("office_action").alias("source"))

        # Merge: prefer OA data when available
        merged = ros_links.join(
            oa_links.select(["query_id", "paper_id", "rejection_type"]),
            on=["query_id", "paper_id"],
            how="left"
        )

        # For links only in OA (not in RoS), add them
        oa_only = oa_links.join(
            ros_links.select(["query_id", "paper_id"]),
            on=["query_id", "paper_id"],
            how="anti"
        )

        if len(oa_only) > 0:
            # Add missing columns to match schema
            if "confidence" not in oa_only.columns:
                oa_only = oa_only.with_columns(pl.lit(10).alias("confidence"))
            merged = pl.concat([merged, oa_only])

        logger.info(
            f"Merged citations: {len(ros_links)} RoS + {len(oa_links)} OA = {len(merged)} total "
            f"({merged['rejection_type'].is_not_null().sum()} with rejection type)"
        )

        return merged

    def create_graded_qrels_from_merged(
        self,
        merged_citations: pl.DataFrame,
        save: bool = True,
    ) -> pl.DataFrame:
        """Create graded qrels from merged RoS + OA citations.

        Args:
            merged_citations: Merged citation links with rejection_type.
            save: Whether to save to disk.

        Returns:
            Graded qrels DataFrame.
        """
        # Calculate relevance for each citation
        relevance_scores = []
        labels = []

        for row in merged_citations.iter_rows(named=True):
            rejection_type = row.get("rejection_type")
            source = row.get("source", "unknown")
            confidence = row.get("confidence", 0)

            # Assign relevance based on spec rules
            if rejection_type == "102":
                rel = RelevanceLevel.NOVELTY_DESTROYING
            elif rejection_type == "103":
                rel = RelevanceLevel.HIGHLY_RELEVANT
            elif source == "examiner" and confidence >= 9:
                rel = RelevanceLevel.HIGHLY_RELEVANT
            elif source == "examiner" and confidence >= 7:
                rel = RelevanceLevel.RELEVANT
            elif source == "applicant" and confidence >= 8:
                rel = RelevanceLevel.RELEVANT
            else:
                rel = RelevanceLevel.NOT_RELEVANT

            relevance_scores.append(int(rel))
            labels.append(RELEVANCE_LABELS.get(rel, "unknown"))

        # Build qrels DataFrame
        qrels = merged_citations.with_columns([
            pl.Series("relevance", relevance_scores),
            pl.Series("label", labels),
        ])

        # Select and rename columns
        qrels = qrels.select([
            pl.col("query_id"),
            pl.col("paper_id").alias("doc_id"),
            pl.col("relevance"),
            pl.col("label"),
            pl.col("source").alias("evidence_source"),
            pl.col("confidence"),
            pl.col("rejection_type"),
        ])

        # Deduplicate (keep highest relevance)
        qrels = qrels.sort("relevance", descending=True).unique(
            subset=["query_id", "doc_id"],
            keep="first"
        )

        # Filter out not_relevant
        qrels = qrels.filter(pl.col("relevance") > 0)

        logger.info(f"Created {len(qrels)} graded relevance judgments")
        self._log_relevance_distribution(qrels)

        if save:
            qrels.write_parquet(self.qrels_path)
            logger.info(f"Saved graded qrels to {self.qrels_path}")

        return qrels

    def assign_dual_corpus_relevance(
        self,
        paper_citations: pl.DataFrame,
        patent_citations: pl.DataFrame,
        office_action_data: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Assign graded relevance for dual-corpus (v2.0).

        Unified relevance assignment for both paper (NPL) and patent citations.
        Same relevance tiers apply to both document types.

        Args:
            paper_citations: Paper/NPL citation links with query_id, paper_id, etc.
            patent_citations: Patent citation links with query_patent_id, cited_patent_id, etc.
            office_action_data: Optional Office Action rejection data.

        Returns:
            Unified qrels DataFrame with doc_type field.
        """
        all_qrels = []

        # Process paper citations
        if len(paper_citations) > 0:
            paper_qrels = self._assign_relevance_to_citations(
                paper_citations,
                id_col="paper_id",
                doc_type=DOC_TYPE_PAPER,
                office_action_data=office_action_data,
            )
            if len(paper_qrels) > 0:
                all_qrels.append(paper_qrels)
            logger.info(f"Assigned relevance to {len(paper_qrels)} paper citations")

        # Process patent citations
        if len(patent_citations) > 0:
            # Standardize column names
            patent_citations_std = patent_citations.rename({
                "query_patent_id": "query_id",
                "cited_patent_id": "doc_id",
            }) if "query_patent_id" in patent_citations.columns else patent_citations

            patent_qrels = self._assign_relevance_to_citations(
                patent_citations_std,
                id_col="doc_id",
                doc_type=DOC_TYPE_PATENT,
                office_action_data=None,  # OA data already in patent_citations
            )
            if len(patent_qrels) > 0:
                all_qrels.append(patent_qrels)
            logger.info(f"Assigned relevance to {len(patent_qrels)} patent citations")

        if not all_qrels:
            logger.warning("No citations to assign relevance to")
            return pl.DataFrame({
                "query_id": [],
                "doc_id": [],
                "relevance": [],
                "label": [],
                "doc_type": [],
                "evidence_source": [],
            })

        # Combine all qrels
        combined = pl.concat(all_qrels)

        # Deduplicate (keep highest relevance per query-doc pair)
        combined = combined.sort("relevance", descending=True).unique(
            subset=["query_id", "doc_id"],
            keep="first"
        )

        # Filter out not_relevant
        combined = combined.filter(pl.col("relevance") > 0)

        logger.info(f"Created {len(combined)} dual-corpus relevance judgments")
        self._log_dual_corpus_distribution(combined)

        return combined

    def _assign_relevance_to_citations(
        self,
        citations: pl.DataFrame,
        id_col: str,
        doc_type: str,
        office_action_data: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Assign relevance scores to a set of citations.

        Internal method for unified relevance assignment.

        Args:
            citations: Citation links DataFrame.
            id_col: Column containing document IDs.
            doc_type: Document type ("paper" or "patent").
            office_action_data: Optional Office Action data.

        Returns:
            Qrels DataFrame with relevance scores.
        """
        relevance_scores = []
        labels = []

        for row in citations.iter_rows(named=True):
            rejection_type = row.get("rejection_type")
            source = row.get("source", "unknown")
            confidence = row.get("confidence", 0) or 0

            # Unified relevance rules for both doc types
            if rejection_type == "102":
                rel = RelevanceLevel.NOVELTY_DESTROYING
            elif rejection_type == "103":
                rel = RelevanceLevel.HIGHLY_RELEVANT
            elif source in ("examiner", "office_action") and confidence >= 9:
                rel = RelevanceLevel.HIGHLY_RELEVANT
            elif source in ("examiner", "office_action") and confidence >= 7:
                rel = RelevanceLevel.RELEVANT
            elif source == "applicant" and confidence >= 8:
                rel = RelevanceLevel.RELEVANT
            elif source == "office_action":
                # Office Action patent citations default to RELEVANT
                rel = RelevanceLevel.RELEVANT
            else:
                rel = RelevanceLevel.NOT_RELEVANT

            relevance_scores.append(int(rel))
            labels.append(RELEVANCE_LABELS.get(rel, "unknown"))

        # Build qrels DataFrame
        qrels = citations.with_columns([
            pl.Series("relevance", relevance_scores),
            pl.Series("label", labels),
            pl.lit(doc_type).alias("doc_type"),
        ])

        # Select output columns
        output_cols = ["query_id", "relevance", "label", "doc_type"]

        # Handle doc_id column name
        if id_col in qrels.columns and id_col != "doc_id":
            qrels = qrels.with_columns(pl.col(id_col).alias("doc_id"))
        output_cols.insert(1, "doc_id")

        # Add evidence source if available
        if "source" in qrels.columns:
            output_cols.append(pl.col("source").alias("evidence_source"))
        elif "evidence_source" in qrels.columns:
            output_cols.append("evidence_source")

        # Add optional columns if present
        for col in ["confidence", "rejection_type"]:
            if col in qrels.columns:
                output_cols.append(col)

        return qrels.select(output_cols)

    def _log_dual_corpus_distribution(self, qrels: pl.DataFrame):
        """Log relevance distribution by document type."""
        # Overall distribution
        self._log_relevance_distribution(qrels)

        # By doc_type
        if "doc_type" in qrels.columns:
            for doc_type in [DOC_TYPE_PAPER, DOC_TYPE_PATENT]:
                subset = qrels.filter(pl.col("doc_type") == doc_type)
                if len(subset) > 0:
                    logger.info(f"\n{doc_type.upper()} distribution:")
                    self._log_relevance_distribution(subset)

    def create_dual_corpus_qrels(
        self,
        paper_citations: pl.DataFrame,
        patent_citations: pl.DataFrame,
        office_action_data: Optional[pl.DataFrame] = None,
        save: bool = True,
    ) -> pl.DataFrame:
        """Create qrels for dual-corpus benchmark (v2.0).

        Args:
            paper_citations: Paper/NPL citation links.
            patent_citations: Patent citation links.
            office_action_data: Optional Office Action data.
            save: Whether to save to disk.

        Returns:
            Dual-corpus qrels DataFrame.
        """
        qrels = self.assign_dual_corpus_relevance(
            paper_citations, patent_citations, office_action_data
        )

        # Log statistics
        stats = self.get_dual_corpus_qrels_stats(qrels)
        logger.info(f"Dual-corpus qrels statistics: {stats}")

        if save:
            qrels.write_parquet(self.qrels_path)
            logger.info(f"Saved dual-corpus qrels to {self.qrels_path}")

        return qrels

    def get_dual_corpus_qrels_stats(self, qrels: pl.DataFrame) -> dict:
        """Get statistics about dual-corpus relevance judgments.

        Args:
            qrels: Qrels DataFrame with doc_type field.

        Returns:
            Statistics dictionary.
        """
        stats = self.get_qrels_stats(qrels)

        # Add doc_type breakdown
        if "doc_type" in qrels.columns:
            stats["by_doc_type"] = {}
            for doc_type in [DOC_TYPE_PAPER, DOC_TYPE_PATENT]:
                subset = qrels.filter(pl.col("doc_type") == doc_type)
                stats["by_doc_type"][doc_type] = {
                    "total": len(subset),
                    "unique_docs": subset["doc_id"].n_unique() if len(subset) > 0 else 0,
                }

        return stats
