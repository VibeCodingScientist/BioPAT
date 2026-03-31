"""Citation linking module.

Handles linking patents to papers via RoS citations and
constructing query-document pairs for the benchmark.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import polars as pl

logger = logging.getLogger(__name__)


class CitationLinker:
    """Links patents to papers and constructs query-document pairs."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def links_path(self) -> Path:
        return self.processed_dir / "citation_links.parquet"

    def create_citation_links(
        self,
        ros_df: pl.DataFrame,
        patents_df: pl.DataFrame,
        papers_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Create citation links between patents and papers.

        Args:
            ros_df: Reliance on Science DataFrame with patent_id, openalex_id.
            patents_df: Processed patents DataFrame with patent_id.
            papers_df: Processed papers DataFrame with paper_id.

        Returns:
            DataFrame of valid citation links.
        """
        # Get valid patent and paper IDs
        valid_patent_ids = set(patents_df["patent_id"].unique().to_list())
        valid_paper_ids = set(papers_df["paper_id"].unique().to_list())

        # Filter RoS to valid patents and papers
        links = ros_df.filter(
            pl.col("patent_id").is_in(valid_patent_ids) &
            pl.col("openalex_id").is_in(valid_paper_ids)
        )

        # Rename for consistency
        links = links.rename({"openalex_id": "paper_id"})

        # Add source information if available
        if "examiner_applicant" in links.columns:
            links = links.with_columns(
                pl.when(pl.col("examiner_applicant").str.to_lowercase().str.contains("examiner"))
                .then(pl.lit("examiner"))
                .when(pl.col("examiner_applicant").str.to_lowercase().str.contains("applicant"))
                .then(pl.lit("applicant"))
                .otherwise(pl.lit("unknown"))
                .alias("source")
            )
        else:
            links = links.with_columns(pl.lit("unknown").alias("source"))

        # Create composite citation ID
        links = links.with_columns(
            (pl.col("patent_id") + "_" + pl.col("paper_id")).alias("citation_id")
        )

        logger.info(f"Created {len(links)} citation links")
        return links

    def expand_to_claims(
        self,
        citation_links: pl.DataFrame,
        claims_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Expand patent-level citations to claim-level.

        For Phase 1, we assign all patent citations to all independent claims
        of that patent. Phase 2 will use Office Action data for claim-level mapping.

        Args:
            citation_links: Patent-paper citation links.
            claims_df: Claims DataFrame with query_id, patent_id, claim_number.

        Returns:
            DataFrame with query_id (claim-level) and paper_id links.
        """
        # Join citations with claims
        expanded = claims_df.select(
            ["query_id", "patent_id", "claim_number", "priority_date"]
        ).join(
            citation_links.select(["patent_id", "paper_id", "confidence", "source"]),
            on="patent_id",
            how="inner"
        )

        logger.info(
            f"Expanded {len(citation_links)} patent citations to "
            f"{len(expanded)} claim-paper pairs"
        )

        return expanded

    def add_paper_dates(
        self,
        expanded_links: pl.DataFrame,
        papers_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Add paper publication dates for temporal validation.

        Args:
            expanded_links: Claim-paper links.
            papers_df: Papers DataFrame with publication_date.

        Returns:
            Links with paper dates added.
        """
        return expanded_links.join(
            papers_df.select(["paper_id", "publication_date"]),
            on="paper_id",
            how="left"
        )

    def validate_temporal_constraint(
        self,
        links_with_dates: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Validate that papers were published before patent priority date.

        Args:
            links_with_dates: Links with both priority_date and publication_date.

        Returns:
            Tuple of (valid_links, invalid_links).
        """
        # Parse dates if needed
        df = links_with_dates.with_columns([
            pl.col("priority_date").cast(pl.Date),
            pl.col("publication_date").cast(pl.Date),
        ])

        # Valid: paper published before patent priority date
        valid = df.filter(
            pl.col("publication_date") < pl.col("priority_date")
        )

        # Invalid: paper published on or after priority date
        invalid = df.filter(
            pl.col("publication_date") >= pl.col("priority_date")
        )

        logger.info(
            f"Temporal validation: {len(valid)} valid, {len(invalid)} invalid "
            f"({len(invalid) / max(1, len(df)) * 100:.1f}% violations)"
        )

        if len(invalid) > 0:
            logger.warning(
                "Found temporal violations - these will be excluded from positive judgments"
            )

        return valid, invalid

    def get_citation_stats(self, citation_links: pl.DataFrame) -> dict:
        """Get statistics about citation links.

        Args:
            citation_links: Citation links DataFrame.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "total_links": len(citation_links),
            "unique_patents": citation_links["patent_id"].n_unique(),
            "unique_papers": citation_links["paper_id"].n_unique(),
        }

        # Citations per patent
        per_patent = citation_links.group_by("patent_id").agg(pl.len().alias("count"))
        stats["mean_citations_per_patent"] = per_patent["count"].mean()
        stats["median_citations_per_patent"] = per_patent["count"].median()
        stats["max_citations_per_patent"] = per_patent["count"].max()

        # Source distribution if available
        if "source" in citation_links.columns:
            source_counts = (
                citation_links.group_by("source")
                .agg(pl.len().alias("count"))
                .to_dicts()
            )
            stats["source_distribution"] = {
                d["source"]: d["count"] for d in source_counts
            }

        # Confidence distribution if available
        if "confidence" in citation_links.columns:
            stats["mean_confidence"] = citation_links["confidence"].mean()
            stats["confidence_distribution"] = (
                citation_links.group_by("confidence")
                .agg(pl.len().alias("count"))
                .sort("confidence")
                .to_dicts()
            )

        return stats

    def process_links(
        self,
        ros_df: pl.DataFrame,
        patents_df: pl.DataFrame,
        papers_df: pl.DataFrame,
        claims_df: pl.DataFrame,
        validate_temporal: bool = True,
        save: bool = True,
    ) -> pl.DataFrame:
        """Full citation linking pipeline.

        Args:
            ros_df: Reliance on Science data.
            patents_df: Processed patents.
            papers_df: Processed papers.
            claims_df: Claims with query_ids.
            validate_temporal: Whether to validate temporal constraints.
            save: Whether to save processed links.

        Returns:
            Processed citation links at claim level.
        """
        logger.info("Starting citation linking pipeline")

        # Create patent-paper links
        citation_links = self.create_citation_links(ros_df, patents_df, papers_df)

        # Expand to claim level
        expanded = self.expand_to_claims(citation_links, claims_df)

        # Add paper dates
        with_dates = self.add_paper_dates(expanded, papers_df)

        # Validate temporal constraints
        if validate_temporal:
            valid_links, invalid_links = self.validate_temporal_constraint(with_dates)

            # Log violations for analysis
            if len(invalid_links) > 0:
                violations_path = self.processed_dir / "temporal_violations.parquet"
                invalid_links.write_parquet(violations_path)
                logger.info(f"Saved {len(invalid_links)} temporal violations to {violations_path}")

            final_links = valid_links
        else:
            final_links = with_dates

        # Print stats
        stats = self.get_citation_stats(citation_links)
        logger.info(f"Citation statistics: {stats}")

        if save:
            final_links.write_parquet(self.links_path)
            logger.info(f"Saved citation links to {self.links_path}")

        return final_links

    def load(self) -> pl.DataFrame:
        """Load processed citation links from disk.

        Returns:
            Citation links DataFrame.
        """
        if not self.links_path.exists():
            raise FileNotFoundError(f"Citation links not found at {self.links_path}")
        return pl.read_parquet(self.links_path)
