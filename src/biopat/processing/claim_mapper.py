"""Claim-rejection mapper module.

Maps Office Action rejections to specific patent claims,
extracting claim numbers from rejection text.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

logger = logging.getLogger(__name__)


class ClaimMapper:
    """Maps Office Action rejections to patent claims."""

    # Patterns for extracting claim numbers from rejection text
    CLAIM_PATTERNS = [
        # "Claims 1-5 and 12 are rejected"
        r"[Cc]laims?\s+([\d,\s\-and]+)\s+(?:is|are)\s+rejected",
        # "Claim 1 is rejected under 35 USC 102"
        r"[Cc]laim\s+(\d+)\s+is\s+rejected",
        # "rejected claims: 1, 3, 5-7"
        r"rejected\s+claims?[:\s]+([\d,\s\-and]+)",
        # Standalone claim references
        r"[Cc]laims?\s+([\d,\s\-and]+)",
    ]

    def __init__(self):
        self._patterns = [re.compile(p) for p in self.CLAIM_PATTERNS]

    def parse_claim_numbers(self, text: str) -> List[int]:
        """Extract claim numbers from rejection text.

        Handles various formats:
        - Single claims: "Claim 1"
        - Lists: "Claims 1, 3, 5"
        - Ranges: "Claims 1-5"
        - Combined: "Claims 1-5, 7, and 10-12"

        Args:
            text: Rejection text containing claim references.

        Returns:
            Sorted list of claim numbers.
        """
        claims: Set[int] = set()

        for pattern in self._patterns:
            for match in pattern.finditer(text):
                claim_text = match.group(1)
                claims.update(self._parse_claim_text(claim_text))

        return sorted(claims)

    def _parse_claim_text(self, claim_text: str) -> Set[int]:
        """Parse claim numbers from extracted text.

        Args:
            claim_text: Text like "1-5, 7, and 10"

        Returns:
            Set of claim numbers.
        """
        claims: Set[int] = set()

        # Remove "and" and extra whitespace
        claim_text = re.sub(r"\s+and\s+", ",", claim_text, flags=re.IGNORECASE)
        claim_text = re.sub(r"\s+", "", claim_text)

        # Split by comma
        parts = claim_text.split(",")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for range (e.g., "1-5")
            range_match = re.match(r"(\d+)-(\d+)", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                claims.update(range(start, end + 1))
            elif part.isdigit():
                claims.add(int(part))

        return claims

    def extract_rejection_type(self, text: str) -> Optional[str]:
        """Extract rejection type (102/103) from text.

        Args:
            text: Rejection text.

        Returns:
            "102", "103", or None.
        """
        # Patterns for 102/103 rejections
        if re.search(r"35\s*U\.?S\.?C\.?\s*§?\s*102", text, re.IGNORECASE):
            return "102"
        if re.search(r"§?\s*102\s*\(", text, re.IGNORECASE):
            return "102"
        if re.search(r"35\s*U\.?S\.?C\.?\s*§?\s*103", text, re.IGNORECASE):
            return "103"
        if re.search(r"§?\s*103\s*\(", text, re.IGNORECASE):
            return "103"

        # Fallback: check for keywords
        if "anticipat" in text.lower() or "anticpat" in text.lower():
            return "102"
        if "obvious" in text.lower():
            return "103"

        return None

    def map_rejections_to_claims(
        self,
        rejections_df: pl.DataFrame,
        rejection_text_col: str = "rejection_fp",
    ) -> pl.DataFrame:
        """Map rejections to their associated claims.

        Args:
            rejections_df: DataFrame with rejection data.
            rejection_text_col: Column containing rejection text.

        Returns:
            DataFrame with claims_list and rejection_type columns added.
        """
        claims_lists = []
        rejection_types = []

        for row in rejections_df.iter_rows(named=True):
            text = row.get(rejection_text_col, "") or ""

            # Extract claims
            claims = self.parse_claim_numbers(text)
            claims_lists.append(claims if claims else None)

            # Extract rejection type (may already exist in df)
            rej_type = row.get("rejection_type")
            if not rej_type:
                rej_type = self.extract_rejection_type(text)
            rejection_types.append(rej_type)

        result = rejections_df.with_columns([
            pl.Series("claims_list", claims_lists),
            pl.Series("rejection_type", rejection_types),
        ])

        # Log statistics
        total = len(result)
        with_claims = result.filter(pl.col("claims_list").is_not_null()).height
        with_102 = result.filter(pl.col("rejection_type") == "102").height
        with_103 = result.filter(pl.col("rejection_type") == "103").height

        logger.info(
            f"Mapped {total} rejections: "
            f"{with_claims} with claim numbers ({with_claims/total*100:.1f}%), "
            f"{with_102} §102, {with_103} §103"
        )

        return result

    def explode_claims(
        self,
        mapped_df: pl.DataFrame,
        claims_col: str = "claims_list",
    ) -> pl.DataFrame:
        """Explode claims list to one row per claim.

        Args:
            mapped_df: DataFrame with claims_list column.
            claims_col: Name of claims list column.

        Returns:
            DataFrame with one row per (rejection, claim) pair.
        """
        # Filter to rows with claims
        with_claims = mapped_df.filter(pl.col(claims_col).is_not_null())

        # Explode the claims list
        exploded = with_claims.explode(claims_col)

        # Rename to claim_number
        exploded = exploded.rename({claims_col: "claim_number"})

        logger.info(f"Exploded to {len(exploded)} claim-level rows")
        return exploded


class ClaimCitationMapper:
    """Maps claims to their cited papers from Office Actions."""

    def __init__(self):
        self.claim_mapper = ClaimMapper()

    def create_claim_paper_mapping(
        self,
        rejections_df: pl.DataFrame,
        linked_citations_df: pl.DataFrame,
        patents_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Create mapping from claims to cited papers.

        Args:
            rejections_df: Rejection data with app_id.
            linked_citations_df: Citations linked to papers.
            patents_df: Patents with patent_id and application_id.

        Returns:
            DataFrame with (query_id, paper_id, rejection_type) mapping.
        """
        # Map rejections to claims
        mapped_rejections = self.claim_mapper.map_rejections_to_claims(rejections_df)

        # Explode to claim level
        claim_level = self.claim_mapper.explode_claims(mapped_rejections)

        # Join with citations on app_id and ifw_number
        if "ifw_number" in claim_level.columns and "ifw_number" in linked_citations_df.columns:
            joined = claim_level.join(
                linked_citations_df.select([
                    "app_id", "ifw_number", "linked_paper_id", "link_method"
                ]),
                on=["app_id", "ifw_number"],
                how="inner"
            )
        else:
            # Fallback: join on app_id only
            joined = claim_level.join(
                linked_citations_df.select([
                    "app_id", "linked_paper_id", "link_method"
                ]).unique(),
                on="app_id",
                how="inner"
            )

        # Filter to linked citations only
        joined = joined.filter(pl.col("linked_paper_id").is_not_null())

        # Map application IDs to patent IDs
        if "application_id" in patents_df.columns:
            # Create mapping from app_id to patent_id
            app_to_patent = patents_df.select([
                pl.col("application_id").alias("app_id"),
                "patent_id"
            ]).unique()

            joined = joined.join(app_to_patent, on="app_id", how="left")
        else:
            # Try direct patent_id column
            joined = joined.with_columns(pl.col("app_id").alias("patent_id"))

        # Create query_id (patent_id + claim_number)
        joined = joined.with_columns(
            (pl.col("patent_id") + "-c" + pl.col("claim_number").cast(pl.Utf8)).alias("query_id")
        )

        # Select final columns
        result = joined.select([
            "query_id",
            "patent_id",
            "claim_number",
            "linked_paper_id",
            "rejection_type",
            "link_method",
        ]).rename({"linked_paper_id": "paper_id"})

        # Deduplicate
        result = result.unique(subset=["query_id", "paper_id"])

        logger.info(
            f"Created {len(result)} claim-paper mappings "
            f"for {result['query_id'].n_unique()} unique claims"
        )

        return result

    def get_mapping_stats(self, mapping_df: pl.DataFrame) -> Dict:
        """Get statistics about the claim-paper mapping.

        Args:
            mapping_df: Claim-paper mapping DataFrame.

        Returns:
            Dict of statistics.
        """
        stats = {
            "total_mappings": len(mapping_df),
            "unique_queries": mapping_df["query_id"].n_unique(),
            "unique_papers": mapping_df["paper_id"].n_unique(),
            "unique_patents": mapping_df["patent_id"].n_unique(),
        }

        # Rejection type distribution
        if "rejection_type" in mapping_df.columns:
            rej_dist = (
                mapping_df
                .group_by("rejection_type")
                .agg(pl.len().alias("count"))
                .to_dicts()
            )
            stats["rejection_types"] = {d["rejection_type"]: d["count"] for d in rej_dist}

        # Link method distribution
        if "link_method" in mapping_df.columns:
            link_dist = (
                mapping_df
                .group_by("link_method")
                .agg(pl.len().alias("count"))
                .to_dicts()
            )
            stats["link_methods"] = {d["link_method"]: d["count"] for d in link_dist}

        # Papers per query
        papers_per_query = mapping_df.group_by("query_id").agg(pl.len().alias("count"))
        stats["mean_papers_per_query"] = papers_per_query["count"].mean()
        stats["median_papers_per_query"] = papers_per_query["count"].median()

        return stats
