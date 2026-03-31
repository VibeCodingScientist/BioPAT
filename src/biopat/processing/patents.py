"""Patent processing module.

Handles patent filtering by IPC codes, claim extraction and parsing,
and identification of independent vs dependent claims.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Optional
import polars as pl

from biopat import compat

logger = logging.getLogger(__name__)


class PatentProcessor:
    """Processor for patent data filtering and claim extraction."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def patents_path(self) -> Path:
        return self.processed_dir / "patents.parquet"

    def filter_by_ipc(
        self,
        patents_df: pl.DataFrame,
        ipc_prefixes: List[str] = ["A61", "C07", "C12"],
    ) -> pl.DataFrame:
        """Filter patents to biomedical IPC classes.

        Args:
            patents_df: DataFrame with patents and ipc_codes column.
            ipc_prefixes: IPC code prefixes to keep.

        Returns:
            Filtered DataFrame.
        """
        def has_biomedical_ipc(ipc_codes) -> bool:
            if ipc_codes is None or (hasattr(ipc_codes, '__len__') and len(ipc_codes) == 0):
                return False
            # Handle both list and other iterable types
            codes = list(ipc_codes) if not isinstance(ipc_codes, list) else ipc_codes
            return any(
                any(str(code).startswith(prefix) for prefix in ipc_prefixes)
                for code in codes
            )

        return patents_df.filter(
            pl.col("ipc_codes").map_elements(
                has_biomedical_ipc,
                return_dtype=pl.Boolean
            )
        )

    def filter_by_citation_count(
        self,
        patents_df: pl.DataFrame,
        citation_counts: pl.DataFrame,
        min_citations: int = 3,
    ) -> pl.DataFrame:
        """Filter patents with minimum citation count.

        Args:
            patents_df: DataFrame with patents.
            citation_counts: DataFrame with patent_id and citation_count columns.
            min_citations: Minimum number of high-confidence citations.

        Returns:
            Filtered DataFrame.
        """
        # Filter citation counts
        eligible_patents = citation_counts.filter(
            pl.col("citation_count") >= min_citations
        ).select("patent_id")

        # Join to filter patents
        return patents_df.join(
            eligible_patents,
            on="patent_id",
            how="inner"
        )

    @staticmethod
    def parse_claim_type(claim_text: str, claim_number: int) -> Tuple[str, Optional[int]]:
        """Determine if claim is independent or dependent.

        Args:
            claim_text: Full text of the claim.
            claim_number: The claim number.

        Returns:
            Tuple of (claim_type, depends_on_claim_number or None).
        """
        # Patterns for dependent claims
        dependent_patterns = [
            r"^\s*(?:The\s+|A\s+)?(?:method|composition|compound|device|system|apparatus|kit|article|process|formulation)\s+(?:of|according\s+to)\s+claim\s+(\d+)",
            r"^\s*A\s+(?:method|composition|compound|device|system|apparatus|kit|article|process|formulation)\s+as\s+(?:claimed|defined|set\s+forth)\s+in\s+claim\s+(\d+)",
            r"^\s*The\s+(?:method|composition|compound|device|system|apparatus|kit|article|process|formulation)\s+as\s+claimed\s+in\s+claim\s+(\d+)",
            r"^\s*(?:The\s+)?claim\s+(\d+)",
        ]

        for pattern in dependent_patterns:
            match = re.match(pattern, claim_text, re.IGNORECASE)
            if match:
                parent_claim = int(match.group(1))
                return ("dependent", parent_claim)

        # Check for "wherein" references to other claims
        wherein_pattern = r"claim\s+(\d+)\s*,?\s*wherein"
        match = re.search(wherein_pattern, claim_text, re.IGNORECASE)
        if match:
            parent_claim = int(match.group(1))
            return ("dependent", parent_claim)

        return ("independent", None)

    def extract_independent_claims(
        self,
        patents_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Extract independent claims from patents.

        Args:
            patents_df: DataFrame with claims column containing list of claim dicts.

        Returns:
            DataFrame with one row per independent claim.
        """
        records = []
        abstract_fallback_count = 0

        for row in compat.iter_rows(patents_df, named=True):
            patent_id = row["patent_id"]
            claims = row.get("claims", [])

            if not claims:
                # Fallback: use patent abstract as query when claims unavailable
                abstract = row.get("abstract", "") or ""
                title = row.get("title", "") or ""
                if abstract or title:
                    query_text = f"{title}. {abstract}".strip(". ")
                    records.append({
                        "query_id": f"{patent_id}-c0",
                        "patent_id": patent_id,
                        "claim_number": 0,
                        "claim_text": query_text,
                        "claim_type": "abstract_fallback",
                        "priority_date": row.get("priority_date"),
                        "ipc_codes": row.get("ipc_codes", []),
                    })
                    abstract_fallback_count += 1
                else:
                    logger.warning(f"No claims or abstract for patent {patent_id}")
                continue

            for claim in claims:
                claim_text = claim.get("claim_text", "")
                claim_number = claim.get("claim_number", 0)

                if not claim_text:
                    continue

                # Determine claim type
                claim_type, depends_on = self.parse_claim_type(claim_text, claim_number)

                # Only include independent claims
                if claim_type == "independent":
                    records.append({
                        "query_id": f"{patent_id}-c{claim_number}",
                        "patent_id": patent_id,
                        "claim_number": claim_number,
                        "claim_text": claim_text,
                        "claim_type": claim_type,
                        "priority_date": row.get("priority_date"),
                        "ipc_codes": row.get("ipc_codes", []),
                    })

        if abstract_fallback_count:
            logger.info(
                f"Used abstract fallback for {abstract_fallback_count} patents "
                f"(claims not available from API)"
            )

        return pl.DataFrame(records)

    @staticmethod
    def get_ipc3(ipc_codes) -> str:
        """Get primary IPC3 code (first 4 characters) for domain stratification.

        Args:
            ipc_codes: List of IPC codes (may be a Python list or Polars Series).

        Returns:
            Primary IPC3 code or empty string.
        """
        # Handle Polars Series from map_elements
        if hasattr(ipc_codes, 'to_list'):
            ipc_codes = ipc_codes.to_list()
        if not ipc_codes or len(ipc_codes) == 0:
            return ""
        # Return first 4 characters of first code (e.g., "A61K")
        first = ipc_codes[0]
        return first[:4] if first else ""

    def add_domain_info(self, claims_df: pl.DataFrame) -> pl.DataFrame:
        """Add domain stratification column based on IPC codes.

        Args:
            claims_df: DataFrame with ipc_codes column.

        Returns:
            DataFrame with added 'domain' column.
        """
        return claims_df.with_columns(
            compat.map_elements(
                pl.col("ipc_codes"),
                self.get_ipc3,
                return_dtype=pl.Utf8
            ).alias("domain")
        )

    def process_patents(
        self,
        patents_df: pl.DataFrame,
        citation_counts: pl.DataFrame,
        ipc_prefixes: List[str] = ["A61", "C07", "C12"],
        min_citations: int = 3,
        save: bool = True,
    ) -> pl.DataFrame:
        """Full patent processing pipeline.

        Args:
            patents_df: Raw patent DataFrame.
            citation_counts: Citation count DataFrame from RoS.
            ipc_prefixes: IPC prefixes for biomedical filtering.
            min_citations: Minimum citations required.
            save: Whether to save processed data.

        Returns:
            Processed patents DataFrame.
        """
        logger.info(f"Starting patent processing with {len(patents_df)} patents")

        # Filter by IPC codes
        filtered = self.filter_by_ipc(patents_df, ipc_prefixes)
        logger.info(f"After IPC filter: {len(filtered)} patents")

        # Filter by citation count
        filtered = self.filter_by_citation_count(filtered, citation_counts, min_citations)
        logger.info(f"After citation filter: {len(filtered)} patents")

        # Extract independent claims
        claims_df = self.extract_independent_claims(filtered)
        logger.info(f"Extracted {len(claims_df)} independent claims")

        # Add domain info
        claims_df = self.add_domain_info(claims_df)

        if save:
            claims_df.write_parquet(self.patents_path)
            logger.info(f"Saved processed patents to {self.patents_path}")

        return claims_df

    def sample_patents(
        self,
        claims_df: pl.DataFrame,
        target_count: int = 2000,
        stratify_by: str = "domain",
        seed: int = 42,
    ) -> pl.DataFrame:
        """Sample patents with stratification by domain.

        Args:
            claims_df: DataFrame with claims.
            target_count: Target number of unique patents.
            stratify_by: Column to stratify by.
            seed: Random seed.

        Returns:
            Sampled DataFrame.
        """
        # Get unique patents
        unique_patents = compat.unique(claims_df.select("patent_id", stratify_by))

        if len(unique_patents) <= target_count:
            logger.info(f"Only {len(unique_patents)} patents available, returning all")
            return claims_df

        # Stratified sampling
        sampled_patents = (
            compat.group_by(unique_patents, stratify_by)
            .agg(pl.col("patent_id"))
            .with_columns(
                pl.col("patent_id").list.sample(
                    fraction=target_count / len(unique_patents),
                    seed=seed
                ).alias("sampled_ids")
            )
            .select(pl.col("sampled_ids").explode().alias("patent_id"))
        )

        # Limit to target count
        sampled_patents = sampled_patents.head(target_count)

        # Filter claims to sampled patents
        return claims_df.join(sampled_patents, on="patent_id", how="inner")

    def load(self) -> pl.DataFrame:
        """Load processed patents from disk.

        Returns:
            Processed patents DataFrame.
        """
        if not self.patents_path.exists():
            raise FileNotFoundError(f"Processed patents not found at {self.patents_path}")
        return pl.read_parquet(self.patents_path)
