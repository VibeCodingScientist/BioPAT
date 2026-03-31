"""Prior patent candidate selection module.

Phase 5 (v2.0): Identifies prior patents to add to the dual-corpus,
matching the real-world workflow of patent examiners who search
both scientific literature and earlier patents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import polars as pl

from biopat import compat
from biopat.benchmark.beir_format import BEIRFormatter, DOC_TYPE_PATENT

logger = logging.getLogger(__name__)


class PriorPatentSelector:
    """Selects prior patent candidates for the v2.0 dual-corpus.

    Prior patents are selected from three sources:
    1. Office Action patent citations (Gold Standard) - examiner-cited
    2. Query-cited patents (Applicant prior art) - applicant-disclosed
    3. Same-IPC patents (Hard negatives) - same subclass, different invention

    All candidates must pass temporal validation (predate query priority date).
    """

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def prior_patents_path(self) -> Path:
        return self.processed_dir / "prior_patents.parquet"

    @property
    def patent_corpus_path(self) -> Path:
        return self.processed_dir / "patent_corpus.parquet"

    def extract_oa_patent_citations(
        self,
        office_action_citations: pl.DataFrame,
    ) -> pl.DataFrame:
        """Extract patent citations from Office Action data.

        These are Gold Standard citations - examiner-validated prior art
        used in 102/103 rejections.

        Args:
            office_action_citations: Office Action citations DataFrame with
                app_id, cite_pat_pgpub_id, rejection_type columns.

        Returns:
            DataFrame with query_patent_id, cited_patent_id, source, rejection_type.
        """
        # Filter to patent citations only (not NPL)
        patent_citations = office_action_citations.filter(
            pl.col("cite_pat_pgpub_id").is_not_null() &
            (pl.col("cite_pat_pgpub_id").str.len_chars() > 0)
        )

        if len(patent_citations) == 0:
            logger.warning("No patent citations found in Office Action data")
            return pl.DataFrame({
                "query_patent_id": [],
                "cited_patent_id": [],
                "source": [],
                "rejection_type": [],
            })

        # Standardize output format
        result = patent_citations.select([
            pl.col("app_id").alias("query_patent_id"),
            pl.col("cite_pat_pgpub_id").alias("cited_patent_id"),
            pl.lit("office_action").alias("source"),
            pl.col("rejection_type").fill_null("unknown"),
        ])

        # Deduplicate
        result = compat.unique(result, subset=["query_patent_id", "cited_patent_id"])

        logger.info(f"Extracted {len(result)} patent citations from Office Actions")
        return result

    def extract_applicant_citations(
        self,
        ros_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Extract applicant-cited patents from Reliance on Science data.

        These are patents cited by the applicant as prior art disclosure.

        Args:
            ros_data: RoS DataFrame with patent_id, cited_patents columns.

        Returns:
            DataFrame with query_patent_id, cited_patent_id, source.
        """
        # Check if cited_patents column exists
        if "cited_patents" not in ros_data.columns:
            logger.info("No cited_patents column in RoS data, skipping applicant citations")
            return pl.DataFrame({
                "query_patent_id": [],
                "cited_patent_id": [],
                "source": [],
                "rejection_type": [],
            })

        # Explode cited patents list
        exploded = ros_data.select([
            pl.col("patent_id").alias("query_patent_id"),
            pl.col("cited_patents").explode().alias("cited_patent_id"),
        ]).filter(
            pl.col("cited_patent_id").is_not_null()
        )

        if len(exploded) == 0:
            return pl.DataFrame({
                "query_patent_id": [],
                "cited_patent_id": [],
                "source": [],
                "rejection_type": [],
            })

        result = exploded.with_columns([
            pl.lit("applicant").alias("source"),
            pl.lit(None).alias("rejection_type"),
        ])

        result = compat.unique(result, subset=["query_patent_id", "cited_patent_id"])

        logger.info(f"Extracted {len(result)} applicant-cited patents")
        return result

    def select_ipc_hard_negatives(
        self,
        query_patents: pl.DataFrame,
        candidate_patents: pl.DataFrame,
        max_per_query: int = 10,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Select hard negative patents from same IPC subclass.

        These are patents in the same technical domain that are NOT
        cited as prior art - useful for training discriminative models.

        Args:
            query_patents: Query patents with patent_id, ipc_codes, priority_date.
            candidate_patents: Pool of candidate patents.
            max_per_query: Maximum negatives per query.
            seed: Random seed for sampling.

        Returns:
            DataFrame with query_patent_id, cited_patent_id, source.
        """
        records = []

        for row in compat.iter_rows(query_patents, named=True):
            query_id = row["patent_id"]
            ipc_codes = row.get("ipc_codes", [])
            priority_date = row.get("priority_date")

            if not ipc_codes:
                continue

            # Get primary IPC4 (e.g., "A61K")
            primary_ipc4 = ipc_codes[0][:4] if ipc_codes[0] else None
            if not primary_ipc4:
                continue

            # Find candidates with same IPC4 prefix
            def has_same_ipc4(candidate_ipc_codes):
                if not candidate_ipc_codes:
                    return False
                return any(
                    str(code).startswith(primary_ipc4)
                    for code in candidate_ipc_codes
                )

            same_ipc_candidates = candidate_patents.filter(
                pl.col("ipc_codes").map_elements(
                    has_same_ipc4, return_dtype=pl.Boolean
                ) &
                (pl.col("patent_id") != query_id)
            )

            # Apply temporal filter if we have priority date
            if priority_date:
                same_ipc_candidates = same_ipc_candidates.filter(
                    pl.col("priority_date") < priority_date
                )

            if len(same_ipc_candidates) == 0:
                continue

            # Sample up to max_per_query
            n_sample = min(max_per_query, len(same_ipc_candidates))
            sampled = compat.sample_df(
                same_ipc_candidates, n=n_sample, seed=seed
            )

            for neg_row in compat.iter_rows(sampled, named=True):
                records.append({
                    "query_patent_id": query_id,
                    "cited_patent_id": neg_row["patent_id"],
                    "source": "ipc_negative",
                    "rejection_type": None,
                })

        if not records:
            return pl.DataFrame({
                "query_patent_id": [],
                "cited_patent_id": [],
                "source": [],
                "rejection_type": [],
            })

        result = pl.DataFrame(records)
        logger.info(f"Selected {len(result)} IPC-based hard negatives")
        return result

    def apply_temporal_filter(
        self,
        citations: pl.DataFrame,
        query_patents: pl.DataFrame,
        cited_patents: pl.DataFrame,
    ) -> pl.DataFrame:
        """Filter citations to ensure prior art predates query priority date.

        Hard temporal constraint: cited patent priority date must be
        BEFORE query patent priority date.

        Args:
            citations: Citations with query_patent_id, cited_patent_id.
            query_patents: Query patents with patent_id, priority_date.
            cited_patents: Cited patents with patent_id, priority_date.

        Returns:
            Temporally valid citations.
        """
        # Join query priority dates
        with_query_dates = citations.join(
            query_patents.select([
                pl.col("patent_id").alias("query_patent_id"),
                pl.col("priority_date").alias("query_priority_date"),
            ]),
            on="query_patent_id",
            how="left"
        )

        # Join cited patent priority dates
        with_all_dates = with_query_dates.join(
            cited_patents.select([
                pl.col("patent_id").alias("cited_patent_id"),
                pl.col("priority_date").alias("cited_priority_date"),
            ]),
            on="cited_patent_id",
            how="left"
        )

        # Filter: cited must predate query
        valid = with_all_dates.filter(
            (pl.col("cited_priority_date").is_null()) |
            (pl.col("query_priority_date").is_null()) |
            (pl.col("cited_priority_date") < pl.col("query_priority_date"))
        )

        removed = len(with_all_dates) - len(valid)
        if removed > 0:
            logger.info(f"Temporal filter removed {removed} citations (postdate query)")

        # Return original columns only
        return valid.select(citations.columns)

    def merge_citation_sources(
        self,
        oa_citations: pl.DataFrame,
        applicant_citations: pl.DataFrame,
        ipc_negatives: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Merge citations from all sources with priority ordering.

        Priority: Office Action > Applicant > IPC Negative

        Args:
            oa_citations: Office Action patent citations.
            applicant_citations: Applicant-disclosed citations.
            ipc_negatives: Optional IPC-based hard negatives.

        Returns:
            Merged citations DataFrame.
        """
        sources = [oa_citations, applicant_citations]
        if ipc_negatives is not None:
            sources.append(ipc_negatives)

        # Filter out empty DataFrames
        sources = [df for df in sources if len(df) > 0]

        if not sources:
            return pl.DataFrame({
                "query_patent_id": [],
                "cited_patent_id": [],
                "source": [],
                "rejection_type": [],
            })

        merged = compat.concat(sources)

        # Deduplicate, keeping highest priority source
        # (Office Action > Applicant > IPC Negative)
        source_priority = {"office_action": 3, "applicant": 2, "ipc_negative": 1}

        merged = merged.with_columns(
            pl.col("source").map_elements(
                lambda s: source_priority.get(s, 0),
                return_dtype=pl.Int32
            ).alias("_priority")
        )

        merged = compat.unique(
            merged.sort("_priority", descending=True),
            subset=["query_patent_id", "cited_patent_id"],
            keep="first"
        ).drop("_priority")

        logger.info(f"Merged {len(merged)} unique patent citations")
        return merged

    def get_unique_cited_patents(
        self,
        citations: pl.DataFrame,
    ) -> List[str]:
        """Get list of unique cited patent IDs for fetching.

        Args:
            citations: Citations DataFrame with cited_patent_id column.

        Returns:
            List of unique patent IDs.
        """
        unique_ids = compat.unique(
            citations.select("cited_patent_id")
        ).to_series().to_list()
        logger.info(f"Found {len(unique_ids)} unique cited patents to fetch")
        return unique_ids

    def build_patent_corpus(
        self,
        patent_metadata: pl.DataFrame,
        citations: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build patent corpus DataFrame for BEIR format.

        Creates corpus entries with combined abstract + first independent claim.

        Args:
            patent_metadata: Patent metadata with patent_id, title, abstract, claims.
            citations: Citations to determine which patents to include.

        Returns:
            Patent corpus DataFrame ready for BEIR formatting.
        """
        # Get unique cited patents
        cited_ids = set(self.get_unique_cited_patents(citations))

        # Filter to cited patents
        corpus_patents = patent_metadata.filter(
            compat.is_in(pl.col("patent_id"), cited_ids)
        )

        # Create corpus text (abstract + first independent claim)
        records = []
        for row in compat.iter_rows(corpus_patents, named=True):
            patent_id = row["patent_id"]
            title = row.get("title", "") or ""
            abstract = row.get("abstract", "") or ""
            claims = row.get("claims", [])

            # Find first independent claim
            first_independent = None
            if claims:
                for claim in claims:
                    claim_text = claim.get("claim_text", "")
                    # Simple heuristic: claims that don't reference other claims
                    if claim_text and "claim" not in claim_text.lower()[:50]:
                        first_independent = claim_text
                        break
                # Fallback to first claim
                if not first_independent and claims:
                    first_independent = claims[0].get("claim_text", "")

            # Combine abstract and first independent claim
            corpus_text = BEIRFormatter.create_patent_corpus_text(
                abstract, first_independent
            )

            records.append({
                "patent_id": patent_id,
                "title": title,
                "corpus_text": corpus_text,
                "priority_date": row.get("priority_date"),
                "ipc_codes": row.get("ipc_codes", []),
            })

        result = pl.DataFrame(records)
        logger.info(f"Built patent corpus with {len(result)} entries")

        return result

    def select_prior_patents(
        self,
        query_patents: pl.DataFrame,
        office_action_citations: Optional[pl.DataFrame] = None,
        ros_data: Optional[pl.DataFrame] = None,
        candidate_patents: Optional[pl.DataFrame] = None,
        include_ipc_negatives: bool = False,
        max_ipc_negatives: int = 10,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Full prior patent selection pipeline.

        Args:
            query_patents: Query patents (benchmark queries).
            office_action_citations: Office Action citation data.
            ros_data: Reliance on Science data.
            candidate_patents: Pool for IPC negative sampling.
            include_ipc_negatives: Whether to include IPC-based negatives.
            max_ipc_negatives: Max IPC negatives per query.
            seed: Random seed.

        Returns:
            Prior patent citations DataFrame.
        """
        all_citations = []

        # Extract Office Action patent citations
        if office_action_citations is not None:
            oa_cites = self.extract_oa_patent_citations(office_action_citations)
            if len(oa_cites) > 0:
                all_citations.append(oa_cites)

        # Extract applicant citations
        if ros_data is not None:
            app_cites = self.extract_applicant_citations(ros_data)
            if len(app_cites) > 0:
                all_citations.append(app_cites)

        # Generate IPC hard negatives
        ipc_negs = None
        if include_ipc_negatives and candidate_patents is not None:
            ipc_negs = self.select_ipc_hard_negatives(
                query_patents, candidate_patents,
                max_per_query=max_ipc_negatives, seed=seed
            )

        # Merge all sources
        if not all_citations:
            logger.warning("No prior patent citations found from any source")
            return pl.DataFrame({
                "query_patent_id": [],
                "cited_patent_id": [],
                "source": [],
                "rejection_type": [],
            })

        merged = self.merge_citation_sources(
            oa_citations=all_citations[0] if len(all_citations) > 0 else pl.DataFrame(),
            applicant_citations=all_citations[1] if len(all_citations) > 1 else pl.DataFrame(),
            ipc_negatives=ipc_negs,
        )

        # Save
        merged.write_parquet(self.prior_patents_path)
        logger.info(f"Saved {len(merged)} prior patent citations to {self.prior_patents_path}")

        return merged

    def get_citation_stats(self, citations: pl.DataFrame) -> Dict[str, Any]:
        """Get statistics about prior patent citations.

        Args:
            citations: Citations DataFrame.

        Returns:
            Statistics dictionary.
        """
        stats = {
            "total_citations": len(citations),
            "unique_query_patents": citations["query_patent_id"].n_unique(),
            "unique_cited_patents": citations["cited_patent_id"].n_unique(),
        }

        # Source distribution
        source_counts = (
            compat.group_by(citations, "source")
            .agg(pl.len().alias("count"))
            .to_dicts()
        )
        stats["by_source"] = {r["source"]: r["count"] for r in source_counts}

        # Rejection type distribution (for OA citations)
        if "rejection_type" in citations.columns:
            oa_only = citations.filter(pl.col("source") == "office_action")
            if len(oa_only) > 0:
                rej_counts = (
                    compat.group_by(oa_only, "rejection_type")
                    .agg(pl.len().alias("count"))
                    .to_dicts()
                )
                stats["by_rejection_type"] = {
                    r["rejection_type"]: r["count"] for r in rej_counts
                }

        return stats

    def load(self) -> pl.DataFrame:
        """Load prior patent citations from disk.

        Returns:
            Prior patent citations DataFrame.
        """
        if not self.prior_patents_path.exists():
            raise FileNotFoundError(
                f"Prior patent citations not found at {self.prior_patents_path}"
            )
        return pl.read_parquet(self.prior_patents_path)
