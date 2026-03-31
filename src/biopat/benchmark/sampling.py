"""Benchmark sampling module.

Handles stratified sampling of queries and corpus documents
to create a balanced and representative benchmark.

REPRODUCIBILITY NOTE: This module uses a fixed seed (42) for all random
operations to ensure deterministic benchmark generation. Any user running
the same code with the same data will produce identical results.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import polars as pl

from biopat import compat
from biopat.reproducibility import REPRODUCIBILITY_SEED

logger = logging.getLogger(__name__)


class BenchmarkSampler:
    """Sampler for creating benchmark datasets with stratification.

    All sampling operations use a fixed seed for reproducibility.
    The default seed (42) ensures that benchmark generation is
    deterministic across different runs and environments.
    """

    def __init__(self, seed: int = REPRODUCIBILITY_SEED):
        """Initialize sampler with fixed seed.

        Args:
            seed: Random seed for reproducibility. Defaults to REPRODUCIBILITY_SEED (42).
                  Changing this will produce different benchmark splits.
        """
        if seed != REPRODUCIBILITY_SEED:
            logger.warning(
                f"Using non-standard seed {seed}. "
                f"For reproducibility, use REPRODUCIBILITY_SEED ({REPRODUCIBILITY_SEED})."
            )
        self.seed = seed

    def sample_queries_stratified(
        self,
        queries_df: pl.DataFrame,
        target_count: int,
        stratify_col: str = "domain",
    ) -> pl.DataFrame:
        """Sample queries with stratification by domain.

        Maintains domain distribution from the original data while
        sampling down to target count.

        Args:
            queries_df: DataFrame with queries and stratification column.
            target_count: Target number of queries.
            stratify_col: Column to stratify by.

        Returns:
            Sampled queries DataFrame.
        """
        total = len(queries_df)

        if total <= target_count:
            logger.info(f"Only {total} queries available, returning all")
            return queries_df

        # Get domain distribution
        domain_counts = compat.group_by(queries_df, stratify_col).agg(pl.len().alias("count"))

        # Calculate sampling fraction per domain to maintain proportions
        sampling_fraction = target_count / total

        # Sample from each domain
        sampled_parts = []
        for row in compat.iter_rows(domain_counts, named=True):
            domain = row[stratify_col]
            domain_count = row["count"]
            sample_n = max(1, int(domain_count * sampling_fraction))

            domain_queries = queries_df.filter(pl.col(stratify_col) == domain)
            if len(domain_queries) > sample_n:
                sampled = compat.sample_df(domain_queries, n=sample_n, seed=self.seed)
            else:
                sampled = domain_queries

            sampled_parts.append(sampled)

        result = compat.concat(sampled_parts)

        # Trim to exact target if needed
        if len(result) > target_count:
            result = compat.sample_df(result, n=target_count, seed=self.seed)

        logger.info(
            f"Sampled {len(result)} queries from {total} "
            f"(stratified by {stratify_col})"
        )

        return result

    def sample_patents_with_min_citations(
        self,
        patents_df: pl.DataFrame,
        qrels_df: pl.DataFrame,
        target_patents: int,
        min_citations_per_patent: int = 3,
        stratify_col: str = "domain",
    ) -> pl.DataFrame:
        """Sample patents ensuring minimum citation coverage.

        Args:
            patents_df: Patents/claims DataFrame.
            qrels_df: Relevance judgments with query_id and doc_id.
            target_patents: Target number of unique patents.
            min_citations_per_patent: Minimum citations required per patent.
            stratify_col: Column to stratify by.

        Returns:
            Sampled patents DataFrame.
        """
        # Get citation counts per patent
        patent_citations = compat.group_by(
            qrels_df.with_columns(
                pl.col("query_id").str.extract(r"^(.+)-c\d+$", 1).alias("patent_id")
            ),
            "patent_id"
        ).agg(pl.col("doc_id").n_unique().alias("citation_count"))

        # Filter to patents with minimum citations
        eligible_patents = patent_citations.filter(
            pl.col("citation_count") >= min_citations_per_patent
        ).select("patent_id")

        logger.info(
            f"{len(eligible_patents)} patents have >= {min_citations_per_patent} citations"
        )

        # Get patents from claims dataframe
        eligible_patent_ids = eligible_patents["patent_id"].to_list()
        eligible_claims = patents_df.filter(
            compat.is_in(pl.col("patent_id"), eligible_patent_ids)
        )

        # Sample with stratification
        if len(eligible_claims) <= target_patents:
            return eligible_claims

        # Get unique patents with their domain
        unique_patents = compat.unique(
            eligible_claims.select(["patent_id", stratify_col])
        )

        # Stratified sample of patents
        sampled_patents = self._stratified_sample(
            unique_patents, target_patents, stratify_col
        )

        # Return all claims for sampled patents
        sampled_patent_ids = sampled_patents["patent_id"].to_list()
        return eligible_claims.filter(
            compat.is_in(pl.col("patent_id"), sampled_patent_ids)
        )

    def _stratified_sample(
        self,
        df: pl.DataFrame,
        target_count: int,
        stratify_col: str,
    ) -> pl.DataFrame:
        """Internal stratified sampling helper.

        Args:
            df: DataFrame to sample from.
            target_count: Target sample size.
            stratify_col: Column to stratify by.

        Returns:
            Sampled DataFrame.
        """
        total = len(df)
        if total <= target_count:
            return df

        sampling_fraction = target_count / total

        sampled_parts = []
        for domain in compat.unique(df.select(stratify_col))[stratify_col].to_list():
            domain_df = df.filter(pl.col(stratify_col) == domain)
            sample_n = max(1, int(len(domain_df) * sampling_fraction))

            if len(domain_df) > sample_n:
                sampled = compat.sample_df(domain_df, n=sample_n, seed=self.seed)
            else:
                sampled = domain_df

            sampled_parts.append(sampled)

        result = compat.concat(sampled_parts)

        if len(result) > target_count:
            result = compat.sample_df(result, n=target_count, seed=self.seed)

        return result

    def filter_corpus_to_queries(
        self,
        corpus_df: pl.DataFrame,
        qrels_df: pl.DataFrame,
        query_ids: List[str],
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Filter corpus to documents relevant to given queries.

        Args:
            corpus_df: Full corpus DataFrame.
            qrels_df: Relevance judgments.
            query_ids: List of query IDs to include.

        Returns:
            Tuple of (filtered_corpus, filtered_qrels).
        """
        # Filter qrels to selected queries
        filtered_qrels = qrels_df.filter(compat.is_in(pl.col("query_id"), query_ids))

        # Get document IDs from filtered qrels
        relevant_doc_ids = filtered_qrels["doc_id"].to_list()

        # Filter corpus to include relevant documents
        filtered_corpus = corpus_df.filter(
            compat.is_in(pl.col("paper_id"), relevant_doc_ids)
        )

        logger.info(
            f"Filtered corpus: {len(filtered_corpus)} docs for "
            f"{len(query_ids)} queries ({len(filtered_qrels)} qrels)"
        )

        return filtered_corpus, filtered_qrels

    def sample_negative_documents(
        self,
        corpus_df: pl.DataFrame,
        positive_doc_ids: Set[str],
        negatives_per_query: int = 100,
        seed: Optional[int] = None,
    ) -> pl.DataFrame:
        """Sample negative documents for evaluation.

        Args:
            corpus_df: Full corpus DataFrame.
            positive_doc_ids: Set of positive document IDs to exclude.
            negatives_per_query: Number of negatives to sample.
            seed: Random seed.

        Returns:
            DataFrame of negative documents.
        """
        if seed is None:
            seed = self.seed

        # Get non-positive documents
        positive_list = list(positive_doc_ids)
        negative_candidates = corpus_df.filter(
            ~compat.is_in(pl.col("paper_id"), positive_list)
        )

        if len(negative_candidates) <= negatives_per_query:
            return negative_candidates

        return compat.sample_df(negative_candidates, n=negatives_per_query, seed=seed)

    def get_sampling_stats(
        self,
        queries_df: pl.DataFrame,
        corpus_df: pl.DataFrame,
        qrels_df: pl.DataFrame,
        stratify_col: str = "domain",
    ) -> dict:
        """Get statistics about the sampled benchmark.

        Args:
            queries_df: Queries DataFrame.
            corpus_df: Corpus DataFrame.
            qrels_df: Qrels DataFrame.
            stratify_col: Stratification column.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "num_queries": len(queries_df),
            "num_documents": len(corpus_df),
            "num_qrels": len(qrels_df),
        }

        # Unique patents
        if "patent_id" in queries_df.columns:
            stats["num_unique_patents"] = queries_df["patent_id"].n_unique()

        # Domain distribution
        if stratify_col in queries_df.columns:
            stats["domain_distribution"] = (
                compat.group_by(queries_df, stratify_col)
                .agg(pl.len().alias("count"))
                .sort("count", reverse=True)
                .to_dicts()
            )

        # Qrels per query
        qrels_per_query = compat.group_by(qrels_df, "query_id").agg(pl.len().alias("count"))
        stats["mean_qrels_per_query"] = qrels_per_query["count"].mean()
        stats["median_qrels_per_query"] = qrels_per_query["count"].median()

        # Relevance distribution
        if "relevance" in qrels_df.columns:
            stats["relevance_distribution"] = (
                compat.group_by(qrels_df, "relevance")
                .agg(pl.len().alias("count"))
                .sort("relevance")
                .to_dicts()
            )

        return stats
