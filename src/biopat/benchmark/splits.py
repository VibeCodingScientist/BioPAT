"""Dataset splitting module.

Handles train/dev/test splitting with stratification to ensure
balanced domain representation across all splits.

REPRODUCIBILITY NOTE: This module uses a fixed seed (42) for all random
operations. The splitting algorithm is deterministic: given the same
input data and seed, the output splits will be identical.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import polars as pl

from biopat import compat
from biopat.reproducibility import REPRODUCIBILITY_SEED

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Splitter for creating train/dev/test sets with stratification.

    Uses a fixed random seed to ensure deterministic splits. Running
    the same code with the same data will always produce identical
    train/dev/test sets.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = REPRODUCIBILITY_SEED,
    ):
        """Initialize splitter with fixed seed.

        Args:
            train_ratio: Fraction of data for training (default 0.7).
            dev_ratio: Fraction of data for development (default 0.15).
            test_ratio: Fraction of data for testing (default 0.15).
            seed: Random seed for reproducibility. Defaults to REPRODUCIBILITY_SEED (42).
                  Changing this will produce different splits.
        """
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        if seed != REPRODUCIBILITY_SEED:
            logger.warning(
                f"Using non-standard seed {seed}. "
                f"For reproducibility, use REPRODUCIBILITY_SEED ({REPRODUCIBILITY_SEED})."
            )

        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_queries_stratified(
        self,
        queries_df: pl.DataFrame,
        stratify_col: str = "domain",
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split queries into train/dev/test with stratification.

        Ensures no query appears in multiple splits and maintains
        domain distribution across splits.

        Args:
            queries_df: Queries DataFrame with stratification column.
            stratify_col: Column to stratify by.

        Returns:
            Tuple of (train_df, dev_df, test_df).
        """
        # Get unique values for stratification
        domains = compat.unique(queries_df.select(stratify_col))[stratify_col].to_list()

        train_parts = []
        dev_parts = []
        test_parts = []

        for domain in domains:
            domain_df = queries_df.filter(pl.col(stratify_col) == domain)

            # Shuffle within domain
            domain_df = compat.sample_df(domain_df, fraction=1.0, seed=self.seed, shuffle=True)

            n = len(domain_df)
            train_n = int(n * self.train_ratio)
            dev_n = int(n * self.dev_ratio)

            # Ensure at least 1 sample per split if domain has enough samples
            if n >= 3:
                train_n = max(1, train_n)
                dev_n = max(1, dev_n)
                test_n = n - train_n - dev_n
            else:
                # Small domains go entirely to train
                train_n = n
                dev_n = 0
                test_n = 0

            train_parts.append(domain_df.head(train_n))
            if dev_n > 0:
                dev_parts.append(domain_df.slice(train_n, dev_n))
            if test_n > 0:
                test_parts.append(domain_df.slice(train_n + dev_n, test_n))

        train = compat.concat(train_parts) if train_parts else pl.DataFrame()
        dev = compat.concat(dev_parts) if dev_parts else pl.DataFrame()
        test = compat.concat(test_parts) if test_parts else pl.DataFrame()

        logger.info(
            f"Split queries: train={len(train)}, dev={len(dev)}, test={len(test)} "
            f"(stratified by {stratify_col})"
        )

        return train, dev, test

    def split_by_patent(
        self,
        queries_df: pl.DataFrame,
        stratify_col: str = "domain",
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split by patent ID to ensure all claims from a patent are in same split.

        This prevents data leakage where different claims from the same
        patent appear in different splits.

        Args:
            queries_df: Queries DataFrame with patent_id and stratification column.
            stratify_col: Column to stratify by.

        Returns:
            Tuple of (train_df, dev_df, test_df).
        """
        # Get unique patents with their domain
        unique_patents = compat.unique(
            queries_df.select(["patent_id", stratify_col])
        )

        # Split patents
        train_patents, dev_patents, test_patents = self.split_queries_stratified(
            unique_patents, stratify_col
        )

        # Map back to queries
        train_ids = train_patents["patent_id"].to_list() if len(train_patents) > 0 else []
        dev_ids = dev_patents["patent_id"].to_list() if len(dev_patents) > 0 else []
        test_ids = test_patents["patent_id"].to_list() if len(test_patents) > 0 else []

        train = queries_df.filter(compat.is_in(pl.col("patent_id"), train_ids)) if train_ids else queries_df.head(0)
        dev = queries_df.filter(compat.is_in(pl.col("patent_id"), dev_ids)) if dev_ids else queries_df.head(0)
        test = queries_df.filter(compat.is_in(pl.col("patent_id"), test_ids)) if test_ids else queries_df.head(0)

        logger.info(
            f"Split by patent: train={len(train)} ({len(train_patents)} patents), "
            f"dev={len(dev)} ({len(dev_patents)} patents), "
            f"test={len(test)} ({len(test_patents)} patents)"
        )

        return train, dev, test

    def split_qrels(
        self,
        qrels_df: pl.DataFrame,
        train_query_ids: Set[str],
        dev_query_ids: Set[str],
        test_query_ids: Set[str],
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split qrels according to query splits.

        Args:
            qrels_df: Qrels DataFrame.
            train_query_ids: Query IDs in train split.
            dev_query_ids: Query IDs in dev split.
            test_query_ids: Query IDs in test split.

        Returns:
            Tuple of (train_qrels, dev_qrels, test_qrels).
        """
        train_qrels = qrels_df.filter(pl.col("query_id").is_in(train_query_ids))
        dev_qrels = qrels_df.filter(pl.col("query_id").is_in(dev_query_ids))
        test_qrels = qrels_df.filter(pl.col("query_id").is_in(test_query_ids))

        logger.info(
            f"Split qrels: train={len(train_qrels)}, dev={len(dev_qrels)}, "
            f"test={len(test_qrels)}"
        )

        return train_qrels, dev_qrels, test_qrels

    def validate_splits(
        self,
        train_df: pl.DataFrame,
        dev_df: pl.DataFrame,
        test_df: pl.DataFrame,
        id_col: str = "query_id",
    ) -> bool:
        """Validate that splits have no overlap.

        Args:
            train_df: Train split.
            dev_df: Dev split.
            test_df: Test split.
            id_col: ID column to check for overlap.

        Returns:
            True if valid (no overlap), False otherwise.
        """
        train_ids = set(train_df[id_col].to_list())
        dev_ids = set(dev_df[id_col].to_list())
        test_ids = set(test_df[id_col].to_list())

        train_dev_overlap = train_ids & dev_ids
        train_test_overlap = train_ids & test_ids
        dev_test_overlap = dev_ids & test_ids

        if train_dev_overlap:
            logger.error(f"Train-dev overlap: {len(train_dev_overlap)} items")
            return False
        if train_test_overlap:
            logger.error(f"Train-test overlap: {len(train_test_overlap)} items")
            return False
        if dev_test_overlap:
            logger.error(f"Dev-test overlap: {len(dev_test_overlap)} items")
            return False

        logger.info("Split validation passed: no overlap between splits")
        return True

    def get_split_stats(
        self,
        train_df: pl.DataFrame,
        dev_df: pl.DataFrame,
        test_df: pl.DataFrame,
        stratify_col: str = "domain",
    ) -> dict:
        """Get statistics about the splits.

        Args:
            train_df: Train split.
            dev_df: Dev split.
            test_df: Test split.
            stratify_col: Stratification column.

        Returns:
            Dictionary of statistics.
        """
        total = len(train_df) + len(dev_df) + len(test_df)

        stats = {
            "train_count": len(train_df),
            "dev_count": len(dev_df),
            "test_count": len(test_df),
            "total_count": total,
            "train_ratio": len(train_df) / max(1, total),
            "dev_ratio": len(dev_df) / max(1, total),
            "test_ratio": len(test_df) / max(1, total),
        }

        # Domain distribution per split
        if stratify_col in train_df.columns:
            stats["train_domains"] = (
                train_df
                .group_by(stratify_col)
                .agg(pl.len().alias("count"))
                .sort(stratify_col)
                .to_dicts()
            )
            stats["dev_domains"] = (
                dev_df
                .group_by(stratify_col)
                .agg(pl.len().alias("count"))
                .sort(stratify_col)
                .to_dicts()
            ) if len(dev_df) > 0 else []
            stats["test_domains"] = (
                test_df
                .group_by(stratify_col)
                .agg(pl.len().alias("count"))
                .sort(stratify_col)
                .to_dicts()
            ) if len(test_df) > 0 else []

        return stats

    def create_splits(
        self,
        queries_df: pl.DataFrame,
        qrels_df: pl.DataFrame,
        stratify_col: str = "domain",
        split_by_patent: bool = True,
    ) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Create all splits for queries and qrels.

        Args:
            queries_df: Queries DataFrame.
            qrels_df: Qrels DataFrame.
            stratify_col: Column to stratify by.
            split_by_patent: If True, ensure all claims from a patent are in same split.

        Returns:
            Dictionary with 'train', 'dev', 'test' keys, each containing
            (queries_df, qrels_df) tuple.
        """
        # Split queries
        if split_by_patent and "patent_id" in queries_df.columns:
            train_q, dev_q, test_q = self.split_by_patent(queries_df, stratify_col)
        else:
            train_q, dev_q, test_q = self.split_queries_stratified(queries_df, stratify_col)

        # Validate no overlap
        assert self.validate_splits(train_q, dev_q, test_q, "query_id"), \
            "Invalid splits: overlap detected"

        # Split qrels
        train_qrels, dev_qrels, test_qrels = self.split_qrels(
            qrels_df,
            set(train_q["query_id"].to_list()),
            set(dev_q["query_id"].to_list()),
            set(test_q["query_id"].to_list()),
        )

        # Log stats
        stats = self.get_split_stats(train_q, dev_q, test_q, stratify_col)
        logger.info(f"Split statistics: {stats}")

        return {
            "train": (train_q, train_qrels),
            "dev": (dev_q, dev_qrels),
            "test": (test_q, test_qrels),
        }
