"""Paper processing module.

Handles paper metadata processing, abstract reconstruction,
and hard negative sampling for the corpus.
"""

import logging
from pathlib import Path
from typing import Optional
import polars as pl

from biopat import compat

logger = logging.getLogger(__name__)


class PaperProcessor:
    """Processor for scientific paper data."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def papers_path(self) -> Path:
        return self.processed_dir / "papers.parquet"

    def validate_papers(self, papers_df: pl.DataFrame) -> pl.DataFrame:
        """Validate paper data and filter invalid entries.

        Args:
            papers_df: Raw papers DataFrame.

        Returns:
            Validated DataFrame with required fields.
        """
        required_columns = ["paper_id", "title", "publication_date"]

        for col in required_columns:
            if col not in papers_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Filter papers with missing critical data
        validated = papers_df.filter(
            pl.col("paper_id").is_not_null() &
            pl.col("title").is_not_null() &
            (compat.str_len_chars(pl.col("title")) > 0)
        )

        logger.info(
            f"Validated {len(validated)}/{len(papers_df)} papers "
            f"({len(papers_df) - len(validated)} invalid removed)"
        )

        return validated

    def filter_by_date(
        self,
        papers_df: pl.DataFrame,
        max_date: Optional[str] = None,
        min_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """Filter papers by publication date range.

        Args:
            papers_df: Papers DataFrame.
            max_date: Maximum publication date (exclusive).
            min_date: Minimum publication date (inclusive).

        Returns:
            Filtered DataFrame.
        """
        result = papers_df

        if max_date:
            result = result.filter(pl.col("publication_date") < max_date)

        if min_date:
            result = result.filter(pl.col("publication_date") >= min_date)

        return result

    def deduplicate(self, papers_df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate papers by paper_id.

        Args:
            papers_df: Papers DataFrame.

        Returns:
            Deduplicated DataFrame.
        """
        original_count = len(papers_df)
        deduplicated = compat.unique(papers_df, subset=["paper_id"])

        if len(deduplicated) < original_count:
            logger.info(
                f"Removed {original_count - len(deduplicated)} duplicate papers"
            )

        return deduplicated

    def enrich_with_text_stats(self, papers_df: pl.DataFrame) -> pl.DataFrame:
        """Add text statistics for filtering and analysis.

        Args:
            papers_df: Papers DataFrame.

        Returns:
            DataFrame with added statistics columns.
        """
        return papers_df.with_columns([
            compat.str_len_chars(pl.col("title")).alias("title_length"),
            compat.str_len_chars(compat.fill_null(pl.col("abstract"), "")).alias("abstract_length"),
            (compat.str_len_chars(compat.fill_null(pl.col("abstract"), "")) > 0).alias("has_abstract"),
        ])

    def filter_quality(
        self,
        papers_df: pl.DataFrame,
        min_abstract_length: int = 50,
        require_abstract: bool = False,
    ) -> pl.DataFrame:
        """Filter papers by quality criteria.

        Args:
            papers_df: Papers DataFrame with text stats.
            min_abstract_length: Minimum abstract length if present.
            require_abstract: If True, only include papers with abstracts.

        Returns:
            Filtered DataFrame.
        """
        # Add stats if not present
        if "abstract_length" not in papers_df.columns:
            papers_df = self.enrich_with_text_stats(papers_df)

        if require_abstract:
            result = papers_df.filter(
                pl.col("abstract_length") >= min_abstract_length
            )
        else:
            # Keep papers without abstracts, but filter short abstracts
            result = papers_df.filter(
                (pl.col("abstract_length") == 0) |
                (pl.col("abstract_length") >= min_abstract_length)
            )

        logger.info(f"After quality filter: {len(result)} papers")
        return result

    def get_concept_distribution(self, papers_df: pl.DataFrame) -> pl.DataFrame:
        """Get distribution of OpenAlex concepts in corpus.

        Args:
            papers_df: Papers DataFrame with concepts column.

        Returns:
            DataFrame with concept counts.
        """
        if "concepts" not in papers_df.columns:
            return pl.DataFrame({"concept_id": [], "concept_name": [], "count": []})

        # Explode concepts and count
        exploded = (
            papers_df
            .select(pl.col("concepts").explode())
            .filter(pl.col("concepts").is_not_null())
            .with_columns([
                pl.col("concepts").struct.field("id").alias("concept_id"),
                pl.col("concepts").struct.field("name").alias("concept_name"),
            ])
        )
        concept_counts = (
            compat.group_by(exploded, ["concept_id", "concept_name"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        return concept_counts

    def select_hard_negatives_by_journal(
        self,
        papers_df: pl.DataFrame,
        positive_papers: pl.DataFrame,
        negatives_per_positive: int = 10,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Select hard negative papers from same journals as positives.

        Args:
            papers_df: Full corpus of papers.
            positive_papers: Papers that are positive examples.
            negatives_per_positive: Number of negatives per positive.
            seed: Random seed.

        Returns:
            DataFrame of hard negative papers.
        """
        # Get journals from positive papers
        positive_journals = compat.unique(
            positive_papers
            .filter(pl.col("journal").is_not_null())
            .select("journal")
        )

        # Get positive paper IDs to exclude
        positive_ids = list(positive_papers["paper_id"].to_list())

        # Find papers from same journals that are not positives
        candidates = (
            papers_df
            .join(positive_journals, on="journal", how="inner")
            .filter(~compat.is_in(pl.col("paper_id"), positive_ids))
        )

        if len(candidates) == 0:
            logger.warning("No hard negative candidates found from journals")
            return pl.DataFrame(schema=papers_df.schema)

        # Sample negatives
        target_count = len(positive_papers) * negatives_per_positive
        if len(candidates) <= target_count:
            return candidates

        return compat.sample_df(candidates, n=target_count, seed=seed)

    def select_hard_negatives_by_concepts(
        self,
        papers_df: pl.DataFrame,
        positive_papers: pl.DataFrame,
        negatives_per_positive: int = 10,
        concept_threshold: float = 0.5,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Select hard negatives based on concept overlap.

        Args:
            papers_df: Full corpus of papers.
            positive_papers: Papers that are positive examples.
            negatives_per_positive: Number of negatives per positive.
            concept_threshold: Minimum concept score to consider.
            seed: Random seed.

        Returns:
            DataFrame of hard negative papers.
        """
        if "concepts" not in papers_df.columns:
            return pl.DataFrame(schema=papers_df.schema)

        # Extract top concepts from positive papers
        positive_concepts = set()
        for concepts in positive_papers["concepts"].to_list():
            if concepts:
                for c in concepts:
                    if c.get("score", 0) >= concept_threshold:
                        positive_concepts.add(c.get("id"))

        if not positive_concepts:
            logger.warning("No concepts found in positive papers")
            return pl.DataFrame(schema=papers_df.schema)

        # Get positive paper IDs to exclude
        positive_ids = list(positive_papers["paper_id"].to_list())

        # Find papers with overlapping concepts
        def has_overlap(concepts):
            if not concepts:
                return False
            paper_concepts = {c.get("id") for c in concepts if c.get("score", 0) >= concept_threshold}
            return bool(paper_concepts & positive_concepts)

        candidates = papers_df.filter(
            ~compat.is_in(pl.col("paper_id"), positive_ids) &
            compat.map_elements(pl.col("concepts"), has_overlap, return_dtype=pl.Boolean)
        )

        # Sample negatives
        target_count = len(positive_papers) * negatives_per_positive
        if len(candidates) <= target_count:
            return candidates

        return compat.sample_df(candidates, n=target_count, seed=seed)

    def process_papers(
        self,
        papers_df: pl.DataFrame,
        require_abstract: bool = False,
        min_abstract_length: int = 50,
        save: bool = True,
    ) -> pl.DataFrame:
        """Full paper processing pipeline.

        Args:
            papers_df: Raw papers DataFrame.
            require_abstract: Whether to require abstracts.
            min_abstract_length: Minimum abstract length.
            save: Whether to save processed data.

        Returns:
            Processed papers DataFrame.
        """
        logger.info(f"Starting paper processing with {len(papers_df)} papers")

        # Validate
        processed = self.validate_papers(papers_df)

        # Deduplicate
        processed = self.deduplicate(processed)

        # Add text stats
        processed = self.enrich_with_text_stats(processed)

        # Quality filter
        processed = self.filter_quality(
            processed,
            min_abstract_length=min_abstract_length,
            require_abstract=require_abstract,
        )

        logger.info(f"Final processed corpus: {len(processed)} papers")

        if save:
            # Remove stats columns before saving
            save_columns = [c for c in processed.columns if c not in ["title_length", "abstract_length", "has_abstract"]]
            processed.select(save_columns).write_parquet(self.papers_path)
            logger.info(f"Saved processed papers to {self.papers_path}")

        return processed

    def build_corpus(
        self,
        positive_papers: pl.DataFrame,
        all_papers: Optional[pl.DataFrame] = None,
        add_hard_negatives: bool = True,
        negatives_per_positive: int = 10,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Build full corpus including hard negatives.

        Args:
            positive_papers: Papers that are positive examples (cited).
            all_papers: Full paper corpus for negative sampling.
            add_hard_negatives: Whether to add hard negatives.
            negatives_per_positive: Number of hard negatives per positive.
            seed: Random seed.

        Returns:
            Full corpus DataFrame.
        """
        corpus_parts = [positive_papers]

        if add_hard_negatives and all_papers is not None:
            # Add journal-based hard negatives
            journal_negatives = self.select_hard_negatives_by_journal(
                all_papers, positive_papers, negatives_per_positive // 2, seed
            )
            if len(journal_negatives) > 0:
                corpus_parts.append(journal_negatives)
                logger.info(f"Added {len(journal_negatives)} journal-based hard negatives")

            # Add concept-based hard negatives
            concept_negatives = self.select_hard_negatives_by_concepts(
                all_papers, positive_papers, negatives_per_positive // 2, seed
            )
            if len(concept_negatives) > 0:
                corpus_parts.append(concept_negatives)
                logger.info(f"Added {len(concept_negatives)} concept-based hard negatives")

        # Combine and deduplicate
        corpus = compat.unique(compat.concat(corpus_parts), subset=["paper_id"])
        logger.info(f"Final corpus size: {len(corpus)} papers")

        return corpus

    def load(self) -> pl.DataFrame:
        """Load processed papers from disk.

        Returns:
            Processed papers DataFrame.
        """
        if not self.papers_path.exists():
            raise FileNotFoundError(f"Processed papers not found at {self.papers_path}")
        return pl.read_parquet(self.papers_path)
