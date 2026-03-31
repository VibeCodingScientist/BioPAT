"""Temporal validation module.

Ensures that only documents published before the patent priority date
are considered valid prior art, following patent law requirements.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import polars as pl

from biopat import compat

logger = logging.getLogger(__name__)


class TemporalValidator:
    """Validates temporal constraints for prior art."""

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = Path(processed_dir) if processed_dir else None

    @staticmethod
    def parse_date(date_value: Union[str, date, datetime, None]) -> Optional[date]:
        """Parse various date formats to date object.

        Args:
            date_value: Date in various formats.

        Returns:
            date object or None if parsing fails.
        """
        if date_value is None:
            return None

        if isinstance(date_value, date):
            return date_value

        if isinstance(date_value, datetime):
            return date_value.date()

        if isinstance(date_value, str):
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d-%m-%Y", "%m/%d/%Y"]:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue

            # Try partial dates (year only, year-month)
            try:
                if len(date_value) == 4:
                    return date(int(date_value), 1, 1)
                if len(date_value) == 7:
                    parts = date_value.split("-")
                    return date(int(parts[0]), int(parts[1]), 1)
            except (ValueError, IndexError):
                pass

        logger.warning(f"Could not parse date: {date_value}")
        return None

    def validate_single(
        self,
        paper_date: Union[str, date, datetime, None],
        patent_priority_date: Union[str, date, datetime, None],
    ) -> bool:
        """Check if a single paper is valid prior art.

        A paper is valid prior art if it was published BEFORE
        the patent's priority date.

        Args:
            paper_date: Publication date of the paper.
            patent_priority_date: Priority date of the patent.

        Returns:
            True if paper is valid prior art, False otherwise.
        """
        paper_dt = self.parse_date(paper_date)
        patent_dt = self.parse_date(patent_priority_date)

        if paper_dt is None or patent_dt is None:
            # If dates are missing, we cannot validate - treat as invalid
            logger.debug(
                f"Missing date for validation: paper={paper_date}, patent={patent_priority_date}"
            )
            return False

        # Paper must be strictly before patent priority date
        return paper_dt < patent_dt

    def validate_dataframe(
        self,
        df: pl.DataFrame,
        paper_date_col: str = "publication_date",
        patent_date_col: str = "priority_date",
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Validate temporal constraints on a DataFrame.

        Args:
            df: DataFrame with paper and patent dates.
            paper_date_col: Name of paper date column.
            patent_date_col: Name of patent date column.

        Returns:
            Tuple of (valid_df, invalid_df).
        """
        # Ensure date columns are in date format
        df = df.with_columns([
            pl.col(paper_date_col).cast(pl.Date),
            pl.col(patent_date_col).cast(pl.Date),
        ])

        # Count rows with missing dates
        missing_paper_dates = df.filter(pl.col(paper_date_col).is_null()).height
        missing_patent_dates = df.filter(pl.col(patent_date_col).is_null()).height

        if missing_paper_dates > 0:
            logger.warning(f"{missing_paper_dates} rows have missing paper dates")
        if missing_patent_dates > 0:
            logger.warning(f"{missing_patent_dates} rows have missing patent dates")

        # Valid: paper published before patent priority date
        # (rows with missing dates are treated as invalid)
        valid = df.filter(
            pl.col(paper_date_col).is_not_null() &
            pl.col(patent_date_col).is_not_null() &
            (pl.col(paper_date_col) < pl.col(patent_date_col))
        )

        # Invalid: paper published on/after priority date OR missing dates
        invalid = df.filter(
            pl.col(paper_date_col).is_null() |
            pl.col(patent_date_col).is_null() |
            (pl.col(paper_date_col) >= pl.col(patent_date_col))
        )

        logger.info(
            f"Temporal validation complete: "
            f"{valid.height} valid ({valid.height / max(1, df.height) * 100:.1f}%), "
            f"{invalid.height} invalid"
        )

        return valid, invalid

    def analyze_violations(
        self,
        invalid_df: pl.DataFrame,
        paper_date_col: str = "publication_date",
        patent_date_col: str = "priority_date",
    ) -> dict:
        """Analyze temporal violations for reporting.

        Args:
            invalid_df: DataFrame of invalid (violating) rows.
            paper_date_col: Name of paper date column.
            patent_date_col: Name of patent date column.

        Returns:
            Dictionary with violation analysis.
        """
        if invalid_df.height == 0:
            return {"total_violations": 0}

        analysis = {
            "total_violations": invalid_df.height,
        }

        # Count by violation type
        missing_paper = invalid_df.filter(pl.col(paper_date_col).is_null()).height
        missing_patent = invalid_df.filter(pl.col(patent_date_col).is_null()).height
        date_violations = invalid_df.filter(
            pl.col(paper_date_col).is_not_null() &
            pl.col(patent_date_col).is_not_null() &
            (pl.col(paper_date_col) >= pl.col(patent_date_col))
        ).height

        analysis["missing_paper_date"] = missing_paper
        analysis["missing_patent_date"] = missing_patent
        analysis["date_violations"] = date_violations

        # Analyze date differences for actual violations
        if date_violations > 0:
            violations_only = invalid_df.filter(
                pl.col(paper_date_col).is_not_null() &
                pl.col(patent_date_col).is_not_null() &
                (pl.col(paper_date_col).cast(pl.Date) >= pl.col(patent_date_col).cast(pl.Date))
            ).with_columns(
                (pl.col(paper_date_col).cast(pl.Date) - pl.col(patent_date_col).cast(pl.Date))
                .dt.total_days()
                .alias("days_after")
            )

            analysis["mean_days_after"] = violations_only["days_after"].mean()
            analysis["max_days_after"] = violations_only["days_after"].max()

            # Distribution of violation severity
            analysis["violations_by_year"] = (
                compat.group_by(
                    violations_only
                    .with_columns((pl.col("days_after") / 365).floor().alias("years_after")),
                    "years_after"
                )
                .agg(pl.len().alias("count"))
                .sort("years_after")
                .to_dicts()
            )

        return analysis

    def get_date_range_stats(
        self,
        df: pl.DataFrame,
        paper_date_col: str = "publication_date",
        patent_date_col: str = "priority_date",
    ) -> dict:
        """Get statistics about date ranges in the data.

        Args:
            df: DataFrame with dates.
            paper_date_col: Name of paper date column.
            patent_date_col: Name of patent date column.

        Returns:
            Dictionary with date range statistics.
        """
        stats = {}

        # Paper dates
        if paper_date_col in df.columns:
            paper_dates = df.filter(pl.col(paper_date_col).is_not_null())
            if paper_dates.height > 0:
                stats["paper_date_min"] = str(paper_dates[paper_date_col].min())
                stats["paper_date_max"] = str(paper_dates[paper_date_col].max())
                stats["papers_with_dates"] = paper_dates.height

        # Patent dates
        if patent_date_col in df.columns:
            patent_dates = df.filter(pl.col(patent_date_col).is_not_null())
            if patent_dates.height > 0:
                stats["patent_date_min"] = str(patent_dates[patent_date_col].min())
                stats["patent_date_max"] = str(patent_dates[patent_date_col].max())
                stats["patents_with_dates"] = patent_dates.height

        return stats

    def filter_corpus_by_date(
        self,
        corpus_df: pl.DataFrame,
        max_date: str,
        date_col: str = "publication_date",
    ) -> pl.DataFrame:
        """Filter corpus to papers published before a cutoff date.

        Useful for ensuring the corpus contains only valid potential prior art.

        Args:
            corpus_df: Papers DataFrame.
            max_date: Maximum publication date (exclusive).
            date_col: Name of date column.

        Returns:
            Filtered DataFrame.
        """
        max_dt = self.parse_date(max_date)
        if max_dt is None:
            raise ValueError(f"Invalid max_date: {max_date}")

        return corpus_df.filter(
            pl.col(date_col).is_not_null() &
            (pl.col(date_col).cast(pl.Date) < max_dt)
        )

    def validate_and_report(
        self,
        df: pl.DataFrame,
        paper_date_col: str = "publication_date",
        patent_date_col: str = "priority_date",
        save_violations: bool = True,
    ) -> Tuple[pl.DataFrame, Dict]:
        """Validate temporal constraints and generate report.

        Args:
            df: DataFrame with dates.
            paper_date_col: Name of paper date column.
            patent_date_col: Name of patent date column.
            save_violations: Whether to save violations to disk.

        Returns:
            Tuple of (valid_df, report_dict).
        """
        # Validate
        valid, invalid = self.validate_dataframe(df, paper_date_col, patent_date_col)

        # Analyze violations
        report = self.analyze_violations(invalid, paper_date_col, patent_date_col)
        report["date_range_stats"] = self.get_date_range_stats(df, paper_date_col, patent_date_col)

        # Log report
        logger.info(f"Temporal validation report: {report}")

        # Save violations if requested
        if save_violations and self.processed_dir and invalid.height > 0:
            violations_path = self.processed_dir / "temporal_violations.parquet"
            invalid.write_parquet(violations_path)
            logger.info(f"Saved {invalid.height} violations to {violations_path}")

        return valid, report
