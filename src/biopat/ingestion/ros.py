"""Reliance on Science dataset loader.

Downloads and parses the RoS dataset from Zenodo which provides
pre-computed patent-to-paper citation links.
"""

import gzip
import logging
import random
import time
from pathlib import Path
from typing import Optional
import httpx
import polars as pl
from tqdm import tqdm

from biopat.reproducibility import ChecksumEngine, AuditLogger

logger = logging.getLogger(__name__)

ROS_ZENODO_URL = "https://zenodo.org/api/records/7996195/files/_pcs_oa.csv/content"
ROS_FILENAME = "_pcs_oa.csv"


class RelianceOnScienceLoader:
    """Loader for the Reliance on Science dataset."""

    def __init__(
        self,
        raw_dir: Path,
        cache_dir: Path,
        checksum_engine: Optional[ChecksumEngine] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checksum_engine = checksum_engine
        self.audit_logger = audit_logger

    @property
    def raw_file_path(self) -> Path:
        return self.raw_dir / ROS_FILENAME

    @property
    def parquet_path(self) -> Path:
        return self.raw_dir / "ros.parquet"

    def download(self, force: bool = False) -> Path:
        """Download RoS dataset from Zenodo.

        Args:
            force: If True, re-download even if file exists.

        Returns:
            Path to downloaded file.
        """
        if self.raw_file_path.exists() and not force:
            logger.info(f"RoS file already exists at {self.raw_file_path}")
            return self.raw_file_path

        logger.info(f"Downloading RoS dataset from {ROS_ZENODO_URL}")

        # Log API call
        if self.audit_logger:
            self.audit_logger.log_api_call(
                service="zenodo",
                endpoint="records/7996195/files/_pcs_oa.csv.gz",
                method="GET",
                params={"file": ROS_FILENAME},
            )

        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                with httpx.stream("GET", ROS_ZENODO_URL, follow_redirects=True, timeout=300.0) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))

                    with open(self.raw_file_path, "wb") as f:
                        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading RoS") as pbar:
                            for chunk in response.iter_bytes(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                break  # success
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Download failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, delay, e,
                )
                time.sleep(delay)

        logger.info(f"Downloaded RoS dataset to {self.raw_file_path}")

        # Log download with checksum
        if self.checksum_engine:
            self.checksum_engine.log_download(
                file_path=self.raw_file_path,
                source_url=ROS_ZENODO_URL,
                compute_hash=True,
            )

        return self.raw_file_path

    def load(
        self,
        confidence_threshold: int = 8,
        examiner_only: bool = False,
        force_reload: bool = False,
    ) -> pl.DataFrame:
        """Load and filter the RoS dataset.

        Args:
            confidence_threshold: Minimum confidence score (1-10).
            examiner_only: If True, only include examiner citations.
            force_reload: If True, re-parse from CSV even if parquet exists.

        Returns:
            Polars DataFrame with filtered citations.
        """
        # Check for cached parquet
        cache_key = f"ros_conf{confidence_threshold}_exam{examiner_only}.parquet"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists() and not force_reload:
            logger.info(f"Loading cached RoS data from {cache_path}")
            return pl.read_parquet(cache_path)

        # Download if needed
        if not self.raw_file_path.exists():
            self.download()

        logger.info("Parsing RoS dataset")

        # Read CSV with appropriate schema
        df = pl.read_csv(self.raw_file_path)

        # Standardize column names to lowercase
        df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})

        # Map RoS column names to internal names
        col_map = {}
        if "oaid" in df.columns:
            col_map["oaid"] = "openalex_id"
        if "patent" in df.columns:
            col_map["patent"] = "patent_id"
        if "confscore" in df.columns:
            col_map["confscore"] = "confidence"
        if "wherefound" in df.columns:
            col_map["wherefound"] = "examiner_applicant"
        if "reftype" in df.columns:
            col_map["reftype"] = "ref_type"
        if col_map:
            df = df.rename(col_map)

        # Cast types
        if "confidence" in df.columns:
            df = df.with_columns(pl.col("confidence").cast(pl.Int32))
        if "patent_id" in df.columns:
            df = df.with_columns(pl.col("patent_id").cast(pl.Utf8))
        if "openalex_id" in df.columns:
            df = df.with_columns(pl.col("openalex_id").cast(pl.Utf8))

        # Normalize patent IDs to PatentsView numeric format
        # RoS uses "us-9186198-b2" style; PatentsView uses "9186198"
        if "patent_id" in df.columns:
            from biopat.ingestion.patentsview import normalize_patent_id
            df = df.with_columns(
                pl.col("patent_id").map_elements(
                    normalize_patent_id, return_dtype=pl.Utf8
                )
            )

        # Normalize OpenAlex IDs to W-prefixed format
        # RoS uses bare numeric "2100342970"; OpenAlex API returns "W2100342970"
        if "openalex_id" in df.columns:
            from biopat.ingestion.openalex import normalize_openalex_id
            df = df.with_columns(
                pl.col("openalex_id").map_elements(
                    normalize_openalex_id, return_dtype=pl.Utf8
                )
            )

        # Filter by confidence
        df = df.filter(pl.col("confidence") >= confidence_threshold)

        # Filter by citation source if requested
        if examiner_only:
            # The examiner_applicant field indicates citation source
            if "examiner_applicant" in df.columns:
                df = df.filter(
                    pl.col("examiner_applicant").str.to_lowercase().str.contains("examiner")
                )

        # Cache the filtered result
        df.write_parquet(cache_path)
        logger.info(f"Cached filtered RoS data to {cache_path}")

        return df

    def get_unique_patents(self, df: Optional[pl.DataFrame] = None) -> pl.Series:
        """Get unique patent IDs from RoS data.

        Args:
            df: Optional pre-loaded DataFrame. If None, loads with defaults.

        Returns:
            Series of unique patent IDs.
        """
        if df is None:
            df = self.load()
        return df.select("patent_id").unique().to_series()

    def get_unique_papers(self, df: Optional[pl.DataFrame] = None) -> pl.Series:
        """Get unique OpenAlex paper IDs from RoS data.

        Args:
            df: Optional pre-loaded DataFrame. If None, loads with defaults.

        Returns:
            Series of unique OpenAlex IDs.
        """
        if df is None:
            df = self.load()
        return df.select("openalex_id").unique().to_series()

    def get_citations_for_patent(
        self, patent_id: str, df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Get all citations for a specific patent.

        Args:
            patent_id: USPTO patent ID.
            df: Optional pre-loaded DataFrame.

        Returns:
            DataFrame of citations for the patent.
        """
        if df is None:
            df = self.load()
        return df.filter(pl.col("patent_id") == patent_id)

    def count_citations_per_patent(
        self, df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Count citations per patent.

        Args:
            df: Optional pre-loaded DataFrame.

        Returns:
            DataFrame with patent_id and citation_count columns.
        """
        if df is None:
            df = self.load()
        return (
            df.group_by("patent_id")
            .agg(pl.len().alias("citation_count"))
            .sort("citation_count", descending=True)
        )
