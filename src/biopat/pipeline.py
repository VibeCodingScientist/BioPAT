"""Phase 1 pipeline orchestration.

Coordinates all components to build the minimum viable benchmark.
"""

import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import polars as pl

from .config import BioPatConfig
from .ingestion import RelianceOnScienceLoader, PatentsViewClient, OpenAlexClient
from .processing import PatentProcessor, PaperProcessor, CitationLinker
from .groundtruth import RelevanceAssigner, TemporalValidator
from .benchmark import BenchmarkSampler, DatasetSplitter, BEIRFormatter
from .evaluation import BM25Evaluator

logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """Orchestrates the Phase 1 benchmark construction pipeline."""

    def __init__(self, config: Optional[BioPatConfig] = None):
        self.config = config or BioPatConfig.load()
        self.config.paths.create_dirs()

        # Initialize components
        self.ros_loader = RelianceOnScienceLoader(
            self.config.paths.raw_dir,
            self.config.paths.cache_dir,
        )
        self.patentsview_client = PatentsViewClient(
            api_keys=self.config.api.patentsview_keys,
            cache_dir=self.config.paths.cache_dir,
        )
        self.openalex_client = OpenAlexClient(
            mailto=self.config.api.openalex_mailto,
            api_key=self.config.api.openalex_api_key,
            cache_dir=self.config.paths.cache_dir,
        )
        self.patent_processor = PatentProcessor(self.config.paths.processed_dir)
        self.paper_processor = PaperProcessor(self.config.paths.processed_dir)
        self.citation_linker = CitationLinker(self.config.paths.processed_dir)
        self.relevance_assigner = RelevanceAssigner(self.config.paths.processed_dir)
        self.temporal_validator = TemporalValidator(self.config.paths.processed_dir)
        self.sampler = BenchmarkSampler(seed=42)
        self.splitter = DatasetSplitter(seed=42)
        self.formatter = BEIRFormatter(self.config.paths.benchmark_dir)

    # --- Checkpoint helpers ---

    def _checkpoint_path(self, name: str) -> Path:
        return self.config.paths.checkpoint_dir / name

    def _has_checkpoint(self, name: str) -> bool:
        return self._checkpoint_path(name).exists()

    def _save_df(self, name: str, df: pl.DataFrame) -> None:
        df.write_parquet(self._checkpoint_path(name))
        logger.info("Checkpoint saved: %s", name)

    def _load_df(self, name: str) -> pl.DataFrame:
        logger.info("Resuming from checkpoint: %s", name)
        return pl.read_parquet(self._checkpoint_path(name))

    def _save_json(self, name: str, data: dict) -> None:
        self._checkpoint_path(name).write_text(json.dumps(data, default=str))
        logger.info("Checkpoint saved: %s", name)

    def _load_json(self, name: str) -> dict:
        logger.info("Resuming from checkpoint: %s", name)
        return json.loads(self._checkpoint_path(name).read_text())

    def _save_splits(self, splits: dict) -> None:
        manifest = {}
        for name, (queries_df, qrels_df) in splits.items():
            q_name = f"step6_{name}_queries.parquet"
            r_name = f"step6_{name}_qrels.parquet"
            self._save_df(q_name, queries_df)
            self._save_df(r_name, qrels_df)
            manifest[name] = {"queries": q_name, "qrels": r_name}
        self._save_json("step6_splits.json", manifest)

    def _load_splits(self) -> dict:
        manifest = self._load_json("step6_splits.json")
        splits = {}
        for name, files in manifest.items():
            splits[name] = (
                self._load_df(files["queries"]),
                self._load_df(files["qrels"]),
            )
        return splits

    def clear_checkpoints(self) -> None:
        """Remove all checkpoint files."""
        cp = self.config.paths.checkpoint_dir
        if cp.exists():
            shutil.rmtree(cp)
            cp.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoints cleared")

    # --- Pipeline steps ---

    async def step1_download_ros(self, force: bool = False) -> pl.DataFrame:
        """Step 1: Download and load Reliance on Science data.

        Args:
            force: Force re-download.

        Returns:
            RoS DataFrame.
        """
        logger.info("Step 1: Downloading Reliance on Science dataset")

        self.ros_loader.download(force=force)
        ros_df = self.ros_loader.load(
            confidence_threshold=self.config.phase1.ros_confidence_threshold,
            examiner_only=False,
        )

        logger.info(f"Loaded {len(ros_df)} high-confidence citations")
        return ros_df

    async def step2_identify_patents(self, ros_df: pl.DataFrame) -> pl.DataFrame:
        """Step 2: Identify biomedical patents from RoS citations.

        Args:
            ros_df: RoS DataFrame.

        Returns:
            Patents DataFrame with metadata.
        """
        logger.info("Step 2: Identifying biomedical patents")

        # Get unique patent IDs with sufficient citations
        citation_counts = self.ros_loader.count_citations_per_patent(ros_df)
        eligible_patents = citation_counts.filter(
            pl.col("citation_count") >= self.config.phase1.min_citations
        )

        logger.info(f"Found {len(eligible_patents)} patents with >= {self.config.phase1.min_citations} citations")

        # Sample if too many
        patent_ids = eligible_patents["patent_id"].to_list()
        if len(patent_ids) > self.config.phase1.target_patent_count * 2:
            # Fetch a larger sample to allow for IPC filtering
            patent_ids = patent_ids[:self.config.phase1.target_patent_count * 3]

        # Fetch patent metadata from PatentsView
        logger.info(f"Fetching metadata for {len(patent_ids)} patents")
        patents = await self.patentsview_client.get_patents_batch(
            patent_ids,
            show_progress=True,
        )

        patents_df = self.patentsview_client.patents_to_dataframe(patents)
        logger.info(f"Retrieved {len(patents_df)} patents from PatentsView")

        return patents_df

    async def step3_filter_and_extract_claims(
        self,
        patents_df: pl.DataFrame,
        ros_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Step 3: Filter patents by IPC and extract claims.

        Args:
            patents_df: Patents DataFrame.
            ros_df: RoS DataFrame for citation counts.

        Returns:
            Claims DataFrame.
        """
        logger.info("Step 3: Filtering patents and extracting claims")

        citation_counts = self.ros_loader.count_citations_per_patent(ros_df)

        claims_df = self.patent_processor.process_patents(
            patents_df,
            citation_counts,
            ipc_prefixes=self.config.phase1.ipc_prefixes,
            min_citations=self.config.phase1.min_citations,
            save=True,
        )

        # Sample if needed
        if claims_df["patent_id"].n_unique() > self.config.phase1.target_patent_count:
            claims_df = self.patent_processor.sample_patents(
                claims_df,
                target_count=self.config.phase1.target_patent_count,
            )

        logger.info(
            f"Final: {len(claims_df)} claims from "
            f"{claims_df['patent_id'].n_unique()} patents"
        )

        return claims_df

    async def step4_build_corpus(
        self,
        ros_df: pl.DataFrame,
        claims_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Step 4: Build literature corpus from citations.

        Args:
            ros_df: RoS DataFrame.
            claims_df: Claims DataFrame.

        Returns:
            Papers DataFrame.
        """
        logger.info("Step 4: Building literature corpus")

        # Get patent IDs in the benchmark
        patent_ids = list(claims_df.select("patent_id").unique()["patent_id"].to_list())

        # Filter RoS to benchmark patents
        ros_filtered = ros_df.filter(pl.col("patent_id").is_in(patent_ids))

        # Get unique paper IDs
        paper_ids = ros_filtered.select("openalex_id").unique()["openalex_id"].to_list()
        logger.info(f"Fetching metadata for {len(paper_ids)} cited papers")

        # Fetch paper metadata from OpenAlex
        papers_df = await self.openalex_client.get_papers_for_citations(
            paper_ids,
            show_progress=True,
        )

        # Process papers
        papers_df = self.paper_processor.process_papers(
            papers_df,
            require_abstract=False,
            save=True,
        )

        logger.info(f"Built corpus with {len(papers_df)} papers")
        return papers_df

    async def step5_create_ground_truth(
        self,
        ros_df: pl.DataFrame,
        claims_df: pl.DataFrame,
        papers_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Step 5: Create ground truth relevance judgments.

        Args:
            ros_df: RoS DataFrame.
            claims_df: Claims DataFrame.
            papers_df: Papers DataFrame.

        Returns:
            Qrels DataFrame.
        """
        logger.info("Step 5: Creating ground truth")

        # Filter RoS to benchmark patents/papers
        patent_ids = list(claims_df.select("patent_id").unique()["patent_id"].to_list())
        paper_ids = list(papers_df.select("paper_id").unique()["paper_id"].to_list())

        ros_filtered = ros_df.filter(
            pl.col("patent_id").is_in(patent_ids) &
            pl.col("openalex_id").is_in(paper_ids)
        )

        # Create patent-level links with patent dates
        # Use priority_date when available, fall back to patent_date (grant date)
        patents_with_dates = claims_df.select(["patent_id", "priority_date"]).unique()

        # If priority_date is entirely null, try to get patent_date from patents checkpoint
        if patents_with_dates["priority_date"].null_count() == len(patents_with_dates):
            patents_cp = self.config.paths.checkpoint_dir / "step2_patents.parquet"
            if patents_cp.exists():
                patents_raw = pl.read_parquet(patents_cp)
                if "patent_date" in patents_raw.columns:
                    logger.warning(
                        "priority_date is null for all patents, falling back to patent_date (grant date)"
                    )
                    date_lookup = patents_raw.select(["patent_id", "patent_date"]).unique()
                    patents_with_dates = patents_with_dates.drop("priority_date").join(
                        date_lookup, on="patent_id", how="left"
                    ).rename({"patent_date": "priority_date"})

        links = ros_filtered.join(
            patents_with_dates,
            on="patent_id",
            how="inner"
        ).join(
            papers_df.select(["paper_id", "publication_date"]).rename({"paper_id": "openalex_id"}),
            on="openalex_id",
            how="inner"
        ).rename({"openalex_id": "paper_id"})

        # Map RoS examiner_applicant (front/body/both) to source (examiner/applicant)
        if "examiner_applicant" in links.columns:
            links = links.with_columns(
                pl.when(pl.col("examiner_applicant").is_in(["front", "both"]))
                .then(pl.lit("examiner"))
                .when(pl.col("examiner_applicant") == "body")
                .then(pl.lit("applicant"))
                .otherwise(pl.lit("unknown"))
                .alias("source")
            )

        # Validate temporal constraints
        valid_links, report = self.temporal_validator.validate_and_report(
            links,
            paper_date_col="publication_date",
            patent_date_col="priority_date",
        )

        # Expand to claim level
        select_cols = ["patent_id", "paper_id", "confidence"]
        if "source" in valid_links.columns:
            select_cols.append("source")
        expanded_links = (
            claims_df.select(["query_id", "patent_id"])
            .join(
                valid_links.select(select_cols),
                on="patent_id",
                how="inner"
            )
        )

        # Add source column if missing
        if "source" not in expanded_links.columns:
            expanded_links = expanded_links.with_columns(pl.lit("unknown").alias("source"))

        # Create qrels
        qrels = self.relevance_assigner.create_qrels(
            expanded_links,
            graded=self.config.phase1.graded_relevance,
            confidence_threshold=self.config.phase1.ros_confidence_threshold,
            save=True,
        )

        logger.info(f"Created {len(qrels)} relevance judgments")
        return qrels

    async def step6_create_splits(
        self,
        claims_df: pl.DataFrame,
        qrels: pl.DataFrame,
    ) -> dict:
        """Step 6: Create train/dev/test splits.

        Args:
            claims_df: Claims DataFrame.
            qrels: Qrels DataFrame.

        Returns:
            Dictionary of splits.
        """
        logger.info("Step 6: Creating train/dev/test splits")

        splits = self.splitter.create_splits(
            claims_df,
            qrels,
            stratify_col="domain",
            split_by_patent=True,
        )

        # Log split statistics
        for split_name, (queries, split_qrels) in splits.items():
            logger.info(
                f"  {split_name}: {len(queries)} queries, "
                f"{queries['patent_id'].n_unique()} patents, "
                f"{len(split_qrels)} qrels"
            )

        return splits

    async def step7_format_output(
        self,
        papers_df: pl.DataFrame,
        claims_df: pl.DataFrame,
        splits: dict,
    ) -> dict:
        """Step 7: Format and write BEIR output.

        Args:
            papers_df: Papers DataFrame.
            claims_df: Claims DataFrame.
            splits: Split dictionaries.

        Returns:
            Benchmark statistics.
        """
        logger.info("Step 7: Formatting BEIR output")

        stats = self.formatter.format_benchmark(
            corpus_df=papers_df,
            queries_df=claims_df,
            splits=splits,
        )

        logger.info(f"Benchmark formatted: {stats}")
        return stats

    def _generate_metadata(
        self,
        papers_df: pl.DataFrame,
        claims_df: pl.DataFrame,
        qrels: pl.DataFrame,
        splits: dict,
        stats: dict,
    ) -> dict:
        """Generate benchmark metadata.json alongside BEIR output."""
        metadata: dict = {"benchmark": "BioPAT", "version": "1.0.0"}

        # Corpus stats
        metadata["corpus"] = {
            "num_queries": int(claims_df["query_id"].n_unique()),
            "num_docs": len(papers_df),
            "num_qrels": len(qrels),
        }

        # Domain distribution
        if "domain" in claims_df.columns:
            domain_counts = (
                claims_df.select("domain")
                .group_by("domain")
                .len()
                .sort("len", descending=True)
            )
            metadata["domain_distribution"] = {
                row["domain"]: int(row["len"]) for row in domain_counts.iter_rows(named=True)
            }

        # Relevance tier distribution
        if "score" in qrels.columns:
            score_counts = (
                qrels.select("score")
                .group_by("score")
                .len()
                .sort("score")
            )
            metadata["relevance_distribution"] = {
                str(row["score"]): int(row["len"]) for row in score_counts.iter_rows(named=True)
            }

        # Splits info
        metadata["splits"] = {
            name: {"queries": len(q), "qrels": len(qr)}
            for name, (q, qr) in splits.items()
        }

        # Temporal coverage
        for col in ["priority_date", "filing_date"]:
            if col in claims_df.columns:
                dates = claims_df[col].drop_nulls()
                if len(dates) > 0:
                    metadata["temporal_coverage"] = {
                        "earliest": str(dates.min()),
                        "latest": str(dates.max()),
                    }
                break

        # SHA256 checksums of output files
        checksums = {}
        for path in sorted(self.config.paths.benchmark_dir.rglob("*")):
            if path.is_file():
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                checksums[str(path.relative_to(self.config.paths.benchmark_dir))] = h
        metadata["checksums"] = checksums

        # Write metadata
        meta_path = self.config.paths.benchmark_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info("Benchmark metadata written to %s", meta_path)

        return metadata

    async def step8_run_baseline(self) -> dict:
        """Step 8: Run BM25 baseline evaluation.

        Returns:
            Baseline metrics.
        """
        logger.info("Step 8: Running BM25 baseline")

        evaluator = BM25Evaluator(
            benchmark_dir=self.config.paths.benchmark_dir,
            results_dir=self.config.paths.benchmark_dir / "results",
        )

        # Evaluate on test split
        metrics = evaluator.run_evaluation(
            split="test",
            top_k=100,
            k_values=[10, 50, 100],
            save_results=True,
        )

        logger.info(f"BM25 baseline metrics: {metrics}")
        return metrics

    async def run(
        self,
        skip_download: bool = False,
        skip_baseline: bool = False,
        fresh: bool = False,
    ) -> dict:
        """Run the complete Phase 1 pipeline with checkpoint/resume.

        Completed steps are saved as checkpoints. On re-run, completed
        steps are loaded from checkpoint instead of re-executing.

        Args:
            skip_download: Skip RoS download if already present.
            skip_baseline: Skip baseline evaluation.
            fresh: Clear all checkpoints and start from scratch.

        Returns:
            Pipeline results including statistics and metrics.
        """
        if fresh:
            self.clear_checkpoints()

        logger.info("Starting Phase 1 pipeline")
        results = {}

        # Step 1: Download RoS
        cp = "step1_ros.parquet"
        if self._has_checkpoint(cp):
            ros_df = self._load_df(cp)
        else:
            ros_df = await self.step1_download_ros(force=not skip_download)
            self._save_df(cp, ros_df)
        results["ros_citations"] = len(ros_df)

        # Step 2: Identify patents
        cp = "step2_patents.parquet"
        if self._has_checkpoint(cp):
            patents_df = self._load_df(cp)
        else:
            patents_df = await self.step2_identify_patents(ros_df)
            self._save_df(cp, patents_df)
        results["patents_fetched"] = len(patents_df)

        # Step 3: Filter and extract claims
        cp = "step3_claims.parquet"
        if self._has_checkpoint(cp):
            claims_df = self._load_df(cp)
        else:
            claims_df = await self.step3_filter_and_extract_claims(patents_df, ros_df)
            self._save_df(cp, claims_df)
        results["claims"] = len(claims_df)
        results["unique_patents"] = claims_df["patent_id"].n_unique()

        # Step 4: Build corpus
        cp = "step4_papers.parquet"
        if self._has_checkpoint(cp):
            papers_df = self._load_df(cp)
        else:
            papers_df = await self.step4_build_corpus(ros_df, claims_df)
            self._save_df(cp, papers_df)
        results["papers"] = len(papers_df)

        # Step 5: Create ground truth
        cp = "step5_qrels.parquet"
        if self._has_checkpoint(cp):
            qrels = self._load_df(cp)
        else:
            qrels = await self.step5_create_ground_truth(ros_df, claims_df, papers_df)
            self._save_df(cp, qrels)
        results["qrels"] = len(qrels)

        # Step 6: Create splits
        cp = "step6_splits.json"
        if self._has_checkpoint(cp):
            splits = self._load_splits()
        else:
            splits = await self.step6_create_splits(claims_df, qrels)
            self._save_splits(splits)
        results["splits"] = {
            name: {"queries": len(q), "qrels": len(qr)}
            for name, (q, qr) in splits.items()
        }

        # Step 7: Format output
        cp = "step7_stats.json"
        if self._has_checkpoint(cp):
            stats = self._load_json(cp)
        else:
            stats = await self.step7_format_output(papers_df, claims_df, splits)
            self._save_json(cp, stats)
        results["benchmark_stats"] = stats

        # Step 7b: Generate benchmark metadata
        cp_meta = "step7b_metadata.json"
        if self._has_checkpoint(cp_meta):
            metadata = self._load_json(cp_meta)
        else:
            metadata = self._generate_metadata(papers_df, claims_df, qrels, splits, stats)
            self._save_json(cp_meta, metadata)
        results["metadata"] = metadata

        # Step 8: Run baseline
        if not skip_baseline:
            cp = "step8_metrics.json"
            if self._has_checkpoint(cp):
                metrics = self._load_json(cp)
            else:
                try:
                    metrics = await self.step8_run_baseline()
                    self._save_json(cp, metrics)
                except Exception as e:
                    logger.warning(f"Baseline evaluation failed: {e}")
                    metrics = {"error": str(e)}
            results["baseline_metrics"] = metrics

        logger.info("Phase 1 pipeline complete")
        logger.info(f"Results: {results}")

        return results


def run_phase1(
    config_path: str = "configs/default.yaml",
    skip_download: bool = False,
    skip_baseline: bool = False,
    fresh: bool = False,
) -> dict:
    """Run Phase 1 pipeline.

    Args:
        config_path: Path to configuration file.
        skip_download: Skip RoS download if present.
        skip_baseline: Skip baseline evaluation.
        fresh: Clear checkpoints and start from scratch.

    Returns:
        Pipeline results.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = BioPatConfig.load(config_path)
    pipeline = Phase1Pipeline(config)

    return asyncio.run(
        pipeline.run(
            skip_download=skip_download,
            skip_baseline=skip_baseline,
            fresh=fresh,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BioPAT Phase 1 pipeline")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip RoS download if already present",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip BM25 baseline evaluation",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear checkpoints, start from scratch",
    )

    args = parser.parse_args()
    results = run_phase1(
        config_path=args.config,
        skip_download=args.skip_download,
        skip_baseline=args.skip_baseline,
        fresh=args.fresh,
    )

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")
