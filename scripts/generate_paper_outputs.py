#!/usr/bin/env python3
"""Generate publication-ready outputs from BioPAT experiment results.

Reads results from data/results/ and produces:
- results/tables/  — LaTeX tables for paper
- results/analysis/ — Detailed JSON analysis reports
- results/summary.md — Markdown summary of all findings

Usage:
    python scripts/generate_paper_outputs.py
    python scripts/generate_paper_outputs.py --results-dir data/results --output-dir results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paper outputs from BioPAT results")
    parser.add_argument("--results-dir", default="data/results", help="Input results directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load results
    summary_path = results_dir / "experiment_summary.json"
    if not summary_path.exists():
        logger.error("No experiment summary found at %s. Run experiments first.", summary_path)
        return 1

    with open(summary_path) as f:
        summary = json.load(f)

    results = summary.get("results", [])
    logger.info("Loaded %d experiment results", len(results))

    # Initialize formatters
    from biopat.evaluation.publication import PublicationFormatter
    from biopat.evaluation.analysis import ResultsAnalyzer

    formatter = PublicationFormatter(output_dir=str(output_dir))
    analyzer = ResultsAnalyzer(results_dir=str(results_dir))

    # Cost-effectiveness analysis
    cost_data = analyzer.cost_effectiveness(results)

    # Generate and save tables
    formatter.save_all(
        results=results,
        cost_data=cost_data,
    )

    # Generate markdown summary
    summary_md = _generate_summary(results, cost_data, summary)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_md)
    logger.info("Summary written to %s", summary_path)

    # Save cost tracker summary if available
    cost_path = results_dir / "cost_tracker.json"
    if cost_path.exists():
        with open(cost_path) as f:
            cost_data_raw = json.load(f)
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "cost_breakdown.json").write_text(
            json.dumps(cost_data_raw.get("summary", {}), indent=2)
        )

    logger.info("All outputs generated in %s", output_dir)
    return 0


def _generate_summary(
    results: list,
    cost_data: list,
    raw_summary: dict,
) -> str:
    """Generate a markdown summary of all findings."""
    lines = [
        "# BioPAT Benchmark — Experiment Summary",
        "",
        f"Total experiments: {len(results)}",
        f"Total cost: ${raw_summary.get('total_cost_usd', 0):.2f}",
        "",
        "## Results by Experiment",
        "",
    ]

    # Group by experiment
    by_exp: dict = {}
    for r in results:
        exp = r.get("experiment", "unknown")
        by_exp.setdefault(exp, []).append(r)

    for exp, exp_results in by_exp.items():
        lines.append(f"### {exp}")
        lines.append("")

        # Table header from first result's metrics
        metrics = list(exp_results[0].get("metrics", {}).keys())
        if metrics:
            header = "| Model | " + " | ".join(metrics[:6]) + " | Cost |"
            sep = "|---|" + "|".join(["---"] * min(len(metrics), 6)) + "|---|"
            lines.append(header)
            lines.append(sep)

            for r in exp_results:
                model = r.get("model", "?")
                vals = []
                for m in metrics[:6]:
                    v = r.get("metrics", {}).get(m, 0)
                    vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                cost = r.get("cost_usd", 0)
                lines.append(f"| {model} | " + " | ".join(vals) + f" | ${cost:.2f} |")

        lines.append("")

    # Cost-effectiveness
    lines.append("## Cost-Effectiveness")
    lines.append("")
    if cost_data:
        lines.append("| Experiment | Model | NDCG@10 | Cost ($) | Metric/$ |")
        lines.append("|---|---|---|---|---|")
        for row in cost_data[:10]:
            mpd = row.get("metric_per_dollar", 0)
            mpd_str = f"{mpd:.2f}" if mpd != float("inf") else "inf"
            lines.append(
                f"| {row['experiment']} | {row['model']} | "
                f"{row['metric']:.4f} | {row['cost_usd']:.2f} | {mpd_str} |"
            )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
