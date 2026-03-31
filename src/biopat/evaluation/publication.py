"""Publication-ready output formatting for BioPAT benchmark results.

Generates LaTeX tables, Markdown summaries, and JSON data files
suitable for Nature-tier venues (NeurIPS D&B, Scientific Data, SIGIR).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PublicationFormatter:
    """Formats experiment results for publication.

    Produces:
    - LaTeX tables for paper inclusion
    - Markdown tables for README
    - JSON data for programmatic access
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def main_results_table(
        self,
        results: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        fmt: str = "latex",
    ) -> str:
        """Generate main results table: rows=methods, cols=metrics.

        Args:
            results: List of ExperimentResult dicts.
            metrics: Metric names to include.
            fmt: Output format ("latex", "markdown", "json").
        """
        metrics = metrics or ["NDCG@10", "NDCG@100", "MAP", "Recall@100"]

        if fmt == "latex":
            return self._latex_table(results, metrics, "Main retrieval results")
        elif fmt == "markdown":
            return self._markdown_table(results, metrics)
        else:
            return json.dumps(
                [{"model": r.get("model"), **{m: r.get("metrics", {}).get(m) for m in metrics}}
                 for r in results], indent=2,
            )

    def per_domain_table(
        self,
        domain_metrics: Dict[str, Dict[str, float]],
        metric: str = "NDCG@10",
        fmt: str = "latex",
    ) -> str:
        """Generate per-domain breakdown table."""
        domains = sorted(domain_metrics.keys())

        if fmt == "latex":
            header = " & ".join(["System"] + domains)
            lines = [
                r"\begin{table}[ht]",
                r"\centering",
                r"\caption{Per-domain " + metric + r" results.}",
                r"\begin{tabular}{l" + "c" * len(domains) + "}",
                r"\toprule",
                header + r" \\",
                r"\midrule",
            ]
            # Each row is a system (we just have one set of domain metrics here)
            vals = " & ".join([f"{domain_metrics[d].get(metric, 0):.4f}" for d in domains])
            lines.append(f"System & {vals} \\\\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
            return "\n".join(lines)

        # Markdown
        header = "| Domain | " + metric + " |"
        sep = "|---|---|"
        rows = [f"| {d} | {domain_metrics[d].get(metric, 0):.4f} |" for d in domains]
        return "\n".join([header, sep] + rows)

    def cost_effectiveness_table(
        self,
        cost_data: List[Dict[str, Any]],
        fmt: str = "latex",
    ) -> str:
        """Generate cost-effectiveness comparison table."""
        if fmt == "latex":
            lines = [
                r"\begin{table}[ht]",
                r"\centering",
                r"\caption{Cost-effectiveness analysis: metric per dollar.}",
                r"\begin{tabular}{llrrr}",
                r"\toprule",
                r"Experiment & Model & Metric & Cost (\$) & Metric/\$ \\",
                r"\midrule",
            ]
            for row in cost_data:
                mpd = row.get("metric_per_dollar", 0)
                mpd_str = f"{mpd:.2f}" if mpd != float("inf") else r"$\infty$"
                lines.append(
                    f"{row['experiment']} & {row['model']} & "
                    f"{row['metric']:.4f} & {row['cost_usd']:.2f} & {mpd_str} \\\\"
                )
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
            return "\n".join(lines)

        # Markdown
        header = "| Experiment | Model | Metric | Cost ($) | Metric/$ |"
        sep = "|---|---|---|---|---|"
        rows = []
        for row in cost_data:
            mpd = row.get("metric_per_dollar", 0)
            mpd_str = f"{mpd:.2f}" if mpd != float("inf") else "inf"
            rows.append(
                f"| {row['experiment']} | {row['model']} | "
                f"{row['metric']:.4f} | {row['cost_usd']:.2f} | {mpd_str} |"
            )
        return "\n".join([header, sep] + rows)

    def significance_matrix_table(
        self,
        sig_matrix: Dict[str, Dict[str, Dict[str, Any]]],
        fmt: str = "latex",
    ) -> str:
        """Generate pairwise significance matrix."""
        systems = list(sig_matrix.keys())

        if fmt == "latex":
            cols = "c" * len(systems)
            header = " & ".join([""] + systems)
            lines = [
                r"\begin{table}[ht]",
                r"\centering",
                r"\caption{Pairwise statistical significance (Bonferroni-corrected p-values).}",
                f"\\begin{{tabular}}{{l{cols}}}",
                r"\toprule",
                header + r" \\",
                r"\midrule",
            ]
            for sys_a in systems:
                vals = []
                for sys_b in systems:
                    if sys_a == sys_b:
                        vals.append("-")
                    else:
                        entry = sig_matrix[sys_a][sys_b]
                        p = entry.get("p_value_corrected", entry.get("p_value", 1))
                        marker = r"$\dagger$" if entry.get("significant") else ""
                        vals.append(f"{p:.3f}{marker}")
                lines.append(f"{sys_a} & " + " & ".join(vals) + r" \\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
            return "\n".join(lines)

        # Markdown
        header = "| | " + " | ".join(systems) + " |"
        sep = "|" + "---|" * (len(systems) + 1)
        rows = []
        for sys_a in systems:
            vals = []
            for sys_b in systems:
                if sys_a == sys_b:
                    vals.append("-")
                else:
                    entry = sig_matrix[sys_a][sys_b]
                    p = entry.get("p_value_corrected", entry.get("p_value", 1))
                    sig = "*" if entry.get("significant") else ""
                    vals.append(f"{p:.3f}{sig}")
            rows.append(f"| {sys_a} | " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)

    def save_all(
        self,
        results: List[Dict[str, Any]],
        domain_metrics: Optional[Dict] = None,
        cost_data: Optional[List[Dict]] = None,
        sig_matrix: Optional[Dict] = None,
    ) -> None:
        """Save all tables in both LaTeX and Markdown formats."""
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Main results
        for fmt, ext in [("latex", "tex"), ("markdown", "md")]:
            table = self.main_results_table(results, fmt=fmt)
            (tables_dir / f"main_results.{ext}").write_text(table)

        # Per-domain
        if domain_metrics:
            for fmt, ext in [("latex", "tex"), ("markdown", "md")]:
                table = self.per_domain_table(domain_metrics, fmt=fmt)
                (tables_dir / f"per_domain.{ext}").write_text(table)

        # Cost-effectiveness
        if cost_data:
            for fmt, ext in [("latex", "tex"), ("markdown", "md")]:
                table = self.cost_effectiveness_table(cost_data, fmt=fmt)
                (tables_dir / f"cost_effectiveness.{ext}").write_text(table)

        # Significance
        if sig_matrix:
            for fmt, ext in [("latex", "tex"), ("markdown", "md")]:
                table = self.significance_matrix_table(sig_matrix, fmt=fmt)
                (tables_dir / f"significance.{ext}").write_text(table)

        # JSON data
        data_dir = self.output_dir / "analysis"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "all_results.json").write_text(json.dumps(results, indent=2, default=str))

        logger.info("Publication outputs saved to %s", self.output_dir)

    # --- Helpers ---

    def _latex_table(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str],
        caption: str,
    ) -> str:
        cols = "c" * len(metrics)
        header = " & ".join(["Method"] + [m.replace("@", r"@") for m in metrics])
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\begin{{tabular}}{{l{cols}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]

        # Find best value per column for bolding
        best = {}
        for m in metrics:
            vals = [r.get("metrics", {}).get(m, 0) for r in results]
            best[m] = max(vals) if vals else 0

        for r in results:
            model = r.get("model", "?")
            vals = []
            for m in metrics:
                v = r.get("metrics", {}).get(m, 0)
                s = f"{v:.4f}"
                if v == best[m] and v > 0:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
            lines.append(f"{model} & " + " & ".join(vals) + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def _markdown_table(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str],
    ) -> str:
        header = "| Method | " + " | ".join(metrics) + " |"
        sep = "|---|" + "|".join(["---"] * len(metrics)) + "|"
        rows = []
        for r in results:
            model = r.get("model", "?")
            vals = [f"{r.get('metrics', {}).get(m, 0):.4f}" for m in metrics]
            rows.append(f"| {model} | " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)
