"""Publication-quality figure generation for the NovEx paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Tol muted palette (colorblind-safe)
TOL = {
    "blue": "#332288",
    "cyan": "#88CCEE",
    "green": "#44AA99",
    "yellow": "#DDCC77",
    "red": "#CC6677",
    "purple": "#AA4499",
    "grey": "#BBBBBB",
    "rose": "#EE8866",
    "olive": "#999933",
    "indigo": "#6699CC",
}

METHOD_COLORS = {
    "BM25": TOL["grey"],
    "Rerank": TOL["blue"],
    "Agent": TOL["rose"],
}

MODEL_COLORS = {
    "GPT-5.2": TOL["cyan"],
    "Claude Sonnet 4.6": TOL["blue"],
    "Gemini 3 Pro": TOL["green"],
}

MODEL_KEY_TO_LABEL = {
    # Hyphen-form (used in tier1_table.json etc.)
    "gpt-5.2": "GPT-5.2",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    # Underscore-form (used in confusion_matrices.json keys)
    "gpt_5.2": "GPT-5.2",
    "claude_sonnet_4_6": "Claude Sonnet 4.6",
    "gemini_3_pro_preview": "Gemini 3 Pro",
}

MODEL_ORDER = ["gpt-5.2", "claude-sonnet-4-6", "gemini-3-pro-preview"]


def _apply_style() -> None:
    """Set global matplotlib rcParams for a clean academic look."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class NovExFigureGenerator:
    """Generates all 6 publication figures from NovEx analysis JSONs."""

    def __init__(self, analysis_dir: str | Path = "data/novex/analysis") -> None:
        self.analysis_dir = Path(analysis_dir)
        self._data: dict[str, object] = {}
        _apply_style()

    # -- data loading -------------------------------------------------------

    def _load(self, name: str) -> object:
        if name not in self._data:
            path = self.analysis_dir / name
            self._data[name] = json.loads(path.read_text())
        return self._data[name]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _label(model_key: str) -> str:
        return MODEL_KEY_TO_LABEL.get(model_key, model_key)

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Fig 1: Tier 1 Retrieval — Recall@10 grouped bar chart
    # -----------------------------------------------------------------------

    def fig_tier1_recall(self, out: Path) -> None:
        rows = self._load("tier1_table.json")

        # Build display order: BM25, then Rerank×3, then Agent×3
        order = [
            ("BM25", "N/A"),
            ("Rerank", "gpt-5.2"),
            ("Rerank", "claude-sonnet-4-6"),
            ("Rerank", "gemini-3-pro-preview"),
            ("Agent", "gpt-5.2"),
            ("Agent", "claude-sonnet-4-6"),
            ("Agent", "gemini-3-pro-preview"),
        ]
        method_map = {"BM25": "bm25", "Rerank": "bm25_rerank", "Agent": "agent"}

        lookup = {(r["method"], r["model"]): r for r in rows}

        labels, vals, lo, hi, colors = [], [], [], [], []
        for method_label, model_key in order:
            r = lookup[(method_map[method_label], model_key)]
            if model_key == "N/A":
                labels.append("BM25")
            else:
                labels.append(f"{method_label}\n{self._label(model_key)}")
            vals.append(r["recall@10"])
            lo.append(r["recall@10"] - r["recall@10_ci_lower"])
            hi.append(r["recall@10_ci_upper"] - r["recall@10"])
            colors.append(METHOD_COLORS[method_label])

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(x, vals, yerr=[lo, hi], color=colors, width=0.65,
               capsize=3, error_kw={"linewidth": 0.8}, edgecolor="white", linewidth=0.5)

        # BM25 baseline dashed line
        bm25_val = lookup[("bm25", "N/A")]["recall@10"]
        ax.axhline(bm25_val, color=TOL["grey"], ls="--", lw=0.8, zorder=0)
        ax.text(len(labels) - 0.5, bm25_val + 0.005, f"BM25 = {bm25_val:.3f}",
                ha="right", va="bottom", fontsize=8, color="0.4")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Recall@10")
        ax.set_title("Tier 1: Prior-Art Retrieval Performance")
        ax.set_ylim(0, max(vals) * 1.15)

        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=l) for l, c in METHOD_COLORS.items()]
        ax.legend(handles=handles, loc="upper right", frameon=False)

        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 2: Tier 2 Relevance — Horizontal dot plot
    # -----------------------------------------------------------------------

    def fig_tier2_comparison(self, out: Path) -> None:
        rows = self._load("tier2_table.json")

        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)

        y_pos = np.arange(len(MODEL_ORDER))
        labels = [self._label(m) for m in MODEL_ORDER]

        for ax, metric, title in [
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "weighted_kappa", "QW-\u03ba"),
        ]:
            for i, mk in enumerate(MODEL_ORDER):
                r = next(row for row in rows if row["model"] == mk)
                val = r[metric]
                color = MODEL_COLORS[self._label(mk)]

                if metric == "accuracy":
                    lo = val - r["accuracy_ci_lower"]
                    hi = r["accuracy_ci_upper"] - val
                    ax.errorbar(val, i, xerr=[[lo], [hi]], fmt="o", color=color,
                                capsize=4, markersize=7, markeredgecolor="white",
                                markeredgewidth=0.5, linewidth=1.2)
                else:
                    ax.plot(val, i, "o", color=color, markersize=7,
                            markeredgecolor="white", markeredgewidth=0.5)

                # Place label to the right of the point, vertically centered
                ax.text(val + 0.008, i, f"{val:.3f}", fontsize=8,
                        va="center", color=color)

            ax.set_title(title, pad=8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlim(0.65, 0.95)

        axes[0].set_xlabel("Accuracy")
        axes[1].set_xlabel("QW-\u03ba")
        fig.suptitle("Tier 2: Relevance Grading Quality", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 3: Context vs Zero-Shot — Dumbbell chart
    # -----------------------------------------------------------------------

    def fig_tier3_context_vs_zs(self, out: Path) -> None:
        rows = self._load("tier3_table.json")

        fig, ax = plt.subplots(figsize=(6, 3.2))
        y_pos = np.arange(len(MODEL_ORDER))
        labels = [self._label(m) for m in MODEL_ORDER]

        for i, mk in enumerate(MODEL_ORDER):
            ctx_row = next(r for r in rows if r["model"] == mk and r["context"] is True)
            zs_row = next(r for r in rows if r["model"] == mk and r["context"] is False)

            ctx_val = ctx_row["accuracy"]
            zs_val = zs_row["accuracy"]
            color = MODEL_COLORS[self._label(mk)]

            # Connecting line
            ax.plot([zs_val, ctx_val], [i, i], color=color, lw=2, zorder=1)

            # CI whiskers for ctx
            ctx_lo = ctx_val - ctx_row["accuracy_ci_lower"]
            ctx_hi = ctx_row["accuracy_ci_upper"] - ctx_val
            ax.errorbar(ctx_val, i, xerr=[[ctx_lo], [ctx_hi]], fmt="o",
                        color=color, capsize=4, markersize=8,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=2)

            # CI whiskers for zs
            zs_lo = zs_val - zs_row["accuracy_ci_lower"]
            zs_hi = zs_row["accuracy_ci_upper"] - zs_val
            ax.errorbar(zs_val, i, xerr=[[zs_lo], [zs_hi]], fmt="s",
                        color=color, capsize=4, markersize=7,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=2)

            # Value annotations — below the markers (positive y = below in inverted axis)
            ax.text(ctx_val, i + 0.3, f"{ctx_val:.1%}", fontsize=8,
                    ha="center", va="top", color=color, fontweight="bold")
            ax.text(zs_val, i + 0.3, f"{zs_val:.1%}", fontsize=8,
                    ha="center", va="top", color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy")
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.7)
        ax.set_title("Tier 3: Context vs Zero-Shot Novelty Classification", pad=10)

        # Legend for markers
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0], [0], marker="o", color="0.4", linestyle="", markersize=7,
                   label="With context"),
            Line2D([0], [0], marker="s", color="0.4", linestyle="", markersize=6,
                   label="Zero-shot"),
        ]
        ax.legend(handles=legend_els, loc="lower right", frameon=False)

        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 4: Per-Class F1 (Novelty, context only)
    # -----------------------------------------------------------------------

    def fig_tier3_per_class_f1(self, out: Path) -> None:
        rows = self._load("tier3_table.json")
        ctx_rows = [r for r in rows if r["context"] is True]

        classes = [
            ("f1_novel", "Novel"),
            ("f1_partially_anticipated", "Partially\nAnticipated"),
            ("f1_anticipated", "Anticipated"),
        ]
        n_models = len(MODEL_ORDER)
        n_classes = len(classes)
        bar_w = 0.22
        x = np.arange(n_classes)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        for j, mk in enumerate(MODEL_ORDER):
            r = next(row for row in ctx_rows if row["model"] == mk)
            label = self._label(mk)
            color = MODEL_COLORS[label]
            vals = [r[cls_key] for cls_key, _ in classes]
            offset = (j - (n_models - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=label, color=color,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in classes])
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Tier 3: Per-Class F1 (With Context)")
        ax.legend(loc="upper left", frameon=False)

        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 5: Paper vs Patent Recall@10
    # -----------------------------------------------------------------------

    def fig_paper_vs_patent(self, out: Path) -> None:
        data = self._load("doc_type_split.json")

        # Display order matching tier1
        keys_order = [
            ("bm25/N/A", "BM25"),
            ("bm25_rerank/gpt-5.2", "Rerank\nGPT-5.2"),
            ("bm25_rerank/claude-sonnet-4-6", "Rerank\nClaude"),
            ("bm25_rerank/gemini-3-pro-preview", "Rerank\nGemini"),
            ("agent/gpt-5.2", "Agent\nGPT-5.2"),
            ("agent/claude-sonnet-4-6", "Agent\nClaude"),
            ("agent/gemini-3-pro-preview", "Agent\nGemini"),
        ]

        paper_vals = [data[k]["paper_recall@10"] for k, _ in keys_order]
        patent_vals = [data[k]["patent_recall@10"] for k, _ in keys_order]
        labels = [lbl for _, lbl in keys_order]

        x = np.arange(len(labels))
        bar_w = 0.35
        fig, ax = plt.subplots(figsize=(7, 3.5))

        ax.bar(x - bar_w / 2, paper_vals, bar_w, label="Paper",
               color=TOL["indigo"], edgecolor="white", linewidth=0.5)
        ax.bar(x + bar_w / 2, patent_vals, bar_w, label="Patent",
               color=TOL["yellow"], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, max(patent_vals) * 1.15)
        # Title intentionally omitted — supplied by the LaTeX caption.
        ax.legend(loc="upper right", frameon=False)

        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 6: Agreement Heatmap (T3 ctx vs T3 zs)
    # -----------------------------------------------------------------------

    def fig_agreement_heatmap(self, out: Path) -> None:
        data = self._load("inter_model_agreement.json")

        short_labels = ["Claude", "Gemini", "GPT"]
        model_keys = ["claude-sonnet-4-6", "gemini-3-pro-preview", "gpt-5.2"]

        def _build_matrix(section: dict) -> np.ndarray:
            n = len(model_keys)
            mat = np.ones((n, n))
            pw = section["pairwise"]
            for key, info in pw.items():
                parts = key.split("_vs_")
                i = model_keys.index(parts[0])
                j = model_keys.index(parts[1])
                mat[i, j] = info["cohens_kappa"]
                mat[j, i] = info["cohens_kappa"]
            return mat

        mat_ctx = _build_matrix(data["tier3_ctx"])
        mat_zs = _build_matrix(data["tier3_zs"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

        vmin, vmax = -0.1, 1.0
        cmap = mpl.colormaps.get_cmap("YlOrRd")

        for ax, mat, title, kappa in [
            (ax1, mat_ctx, "With Context", data["tier3_ctx"]["fleiss_kappa"]),
            (ax2, mat_zs, "Zero-Shot", data["tier3_zs"]["fleiss_kappa"]),
        ]:
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, fontsize=9)
            ax.set_yticks(range(len(short_labels)))
            ax.set_yticklabels(short_labels, fontsize=9)
            ax.set_title(f"{title}  (\u03ba = {kappa:.3f})", fontsize=10, pad=8)

            # Annotate cells
            for i in range(len(model_keys)):
                for j in range(len(model_keys)):
                    val = mat[i, j]
                    text_color = "white" if val > 0.6 else "black"
                    label = f"{val:.2f}" if i != j else "\u2014"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=9, color=text_color)

        cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.85, pad=0.04)
        cbar.set_label("Cohen's \u03ba", fontsize=10)
        fig.suptitle("Tier 3: Inter-Model Agreement", fontsize=12)
        fig.subplots_adjust(left=0.08, right=0.82, top=0.82, bottom=0.08, wspace=0.4)
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 7: Tier Correlation — BM25 R@10 vs T3 Accuracy scatter
    # -----------------------------------------------------------------------

    def fig_tier_correlation(self, out: Path) -> None:
        data = self._load("tier_correlation.json")
        if "error" in data or "per_model" not in data:
            return

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        for mk in MODEL_ORDER:
            entry = data["per_model"].get(mk)
            if not entry:
                continue
            label = self._label(mk)
            color = MODEL_COLORS[label]
            pq = entry["per_query"]
            r10 = [p["bm25_r10"] for p in pq]
            acc = [p["t3_correct"] for p in pq]
            # Jitter y slightly for visibility (binary 0/1)
            rng = np.random.RandomState(42)
            jitter = rng.uniform(-0.03, 0.03, len(acc))
            ax.scatter(r10, np.array(acc) + jitter, s=18, alpha=0.5,
                       color=color, label=f"{label} (r={entry['pearson']:.2f})",
                       edgecolors="none")

        ax.set_xlabel("BM25 Recall@10")
        ax.set_ylabel("T3 Correct (with context)")
        ax.set_title("Tier Correlation: Retrieval Quality vs Novelty Accuracy")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])
        ax.legend(loc="lower right", frameon=False, fontsize=8)

        # Pooled correlation annotation
        ax.text(0.02, 0.92, f"Pooled r = {data['pooled_pearson']:.3f}",
                transform=ax.transAxes, fontsize=9, color="0.3")

        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 8: Cost-Performance Pareto scatter
    # -----------------------------------------------------------------------

    def fig_cost_pareto(self, out: Path) -> None:
        data = self._load("cost_performance.json")

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

        # T1: cost vs recall@10
        ax = axes[0]
        for entry in data["tier1"]:
            method = entry["method"]
            color = METHOD_COLORS.get(
                {"bm25": "BM25", "bm25_rerank": "Rerank", "agent": "Agent"}.get(method, "BM25"),
                TOL["grey"],
            )
            ax.scatter(entry["cost_usd"], entry["recall@10"], s=50, color=color,
                       edgecolors="white", linewidth=0.5, zorder=2)
            model_short = self._label(entry["model"]).split()[-1] if entry["model"] != "N/A" else "BM25"
            ax.annotate(model_short, (entry["cost_usd"], entry["recall@10"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7, color="0.3")
        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("Recall@10")
        ax.set_title("Tier 1: Retrieval")

        # T2: cost vs accuracy
        ax = axes[1]
        for entry in data["tier2"]:
            label = self._label(entry["model"])
            color = MODEL_COLORS.get(label, TOL["grey"])
            ax.scatter(entry["cost_usd"], entry["accuracy"], s=50, color=color,
                       edgecolors="white", linewidth=0.5, zorder=2)
            ax.annotate(label.split()[-1], (entry["cost_usd"], entry["accuracy"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7, color="0.3")
        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Tier 2: Relevance")

        # T3: cost vs macro_f1
        ax = axes[2]
        for entry in data["tier3"]:
            label = self._label(entry["model"])
            color = MODEL_COLORS.get(label, TOL["grey"])
            marker = "o" if entry["context"] == "ctx" else "s"
            ax.scatter(entry["cost_usd"], entry["macro_f1"], s=50, color=color,
                       marker=marker, edgecolors="white", linewidth=0.5, zorder=2)
            short = label.split()[-1]
            suffix = "" if entry["context"] == "ctx" else " (zs)"
            ax.annotate(f"{short}{suffix}", (entry["cost_usd"], entry["macro_f1"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7, color="0.3")
        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("Macro F1")
        ax.set_title("Tier 3: Novelty")

        fig.suptitle("Cost vs Performance", fontsize=12, y=1.02)
        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 9: T3 Confusion Matrices (Phase 2)
    # -----------------------------------------------------------------------

    def fig_confusion_matrices(self, out: Path) -> None:
        data = self._load("confusion_matrices.json")
        if not data:
            return

        labels = ["NOVEL", "PARTIALLY_ANTICIPATED", "ANTICIPATED"]
        short_labels = ["N", "PA", "A"]

        # Only show context results
        ctx_keys = [k for k in data if k.endswith("_ctx") and "matrix" in data[k]]
        if not ctx_keys:
            return

        n = len(ctx_keys)
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
        if n == 1:
            axes = [axes]

        cmap = mpl.colormaps.get_cmap("Blues")
        for ax, key in zip(axes, sorted(ctx_keys)):
            mat = data[key]["matrix"]
            arr = np.array([[mat[gt][pr] for pr in labels] for gt in labels], dtype=float)
            # Normalize rows (per-GT-class)
            row_sums = arr.sum(axis=1, keepdims=True)
            norm = np.divide(arr, row_sums, where=row_sums > 0, out=np.zeros_like(arr))

            im = ax.imshow(norm, vmin=0, vmax=1, cmap=cmap, aspect="equal")
            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, fontsize=9)
            ax.set_yticks(range(len(short_labels)))
            ax.set_yticklabels(short_labels, fontsize=9)

            model_name = key.replace("_ctx", "")
            ax.set_title(self._label(model_name), fontsize=10, pad=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")

            for i in range(len(labels)):
                for j in range(len(labels)):
                    count = int(arr[i, j])
                    pct = norm[i, j]
                    text_color = "white" if pct > 0.5 else "black"
                    ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                            fontsize=8, color=text_color)

        # suptitle intentionally omitted — supplied by the LaTeX caption.
        fig.tight_layout(rect=[0, 0, 0.92, 1.0])
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.04)
        cbar.set_label("Row-Normalized", fontsize=9)
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Fig 10: Context Ablation curve (Phase 2)
    # -----------------------------------------------------------------------

    def fig_context_ablation(self, out: Path) -> None:
        data = self._load("context_ablation.json")
        if not data:
            return

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        for mk in MODEL_ORDER:
            if mk not in data:
                continue
            label = self._label(mk)
            color = MODEL_COLORS[label]
            kv = data[mk]
            ks = sorted(int(k) for k in kv)
            accs = [kv[str(k)] if str(k) in kv else kv[k] for k in ks]
            ax.plot(ks, accs, "o-", color=color, label=label, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5, linewidth=1.5)
            for k_val, acc in zip(ks, accs):
                ax.text(k_val, acc + 0.01, f"{acc:.1%}", ha="center", va="bottom",
                        fontsize=7, color=color)

        ax.set_xlabel("Number of Context Documents (k)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Tier 3: Context Quantity Ablation")
        ax.legend(loc="lower right", frameon=False)
        ax.set_xticks([1, 3, 5, 10, 20])

        fig.tight_layout()
        self._save(fig, out)

    # -----------------------------------------------------------------------
    # Generate all
    # -----------------------------------------------------------------------

    def generate_all(
        self,
        output_dir: str | Path = "data/novex/analysis/figures",
        fmt: Literal["pdf", "png"] = "pdf",
    ) -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        figures = [
            ("fig1_tier1_recall", self.fig_tier1_recall),
            ("fig2_tier2_comparison", self.fig_tier2_comparison),
            ("fig3_tier3_context_vs_zs", self.fig_tier3_context_vs_zs),
            ("fig4_tier3_per_class_f1", self.fig_tier3_per_class_f1),
            ("fig5_paper_vs_patent", self.fig_paper_vs_patent),
            ("fig6_agreement_heatmap", self.fig_agreement_heatmap),
            ("fig7_tier_correlation", self.fig_tier_correlation),
            ("fig8_cost_pareto", self.fig_cost_pareto),
            ("fig9_confusion_matrices", self.fig_confusion_matrices),
            ("fig10_context_ablation", self.fig_context_ablation),
        ]

        paths = []
        for name, method in figures:
            p = out / f"{name}.{fmt}"
            try:
                method(p)
                paths.append(p)
            except (FileNotFoundError, KeyError) as exc:
                import logging
                logging.getLogger(__name__).warning("Skipping %s: %s", name, exc)
        return paths
