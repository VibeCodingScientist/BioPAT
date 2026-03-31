#!/usr/bin/env python3
"""Generate all NovEx publication figures.

Usage:
    PYTHONPATH=src python3.12 scripts/generate_figures.py
    PYTHONPATH=src python3.12 scripts/generate_figures.py --format png --dpi 150
    PYTHONPATH=src python3.12 scripts/generate_figures.py --output-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NovEx paper figures")
    parser.add_argument(
        "--format", choices=["pdf", "png"], default="pdf",
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)",
    )
    parser.add_argument(
        "--output-dir", default="data/novex/analysis/figures",
        help="Output directory (default: data/novex/analysis/figures)",
    )
    parser.add_argument(
        "--analysis-dir", default="data/novex/analysis",
        help="Analysis data directory (default: data/novex/analysis)",
    )
    args = parser.parse_args()

    # Set DPI before importing figure generator (which sets rcParams)
    mpl.rcParams["savefig.dpi"] = args.dpi

    # Import directly to avoid __init__.py pulling in polars/torch etc.
    spec = importlib.util.spec_from_file_location(
        "figures", Path(__file__).resolve().parent.parent / "src" / "biopat" / "novex" / "figures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    NovExFigureGenerator = mod.NovExFigureGenerator

    gen = NovExFigureGenerator(analysis_dir=args.analysis_dir)
    paths = gen.generate_all(output_dir=args.output_dir, fmt=args.format)

    print(f"Generated {len(paths)} figures in {args.output_dir}/:")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
