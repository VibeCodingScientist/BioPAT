"""Allow running BioPAT pipeline via `python -m biopat`."""

import argparse
import sys

from biopat.pipeline import run_phase1


def main():
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


if __name__ == "__main__":
    main()
