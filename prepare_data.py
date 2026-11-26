#!/usr/bin/env python3
"""Validate, clean, split, and report locally available VoxSentinel datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.data.preparation import prepare_all_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generated splits")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = prepare_all_datasets(PROJECT_ROOT, seed=args.seed)
    print(f"Prepared datasets: {', '.join(report['available_datasets']) or 'none'}")
    print(f"Quality report: {PROJECT_ROOT / 'data' / 'processed' / 'data_quality_report.md'}")
    for name in report["available_datasets"]:
        dataset_report = report["datasets"][name]
        print(f"  {name}: {dataset_report['source_size']:,} source rows, "
              f"{dataset_report['valid_rows_after_cleaning']:,} valid rows, "
              f"splits={dataset_report['split_sizes']}")
    for name, status in report["unavailable_datasets"].items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
