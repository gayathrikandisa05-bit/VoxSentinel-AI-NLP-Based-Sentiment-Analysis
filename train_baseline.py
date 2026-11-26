#!/usr/bin/env python3
"""Train and benchmark classical sentiment models on prepared dataset splits."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.sentiment.baselines import run_dataset_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imdb", "tweet_eval", "all"), default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    datasets = ("imdb", "tweet_eval") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        result = run_dataset_benchmarks(PROJECT_ROOT, dataset, seed=args.seed)
        print(f"{dataset}: selected {result['selected_model_by_validation_macro_f1']} by validation macro F1")
        for model_name, metrics in result["models"].items():
            test = metrics["test"]
            print(f"  {model_name}: test accuracy={test['accuracy']:.4f}, macro_f1={test['f1_macro']:.4f}, weighted_f1={test['f1_weighted']:.4f}")
        print(f"  report: artifacts/reports/{dataset}_classical_benchmark.md")


if __name__ == "__main__":
    main()
