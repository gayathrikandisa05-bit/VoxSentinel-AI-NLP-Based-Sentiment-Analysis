#!/usr/bin/env python3
"""
VoxSentinel AI dataset downloader.

Downloads:
  - IMDb (Hugging Face)
  - TweetEval sentiment (Hugging Face)
  - GoEmotions simplified (Hugging Face)
  - SemEval-2014 Task 4 archive (Zenodo)

Optional:
  - Amazon Reviews 2023 category subset (Hugging Face streaming)

SARC is intentionally not auto-downloaded: use the official Princeton page,
review its access/usage terms, and place the downloaded files in data/raw/sarc/.
"""

from pathlib import Path
import argparse
import json
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def ensure_dependencies():
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Missing dependency. Install with: pip install -r requirements-data.txt")
        sys.exit(1)


def save_hf_dataset(dataset_id, output_dir, config=None, splits=None):
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {dataset_id} {f'({config})' if config else ''} ...")
    ds = load_dataset(dataset_id, config)

    selected = splits or list(ds.keys())
    manifest = {
        "dataset": dataset_id,
        "config": config,
        "splits": {},
        "source": f"https://huggingface.co/datasets/{dataset_id}",
    }

    for split in selected:
        if split not in ds:
            print(f"  - Split not present, skipping: {split}")
            continue
        split_dir = output_dir / split
        ds[split].to_parquet(str(split_dir.with_suffix(".parquet")))
        manifest["splits"][split] = {
            "rows": len(ds[split]),
            "columns": ds[split].column_names,
            "file": str(split_dir.with_suffix(".parquet").relative_to(ROOT)),
        }
        print(f"  - Saved {split}: {len(ds[split]):,} rows")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def download_semeval():
    out = RAW / "semeval_absa"
    out.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/14655773/files/archive.zip?download=1"
    archive = out / "semeval2014_task4_archive.zip"
    print("\nDownloading SemEval-2014 Task 4 archive ...")
    try:
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out / "extracted")
        print(f"  - Archive and extracted files saved under: {out}")
        print("  - Inspect the extracted contents and preserve the original splits.")
    except Exception as exc:
        print(f"  - Automatic download failed: {exc}")
        print("  - Download manually from https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools")
        print("  - A Zenodo mirror is available at https://zenodo.org/records/14655773")


def download_amazon_subset(category, max_rows):
    from datasets import load_dataset

    out = RAW / "amazon_reviews" / category.lower()
    out.mkdir(parents=True, exist_ok=True)
    config = f"raw_review_{category}"
    print(f"\nStreaming Amazon Reviews 2023 category: {category}")
    print("Only a limited sample will be saved; the full repository is very large.")
    try:
        stream = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            config,
            split="full",
            streaming=True,
        )
        rows = []
        for row in stream.take(max_rows):
            rows.append(row)
        import pandas as pd
        df = pd.DataFrame(rows)
        target = out / f"{category.lower()}_sample.parquet"
        df.to_parquet(target, index=False)
        print(f"  - Saved {len(df):,} rows to {target}")
    except Exception as exc:
        print(f"  - Amazon subset download failed: {exc}")
        print("  - Check the dataset card/config names and access requirements:")
        print("    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon", action="store_true",
                        help="Also stream a small Amazon Reviews 2023 category sample")
    parser.add_argument("--amazon-category", default="All_Beauty")
    parser.add_argument("--amazon-rows", type=int, default=5000)
    args = parser.parse_args()

    ensure_dependencies()
    RAW.mkdir(parents=True, exist_ok=True)

    # The canonical IMDb dataset has train/test/unsupervised splits.
    save_hf_dataset("stanfordnlp/imdb", RAW / "imdb")

    # TweetEval is a multi-task benchmark; explicitly select sentiment.
    save_hf_dataset(
        "cardiffnlp/tweet_eval",
        RAW / "tweet_eval",
        config="sentiment",
        splits=["train", "validation", "test"],
    )

    # GoEmotions simplified: 27 emotion labels plus neutral, multi-label format.
    save_hf_dataset(
        "google-research-datasets/go_emotions",
        RAW / "go_emotions",
        config="simplified",
        splits=["train", "validation", "test"],
    )

    download_semeval()

    if args.amazon:
        download_amazon_subset(args.amazon_category, args.amazon_rows)

    print("\nSARC manual retrieval:")
    print("  Official page: https://nlp.cs.princeton.edu/SARC/")
    print(f"  Place the permitted downloaded files in: {RAW / 'sarc'}")
    print("\nDataset retrieval script finished. Review each manifest and license/terms.")
    print("Do not combine task-specific labels into one training target.")


if __name__ == "__main__":
    main()
