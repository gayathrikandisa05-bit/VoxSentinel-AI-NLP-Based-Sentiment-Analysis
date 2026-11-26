"""Raw dataset discovery and Parquet loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_manifest_dataset(raw_root: Path, dataset_name: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Load every manifest-declared Parquet split for one dataset.

    The manifest's relative paths are resolved from the project root (the parent
    of ``data/raw``), matching the existing downloader's output.
    """
    dataset_dir = raw_root / dataset_name
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("splits"), dict) or not manifest["splits"]:
        raise ValueError(f"Manifest has no split definitions: {manifest_path}")

    project_root = raw_root.parent.parent
    frames: dict[str, pd.DataFrame] = {}
    for split_name, split_info in manifest["splits"].items():
        relative_path = split_info.get("file")
        if not relative_path:
            raise ValueError(f"Missing file path for {dataset_name}/{split_name}")
        path = (project_root / relative_path).resolve()
        if not path.is_relative_to(project_root.resolve()):
            raise ValueError(f"Manifest path escapes project root: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset_name}/{split_name} file: {path}")
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported data format for {path}; expected Parquet")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"Dataset split is empty: {path}")
        expected_rows = split_info.get("rows")
        if expected_rows is not None and int(expected_rows) != len(frame):
            raise ValueError(
                f"Manifest row count mismatch for {dataset_name}/{split_name}: "
                f"expected {expected_rows}, found {len(frame)}"
            )
        expected_columns = split_info.get("columns")
        if expected_columns is not None and set(expected_columns) != set(frame.columns):
            raise ValueError(
                f"Manifest column mismatch for {dataset_name}/{split_name}: "
                f"expected {expected_columns}, found {frame.columns.tolist()}"
            )
        frames[split_name] = frame
    return manifest, frames


def discover_available_datasets(raw_root: Path) -> dict[str, str]:
    """Report known dataset availability without treating absent data as errors."""
    locations = {
        "imdb": raw_root / "imdb",
        "tweet_eval": raw_root / "tweet_eval",
        "go_emotions": raw_root / "go_emotions",
        "semeval_absa": raw_root / "semeval_absa",
        "sarc": raw_root / "sarc",
        "amazon_reviews": raw_root / "amazon_reviews",
    }
    result = {}
    for name, path in locations.items():
        if path.is_dir() and any(p.is_file() for p in path.rglob("*")):
            result[name] = "available"
        else:
            result[name] = "unavailable"
    return result
