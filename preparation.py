"""Dataset-specific preparation orchestration and honest data quality reports."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from app.data.cleaning import clean_text
from app.data.ingestion import discover_available_datasets, load_manifest_dataset
from app.data.splitting import remove_cross_split_duplicates, stratified_train_validation_split
from app.data.validation import normalize_target, validate_schema

LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = ("imdb", "tweet_eval", "go_emotions")


def _label_distribution(series: pd.Series, multi_label: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in series:
        if multi_label and isinstance(value, list):
            counts.update(value)
        elif value is not None and not pd.isna(value):
            counts[str(value)] += 1
        else:
            counts["unlabeled"] += 1
    return dict(sorted(counts.items()))


def _prepare_one_dataset(raw_root: Path, output_root: Path, dataset_name: str, seed: int) -> dict:
    manifest, source_splits = load_manifest_dataset(raw_root, dataset_name)
    all_rows: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    missing_values: dict[str, dict[str, int]] = {}
    invalid_records: Counter[str] = Counter()
    for source_split, raw_frame in source_splits.items():
        validate_schema(raw_frame, dataset_name)
        source_counts[source_split] = len(raw_frame)
        missing_values[source_split] = {
            str(column): int(count)
            for column, count in raw_frame.isna().sum().items()
            if count
        }
        frame = raw_frame.copy()
        frame["text_original"] = frame["text"]
        frame["text"] = frame["text"].map(clean_text)
        frame["label"] = frame["labels"].map(lambda value: normalize_target(dataset_name, value, source_split)) if dataset_name == "go_emotions" else frame["label"].map(lambda value: normalize_target(dataset_name, value, source_split))
        if dataset_name == "go_emotions":
            frame = frame.drop(columns=["labels"])
        frame["source_split"] = source_split
        frame["split"] = "unlabeled" if dataset_name == "imdb" and source_split == "unsupervised" else source_split
        blank_text = frame["text"].eq("")
        non_string_text = frame["text_original"].map(lambda value: not pd.isna(value) and not isinstance(value, str))
        invalid_label = frame["label"].isna() & (frame["split"] != "unlabeled")
        invalid_records["missing_or_blank_text"] += int(frame["text"].eq("").sum())
        invalid_records["non_string_text"] += int(non_string_text.sum())
        invalid_records["invalid_or_missing_label"] += int(invalid_label.sum())
        frame = frame.loc[~blank_text & ~non_string_text & ~invalid_label].copy()
        all_rows.append(frame)

    data = pd.concat(all_rows, ignore_index=True)
    duplicate_text_rows_before_dedup = int(data["text"].map(lambda text: text.casefold()).duplicated().sum())

    # The source provides no validation split for IMDb. Carve one from its
    # canonical train set, preserving its official test set.
    if dataset_name == "imdb" and "validation" not in source_splits:
        original_train = data.loc[data["split"] == "train"].reset_index(drop=True)
        train, validation = stratified_train_validation_split(original_train, seed=seed)
        data = data.loc[data["split"] != "train"].copy()
        train["split"] = "train"
        validation["split"] = "validation"
        data = pd.concat([data, train, validation], ignore_index=True)

    data, duplicate_removed, duplicate_label_conflicts = remove_cross_split_duplicates(data)
    invalid_records["duplicate_rows_removed"] = duplicate_removed
    invalid_records["duplicate_text_label_conflicts"] = duplicate_label_conflicts
    data["text_length_chars"] = data["text"].str.len()

    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    class_distribution: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test", "unlabeled"):
        part = data.loc[data["split"] == split_name].drop(columns=["split", "source_split"], errors="ignore")
        split_counts[split_name] = len(part)
        if not part.empty:
            part.to_parquet(out_dir / f"{split_name}.parquet", index=False)
            class_distribution[split_name] = _label_distribution(part["label"], dataset_name == "go_emotions")

    labeled = data.loc[data["split"] != "unlabeled"]
    lengths = data["text_length_chars"]
    report = {
        "dataset": dataset_name,
        "source": manifest.get("source"),
        "dataset_id": manifest.get("dataset"),
        "task": "multi_label_emotion_classification" if dataset_name == "go_emotions" else "binary_sentiment_classification" if dataset_name == "imdb" else "three_class_sentiment_classification",
        "label_column": "label",
        "source_split_sizes": source_counts,
        "source_size": sum(source_counts.values()),
        "valid_rows_after_cleaning": len(data),
        "labeled_rows": len(labeled),
        "missing_values_by_source_split": missing_values,
        "duplicate_text_rows_before_dedup": duplicate_text_rows_before_dedup,
        "duplicate_rows_removed": duplicate_removed,
        "duplicate_text_label_conflicts": duplicate_label_conflicts,
        "invalid_records": dict(invalid_records),
        "class_distribution_by_split": class_distribution,
        "split_sizes": split_counts,
        "text_length_characters": {
            "scope": "all retained records, including unlabeled rows",
            "count": int(len(lengths)),
            "min": int(lengths.min()) if len(lengths) else None,
            "mean": round(float(lengths.mean()), 2) if len(lengths) else None,
            "median": float(lengths.median()) if len(lengths) else None,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else None,
            "max": int(lengths.max()) if len(lengths) else None,
        },
        "split_method": "official source splits preserved; exact normalized text duplicates removed across splits, preferring test then validation then train" if dataset_name != "imdb" else f"official test preserved; stratified train/validation split (validation=10%, seed={seed}); cross-split duplicate text removed preferring test then validation then train",
        "seed": seed,
    }
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def prepare_all_datasets(project_root: Path, seed: int = 42) -> dict:
    """Prepare every available supported dataset and write aggregate reports."""
    raw_root = project_root / "data" / "raw"
    output_root = project_root / "data" / "processed"
    availability = discover_available_datasets(raw_root)
    reports = {}
    unavailable = {name: state for name, state in availability.items() if state == "unavailable"}
    for name in SUPPORTED_DATASETS:
        if availability.get(name) == "available":
            LOGGER.info("Preparing dataset %s", name)
            reports[name] = _prepare_one_dataset(raw_root, output_root, name, seed)
        else:
            unavailable[name] = "unavailable (no local files)"
    aggregate = {
        "seed": seed,
        "available_datasets": list(reports),
        "unavailable_datasets": unavailable,
        "datasets": reports,
        "notes": [
            "Dataset tasks and target vocabularies remain separate.",
            "GoEmotions targets are multi-label emotion names; sentiment labels are not inferred.",
            "URLs become <URL>; hashtags, mentions, contractions, negation, punctuation, emoji, and case are retained.",
            "Unlabeled IMDb rows are written separately and excluded from supervised class distributions.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data_quality_report.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Data Quality Report", "", f"Random seed: `{seed}`", "", "## Availability", "", "| Dataset | Status |", "|---|---|"]
    for name, status in availability.items():
        lines.append(f"| {name} | {'prepared' if name in reports else 'unavailable: no local files'} |")
    for name, report in reports.items():
        lines += ["", f"## {name}", "", f"- Task: `{report['task']}`", f"- Raw rows: {report['source_size']:,}", f"- Valid rows: {report['valid_rows_after_cleaning']:,}", f"- Missing values: `{json.dumps(report['missing_values_by_source_split'], ensure_ascii=False)}`", f"- Duplicate text rows before deduplication: {report['duplicate_text_rows_before_dedup']:,}", f"- Duplicate rows removed: {report['duplicate_rows_removed']:,}", f"- Duplicate-text label conflicts: {report['duplicate_text_label_conflicts']:,}", f"- Invalid records: `{json.dumps(report['invalid_records'], ensure_ascii=False)}`", f"- Split sizes: `{json.dumps(report['split_sizes'], ensure_ascii=False)}`", f"- Class distribution by split: `{json.dumps(report['class_distribution_by_split'], ensure_ascii=False)}`", f"- Text length (characters): `{json.dumps(report['text_length_characters'], ensure_ascii=False)}`"]
    (output_root / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate
