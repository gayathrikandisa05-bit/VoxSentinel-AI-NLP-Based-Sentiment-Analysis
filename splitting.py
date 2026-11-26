"""Deterministic, stratified validation split and duplicate decontamination."""

from __future__ import annotations

import hashlib

import pandas as pd


def stable_text_key(text: str) -> str:
    """Hash normalized text to support split leakage checks without storing keys."""
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


def stratified_train_validation_split(
    frame: pd.DataFrame, validation_fraction: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled single-label rows reproducibly, preserving class proportions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if frame["label"].isna().any() or frame["label"].map(lambda x: isinstance(x, list)).any():
        raise ValueError("Stratified fallback split requires complete single labels")
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index, (_, group) in enumerate(frame.groupby("label", sort=True)):
        shuffled = group.sample(frac=1, random_state=seed + class_index)
        count = int(round(len(shuffled) * validation_fraction))
        if len(shuffled) > 1:
            count = min(max(count, 1), len(shuffled) - 1)
        validation_indices.extend(shuffled.index[:count].tolist())
        train_indices.extend(shuffled.index[count:].tolist())
    return frame.loc[train_indices].sort_index().copy(), frame.loc[validation_indices].sort_index().copy()


def remove_cross_split_duplicates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Keep each normalized text once, preferring held-out splits over train.

    Rows with repeated text across official splits are retained in test, then
    validation, then train. Duplicate rows within a split keep their first row.
    Returns cleaned rows, number removed, and number of duplicate-label conflicts.
    """
    priority = {"test": 0, "validation": 1, "train": 2, "unlabeled": 3}
    work = frame.copy()
    work["_dedup_key"] = work["text"].map(stable_text_key)
    work["_priority"] = work["split"].map(priority).fillna(4)
    work["_row_order"] = range(len(work))
    sorted_work = work.sort_values(["_priority", "_row_order"], kind="stable")
    conflicts = 0
    for _, group in sorted_work.groupby("_dedup_key", sort=False):
        if len(group) > 1:
            first_label = group.iloc[0]["label"]
            if any(str(label) != str(first_label) for label in group["label"].iloc[1:]):
                conflicts += 1
    result = sorted_work.drop_duplicates("_dedup_key", keep="first")
    removed = len(work) - len(result)
    return result.sort_values("_row_order").drop(columns=["_dedup_key", "_priority", "_row_order"]), removed, conflicts
