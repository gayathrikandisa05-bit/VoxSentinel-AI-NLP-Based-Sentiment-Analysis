"""Schema and target validation for supported datasets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

GO_EMOTIONS_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)
TWEET_SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
IMDB_SENTIMENT_LABELS = {0: "negative", 1: "positive"}


def validate_schema(frame: pd.DataFrame, dataset_name: str) -> None:
    """Raise a useful error when required dataset fields are absent."""
    target = "labels" if dataset_name == "go_emotions" else "label"
    missing = {"text", target} - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {sorted(missing)}")


def _single_label(value: object, labels: Mapping[int, str]) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
        if not numeric.is_integer():
            return None
        return labels.get(int(numeric))
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_target(dataset_name: str, value: object, source_split: str) -> object:
    """Convert benchmark IDs into dataset-specific named labels."""
    if dataset_name == "go_emotions":
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
            return None
        result = []
        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError, OverflowError):
                return None
            if index < 0 or index >= len(GO_EMOTIONS_LABELS):
                return None
            label = GO_EMOTIONS_LABELS[index]
            if label not in result:
                result.append(label)
        return result or None
    if dataset_name == "imdb" and source_split == "unsupervised":
        try:
            if int(value) == -1:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    if dataset_name == "imdb":
        return _single_label(value, IMDB_SENTIMENT_LABELS)
    if dataset_name == "tweet_eval":
        return _single_label(value, TWEET_SENTIMENT_LABELS)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
