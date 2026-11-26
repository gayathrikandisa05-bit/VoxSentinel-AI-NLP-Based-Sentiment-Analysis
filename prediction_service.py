"""Inference helpers for the saved classical sentiment models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.data.cleaning import clean_text

DATASET_NAMES = {
    "imdb": "IMDb movie reviews",
    "tweet_eval": "TweetEval social posts",
}
MODEL_NAMES = {
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
}


def model_artifact_path(project_root: Path, dataset: str, model_name: str) -> Path:
    """Return a checked-in artifact path for a supported model choice."""
    if dataset not in DATASET_NAMES or model_name not in MODEL_NAMES:
        raise ValueError("Unsupported dataset or model selection")
    return project_root / "artifacts" / "models" / f"{dataset}_{model_name}.joblib"


def load_model(project_root: Path, dataset: str, model_name: str) -> Any:
    """Load one previously trained model, raising a clear error if it is absent."""
    path = model_artifact_path(project_root, dataset, model_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Train the baseline models with "
            "`venv/bin/python scripts/train_baseline.py --dataset all`."
        )
    return joblib.load(path)


def analyze_text(model: Any, text: str) -> dict[str, Any]:
    """Predict one text and return scores with the right probability semantics."""
    normalized = clean_text(text)
    if not normalized:
        raise ValueError("Enter non-empty text to analyze")
    label = str(model.predict([normalized])[0])
    classes = [str(value) for value in model.classes_]
    result: dict[str, Any] = {"label": label, "classes": classes}
    if hasattr(model, "predict_proba"):
        values = model.predict_proba([normalized])[0]
        result["score_type"] = "probability (not calibrated)"
        result["scores"] = {name: float(value) for name, value in zip(classes, values)}
    elif hasattr(model, "decision_function"):
        values = np.asarray(model.decision_function([normalized])).reshape(-1)
        if len(values) == 1 and len(classes) == 2:
            values = np.array([-values[0], values[0]])
        if len(values) != len(classes):
            raise ValueError("Model returned an unexpected decision score shape")
        result["score_type"] = "raw decision score (not probability or confidence)"
        result["scores"] = {name: float(value) for name, value in zip(classes, values)}
    return result


def analyze_many(model: Any, texts: list[str]) -> dict[str, Any]:
    """Batch-predict non-empty texts and return per-class model scores."""
    normalized_texts = [clean_text(text) for text in texts]
    if not normalized_texts:
        return {"labels": [], "classes": [str(value) for value in model.classes_]}
    if any(not text for text in normalized_texts):
        raise ValueError("analyze_many expects non-empty text values")
    labels = [str(value) for value in model.predict(normalized_texts)]
    classes = [str(value) for value in model.classes_]
    scores = None
    score_type = None
    if hasattr(model, "predict_proba"):
        matrix = model.predict_proba(normalized_texts)
        scores = [{name: float(value) for name, value in zip(classes, row)} for row in matrix]
        score_type = "probability (not calibrated)"
    elif hasattr(model, "decision_function"):
        matrix = np.asarray(model.decision_function(normalized_texts))
        if matrix.ndim == 1:
            matrix = np.column_stack((-matrix, matrix))
        scores = [{name: float(value) for name, value in zip(classes, row)} for row in matrix]
        score_type = "raw decision score (not probability or confidence)"
    return {"labels": labels, "classes": classes, "scores": scores, "score_type": score_type}
