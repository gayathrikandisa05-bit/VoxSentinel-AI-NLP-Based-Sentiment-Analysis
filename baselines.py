"""TF-IDF sentiment baselines and benchmark evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


def _load_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "model_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _make_pipeline(model_name: str, config: dict[str, Any], seed: int) -> Pipeline:
    classical = config.get("classical", {})
    vectorizer = TfidfVectorizer(
        max_features=int(classical.get("max_features", 50_000)),
        ngram_range=tuple(classical.get("ngram_range", [1, 2])),
        dtype=np.float32,
        token_pattern=r"(?u)<url>|[@#]?\w+(?:['’]\w+)*|[^\w\s]",
        lowercase=True,
        sublinear_tf=True,
    )
    if model_name == "logistic_regression":
        params = classical.get("logistic_regression", {})
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=seed,
        )
    elif model_name == "linear_svm":
        params = classical.get("linear_svm", {})
        estimator = LinearSVC(C=float(params.get("C", 1.0)), random_state=seed)
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", estimator)])


def _load_split(data_root: Path, dataset: str, split: str) -> pd.DataFrame:
    path = data_root / dataset / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared {split} split: {path}")
    frame = pd.read_parquet(path)
    if "text" not in frame or "label" not in frame:
        raise ValueError(f"Prepared {dataset}/{split} must contain text and label columns")
    frame = frame.dropna(subset=["text", "label"])
    if frame.empty:
        raise ValueError(f"Prepared {dataset}/{split} has no labeled records")
    if frame["label"].map(lambda value: isinstance(value, list)).any():
        raise ValueError(f"{dataset}/{split} contains multi-label targets; this phase is sentiment-only")
    return frame


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def run_dataset_benchmarks(project_root: Path, dataset: str, seed: int | None = None) -> dict[str, Any]:
    """Fit two baselines, select by validation macro F1, and report held-out test metrics."""
    config = _load_config(project_root)
    random_seed = int(seed if seed is not None else config.get("seed", 42))
    data_root = project_root / "data" / "processed"
    train = _load_split(data_root, dataset, "train")
    validation = _load_split(data_root, dataset, "validation")
    test = _load_split(data_root, dataset, "test")
    labels = sorted(set(train["label"]) | set(validation["label"]) | set(test["label"]))
    if len(labels) < 2:
        raise ValueError(f"Expected at least two sentiment classes for {dataset}; found {labels}")

    models_dir = project_root / "artifacts" / "models"
    reports_dir = project_root / "artifacts" / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "dataset": dataset,
        "task": "sentiment_classification",
        "model_version": "tfidf-linear-baseline-v1",
        "scikit_learn_version": sklearn.__version__,
        "seed": random_seed,
        "classes": labels,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selection_metric": "validation.f1_macro",
        "models": {},
    }
    for model_name in ("logistic_regression", "linear_svm"):
        model = _make_pipeline(model_name, config, random_seed)
        started = time.perf_counter()
        model.fit(train["text"].tolist(), train["label"].to_numpy())
        fit_seconds = time.perf_counter() - started

        validation_prediction = model.predict(validation["text"].tolist())
        validation_metrics = _metrics(validation["label"].to_numpy(), validation_prediction, labels)

        artifact_path = models_dir / f"{dataset}_{model_name}.joblib"
        joblib.dump(model, artifact_path, compress=3)

        # Warm the pipeline before timing a full held-out batch.
        model.predict(test["text"].iloc[: min(64, len(test))].tolist())
        started = time.perf_counter()
        test_prediction = model.predict(test["text"].tolist())
        inference_seconds = time.perf_counter() - started
        test_metrics = _metrics(test["label"].to_numpy(), test_prediction, labels)
        vectorizer = model.named_steps["tfidf"]
        result["models"][model_name] = {
            "parameters": {
                "tfidf": {
                    "max_features": vectorizer.max_features,
                    "ngram_range": list(vectorizer.ngram_range),
                    "sublinear_tf": vectorizer.sublinear_tf,
                    "lowercase": vectorizer.lowercase,
                    "token_pattern": vectorizer.token_pattern,
                },
                "classifier": model.named_steps["classifier"].get_params(),
            },
            "fit_seconds": fit_seconds,
            "artifact_path": str(artifact_path.relative_to(project_root)),
            "artifact_size_bytes": artifact_path.stat().st_size,
            "validation": validation_metrics,
            "test": test_metrics,
            "test_inference_seconds": inference_seconds,
            "test_inference_records_per_second": len(test) / inference_seconds if inference_seconds else None,
        }
        LOGGER.info("%s/%s validation macro F1 %.4f", dataset, model_name, validation_metrics["f1_macro"])

    selected = max(result["models"], key=lambda name: result["models"][name]["validation"]["f1_macro"])
    result["selected_model_by_validation_macro_f1"] = selected
    result["limitations"] = [
        "TF-IDF and linear classifiers are lexical baselines; they do not model word order beyond unigrams and bigrams.",
        "The vectorizer lowercases text; it does not preserve casing as a feature.",
        "Reported confidence is not calibrated; LinearSVC does not provide class probabilities.",
        "IMDb and TweetEval use different source domains and label taxonomies; scores are not directly comparable across datasets.",
    ]
    report_path = reports_dir / f"{dataset}_classical_benchmark.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(result, reports_dir / f"{dataset}_classical_benchmark.md")
    return result


def _write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    rows = [
        f"# {result['dataset']} Classical Sentiment Baselines",
        "",
        f"- Train / validation / test: {result['split_sizes']['train']:,} / {result['split_sizes']['validation']:,} / {result['split_sizes']['test']:,}",
        f"- Classes: {', '.join(result['classes'])}",
        f"- Model selection: validation macro F1; selected `{result['selected_model_by_validation_macro_f1']}`.",
        "",
        "| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 | Fit seconds | Test records/sec | Artifact MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        validation, test = model["validation"], model["test"]
        rows.append(
            f"| {name} | {validation['accuracy']:.4f} | {validation['f1_macro']:.4f} | "
            f"{test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['f1_weighted']:.4f} | "
            f"{model['fit_seconds']:.2f} | {model['test_inference_records_per_second']:.1f} | "
            f"{model['artifact_size_bytes'] / (1024 * 1024):.2f} |"
        )
    rows.extend(["", "## Test confusion matrices", ""])
    for name, model in result["models"].items():
        rows.extend([f"### {name}", "", f"Label order: `{result['classes']}`", "", "```text", str(model["test"]["confusion_matrix"]), "```", ""])
    rows.extend(["## Limitations", ""] + [f"- {item}" for item in result["limitations"]])
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
