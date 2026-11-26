import json

import pandas as pd
import pytest

from app.data.cleaning import clean_text
from app.data.ingestion import discover_available_datasets, load_manifest_dataset
from app.data.splitting import remove_cross_split_duplicates, stable_text_key, stratified_train_validation_split
from app.data.validation import GO_EMOTIONS_LABELS, normalize_target, validate_schema


def test_cleaning_keeps_sentiment_bearing_text():
    cleaned = clean_text("<p>I don't hate it 😍 #WorthIt https://example.com</p>")
    assert "don't" in cleaned
    assert "😍" in cleaned
    assert "#WorthIt" in cleaned
    assert "<URL>" in cleaned
    assert "https://example.com" not in cleaned
    assert clean_text(pd.NA) == ""


def test_go_emotions_labels_remain_multilabel():
    assert normalize_target("go_emotions", [0, 27], "train") == ["admiration", "neutral"]
    assert normalize_target("go_emotions", [28], "train") is None
    assert len(GO_EMOTIONS_LABELS) == 28


def test_label_normalization_is_task_specific():
    assert normalize_target("imdb", 0, "train") == "negative"
    assert normalize_target("tweet_eval", 2, "train") == "positive"
    assert normalize_target("imdb", -1, "train") is None
    assert normalize_target("imdb", -1, "unsupervised") is None


def test_schema_validation_reports_missing_target():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"text": ["hello"]}), "tweet_eval")


def test_manifest_loading_checks_files_and_parquet(tmp_path):
    raw = tmp_path / "data" / "raw"
    dataset = raw / "tiny"
    dataset.mkdir(parents=True)
    source = pd.DataFrame({"text": ["great"], "label": [1]})
    (tmp_path / "data" / "raw" / "tiny" / "train.parquet").parent.mkdir(parents=True, exist_ok=True)
    source.to_parquet(dataset / "train.parquet", index=False)
    (dataset / "manifest.json").write_text(json.dumps({"splits": {"train": {"file": "data/raw/tiny/train.parquet"}}}))
    manifest, splits = load_manifest_dataset(raw, "tiny")
    assert manifest["splits"].keys() == splits.keys()
    assert len(splits["train"]) == 1
    assert discover_available_datasets(raw)["sarc"] == "unavailable"
    manifest["splits"]["train"]["rows"] = 2
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row count mismatch"):
        load_manifest_dataset(raw, "tiny")


def test_manifest_loader_rejects_missing_file(tmp_path):
    raw = tmp_path / "data" / "raw"
    dataset = raw / "missing"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps({"splits": {"train": {"file": "data/raw/missing/train.parquet"}}}))
    with pytest.raises(FileNotFoundError, match="Missing missing/train file"):
        load_manifest_dataset(raw, "missing")


def test_stratified_split_is_reproducible_and_disjoint():
    frame = pd.DataFrame({"text": [f"review {i}" for i in range(40)], "label": ["negative"] * 20 + ["positive"] * 20})
    train_a, validation_a = stratified_train_validation_split(frame, seed=7)
    train_b, validation_b = stratified_train_validation_split(frame, seed=7)
    assert train_a.index.tolist() == train_b.index.tolist()
    assert validation_a.index.tolist() == validation_b.index.tolist()
    assert set(train_a.index).isdisjoint(validation_a.index)
    assert validation_a["label"].value_counts().to_dict() == {"negative": 2, "positive": 2}


def test_duplicate_texts_are_decontaminated_preferring_test():
    frame = pd.DataFrame({
        "text": ["same opinion", "SAME OPINION", "another"],
        "label": ["negative", "positive", "positive"],
        "split": ["train", "test", "train"],
    })
    result, removed, conflicts = remove_cross_split_duplicates(frame)
    assert removed == 1
    assert conflicts == 1
    assert result.loc[result["text"].str.lower() == "same opinion", "split"].item() == "test"
    assert len({stable_text_key(text) for text in result["text"]}) == len(result)
