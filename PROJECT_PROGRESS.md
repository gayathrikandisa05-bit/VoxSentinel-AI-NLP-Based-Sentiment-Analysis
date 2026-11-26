# VoxSentinel AI Project Progress

## Phase 1 — Environment & Project Foundation

**Status:** Complete. Foundation details are retained in the Phase 1 history below.

## Phase 2 — Dataset Preparation and Data Quality

**Status:** Complete

### Implementation

- Added manifest-based Parquet loading with file, format, schema, manifest row-count, and manifest-column checks.
- Added task-specific label normalization for IMDb binary sentiment, TweetEval three-class sentiment, and GoEmotions multi-label emotion classification.
- Added conservative text normalization: HTML entity/tag handling, Unicode normalization, URL replacement, whitespace normalization. It retains negation, emoji, mentions, hashtags, contractions, punctuation, and case. Original text is retained alongside cleaned text.
- Added missing/blank text and invalid-target handling, duplicate detection, duplicate-label conflict reporting, and split decontamination by normalized text.
- Preserved official TweetEval and GoEmotions splits. Created a deterministic 10% stratified validation split from IMDb train data, retaining its official test split. Duplicate rows are preferentially retained in test, then validation, then train.
- Retained IMDb unsupervised examples in their own `unlabeled.parquet` file, outside labeled sentiment split statistics.
- Added JSON and Markdown data quality reports with per-split sizes, missing values, duplicate counts/conflicts, invalid counts, class distribution, and text length statistics.
- Text-length statistics cover all retained records, including unlabeled IMDb rows.
- Added a source/license inventory. No datasets were downloaded during Phase 2.

### Dataset availability and actual results

| Dataset | Availability | Source rows | Rows written | Split sizes (train / validation / test / unlabeled) |
|---|---|---:|---:|---|
| IMDb | Available | 100,000 | 98,463 | 22,290 / 2,489 / 24,799 / 48,885 |
| TweetEval sentiment | Available | 59,899 | 59,865 | 45,585 / 2,000 / 12,280 / 0 |
| GoEmotions simplified | Available | 54,263 | 53,943 | 43,112 / 5,410 / 5,421 / 0 |
| SemEval ABSA | Unavailable locally | — | — | No data fabricated or downloaded |
| SARC | Unavailable locally | — | — | No data fabricated or downloaded |
| Amazon Reviews | Unavailable locally | — | — | No data fabricated or downloaded |

Rows written are after invalid-record filtering and exact normalized-text deduplication. IMDb source rows include its 50,000 unlabeled examples.

Observed duplicate rows removed: IMDb 1,537 (620 duplicate-text label conflicts), TweetEval 34 (6 conflicts), GoEmotions 320 (81 conflicts). These rows are counted in the quality reports; no missing or invalid text/labels were found in these local files.

### Files added

- `app/data/cleaning.py`
- `app/data/ingestion.py`
- `app/data/preparation.py`
- `app/data/splitting.py`
- `app/data/validation.py`
- `scripts/prepare_data.py`
- `tests/test_data_preparation.py`
- `docs/data_sources.md`
- `data/processed/data_quality_report.json`
- `data/processed/data_quality_report.md`
- Per dataset under `data/processed/{imdb,tweet_eval,go_emotions}/`: train/validation/test Parquet files as available, IMDb `unlabeled.parquet`, and `quality_report.json`.

### Files modified

- `README.md` — added Phase 2 status and data preparation command.
- `PROJECT_PROGRESS.md` — recorded Phase 2 outputs, results, decisions, and limitations.

### Commands executed and verification

- Inspected manifests and read every existing raw Parquet split with pandas/PyArrow.
- `venv/bin/python scripts/prepare_data.py` — prepared the three available datasets and generated the quality reports.
- Verified zero cleaned text overlap between every labeled train/validation/test pair in each processed dataset.
- `venv/bin/python -m pytest -q` — **10 passed**.
- The local PyArrow build emits restricted macOS `sysctlbyname` warnings while reading Parquet; processing succeeded.

### Known limitations

- SemEval ABSA, SARC, and Amazon Reviews data are absent, so no processed splits exist for them.
- IMDb's metadata labels its license as `other`; TweetEval's current card has conflicting license metadata and subset-level terms. Review source terms before redistributing data. Amazon Reviews authors do not assign a dataset license. See `docs/data_sources.md`.
- Exact decontamination is based on case-folded cleaned text; paraphrase or near-duplicate detection is outside this phase.
- GoEmotions remains multi-label and is not transformed into a sentiment target.
- No model training, dashboard, or API was implemented in this phase.

### Next phase

Phase 3 — Classical ML Baselines. Completed below.

## Phase 3 — Classical ML Baselines

**Status:** Complete

### Implementation and decisions

- Trained TF-IDF + Logistic Regression and TF-IDF + Linear SVM independently for IMDb and TweetEval sentiment.
- Used only the prepared labeled train/validation/test splits. GoEmotions was not included because it is a separate multi-label emotion task.
- Selected models by validation macro-F1. Test metrics are reported for both estimators, and the test split was not used for selection.
- Recorded per-class precision/recall/F1, confusion matrices, accuracy, macro/weighted F1, fit time, test batch throughput, artifact size, configuration, random seed, and scikit-learn version.
- Used configured TF-IDF n-grams/features and classifier hyperparameters from `configs/model_config.yaml`.

### Actual benchmark results

| Dataset | Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---:|---:|---:|---:|
| IMDb | Logistic Regression | 0.8931 | 0.9017 | 0.9017 | 0.9017 |
| IMDb | Linear SVM | 0.9012 | 0.9007 | 0.9007 | 0.9007 |
| TweetEval sentiment | Logistic Regression | 0.6450 | 0.5934 | 0.5569 | 0.5728 |
| TweetEval sentiment | Linear SVM | 0.6527 | 0.5843 | 0.5698 | 0.5808 |

Linear SVM was selected by validation macro-F1 on both datasets. On IMDb, Logistic Regression scored slightly higher on the test set; on TweetEval, Linear SVM had higher test macro-F1 while Logistic Regression had higher accuracy. The test results are reported without changing validation-based selection.

### Files added/generated

- `app/models/sentiment/baselines.py`
- `scripts/train_baseline.py`
- `docs/classical_baselines.md`
- `artifacts/models/imdb_logistic_regression.joblib`
- `artifacts/models/imdb_linear_svm.joblib`
- `artifacts/models/tweet_eval_logistic_regression.joblib`
- `artifacts/models/tweet_eval_linear_svm.joblib`
- JSON and Markdown benchmark reports under `artifacts/reports/` for IMDb and TweetEval.

### Files modified

- `README.md` — added the baseline training command and Phase 3 status.
- `PROJECT_PROGRESS.md` — recorded benchmark results, decisions, files, and limitations.

### Commands executed and verification

- `venv/bin/python scripts/train_baseline.py --dataset all` — trained and benchmarked four estimators.
- Loaded each serialized estimator with joblib and ran a prediction smoke check successfully.
- Verified output JSON reports include the scikit-learn version and metrics, including per-class results and confusion matrices.

### Known limitations

- These are sparse lexical baselines and do not capture long-range context or robust sarcasm/emotion interpretation.
- The TweetEval test macro-F1 is lower than its validation macro-F1; later error analysis should examine per-class behavior.
- These benchmark measurements do not establish production readiness. Model confidence is not calibrated.
- The classifiers and task datasets are not comparable across IMDb and TweetEval because domains and labels differ.

### Next phase

Phase 4 — Transformer Modeling, after checking available memory/compute and selecting a feasible fine-tuning setup. Compare it against these saved classical baselines.

## User-requested GUI integration — sentiment analyzer

**Status:** Implemented

### Features

- Added a Streamlit GUI connected to the saved IMDb and TweetEval models.
- Single-text analysis shows the predicted sentiment and per-class score chart.
- CSV analysis supports text-column selection, batch prediction, preservation of rows with empty text, and CSV download; local limit is 10,000 records per upload.
- Model loading is cached for the Streamlit session.
- Logistic Regression values are identified as uncalibrated probabilities. Linear SVM values are identified as raw decision scores, never confidence or probabilities.
- Single and bulk inputs use the same conservative text normalizer used during data preparation.
- The UI states that emotion, aspect, sarcasm, and calibrated confidence outputs are not available.

### Files added

- `app/dashboard/main.py`
- `app/services/prediction_service.py`

### Files modified

- `README.md` — added GUI launch instructions and current capability limits.
- `PROJECT_PROGRESS.md` — recorded GUI features and verification.

### Verification

- Loaded each of the four saved models and ran single and batch predictions through the GUI inference service.
- Confirmed binary SVM decision scores are expanded and labeled as raw scores, not probabilities.
- `venv/bin/python -m py_compile app/dashboard/main.py app/services/prediction_service.py` — completed successfully.
- Streamlit, Plotly, pandas, scikit-learn, and joblib are available in the existing `venv/`.
- Started the Streamlit server successfully. The restricted command environment could not connect to its health endpoint from a separate process; the app process itself reported a running URL and was stopped cleanly after the startup check.

### Run command

`./venv/bin/streamlit run app/dashboard/main.py --server.address 127.0.0.1`

This is a focused sentiment analyzer connected to the current baselines; it does not implement the full multi-page analytics dashboard.

---

## Phase 1 — Environment & Project Foundation (history)

**Status:** Complete (foundation verified; full application dependencies intentionally staged)

### Environment

- macOS 26.6.2, Apple Silicon arm64
- Python 3.13.5; 10 logical CPUs
- 217 GiB free disk at initial inspection; physical RAM query restricted
- Existing `venv/` reused; confirmed NumPy 2.5.5, pandas 2.3.3, PyArrow 21.0.0, PyYAML 6.0.4, pytest 8.4.2

### Verification

- Initial foundation suite: 2 passed.
- `venv/bin/python -m pip check`: no broken requirements.
- Dataset files were preserved during Phase 1 and inspected only in Phase 2.
