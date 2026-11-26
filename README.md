# VoxSentinel AI

Context-aware sentiment, emotion, and customer opinion intelligence platform.

VoxSentinel AI is being built in documented phases. Phase 2 prepares the locally available IMDb, TweetEval sentiment, and GoEmotions data with task-specific labels, leakage-aware splits, and quality reports. Phase 3 adds classical TF-IDF sentiment baselines for IMDb and TweetEval with held-out benchmarks. A Streamlit GUI now runs those saved models for single-text and CSV sentiment analysis. Transformer models, emotion/aspect/sarcasm inference, and the REST API are not yet implemented.

## Current status

- Phase 1 foundation verified and Phase 2 data preparation completed; see [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) and [dataset source notes](docs/data_sources.md).
- Phase 3 classical sentiment baselines trained and evaluated; see [baseline methodology and results](docs/classical_baselines.md). Results are benchmark measurements, not production performance claims.
- Streamlit sentiment GUI is available for IMDb and TweetEval baseline models; it shows model scores without treating them as calibrated confidence.
- Processed data and reports are under `data/processed/`. IMDb, TweetEval, and GoEmotions are present locally. SemEval ABSA, SARC, and Amazon Reviews are reported unavailable and were not fabricated or downloaded.
- The local environment has the foundation dependencies installed. The complete stack in `requirements.txt` is staged for later phases and has not been installed or verified as a whole.

## Environment

The current development environment is macOS on Apple Silicon with Python 3.13.5. Python 3.11 or a compatible version is the project target. Verify package support on your own platform before installing the full machine learning stack.

## Local setup

Create an isolated environment and install the project dependencies:

```bash
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

The full requirements include heavyweight machine learning packages. During phased development, the currently verified foundation packages can be installed with:

```bash
./venv/bin/python -m pip install 'numpy>=2.1,<3' 'pandas>=2.2,<3' 'pyarrow>=17,<22' 'PyYAML>=6,<7' 'pytest>=8,<9'
```

Run the current test suite from the project root:

```bash
./venv/bin/python -m pytest -q
```

Prepare the available local datasets and regenerate processed Parquet files and data quality reports:

```bash
./venv/bin/python scripts/prepare_data.py --seed 42
```

This step does not train models. It preserves the source train/validation/test splits where available, creates a stratified validation split from IMDb train data, removes duplicate text across splits, and keeps IMDb's unlabeled records in a separate file.

Train and benchmark the IMDb and TweetEval classical baselines:

```bash
./venv/bin/python scripts/train_baseline.py --dataset all
```

Choose one dataset with `--dataset imdb` or `--dataset tweet_eval`. Models are saved under `artifacts/models/`; benchmark reports are written under `artifacts/reports/`.

Launch the GUI from the project root:

```bash
./venv/bin/streamlit run app/dashboard/main.py --server.address 127.0.0.1
```

The page supports single text prediction and CSV batch export. Select IMDb for binary movie-review sentiment or TweetEval for negative/neutral/positive social-post sentiment. The trained model artifacts must exist under `artifacts/models/`.

## Project layout

```text
app/         Application packages for core, data, models, services, API, and dashboard
configs/     Application and model configuration
data/        Raw, processed, and sample data
docs/        Project documentation
scripts/     Reproducible data and training commands
artifacts/   Model and report outputs
tests/       Automated tests
```

## Planned capabilities

The remaining roadmap covers transformer fine-tuning, emotion and aspect analysis, sarcasm and uncertainty signals, explainability, experimental Customer Pain Score, topic and trend analytics, expanded dashboard pages, FastAPI endpoints, persistence, and deployment. Each capability will be documented as implemented only after it exists and has been verified.
