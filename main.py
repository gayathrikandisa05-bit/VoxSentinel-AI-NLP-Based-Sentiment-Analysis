"""Streamlit interface for trained VoxSentinel sentiment baselines."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.data.cleaning import clean_text  # noqa: E402
from app.services.prediction_service import (  # noqa: E402
    DATASET_NAMES,
    MODEL_NAMES,
    analyze_many,
    analyze_text,
    load_model,
)

st.set_page_config(
    page_title="VoxSentinel AI | Sentiment Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #0b1220; color: #e6edf7; }
      [data-testid="stSidebar"] { background: #111b2e; }
      [data-testid="stMetric"] { background: #131f33; border: 1px solid #26364d;
        padding: 16px; border-radius: 12px; }
      .hero { padding: 1.4rem 1.6rem; border: 1px solid #26364d; border-radius: 16px;
        background: linear-gradient(120deg, #14233a, #111b2e); margin-bottom: 1.2rem; }
      .muted { color: #9aabc2; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained model…")
def cached_model(dataset: str, model_name: str):
    return load_model(PROJECT_ROOT, dataset, model_name)


def benchmark_summary(dataset: str, model_name: str) -> tuple[float | None, str | None]:
    path = PROJECT_ROOT / "artifacts" / "reports" / f"{dataset}_classical_benchmark.json"
    if not path.is_file():
        return None, None
    try:
        import json

        report = json.loads(path.read_text(encoding="utf-8"))
        entry = report["models"][model_name]
        return float(entry["test"]["f1_macro"]), report.get("selected_model_by_validation_macro_f1")
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


st.markdown(
    """
    <div class="hero">
      <h1>🛡️ VoxSentinel AI</h1>
      <p class="muted">Review sentiment analysis, powered by the models trained in this project.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model settings")
    dataset = st.selectbox(
        "Dataset / label set",
        options=list(DATASET_NAMES),
        format_func=lambda value: DATASET_NAMES[value],
        help="IMDb predicts positive or negative. TweetEval predicts negative, neutral, or positive.",
    )
    selected_model = st.selectbox(
        "Trained classifier",
        options=list(MODEL_NAMES),
        format_func=lambda value: MODEL_NAMES[value],
        index=0,
    )
    score, validation_selected = benchmark_summary(dataset, selected_model)
    if validation_selected:
        st.caption(
            f"Validation macro-F1 selection: {MODEL_NAMES[validation_selected]}. "
            f"Current model test macro-F1: {score:.3f}."
            if score is not None
            else f"Validation macro-F1 selection: {MODEL_NAMES[validation_selected]}."
        )
    st.divider()
    st.caption("This interface runs the saved classical sentiment models. Emotion, aspect, sarcasm, and calibrated confidence outputs are not available here.")

try:
    model = cached_model(dataset, selected_model)
except (OSError, ValueError, ImportError) as exc:
    st.error(f"Could not load the selected model: {exc}")
    st.stop()

single_tab, bulk_tab = st.tabs(["Single review", "CSV batch"])

with single_tab:
    st.subheader("Analyze a review or post")
    text = st.text_area(
        "Text to analyze",
        height=180,
        placeholder="Paste customer feedback here…",
        help="Predictions come from the selected saved model; the interface does not generate example results.",
    )
    analyze_clicked = st.button("Analyze sentiment", type="primary", disabled=not text.strip())
    if analyze_clicked:
        try:
            result = analyze_text(model, text)
            left, right = st.columns([1, 2])
            with left:
                st.metric("Predicted sentiment", result["label"].title())
                st.caption(f"Dataset: {DATASET_NAMES[dataset]}")
                st.caption(f"Model: {MODEL_NAMES[selected_model]}")
            with right:
                st.markdown(f"**Model scores** · {result['score_type']}")
                score_frame = pd.DataFrame(
                    {"Sentiment": list(result["scores"]), "Score": list(result["scores"].values())}
                ).sort_values("Score", ascending=True)
                chart = px.bar(
                    score_frame,
                    x="Score",
                    y="Sentiment",
                    orientation="h",
                    color="Score",
                    color_continuous_scale=["#37506e", "#53b4a0"],
                )
                chart.update_layout(
                    height=230,
                    margin=dict(l=0, r=0, t=8, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf7",
                    coloraxis_showscale=False,
                    xaxis_title=None,
                    yaxis_title=None,
                )
                st.plotly_chart(chart, use_container_width=True)
                if selected_model == "logistic_regression":
                    st.caption("These are Logistic Regression probabilities and are not calibrated confidence estimates.")
                else:
                    st.caption("Linear SVM scores are raw decision values, not probabilities or confidence estimates.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

with bulk_tab:
    st.subheader("Analyze a CSV file")
    st.write("Upload a CSV containing one text column. The app adds sentiment predictions and model scores.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(BytesIO(uploaded.getvalue()))
            if batch.empty:
                st.warning("The uploaded CSV has no rows.")
            elif len(batch) > 10_000:
                st.error("For this local interface, batch files are limited to 10,000 rows.")
            else:
                text_column = st.selectbox("Text column", options=list(batch.columns))
                st.dataframe(batch.head(8), use_container_width=True)
                if st.button("Analyze CSV", type="primary"):
                    texts = batch[text_column].fillna("").astype(str).str.strip().tolist()
                    valid_positions = [index for index, value in enumerate(texts) if clean_text(value)]
                    results = analyze_many(model, [texts[index] for index in valid_positions])
                    batch["predicted_sentiment"] = ""
                    batch["predicted_sentiment"] = batch["predicted_sentiment"].astype("string")
                    batch.loc[valid_positions, "predicted_sentiment"] = results["labels"]
                    if results["scores"] is not None:
                        for label in results["classes"]:
                            column = f"score_{label}"
                            batch[column] = pd.NA
                            batch.loc[valid_positions, column] = [row[label] for row in results["scores"]]
                    st.success(f"Analyzed {len(valid_positions):,} non-empty rows.")
                    st.caption(f"Score type: {results['score_type'] or 'not available for empty input'}")
                    st.dataframe(batch.head(100), use_container_width=True)
                    csv_bytes = batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download analyzed CSV",
                        data=csv_bytes,
                        file_name="voxsentinel_analysis.csv",
                        mime="text/csv",
                    )
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            st.error(f"Could not read this CSV: {exc}")

st.divider()
st.caption("Predictions are model outputs, not guarantees of correctness. Do not treat sentiment scores as calibrated certainty.")
