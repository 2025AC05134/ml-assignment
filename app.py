"""Streamlit app for ML Assignment 2.

Required features (from the PDF, Step 6):
    a. Dataset upload option (CSV) - test data only
    b. Model selection dropdown
    c. Display of evaluation metrics
    d. Confusion matrix / classification report

Optional enhancement:
    - "All models comparison" tab that evaluates all 5 trained models
      on the uploaded test CSV in a single view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

sys.path.insert(0, str(MODEL_DIR))
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    return models


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def predict_with_proba(model, x_eval):
    y_pred = model.predict(x_eval)
    try:
        y_proba = model.predict_proba(x_eval)[:, 1]
    except Exception:
        y_proba = None
    return y_pred, y_proba


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        st.error(f"Missing required feature columns: {missing_features}")
        return None
    if TARGET_COLUMN not in df.columns:
        st.error(f"Missing required target column: `{TARGET_COLUMN}` (0/1).")
        return None
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def render_sidebar() -> str:
    st.sidebar.title("ML Assignment 2")
    st.sidebar.markdown(
        "**Dataset:** UCI Adult / Census Income  \n"
        "**Task:** Binary Classification (income >50K vs ≤50K)"
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Feature b) Model selection")
    return st.sidebar.selectbox("Choose a trained model", list(MODEL_FILES.keys()))


def render_upload_section() -> pd.DataFrame | None:
    st.subheader("Feature a) Upload test dataset (CSV)")
    st.caption(
        "Upload the provided `test_data.csv` (or any CSV with the same schema: "
        "14 features + a `class` target column of 0/1)."
    )
    uploaded_file = st.file_uploader("Test dataset (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Waiting for a CSV upload. Use `test_data.csv` shipped with this repo.")
        return None
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded CSV with shape {df.shape}")
    with st.expander("Preview first 5 rows"):
        st.dataframe(df.head(5), width="stretch")
    return df


def render_single_model_view(models: dict, selected_model_name: str, x_eval, y_eval) -> None:
    model = models[selected_model_name]
    y_pred, y_proba = predict_with_proba(model, x_eval)
    metrics = compute_metrics(y_eval, y_pred, y_proba)

    st.markdown(f"### Selected model: **{selected_model_name}**")

    st.subheader("Feature c) Evaluation metrics")
    cols = st.columns(6)
    for col, label in zip(cols, METRIC_ORDER):
        value = metrics[label]
        if isinstance(value, float) and not np.isnan(value):
            col.metric(label, f"{value:.4f}")
        else:
            col.metric(label, "N/A")

    metrics_df = pd.DataFrame(
        [
            {"Metric": k, "Value": round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k]}
            for k in METRIC_ORDER
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Feature d) Confusion matrix & classification report")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_eval, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: ≤50K (0)", "Actual: >50K (1)"],
            columns=["Predicted: ≤50K (0)", "Predicted: >50K (1)"],
        )
        st.dataframe(cm_df, use_container_width=True)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y_eval, y_pred, digits=4)
        st.code(report, language="text")


def render_all_models_view(models: dict, x_eval, y_eval) -> None:
    st.markdown("### All 5 models evaluated on the uploaded test CSV")
    st.caption(
        "Same dataset for every model, as required by the assignment. "
        "The best value in each metric column is highlighted below."
    )

    rows = []
    predictions = {}
    for name, model in models.items():
        y_pred, y_proba = predict_with_proba(model, x_eval)
        metrics = compute_metrics(y_eval, y_pred, y_proba)
        predictions[name] = y_pred
        rows.append({"ML Model Name": name, **{m: metrics[m] for m in METRIC_ORDER}})

    comparison_df = pd.DataFrame(rows)

    styled = comparison_df.style.format({m: "{:.4f}" for m in METRIC_ORDER})
    for metric in METRIC_ORDER:
        styled = styled.highlight_max(subset=[metric], color="#c8f7c5")
    st.dataframe(styled, use_container_width=True)

    st.download_button(
        label="Download comparison as CSV",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="comparison_metrics.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Confusion matrices (side-by-side)")
    cols = st.columns(len(models))
    for col, (name, y_pred) in zip(cols, predictions.items()):
        with col:
            st.markdown(f"**{name}**")
            cm = confusion_matrix(y_eval, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["≤50K", ">50K"],
                columns=["Pred ≤50K", "Pred >50K"],
            )
            st.dataframe(cm_df, use_container_width=True)

    winner_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    st.success(
        f"**Best accuracy on this test set:** "
        f"{winner_row['ML Model Name']} → {winner_row['Accuracy']:.4f}"
    )


def main() -> None:
    st.set_page_config(page_title="ML Assignment 2", layout="wide")
    st.title("ML Assignment 2 - Classification Explorer")
    st.markdown(
        "This app demonstrates 5 classification models trained on the "
        "**UCI Adult (Census Income)** dataset. Upload test data, pick a model, "
        "and view its metrics, confusion matrix and classification report — or "
        "switch to the *All models comparison* tab to see all 5 at once."
    )

    models = load_models()
    if not models:
        st.error("No trained models found in `model/`. Run `python train_all.py` first.")
        return

    selected_model_name = render_sidebar()
    df = render_upload_section()
    if df is None:
        return

    validated = validate_dataframe(df)
    if validated is None:
        return
    x_eval, y_eval = validated

    st.markdown("---")
    tab1, tab2 = st.tabs(["Single model", "All models comparison"])
    with tab1:
        render_single_model_view(models, selected_model_name, x_eval, y_eval)
    with tab2:
        render_all_models_view(models, x_eval, y_eval)

    st.markdown("---")
    with st.expander("Assignment compliance mapping"):
        st.markdown(
            "- **a) Dataset upload (CSV):** file uploader above  \n"
            "- **b) Model selection dropdown:** left sidebar  \n"
            "- **c) Evaluation metrics:** Accuracy / AUC / Precision / Recall / F1 / MCC  \n"
            "- **d) Confusion matrix + Classification report:** shown in the *Single model* tab  \n"
            "- **Bonus:** *All models comparison* tab shows every model's metrics side-by-side"
        )


if __name__ == "__main__":
    main()
