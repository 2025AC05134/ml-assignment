"""End-to-end training script for ML Assignment 2 (UCI Adult / Census Income).

Fetches the dataset, performs a stratified 80/20 split, trains all five
required classifiers, saves the fitted pipelines as `.joblib` files, and
writes `test_data.csv` plus `metrics.csv` for use by the Streamlit app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(MODEL_DIR))

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
import decision_tree  # noqa: E402
import knn  # noqa: E402
import logistic_regression  # noqa: E402
import naive_bayes  # noqa: E402
import random_forest  # noqa: E402


RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Fetch UCI Adult (Census Income) from OpenML and return a tidy DataFrame."""
    dataset = fetch_openml("adult", version=2, as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"class": TARGET_COLUMN})
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].astype(str).str.strip() == ">50K").astype(int)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def evaluate(pipeline, x_eval, y_eval) -> dict:
    y_pred = pipeline.predict(x_eval)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(x_eval)[:, 1]
    else:
        y_proba = y_pred
    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_proba),
        "Precision": precision_score(y_eval, y_pred),
        "Recall": recall_score(y_eval, y_pred),
        "F1": f1_score(y_eval, y_pred),
        "MCC": matthews_corrcoef(y_eval, y_pred),
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading UCI Adult (Census Income) dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    df.to_csv(DATA_DIR / "adult_full.csv", index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)
    train_df.to_csv(DATA_DIR / "train_data.csv", index=False)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    models = [
        ("Logistic Regression", logistic_regression),
        ("Decision Tree", decision_tree),
        ("KNN", knn),
        ("Naive Bayes", naive_bayes),
        ("Random Forest", random_forest),
    ]

    metrics_rows = []
    for display_name, module in models:
        print(f"\nTraining {display_name}...")
        pipeline = module.train_and_save(x_train, y_train, MODEL_DIR)
        metrics = evaluate(pipeline, x_test, y_test)
        metrics_rows.append({"ML Model Name": display_name, **metrics})
        cm = confusion_matrix(y_test, pipeline.predict(x_test))
        print(f"  Accuracy = {metrics['Accuracy']:.4f} | AUC = {metrics['AUC']:.4f}")
        print(f"  Confusion Matrix (rows=actual, cols=predicted):\n{cm}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(BASE_DIR / "metrics.csv", index=False)
    print("\nSaved:")
    print(f"  Models -> {MODEL_DIR}")
    print(f"  Test data -> {BASE_DIR / 'test_data.csv'}")
    print(f"  Metrics -> {BASE_DIR / 'metrics.csv'}")
    print("\nComparison Table:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
