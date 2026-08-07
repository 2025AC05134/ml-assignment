"""Train Logistic Regression on UCI Adult and persist the fitted pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor

RANDOM_STATE = 42
MODEL_NAME = "Logistic Regression"
MODEL_FILE = "logistic_regression.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_save(x_train, y_train, output_dir: Path) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, output_dir / MODEL_FILE, compress=3)
    return pipeline
