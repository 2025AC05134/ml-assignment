"""Train Gaussian Naive Bayes on UCI Adult and persist the fitted pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor

MODEL_NAME = "Naive Bayes"
MODEL_FILE = "naive_bayes.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", GaussianNB()),
        ]
    )


def train_and_save(x_train, y_train, output_dir: Path) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, output_dir / MODEL_FILE, compress=3)
    return pipeline
