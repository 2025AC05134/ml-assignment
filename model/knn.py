"""Train K-Nearest Neighbours on UCI Adult and persist the fitted pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor

MODEL_NAME = "KNN"
MODEL_FILE = "knn.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
            ),
        ]
    )


def train_and_save(x_train, y_train, output_dir: Path) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, output_dir / MODEL_FILE, compress=3)
    return pipeline
