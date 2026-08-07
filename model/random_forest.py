"""Train Random Forest (Ensemble) on UCI Adult and persist the fitted pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor

RANDOM_STATE = 42
MODEL_NAME = "Random Forest"
MODEL_FILE = "random_forest.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_save(x_train, y_train, output_dir: Path) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, output_dir / MODEL_FILE, compress=3)
    return pipeline
