"""Train Decision Tree on UCI Adult and persist the fitted pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import build_preprocessor

RANDOM_STATE = 42
MODEL_NAME = "Decision Tree"
MODEL_FILE = "decision_tree.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=12,
                    min_samples_leaf=20,
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
