"""Shared helpers for revision modules.

Provides:
    - Path constants pointing at recent-review-comments/revision/output/.
    - load_data(): loads preprocessed X, y, feature_names from existing pipeline outputs;
      falls back to running the preprocessor if cached arrays are absent.
    - load_raw_with_targets(): returns the cleaned + engineered DataFrame plus the
      three target arrays, useful for fairness/PSM modules that need raw demographic
      columns alongside the encoded feature matrix.
    - cv_iter(): a single source of truth for the 5-fold StratifiedKFold splits
      used across all revision modules so results are directly comparable.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

# Make the parent project importable when running scripts directly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import (  # noqa: E402
    PREPROCESSED_DIR,
    RANDOM_SEED,
    N_FOLDS,
)

logger = logging.getLogger(__name__)

REVISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REVISION_ROOT, "output")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

for _d in (OUTPUT_DIR, TABLES_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)


def load_preprocessed():
    """Load the preprocessed arrays produced by `python -m src.pipeline --stage preprocess`.

    Returns:
        X: (N, d) float32 feature matrix.
        y_cgpa, y_at_risk, y_career_ready: int64 target arrays.
        feature_names: list[str] in column order of X.
        encoders: dict of fitted encoders (may be None if encoders.pkl missing).
    """
    x_path = os.path.join(PREPROCESSED_DIR, "X.npy")
    if not os.path.exists(x_path):
        logger.warning(
            "Preprocessed X.npy not found at %s — running the preprocessor now.",
            x_path,
        )
        from src.data.preprocessor import preprocess_pipeline

        preprocess_pipeline()

    X = np.load(os.path.join(PREPROCESSED_DIR, "X.npy"))
    y_cgpa = np.load(os.path.join(PREPROCESSED_DIR, "y_cgpa.npy"))
    y_at_risk = np.load(os.path.join(PREPROCESSED_DIR, "y_at_risk.npy"))
    y_career_ready = np.load(os.path.join(PREPROCESSED_DIR, "y_career_ready.npy"))

    with open(os.path.join(PREPROCESSED_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    encoders = None
    enc_path = os.path.join(PREPROCESSED_DIR, "encoders.pkl")
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)

    return X, y_cgpa, y_at_risk, y_career_ready, feature_names, encoders


def load_cleaned_dataframe() -> pd.DataFrame:
    """Returns the cleaned + feature-engineered DataFrame *before* encoding.

    Used by fairness/PSM modules that need raw demographic columns
    (gender, residence_type, family_income_bracket) alongside numeric features.
    """
    from src.data.feature_engineering import engineer_features
    from src.data.loader import load_dataset
    from src.data.preprocessor import clean_data

    df = load_dataset()
    df = clean_data(df)
    df = engineer_features(df)
    return df


def cv_iter(y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Single canonical 5-fold StratifiedKFold iterator (seed=42).

    Identical to the splits used in src/models/baselines.py, ensuring
    revision-module results are directly comparable to Table 4 of the paper.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return skf.split(np.zeros(len(y)), y)


def write_markdown_table(rows, header, path: str, title: str | None = None) -> None:
    """Write a list-of-rows table as a Markdown file (also CSV alongside)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")

    csv_path = os.path.splitext(path)[0] + ".csv"
    pd.DataFrame(rows, columns=header).to_csv(csv_path, index=False)
    logger.info("Wrote %s and %s", path, csv_path)


def configure_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
