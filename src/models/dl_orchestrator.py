"""Orchestrator for all deep learning models (TabNet, FT-Transformer, SAINT)."""

import logging
import os

import pandas as pd

from src.config import TABLES_DIR

logger = logging.getLogger(__name__)


def run_all_dl_models(X, y_cgpa, y_at_risk, y_career_ready, feature_names, params=None):
    """Train all DL models for both objectives.

    Args:
        X: Feature array.
        y_cgpa: CGPA target.
        y_at_risk: At-risk target (CGPA<7.0 OR backlogs).
        y_career_ready: Career readiness target (internship + projects).
        feature_names: List of feature names.
        params: Optional dict of tuned hyperparameters (from best_hyperparams.json).

    Returns:
        List of aggregated result dicts for all DL models.
    """
    # TabNet (returns (results_list, models_dict))
    from src.models.tabnet_model import run_tabnet
    logger.info("=== Training TabNet ===")
    tabnet_results, tabnet_models = run_tabnet(X, y_cgpa, y_at_risk, y_career_ready, feature_names)

    # FT-Transformer (returns results_list)
    from src.models.ft_transformer import run_ft_transformer
    logger.info("=== Training FT-Transformer ===")
    ft_results = run_ft_transformer(X, y_cgpa, y_at_risk, y_career_ready, feature_names, params=params)

    # SAINT (returns results_list)
    from src.models.saint_model import run_saint
    logger.info("=== Training SAINT ===")
    saint_results = run_saint(X, y_cgpa, y_at_risk, y_career_ready, feature_names, params=params)

    # Save combined DL results from individual CSVs
    os.makedirs(TABLES_DIR, exist_ok=True)
    dl_csvs = [
        os.path.join(TABLES_DIR, "tabnet_results.csv"),
        os.path.join(TABLES_DIR, "ft_transformer_results.csv"),
        os.path.join(TABLES_DIR, "saint_results.csv"),
    ]
    frames = [pd.read_csv(p) for p in dl_csvs if os.path.exists(p)]
    if frames:
        dl_df = pd.concat(frames, ignore_index=True)
        dl_path = os.path.join(TABLES_DIR, "dl_results.csv")
        dl_df.to_csv(dl_path, index=False)
        logger.info("Combined DL results saved to %s", dl_path)

    return tabnet_results + ft_results + saint_results
