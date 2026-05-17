"""Main pipeline orchestrator for the Student Performance Prediction system."""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from src.config import (
    BASELINES_DIR,
    FT_TRANSFORMER_DIR,
    MODELS_DIR,
    OBJECTIVES,
    PLOTS_DIR,
    PREPROCESSED_DIR,
    RANDOM_SEED,
    RECOMMENDATIONS_DIR,
    RESULTS_DIR,
    SAINT_DIR,
    SHAP_DIR,
    TABLES_DIR,
    PAPER_DIR,
    PAPER_FIGURES_DIR,
    PAPER_TABLES_DIR,
    N_FOLDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_dirs():
    """Create all output directories."""
    for d in [
        PREPROCESSED_DIR, BASELINES_DIR, SHAP_DIR, TABLES_DIR,
        PLOTS_DIR, RECOMMENDATIONS_DIR, PAPER_DIR, PAPER_FIGURES_DIR,
        PAPER_TABLES_DIR,
        os.path.join(MODELS_DIR, "tabnet"),
        FT_TRANSFORMER_DIR, SAINT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def load_preprocessed():
    """Load preprocessed data from disk (3 objectives: cgpa, at_risk, career_ready)."""
    X = np.load(os.path.join(PREPROCESSED_DIR, "X.npy"))
    y_cgpa = np.load(os.path.join(PREPROCESSED_DIR, "y_cgpa.npy"))
    y_at_risk = np.load(os.path.join(PREPROCESSED_DIR, "y_at_risk.npy"))
    y_career_ready = np.load(os.path.join(PREPROCESSED_DIR, "y_career_ready.npy"))
    with open(os.path.join(PREPROCESSED_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    return X, y_cgpa, y_at_risk, y_career_ready, feature_names


def run_stage(stage, args):
    """Run a single pipeline stage."""
    t0 = time.time()
    logger.info("=== Starting stage: %s ===", stage)

    if stage == "preprocess":
        from src.data.preprocessor import preprocess_pipeline
        preprocess_pipeline()

    elif stage == "tune":
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        from src.models.tuning import tune_all_models
        n_trials = getattr(args, "n_trials", None)
        tune_all_models(X, y_cgpa, y_at_risk, y_career_ready, n_trials=n_trials)

    elif stage == "baselines":
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        from src.models.baselines import run_all_baselines
        run_all_baselines(X, y_cgpa, y_at_risk, y_career_ready, feature_names)

    elif stage == "tabnet":
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        from src.models.tabnet_model import run_tabnet
        run_tabnet(X, y_cgpa, y_at_risk, y_career_ready, feature_names)

    elif stage == "dlmodels":
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        params = None
        params_path = os.path.join(TABLES_DIR, "best_hyperparams.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                params = json.load(f)
            logger.info("Loaded tuned hyperparameters from %s", params_path)
        from src.models.dl_orchestrator import run_all_dl_models
        run_all_dl_models(X, y_cgpa, y_at_risk, y_career_ready, feature_names, params=params)

    elif stage == "shap":
        if getattr(args, "skip_shap", False):
            logger.info("Skipping SHAP stage (--skip-shap)")
            return
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        from src.explainability.global_shap import run_shap_analysis
        run_shap_analysis(X, y_cgpa, y_at_risk, y_career_ready, feature_names)

    elif stage == "recommend":
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        from src.recommendations.engine import run_recommendations
        import pickle
        with open(os.path.join(PREPROCESSED_DIR, "encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        run_recommendations(X, y_cgpa, y_at_risk, y_career_ready, feature_names, encoders)

    elif stage == "visualize":
        X, y_cgpa, y_at_risk, y_career_ready, feature_names = load_preprocessed()
        from src.visualization.plots import run_visualizations
        run_visualizations(feature_names)

    elif stage == "paper":
        from src.paper.generator import generate_paper
        generate_paper(RESULTS_DIR, PAPER_DIR)

    else:
        logger.error("Unknown stage: %s", stage)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("=== Stage %s completed in %.1f seconds ===", stage, elapsed)


def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction Pipeline")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["preprocess", "tune", "baselines", "tabnet", "dlmodels", "shap", "recommend", "visualize", "paper", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--objective", type=str, default="all", choices=["cgpa_category", "at_risk", "career_ready", "all"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials (default: config)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set seeds
    import random as _random
    np.random.seed(args.seed)
    _random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except (ImportError, OSError):
        pass

    ensure_dirs()

    total_t0 = time.time()

    if args.stage == "all":
        stages = ["preprocess", "tune", "baselines", "dlmodels", "shap", "recommend", "visualize", "paper"]
    else:
        stages = [args.stage]

    for stage in stages:
        run_stage(stage, args)

    total_elapsed = time.time() - total_t0
    logger.info("=== Pipeline complete. Total time: %.1f seconds ===", total_elapsed)


if __name__ == "__main__":
    main()
