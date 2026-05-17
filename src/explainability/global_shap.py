"""Global SHAP computation for all models and SHAP analysis orchestrator."""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from src.config import (
    BASELINES_DIR,
    N_FOLDS,
    RANDOM_SEED,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_DIR,
    SHAP_NSAMPLES,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)


# Max test samples for slow explainers (KernelExplainer)
KERNEL_MAX_SAMPLES = 20


def _extract_shap_array(shap_result):
    """Extract a plain numpy array from various SHAP return types."""
    if hasattr(shap_result, "values"):
        # shap.Explanation object
        return shap_result.values
    return shap_result


def compute_global_shap(model, X_train, X_test, model_type, feature_names, objective):
    """Compute global SHAP values for a model.

    Args:
        model: Trained model with predict_proba.
        X_train: Training data for background.
        X_test: Test data to explain.
        model_type: One of 'LogisticRegression', 'RandomForest', 'XGBoost', 'SVM', 'TabNet'.
        feature_names: List of feature names.
        objective: 'cgpa_category', 'at_risk', or 'career_ready'.

    Returns:
        shap_values: SHAP values array (2D or 3D).
        global_importance: Mean absolute SHAP per feature (1D).
    """
    logger.info("  Computing SHAP for %s on %s (test size: %d)...", model_type, objective, len(X_test))

    if model_type in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_test)
        shap_values = _extract_shap_array(raw)
    elif model_type == "LogisticRegression":
        # Use masker-based Explainer to avoid multi-class crash
        background = X_train[np.random.choice(len(X_train), min(100, len(X_train)), replace=False)]
        explainer = shap.Explainer(model, background)
        explanation = explainer(X_test)
        shap_values = explanation.values  # (n_samples, n_features) or (n_samples, n_features, n_classes)
    else:
        # SVM, TabNet — KernelExplainer on a small subsample
        n_sub = min(KERNEL_MAX_SAMPLES, len(X_test))
        X_sub = X_test[:n_sub]
        logger.info("    Using %d/%d test samples for KernelExplainer", n_sub, len(X_test))
        background = shap.kmeans(X_train, min(SHAP_BACKGROUND_SAMPLES, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(X_sub, nsamples=min(SHAP_NSAMPLES, 500))
        shap_values = _extract_shap_array(raw)

    # Compute global importance from whatever shape we got
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv) for sv in shap_values])
        global_importance = stacked.mean(axis=(0, 1))
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        global_importance = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.shape[2] < shap_values.shape[1] \
            else np.abs(shap_values).mean(axis=(0, 1))
    else:
        global_importance = np.abs(shap_values).mean(axis=0)

    # Ensure global_importance length matches features
    if len(global_importance) != len(feature_names):
        logger.warning("  SHAP importance length %d != features %d, attempting axis transpose",
                       len(global_importance), len(feature_names))
        if shap_values.ndim == 3:
            global_importance = np.abs(shap_values).mean(axis=(0, -1)) if shap_values.shape[-1] != len(feature_names) \
                else np.abs(shap_values).mean(axis=0).mean(axis=-1)

    # Final safety check
    if len(global_importance) != len(feature_names):
        logger.error("  Cannot align SHAP importance (%d) with features (%d), skipping save",
                     len(global_importance), len(feature_names))
        return shap_values, global_importance

    # Save global importance
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_importance,
    }).sort_values("mean_abs_shap", ascending=False)
    imp_path = os.path.join(SHAP_DIR, f"global_importance_{model_type}_{objective}.csv")
    imp_df.to_csv(imp_path, index=False)
    logger.info("  Saved global importance to %s", imp_path)

    return shap_values, global_importance


def run_shap_analysis(X, y_cgpa, y_at_risk, y_career_ready, feature_names):
    """Compute SHAP for all trained models (baselines + TabNet) per objective."""
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}
    # Skip SVM (KernelExplainer too slow) and TabNet (needs re-train) from full SHAP
    # Focus on tree models + LR which have fast native explainers
    baseline_types = ["LogisticRegression", "RandomForest", "XGBoost"]

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for obj_name, y in objectives.items():
        logger.info("Running SHAP analysis for %s...", obj_name)
        splits = list(cv.split(X, y))
        train_idx, test_idx = splits[0]
        X_train, X_test = X[train_idx], X[test_idx]

        # --- Baseline models (fast) ---
        for model_type in baseline_types:
            model_path = os.path.join(
                BASELINES_DIR, f"{model_type.lower()}_{obj_name}_best.pkl"
            )
            if not os.path.exists(model_path):
                logger.warning("Model file not found: %s, skipping", model_path)
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            try:
                shap_values, global_imp = compute_global_shap(
                    model, X_train, X_test, model_type, feature_names, obj_name
                )
                # Save local SHAP values
                local_path = os.path.join(SHAP_DIR, f"local_shap_{model_type}_{obj_name}_fold0.npy")
                if isinstance(shap_values, list):
                    np.save(local_path, np.stack(shap_values))
                else:
                    np.save(local_path, shap_values)
            except Exception as e:
                logger.error("SHAP failed for %s/%s: %s", model_type, obj_name, e)

        # --- TabNet (train a quick 2-fold model, KernelExplainer on subsample) ---
        try:
            from src.models.tabnet_model import train_tabnet_cv
            logger.info("  Training TabNet for SHAP (2-fold)...")
            fold_results, _, fold_models = train_tabnet_cv(X, y, obj_name, n_folds=2, seed=RANDOM_SEED)
            model = fold_models[0]

            shap_values, global_imp = compute_global_shap(
                model, X_train, X_test, "TabNet", feature_names, obj_name
            )
            local_path = os.path.join(SHAP_DIR, f"local_shap_TabNet_{obj_name}_fold0.npy")
            if isinstance(shap_values, list):
                np.save(local_path, np.stack(shap_values))
            else:
                np.save(local_path, shap_values)

            # Spearman correlation: attention vs SHAP
            att_path = os.path.join(SHAP_DIR, f"attention_importance_tabnet_{obj_name}.csv")
            if os.path.exists(att_path) and len(global_imp) == len(feature_names):
                att_df = pd.read_csv(att_path)
                att_imp = att_df.set_index("feature")["attention_importance"]
                shap_imp = pd.Series(global_imp, index=feature_names)
                common = att_imp.index.intersection(shap_imp.index)
                if len(common) > 5:
                    corr, pval = spearmanr(att_imp.loc[common].values, shap_imp.loc[common].values)
                    logger.info("  TabNet attention vs SHAP Spearman: r=%.4f, p=%.4e (%s)", corr, pval, obj_name)
        except Exception as e:
            logger.error("TabNet SHAP failed for %s: %s", obj_name, e)

    # Generate cross-model comparison of top features
    _generate_shap_comparison(feature_names)

    logger.info("SHAP analysis complete.")


def _generate_shap_comparison(feature_names):
    """Generate a cross-model top-feature comparison from saved SHAP importance files."""
    import glob
    files = glob.glob(os.path.join(SHAP_DIR, "global_importance_*.csv"))
    if not files:
        return

    records = []
    for f in files:
        parts = os.path.basename(f).replace("global_importance_", "").replace(".csv", "").rsplit("_", 1)
        if len(parts) == 2:
            model_name, objective = parts
            df = pd.read_csv(f)
            top10 = df.nlargest(10, "mean_abs_shap")["feature"].tolist()
            records.append({"model": model_name, "objective": objective, "top_10_features": top10})

    if records:
        # Check SC-002: 70% overlap across folds/models for same objective
        for obj in ["cgpa_category", "at_risk", "career_ready"]:
            obj_records = [r for r in records if r["objective"] == obj]
            if len(obj_records) >= 2:
                sets = [set(r["top_10_features"]) for r in obj_records]
                pairwise_overlaps = []
                for i in range(len(sets)):
                    for j in range(i + 1, len(sets)):
                        overlap = len(sets[i] & sets[j]) / 10.0
                        pairwise_overlaps.append(overlap)
                avg_overlap = np.mean(pairwise_overlaps)
                logger.info(
                    "SC-002 check (%s): avg top-10 feature overlap across models: %.1f%%",
                    obj, avg_overlap * 100,
                )
