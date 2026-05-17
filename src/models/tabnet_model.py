"""TabNet deep learning model training with manual CV for SMOTE + early stopping."""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    N_FOLDS,
    RANDOM_SEED,
    TABLES_DIR,
    TABNET_BATCH_SIZE,
    TABNET_CLIP_VALUE,
    TABNET_GAMMA,
    TABNET_LAMBDA_SPARSE,
    TABNET_LR,
    TABNET_MASK_TYPE,
    TABNET_MAX_EPOCHS,
    TABNET_MOMENTUM,
    TABNET_N_A,
    TABNET_N_D,
    TABNET_N_STEPS,
    TABNET_PATIENCE,
    TABNET_SCHEDULER_GAMMA,
    TABNET_SCHEDULER_STEP,
    TABNET_VALID_SPLIT,
    SHAP_DIR,
)
from src.evaluation.metrics import compute_metrics, aggregate_fold_results

logger = logging.getLogger(__name__)

TABNET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "models", "tabnet")


def train_tabnet_cv(X, y, objective, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """Train TabNet with manual StratifiedKFold CV, SMOTE, and early stopping.

    Returns:
        fold_results: list of per-fold BenchmarkResult dicts
        attention_importances: list of per-fold attention importance arrays
        fold_models: list of trained TabNet models
    """
    n_classes = len(np.unique(y))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    attention_importances = []
    fold_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info("  TabNet | %s | Fold %d/%d", objective, fold_idx + 1, n_folds)

        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Split train into train/valid for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=TABNET_VALID_SPLIT,
            stratify=y_train_fold,
            random_state=seed,
        )

        # Apply SMOTE on training portion only
        smote = SMOTE(random_state=seed)
        try:
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        except ValueError:
            logger.warning("SMOTE failed for fold %d, using original data", fold_idx)
            X_tr_res, y_tr_res = X_tr, y_tr

        # Compute class weights
        classes = np.unique(y_tr)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        weights_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

        # Instantiate fresh TabNet
        model = TabNetClassifier(
            n_d=TABNET_N_D,
            n_a=TABNET_N_A,
            n_steps=TABNET_N_STEPS,
            gamma=TABNET_GAMMA,
            n_independent=1,
            n_shared=1,
            lambda_sparse=TABNET_LAMBDA_SPARSE,
            momentum=TABNET_MOMENTUM,
            clip_value=TABNET_CLIP_VALUE,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=TABNET_LR),
            scheduler_params={"step_size": TABNET_SCHEDULER_STEP, "gamma": TABNET_SCHEDULER_GAMMA},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=TABNET_MASK_TYPE,
            seed=seed,
            verbose=0,
        )

        # Train with eval_set for early stopping
        model.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val, y_val)],
            eval_metric=["balanced_accuracy"],
            max_epochs=TABNET_MAX_EPOCHS,
            patience=TABNET_PATIENCE,
            batch_size=TABNET_BATCH_SIZE,
            weights=weights_dict,
            drop_last=False,
        )

        # Predict on held-out test fold
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)

        # Compute metrics
        metrics = compute_metrics(y_test_fold, y_pred, y_proba, objective)
        metrics["model_name"] = "TabNet"
        metrics["objective"] = objective
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        # Extract attention masks
        try:
            explain_matrix, masks = model.explain(X_test_fold)
            attention_imp = explain_matrix.mean(axis=0)
            attention_importances.append(attention_imp)
        except Exception as e:
            logger.warning("Could not extract attention masks: %s", e)
            attention_importances.append(None)

        fold_models.append(model)

        logger.info(
            "    Fold %d: accuracy=%.4f, f1_macro=%.4f, balanced_acc=%.4f",
            fold_idx, metrics["accuracy"], metrics["f1_macro"], metrics["balanced_accuracy"],
        )

    return fold_results, attention_importances, fold_models


def run_tabnet(X, y_cgpa, y_at_risk, y_career_ready, feature_names):
    """Train TabNet for all objectives. Save results and attention importance."""
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}
    all_results = []
    all_models = {}

    for obj_name, y in objectives.items():
        logger.info("Training TabNet for %s...", obj_name)
        fold_results, attention_imps, fold_models = train_tabnet_cv(X, y, obj_name)
        all_results.extend(fold_results)
        all_models[f"TabNet_{obj_name}"] = fold_models

        # Save attention importance
        valid_imps = [a for a in attention_imps if a is not None]
        if valid_imps:
            mean_attention = np.mean(np.stack(valid_imps), axis=0)
            att_df = pd.DataFrame({
                "feature": feature_names,
                "attention_importance": mean_attention,
            }).sort_values("attention_importance", ascending=False)
            att_path = os.path.join(SHAP_DIR, f"attention_importance_tabnet_{obj_name}.csv")
            att_df.to_csv(att_path, index=False)

        # Log aggregated
        agg = aggregate_fold_results(fold_results)
        logger.info(
            "  TabNet | %s | Mean F1-macro: %.4f (+/- %.4f)",
            obj_name, agg.get("f1_macro_mean", 0), agg.get("f1_macro_std", 0),
        )

    # Save all fold-level results to CSV
    rows = []
    for r in all_results:
        rows.append({
            "model_name": r["model_name"],
            "objective": r["objective"],
            "fold": r["fold"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "balanced_accuracy": r["balanced_accuracy"],
            "auc_roc": r.get("auc_roc"),
        })

    df_results = pd.DataFrame(rows)
    results_path = os.path.join(TABLES_DIR, "tabnet_results.csv")
    df_results.to_csv(results_path, index=False)
    logger.info("TabNet results saved to %s", results_path)

    return all_results, all_models
