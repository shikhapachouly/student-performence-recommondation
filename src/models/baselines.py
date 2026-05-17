"""Baseline model training: LR, RF, SVM, XGBoost with Optuna tuning + SMOTE + StratifiedKFold CV.

Optuna (TPE sampler, 50 trials) tunes hyperparameters using inner 3-fold CV
on each outer training fold. This prevents information leakage between
hyperparameter selection and performance evaluation.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, make_scorer

from src.config import (
    BASELINES_DIR,
    N_FOLDS,
    RANDOM_SEED,
    TABLES_DIR,
)
from src.evaluation.metrics import compute_metrics, aggregate_fold_results

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_N_TRIALS = 50
OPTUNA_INNER_FOLDS = 3


def _build_model(model_name, n_classes, params):
    """Build a model with the given hyperparameters."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            multi_class="multinomial" if n_classes > 2 else "auto",
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
        )
    elif model_name == "SVM":
        return SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            random_state=RANDOM_SEED,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            use_label_encoder=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _suggest_params(trial, model_name):
    """Suggest hyperparameters for an Optuna trial."""
    if model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }
    elif model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    elif model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _tune_hyperparams(X_train, y_train, model_name, n_classes):
    """Run Optuna TPE optimization with inner 3-fold CV on training data.

    Returns the best hyperparameter dict.
    """
    f1_macro_scorer = make_scorer(f1_score, average="macro")

    def objective(trial):
        params = _suggest_params(trial, model_name)
        model = _build_model(model_name, n_classes, params)
        inner_cv = StratifiedKFold(n_splits=OPTUNA_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=inner_cv, scoring=f1_macro_scorer, n_jobs=1)
            return scores.mean()
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    logger.info("    Optuna best trial: F1-macro=%.4f, params=%s", study.best_value, study.best_params)
    return study.best_params


def get_model(model_name, n_classes):
    """Return a model with default hyperparameters (fallback, not used in tuned pipeline)."""
    return _build_model(model_name, n_classes, _get_defaults(model_name))


def _get_defaults(model_name):
    """Default hyperparameters (used only if Optuna is disabled)."""
    if model_name == "LogisticRegression":
        return {"C": 1.0, "solver": "lbfgs"}
    elif model_name == "RandomForest":
        return {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5}
    elif model_name == "SVM":
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    elif model_name == "XGBoost":
        return {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 1.0}
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_baseline_cv(X, y, model_name, objective, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """Train a single baseline model with StratifiedKFold CV + SMOTE + Optuna tuning.

    For each outer fold:
      1. SMOTE the training data
      2. Run Optuna (50 trials, inner 3-fold CV) to find best hyperparameters
      3. Train final model with best params on full SMOTE'd training set
      4. Evaluate on held-out test fold

    Returns:
        list of per-fold BenchmarkResult dicts, list of trained models.
    """
    n_classes = len(np.unique(y))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    fold_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply SMOTE on training data
        smote = SMOTE(random_state=seed)
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except ValueError:
            logger.warning("SMOTE failed for fold %d, using original data", fold_idx)
            X_train_res, y_train_res = X_train, y_train

        # Optuna hyperparameter tuning on SMOTE'd training data
        logger.info("  %s | %s | Fold %d: tuning (%d trials)...",
                     model_name, objective, fold_idx, OPTUNA_N_TRIALS)
        best_params = _tune_hyperparams(X_train_res, y_train_res, model_name, n_classes)

        # Train final model with best params
        model = _build_model(model_name, n_classes, best_params)

        if model_name == "XGBoost":
            sample_weights = compute_sample_weight("balanced", y_train_res)
            model.fit(X_train_res, y_train_res, sample_weight=sample_weights)
        else:
            model.fit(X_train_res, y_train_res)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba, objective)
        metrics["model_name"] = model_name
        metrics["objective"] = objective
        metrics["fold"] = fold_idx
        fold_results.append(metrics)
        fold_models.append(model)

        logger.info(
            "  %s | %s | Fold %d: accuracy=%.4f, f1_macro=%.4f",
            model_name, objective, fold_idx, metrics["accuracy"], metrics["f1_macro"],
        )

    return fold_results, fold_models


def run_all_baselines(X, y_cgpa, y_at_risk, y_career_ready, feature_names):
    """Train all baseline models for all objectives. Save results."""
    model_names = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}

    all_results = []
    all_models = {}

    for model_name in model_names:
        for obj_name, y in objectives.items():
            logger.info("Training %s for %s...", model_name, obj_name)
            fold_results, fold_models = train_baseline_cv(X, y, model_name, obj_name)
            all_results.extend(fold_results)
            all_models[f"{model_name}_{obj_name}"] = fold_models

            # Save best fold model
            best_fold = max(range(len(fold_results)), key=lambda i: fold_results[i]["f1_macro"])
            model_path = os.path.join(BASELINES_DIR, f"{model_name.lower()}_{obj_name}_best.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(fold_models[best_fold], f)

            # Log aggregated
            agg = aggregate_fold_results(fold_results)
            logger.info(
                "  %s | %s | Mean F1-macro: %.4f (+/- %.4f)",
                model_name, obj_name,
                agg.get("f1_macro_mean", 0), agg.get("f1_macro_std", 0),
            )

    # Save all fold-level results to CSV
    rows = []
    for r in all_results:
        row = {
            "model_name": r["model_name"],
            "objective": r["objective"],
            "fold": r["fold"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "balanced_accuracy": r["balanced_accuracy"],
            "auc_roc": r.get("auc_roc"),
        }
        rows.append(row)

    df_results = pd.DataFrame(rows)
    results_path = os.path.join(TABLES_DIR, "baseline_results.csv")
    df_results.to_csv(results_path, index=False)
    logger.info("Baseline results saved to %s", results_path)

    return all_results, all_models
