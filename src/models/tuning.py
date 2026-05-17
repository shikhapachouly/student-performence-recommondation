"""Optuna Bayesian hyperparameter optimization for all models."""

import json
import logging
import os

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    OPTUNA_INNER_FOLDS,
    OPTUNA_N_TRIALS,
    OPTUNA_SEARCH_SPACES,
    RANDOM_SEED,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial, model_type):
    """Suggest hyperparameters from Optuna search spaces."""
    space = OPTUNA_SEARCH_SPACES[model_type]
    params = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec, tuple) and len(spec) == 3 and spec[2] == "log":
            params[name] = trial.suggest_float(name, spec[0], spec[1], log=True)
        elif isinstance(spec, tuple) and len(spec) == 2:
            if isinstance(spec[0], int) and isinstance(spec[1], int):
                params[name] = trial.suggest_int(name, spec[0], spec[1])
            else:
                params[name] = trial.suggest_float(name, spec[0], spec[1])
    return params


def _build_baseline_model(model_type, params):
    """Build a sklearn model with given params."""
    if model_type == "LogisticRegression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED, multi_class="multinomial",
        )
    elif model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            class_weight="balanced_subsample", random_state=RANDOM_SEED,
        )
    elif model_type == "SVM":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            class_weight="balanced", probability=True, random_state=RANDOM_SEED,
        )
    elif model_type == "XGBoost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            random_state=RANDOM_SEED, eval_metric="mlogloss",
            use_label_encoder=False,
        )
    raise ValueError(f"Unknown baseline model: {model_type}")


def _baseline_objective(trial, X, y, model_type):
    """Optuna objective for baseline ML models using inner CV."""
    params = _suggest_params(trial, model_type)
    model = _build_baseline_model(model_type, params)
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", model),
    ])
    inner_cv = StratifiedKFold(n_splits=OPTUNA_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X, y, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


def _tabnet_objective(trial, X, y, objective_name):
    """Optuna objective for TabNet using inner CV."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    params = _suggest_params(trial, "TabNet")
    n_d = params["n_d"]

    inner_cv = StratifiedKFold(n_splits=OPTUNA_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    for train_idx, test_idx in inner_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_t, X_v, y_t, y_v = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=RANDOM_SEED)
        sm = SMOTE(random_state=RANDOM_SEED)
        X_t, y_t = sm.fit_resample(X_t, y_t)
        cw = compute_class_weight("balanced", classes=np.unique(y_t), y=y_t)
        wd = dict(enumerate(cw))
        model = TabNetClassifier(
            n_d=n_d, n_a=n_d, n_steps=params["n_steps"],
            lambda_sparse=params["lambda_sparse"],
            optimizer_params=dict(lr=params["lr"]),
            seed=RANDOM_SEED, verbose=0,
        )
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], eval_metric=["balanced_accuracy"],
                  max_epochs=100, patience=10, batch_size=128, weights=wd, drop_last=False)
        from sklearn.metrics import f1_score
        y_pred = model.predict(X_te)
        fold_scores.append(f1_score(y_te, y_pred, average="macro"))
    return np.mean(fold_scores)


def tune_model(X, y, model_type, objective_name, n_trials=None):
    """Run Optuna tuning for a single model/objective.

    Returns:
        dict: Best hyperparameters.
    """
    if n_trials is None:
        n_trials = OPTUNA_N_TRIALS

    logger.info("Tuning %s for %s (%d trials)...", model_type, objective_name, n_trials)

    if model_type in ("LogisticRegression", "RandomForest", "SVM", "XGBoost"):
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(lambda trial: _baseline_objective(trial, X, y, model_type),
                       n_trials=n_trials, show_progress_bar=False)
    elif model_type == "TabNet":
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(lambda trial: _tabnet_objective(trial, X, y, objective_name),
                       n_trials=n_trials, show_progress_bar=False)
    elif model_type in ("FTTransformer", "SAINT"):
        # DL models: use default params (Optuna tuning for these is slow)
        # Return default search space midpoints
        logger.info("  Using default hyperparameters for %s (skip Optuna for speed)", model_type)
        return _default_dl_params(model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info("  Best F1-macro: %.4f | Params: %s", study.best_value, study.best_params)
    return study.best_params


def _default_dl_params(model_type):
    """Return reasonable default params for DL models."""
    if model_type == "FTTransformer":
        return {"n_blocks": 2, "d_token": 64, "attention_dropout": 0.2,
                "ffn_dropout": 0.1, "lr": 1e-3}
    elif model_type == "SAINT":
        return {"depth": 3, "heads": 4, "dim": 32,
                "attn_dropout": 0.1, "lr": 1e-3}
    return {}


def tune_all_models(X, y_cgpa, y_at_risk, y_career_ready, n_trials=None):
    """Tune all models for both objectives. Save results to JSON."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    all_params = {}
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}

    for obj_name, y in objectives.items():
        all_params[obj_name] = {}
        for model_type in ["LogisticRegression", "RandomForest", "SVM", "XGBoost",
                           "TabNet", "FTTransformer", "SAINT"]:
            best = tune_model(X, y, model_type, obj_name, n_trials=n_trials)
            all_params[obj_name][model_type] = best

    out_path = os.path.join(TABLES_DIR, "best_hyperparams.json")
    with open(out_path, "w") as f:
        json.dump(all_params, f, indent=2, default=str)
    logger.info("All tuned hyperparameters saved to %s", out_path)
    return all_params
