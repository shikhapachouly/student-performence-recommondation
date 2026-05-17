"""Replication-package tables (Reviewer 2, R2.3).

Reviewer concern:
    "[The paper] omits the complete hyperparameter search spaces, final best
     parameters per objective, preprocessing mappings, feature-generation code,
     SMOTE parameters, fold indices, software versions, and exact availability
     conditions for the dataset."

This module reads the (read-only) `src/config.py` and any cached
`results/tables/best_hyperparams.json` and produces:

    Table A1  Optuna search spaces per model
    Table A2  Best hyperparameters per (model x objective)
    Table A3  Preprocessing mappings (ordinal encodings, binary columns,
              ordinal scale columns, low-variance drops)
    Table A4  CV / SMOTE / runtime configuration

All outputs land in revision/output/tables/.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from typing import Iterable

try:
    from revision.code.common import (
        TABLES_DIR,
        write_markdown_table,
    )
except ImportError:
    from common import (  # type: ignore
        TABLES_DIR,
        write_markdown_table,
    )

logger = logging.getLogger(__name__)


def _format_search_space_value(v):
    if isinstance(v, tuple):
        if len(v) == 3:
            lo, hi, scale = v
            return f"[{lo}, {hi}] ({scale})"
        if len(v) == 2:
            lo, hi = v
            return f"[{lo}, {hi}]"
    if isinstance(v, list):
        return "{" + ", ".join(str(x) for x in v) + "}"
    return str(v)


def write_search_spaces():
    from src.config import OPTUNA_SEARCH_SPACES

    rows = []
    for model_name, params in OPTUNA_SEARCH_SPACES.items():
        for k, v in params.items():
            rows.append([model_name, k, _format_search_space_value(v)])
    write_markdown_table(
        rows,
        ["Model", "Hyperparameter", "Search space"],
        os.path.join(TABLES_DIR, "tableA1_search_spaces.md"),
        title="Table A1. Optuna search spaces per model (TPE sampler, 50 trials, inner 3-fold CV).",
    )


def write_best_hyperparams():
    from src.config import TABLES_DIR as SRC_TABLES_DIR

    best_path = os.path.join(SRC_TABLES_DIR, "best_hyperparams.json")
    if not os.path.exists(best_path):
        rows = [["—", "—", "—", "Run `python -m src.pipeline --stage tune` to produce best_hyperparams.json"]]
        write_markdown_table(
            rows,
            ["Model", "Objective", "Hyperparameter", "Value"],
            os.path.join(TABLES_DIR, "tableA2_best_hyperparams.md"),
            title="Table A2. Best hyperparameters per (model, objective) — placeholder.",
        )
        return

    with open(best_path) as f:
        data = json.load(f)

    rows = []
    for key, params in data.items():
        if "_" in key:
            model_name, *obj_parts = key.split("_")
            obj = "_".join(obj_parts)
        else:
            model_name, obj = key, ""
        if isinstance(params, dict):
            for hp, val in params.items():
                rows.append([model_name, obj, hp, _format_value(val)])
        else:
            rows.append([model_name, obj, "—", _format_value(params)])

    write_markdown_table(
        rows,
        ["Model", "Objective", "Hyperparameter", "Value"],
        os.path.join(TABLES_DIR, "tableA2_best_hyperparams.md"),
        title="Table A2. Best hyperparameters per (model, objective). Source: results/tables/best_hyperparams.json.",
    )


def _format_value(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_preprocessing_mappings():
    from src.config import (
        BINARY_COLUMNS,
        DROP_FEATURES,
        LOW_VARIANCE_DROP,
        ORDINAL_ENCODINGS,
        ORDINAL_SCALE_COLUMNS,
        STUDY_HOURS_ORDINAL,
        TARGET_FEATURES,
    )

    rows: list[list[str]] = []

    rows.append(["Identifiers dropped (`DROP_FEATURES`)", _format_iter(DROP_FEATURES)])
    rows.append(["Target-derived features dropped (`TARGET_FEATURES`)", _format_iter(TARGET_FEATURES)])
    rows.append(["Low-variance columns dropped (`LOW_VARIANCE_DROP`)", _format_iter(LOW_VARIANCE_DROP)])
    rows.append(["Binary Yes/No columns (`BINARY_COLUMNS`)", _format_iter(BINARY_COLUMNS)])
    rows.append(["Ordinal 1–5 scale columns (`ORDINAL_SCALE_COLUMNS`)", _format_iter(ORDINAL_SCALE_COLUMNS)])

    for col, levels in ORDINAL_ENCODINGS.items():
        rows.append([f"Ordinal encoding for `{col}`", " < ".join(levels)])
    rows.append([
        "study_hours ordinal map",
        ", ".join(f"{k} → {v}" for k, v in STUDY_HOURS_ORDINAL.items()),
    ])

    write_markdown_table(
        rows,
        ["Mapping", "Definition"],
        os.path.join(TABLES_DIR, "tableA3_preprocessing.md"),
        title="Table A3. Preprocessing mappings (canonical source: `src/config.py`).",
    )


def write_runtime_config():
    from src.config import (
        AT_RISK_CGPA_THRESHOLD,
        CGPA_HIGH_THRESHOLD,
        CGPA_LOW_THRESHOLD,
        N_FOLDS,
        OPTUNA_INNER_FOLDS,
        OPTUNA_N_TRIALS,
        RANDOM_SEED,
        SHAP_BACKGROUND_SAMPLES,
        SHAP_NSAMPLES,
    )

    rows = [
        ["Random seed", str(RANDOM_SEED)],
        ["Outer CV folds", str(N_FOLDS)],
        ["Outer CV class", "sklearn.model_selection.StratifiedKFold(shuffle=True)"],
        ["Inner CV folds (Optuna)", str(OPTUNA_INNER_FOLDS)],
        ["Optuna trials per model", str(OPTUNA_N_TRIALS)],
        ["Optuna sampler", "TPESampler"],
        ["SMOTE k_neighbors", "5"],
        ["SMOTE random_state", "42"],
        ["SMOTE applied", "Within each training fold only (never on validation/test)"],
        ["SHAP background samples", str(SHAP_BACKGROUND_SAMPLES)],
        ["SHAP nsamples (KernelExplainer)", str(SHAP_NSAMPLES)],
        ["CGPA Low threshold", f"< {CGPA_LOW_THRESHOLD}"],
        ["CGPA High threshold", f">= {CGPA_HIGH_THRESHOLD}"],
        ["At-risk CGPA threshold", f"< {AT_RISK_CGPA_THRESHOLD}"],
        ["Python", sys.version.split()[0]],
        ["Platform", platform.platform()],
    ]

    for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "shap", "optuna", "torch", "imbalanced-learn", "matplotlib"]:
        try:
            import importlib

            mod_name = {
                "scikit-learn": "sklearn",
                "imbalanced-learn": "imblearn",
            }.get(pkg, pkg)
            module = importlib.import_module(mod_name)
            ver = getattr(module, "__version__", "unknown")
            rows.append([pkg, ver])
        except (ImportError, OSError):
            rows.append([pkg, "not installed"])

    write_markdown_table(
        rows,
        ["Setting", "Value"],
        os.path.join(TABLES_DIR, "tableA4_runtime.md"),
        title="Table A4. Runtime / CV / SMOTE configuration and software versions.",
    )


def _format_iter(it: Iterable[str]) -> str:
    return ", ".join(f"`{x}`" for x in it)


def run_replication_tables():
    write_search_spaces()
    write_best_hyperparams()
    write_preprocessing_mappings()
    write_runtime_config()


if __name__ == "__main__":
    try:
        from revision.code.common import configure_logging
    except ImportError:
        from common import configure_logging  # type: ignore

    configure_logging()
    run_replication_tables()
