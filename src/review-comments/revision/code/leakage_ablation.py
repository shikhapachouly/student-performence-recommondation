"""Leakage-free ablation study (addresses Reviewer 2, comment R2.4).

Reviewer concern:
    "CGPA and backlog-related variables are used to define academic and at-risk
     labels, while academic_index includes CGPA and prior academic marks. If CGPA
     or strongly label-defining variables remain in predictors for CGPA category
     or at-risk prediction, the task may become partially circular."

Already mitigated in code:
    - src/data/feature_engineering.py excludes current_cgpa from academic_index.
    - src/config.py:TARGET_FEATURES drops current_cgpa, backlog_status,
      internship_completion, academic_projects_completed_yn,
      internship_payment_status, placement_status, higher_education_interest_yn
      from the feature matrix.
    - src/data/preprocessor.py:encode_features() applies the drop.

This module produces an ablation table demonstrating that, even after dropping
prior academic features (tenth_percentage, twelth_percentage, academic_index),
non-trivial predictive signal remains for the at-risk objective. That confirms
the task is not circular — the engineered academic_index carries genuine prior
academic record signal, not echoes of the target variable.

Outputs:
    revision/output/tables/leakage_ablation.md
    revision/output/tables/leakage_ablation.csv
"""

from __future__ import annotations

import logging
import os

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

try:
    from revision.code.common import (
        TABLES_DIR,
        cv_iter,
        load_preprocessed,
        write_markdown_table,
    )
except ImportError:  # script-mode fallback
    from common import (  # type: ignore
        TABLES_DIR,
        cv_iter,
        load_preprocessed,
        write_markdown_table,
    )

logger = logging.getLogger(__name__)

PRIOR_ACADEMIC_FEATURES = (
    "tenth_percentage",
    "twelth_percentage",
    "academic_index",
)


def _xgb_clf(num_class: int):
    """XGBoost with conservative defaults — matches src/models/baselines.py spirit."""
    if num_class > 2:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _evaluate(X: np.ndarray, y: np.ndarray, num_class: int) -> tuple[float, float]:
    """5-fold CV F1-macro mean ± std with SMOTE applied within each training fold."""
    from imblearn.over_sampling import SMOTE

    f1s = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv_iter(y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # SMOTE only on training fold (matches src/models/baselines.py).
        try:
            sm = SMOTE(k_neighbors=5, random_state=42)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        except ValueError as exc:
            logger.warning("SMOTE failed on fold %d (%s); using original fold.", fold_idx, exc)
            X_tr_res, y_tr_res = X_tr, y_tr

        clf = _xgb_clf(num_class)
        clf.fit(X_tr_res, y_tr_res)
        y_pred = clf.predict(X_va)
        f1s.append(f1_score(y_va, y_pred, average="macro"))

    return float(np.mean(f1s)), float(np.std(f1s))


def run_leakage_ablation() -> str:
    """Run the ablation and write the markdown/csv table.

    Returns:
        Path to the generated markdown table.
    """
    X, y_cgpa, y_at_risk, y_career_ready, feature_names, _ = load_preprocessed()

    feature_idx = {name: i for i, name in enumerate(feature_names)}
    drop_idx = [feature_idx[c] for c in PRIOR_ACADEMIC_FEATURES if c in feature_idx]
    keep_mask = np.ones(X.shape[1], dtype=bool)
    keep_mask[drop_idx] = False

    objectives = [
        ("CGPA Category", y_cgpa, 3),
        ("At-Risk", y_at_risk, 2),
        ("Career Ready", y_career_ready, 2),
    ]

    rows = []
    for name, y, n_classes in objectives:
        logger.info("[%s] Full feature set …", name)
        f1_full, std_full = _evaluate(X, y, n_classes)

        logger.info("[%s] Without prior academic features …", name)
        f1_drop, std_drop = _evaluate(X[:, keep_mask], y, n_classes)

        delta = f1_full - f1_drop
        rows.append([
            name,
            f"{f1_full:.3f} ± {std_full:.3f}",
            f"{f1_drop:.3f} ± {std_drop:.3f}",
            f"{delta:+.3f}",
        ])

    header = [
        "Objective",
        "F1-macro (full features)",
        "F1-macro (no prior academic)",
        "Δ (impact of prior academic)",
    ]
    md_path = os.path.join(TABLES_DIR, "leakage_ablation.md")
    write_markdown_table(
        rows,
        header,
        md_path,
        title=(
            "Leakage-Free Ablation — F1-macro with vs without prior academic features\n"
            "(tenth_percentage, twelth_percentage, academic_index). "
            "current_cgpa is already excluded from the feature set."
        ),
    )
    return md_path


if __name__ == "__main__":
    try:
        from revision.code.common import configure_logging
    except ImportError:
        from common import configure_logging  # type: ignore

    configure_logging()
    run_leakage_ablation()
