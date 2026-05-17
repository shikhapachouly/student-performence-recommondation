"""Fairness and subgroup analysis (addresses Reviewer 2, comment R2.5).

Reviewer concern:
    "Gender appears among top SHAP features, but the paper does not evaluate
     subgroup performance, calibration, disparate error rates, or whether
     recommendations differ unfairly across demographic or socioeconomic groups."

Reports per-subgroup, for each of the three objectives:
    - F1-macro
    - False-Negative Rate (FNR) for the at-risk class — the costliest error
      type in an educational decision-support setting
    - Brier score (calibration)
    - Equalised-odds gap = max FNR − min FNR across subgroups

Subgroups analysed (raw demographic columns from the cleaned DataFrame):
    - gender
    - residence_type
    - family_income_bracket

Outputs:
    revision/output/tables/fairness_subgroups.md
    revision/output/tables/fairness_subgroups.csv
    revision/output/figures/fairness_calibration.png
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score
from xgboost import XGBClassifier

try:
    from revision.code.common import (
        FIGURES_DIR,
        TABLES_DIR,
        cv_iter,
        load_cleaned_dataframe,
        load_preprocessed,
        write_markdown_table,
    )
except ImportError:
    from common import (  # type: ignore
        FIGURES_DIR,
        TABLES_DIR,
        cv_iter,
        load_cleaned_dataframe,
        load_preprocessed,
        write_markdown_table,
    )

logger = logging.getLogger(__name__)

SUBGROUP_COLUMNS = ["gender", "residence_type", "family_income_bracket"]


def _xgb_for(num_class: int):
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


def _oof_predictions(X: np.ndarray, y: np.ndarray, num_class: int):
    """Generate out-of-fold predictions and probabilities under the canonical CV."""
    from imblearn.over_sampling import SMOTE

    n = len(y)
    y_pred = np.full(n, -1, dtype=int)
    y_prob = np.zeros((n, num_class), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(cv_iter(y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]

        try:
            sm = SMOTE(k_neighbors=5, random_state=42)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        except ValueError:
            X_tr_res, y_tr_res = X_tr, y_tr

        clf = _xgb_for(num_class)
        clf.fit(X_tr_res, y_tr_res)

        y_pred[va_idx] = clf.predict(X_va)
        proba = clf.predict_proba(X_va)
        if proba.shape[1] < num_class:
            full = np.zeros((proba.shape[0], num_class))
            full[:, : proba.shape[1]] = proba
            proba = full
        y_prob[va_idx] = proba

    return y_pred, y_prob


def _subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    positive_class: int,
    is_binary: bool,
) -> List[Dict]:
    """Compute F1-macro, FNR (for positive_class), and Brier per subgroup.

    For multi-class, Brier is one-vs-rest averaged; FNR is reported for the
    minority/at-risk class only.
    """
    rows = []
    for name, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < 5:
            continue

        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_prob[idx]

        f1m = f1_score(y_t, y_p, average="macro", zero_division=0)

        # FNR for positive_class: of true positives, how many were missed?
        true_pos_mask = y_t == positive_class
        if true_pos_mask.sum() > 0:
            fnr = float(np.mean(y_p[true_pos_mask] != positive_class))
        else:
            fnr = float("nan")

        # Brier: binary uses prob of positive_class; multiclass uses OvR mean.
        if is_binary:
            brier = brier_score_loss((y_t == positive_class).astype(int), y_pr[:, positive_class])
        else:
            briers = []
            for c in range(y_pr.shape[1]):
                briers.append(
                    brier_score_loss((y_t == c).astype(int), y_pr[:, c])
                )
            brier = float(np.mean(briers))

        rows.append({
            "subgroup": str(name),
            "n": int(len(idx)),
            "f1_macro": round(float(f1m), 3),
            "fnr_positive": round(float(fnr), 3) if not np.isnan(fnr) else "n/a",
            "brier": round(float(brier), 3),
        })
    return rows


def run_fairness_analysis() -> str:
    """Run subgroup analysis for the three objectives, write table and calibration figure.

    Returns:
        Path to the generated markdown table.
    """
    X, y_cgpa, y_at_risk, y_career_ready, feature_names, _ = load_preprocessed()

    df_clean = load_cleaned_dataframe()
    if len(df_clean) != len(y_cgpa):
        logger.warning(
            "Cleaned DataFrame length (%d) != target length (%d). "
            "Subgroup mapping may misalign — verify preprocessing.",
            len(df_clean),
            len(y_cgpa),
        )

    objectives = [
        ("CGPA Category", y_cgpa, 3, 0, False),     # Low (=0) is the minority class
        ("At-Risk", y_at_risk, 2, 1, True),          # at-risk (=1) is the positive class
        ("Career Ready", y_career_ready, 2, 1, True),
    ]

    table_rows = []

    for name, y, n_classes, pos_class, is_binary in objectives:
        logger.info("[Fairness] %s — generating out-of-fold predictions …", name)
        y_pred, y_prob = _oof_predictions(X, y, n_classes)

        for col in SUBGROUP_COLUMNS:
            if col not in df_clean.columns:
                logger.warning("Subgroup column %s missing from cleaned DF; skipping.", col)
                continue
            groups = df_clean[col].astype(str).reset_index(drop=True)
            sub_rows = _subgroup_metrics(y, y_pred, y_prob, groups, pos_class, is_binary)

            if not sub_rows:
                continue

            fnrs = [r["fnr_positive"] for r in sub_rows if isinstance(r["fnr_positive"], float)]
            eo_gap = (max(fnrs) - min(fnrs)) if len(fnrs) >= 2 else float("nan")

            for r in sub_rows:
                table_rows.append([
                    name,
                    col,
                    r["subgroup"],
                    r["n"],
                    r["f1_macro"],
                    r["fnr_positive"],
                    r["brier"],
                    f"{eo_gap:.3f}" if not np.isnan(eo_gap) else "n/a",
                ])

    md_path = os.path.join(TABLES_DIR, "fairness_subgroups.md")
    write_markdown_table(
        table_rows,
        [
            "Objective",
            "Subgroup attribute",
            "Subgroup",
            "n",
            "F1-macro",
            "FNR (positive class)",
            "Brier score",
            "Equalised-odds gap (FNR)",
        ],
        md_path,
        title="Fairness and Subgroup Analysis (5-fold OOF predictions, XGBoost)",
    )

    # Calibration plot for at-risk objective by gender (most-cited subgroup).
    try:
        _plot_calibration(X, y_at_risk, df_clean["gender"].astype(str).reset_index(drop=True))
    except Exception as exc:  # pragma: no cover — figure is best-effort
        logger.warning("Calibration plot skipped: %s", exc)

    return md_path


def _plot_calibration(X, y, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    _, y_prob = _oof_predictions(X, y, 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    for g in sorted(groups.unique()):
        idx = (groups == g).values
        if idx.sum() < 30:
            continue
        prob_true, prob_pred = calibration_curve(y[idx], y_prob[idx, 1], n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=f"{g} (n={idx.sum()})")

    ax.set_xlabel("Mean predicted P(at-risk)", fontsize=11)
    ax.set_ylabel("Observed P(at-risk)", fontsize=11)
    ax.set_title("At-Risk Calibration by Gender", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fairness_calibration.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved calibration plot to %s", out)


if __name__ == "__main__":
    try:
        from revision.code.common import configure_logging
    except ImportError:
        from common import configure_logging  # type: ignore

    configure_logging()
    run_fairness_analysis()
