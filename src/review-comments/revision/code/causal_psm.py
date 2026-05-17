"""Propensity-Score Matching causal-plausibility analysis (Reviewer 2, R2.2).

Reviewer concern:
    "The recommendation engine treats SHAP attribution as if it can support
     intervention prioritization, but SHAP only explains model predictions and
     does not establish causal effects."

This module provides an *observational* propensity-score matching ablation for
the top three modifiable features identified by the SHAP-based recommendation
engine:
    - workshop_seminar_participation_yn
    - peer_group_quality_scale_1to5     (binarised at >= 4)
    - stress_coping_yn

Outcome: current_cgpa (continuous, taken from the cleaned DataFrame).

Method:
    1. Estimate propensity P(T=1 | X_confounders) via logistic regression on
       confounder features (everything except the treatment and outcome).
    2. 1:1 nearest-neighbour matching on the logit of the propensity, with
       caliper = 0.2 * SD(logit-propensity).
    3. Estimate ATT = mean(Y_treated - Y_matched_control).
    4. Bootstrap a 95% confidence interval (1000 resamples).

The framing in the response letter must remain that this is *consistent with*
the SHAP hypothesis directions but does not establish causation; it is a
plausibility check, not an RCT.

Outputs:
    revision/output/tables/causal_psm.md
    revision/output/tables/causal_psm.csv
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from revision.code.common import (
        TABLES_DIR,
        load_cleaned_dataframe,
        write_markdown_table,
    )
except ImportError:
    from common import (  # type: ignore
        TABLES_DIR,
        load_cleaned_dataframe,
        write_markdown_table,
    )

logger = logging.getLogger(__name__)

TREATMENTS: Dict[str, Dict] = {
    "workshop_seminar_participation_yn": {
        "display": "Workshop participation",
        "binarise": lambda s: s.astype(str).str.upper().str.strip().eq("YES").astype(int),
    },
    "peer_group_quality_scale_1to5": {
        "display": "High peer-group quality (>= 4 of 5)",
        "binarise": lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) >= 4).astype(int),
    },
    "stress_coping_yn": {
        "display": "Stress coping",
        "binarise": lambda s: s.astype(str).str.upper().str.strip().eq("YES").astype(int),
    },
}

OUTCOME_COL = "current_cgpa"


def _build_confounder_matrix(df: pd.DataFrame, treatment_col: str) -> tuple[np.ndarray, list[str]]:
    """Numeric confounder matrix, dropping treatment and outcome and any obvious leakage."""
    drop_cols = {
        treatment_col,
        OUTCOME_COL,
        "backlog_status",
        "internship_completion",
        "academic_projects_completed_yn",
        "internship_payment_status",
        "placement_status",
        "higher_education_interest_yn",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # One-hot encode categorical columns; numerics pass through.
    X = pd.get_dummies(X, drop_first=True, dummy_na=False)

    # Coerce booleans to int and fill any residual NaN with column means.
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean(numeric_only=True))
    X = X.fillna(0.0)

    feat_names = X.columns.tolist()
    return X.values.astype(float), feat_names


def _propensity_match(
    treatment: np.ndarray,
    confounders: np.ndarray,
    outcome: np.ndarray,
    caliper_sd: float = 0.2,
    seed: int = 42,
) -> dict:
    """Estimate ATT via 1:1 nearest-neighbour matching on logit-propensity."""
    scaler = StandardScaler()
    Xc = scaler.fit_transform(confounders)

    lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    lr.fit(Xc, treatment)
    p = lr.predict_proba(Xc)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))

    caliper = caliper_sd * np.std(logit_p)

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return {"att": np.nan, "n_treated": int(len(treated_idx)), "n_matched": 0}

    control_logits = logit_p[control_idx]
    used_control = np.zeros(len(control_idx), dtype=bool)

    matched_pairs = []
    for ti in treated_idx:
        diffs = np.abs(control_logits - logit_p[ti])
        diffs[used_control] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            matched_pairs.append((ti, control_idx[j]))
            used_control[j] = True

    if not matched_pairs:
        return {"att": np.nan, "n_treated": int(len(treated_idx)), "n_matched": 0}

    treated_y = np.array([outcome[pair[0]] for pair in matched_pairs])
    control_y = np.array([outcome[pair[1]] for pair in matched_pairs])
    att = float(np.mean(treated_y - control_y))

    return {
        "att": att,
        "n_treated": int(len(treated_idx)),
        "n_matched": int(len(matched_pairs)),
        "matched_pairs": matched_pairs,
        "treated_y": treated_y,
        "control_y": control_y,
    }


def _bootstrap_ci(treated_y: np.ndarray, control_y: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Paired bootstrap of the ATT 95% CI."""
    rng = np.random.default_rng(seed)
    diffs = treated_y - control_y
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        boot[b] = sample.mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_causal_psm() -> str:
    df = load_cleaned_dataframe()

    if OUTCOME_COL not in df.columns:
        raise RuntimeError(f"Outcome column {OUTCOME_COL} missing from cleaned DataFrame.")

    rows = []
    for col, meta in TREATMENTS.items():
        if col not in df.columns:
            logger.warning("Treatment column %s missing; skipping.", col)
            continue

        treat = meta["binarise"](df[col]).values
        if treat.sum() == 0 or treat.sum() == len(treat):
            logger.warning("Treatment %s has no variation; skipping.", col)
            continue

        confounders, _ = _build_confounder_matrix(df, col)
        outcome = pd.to_numeric(df[OUTCOME_COL], errors="coerce").fillna(df[OUTCOME_COL].median()).values

        result = _propensity_match(treat, confounders, outcome)

        if result["n_matched"] == 0 or np.isnan(result["att"]):
            rows.append([
                meta["display"],
                int(treat.sum()),
                0,
                "n/a",
                "n/a",
                "no acceptable matches within caliper",
            ])
            continue

        ci_lo, ci_hi = _bootstrap_ci(result["treated_y"], result["control_y"])
        sig = "yes" if (ci_lo > 0) or (ci_hi < 0) else "no"
        rows.append([
            meta["display"],
            result["n_treated"],
            result["n_matched"],
            f"{result['att']:+.3f}",
            f"[{ci_lo:+.3f}, {ci_hi:+.3f}]",
            sig,
        ])

    md_path = os.path.join(TABLES_DIR, "causal_psm.md")
    write_markdown_table(
        rows,
        [
            "Treatment",
            "n treated",
            "n matched (1:1, caliper 0.2 SD)",
            "ATT (CGPA points)",
            "95% bootstrap CI",
            "CI excludes 0",
        ],
        md_path,
        title=(
            "Propensity-Score Matching for top modifiable features.\n"
            "ATT is the Average Treatment Effect on the Treated; outcome = current_cgpa.\n"
            "Note: observational; not a substitute for randomised intervention evidence."
        ),
    )
    return md_path


if __name__ == "__main__":
    try:
        from revision.code.common import configure_logging
    except ImportError:
        from common import configure_logging  # type: ignore

    configure_logging()
    run_causal_psm()
