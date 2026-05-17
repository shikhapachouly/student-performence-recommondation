"""SHAP-based recommendation engine for student performance improvement."""

import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import (
    CONTEXTUAL_FEATURES,
    IMMUTABLE_FEATURES,
    MODIFIABLE_FEATURES,
    N_FOLDS,
    RANDOM_SEED,
    RECOMMENDATIONS_DIR,
    SHAP_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    rank: int
    feature_name: str
    display_name: str
    category: str
    current_value: str
    suggested_value: str
    shap_impact: float
    ranking_score: float
    direction: str
    narrative: str
    objective: str
    confidence: str


def compute_ranking_score(feature_name, shap_value, current_value, feature_meta):
    """Compute composite ranking score: abs(neg_SHAP) x (0.3 + 0.7 x headroom)."""
    ordinal_order = feature_meta.get("ordinal_order", [])

    # Compute headroom (how much room for improvement)
    headroom = 1.0
    if ordinal_order and current_value is not None:
        try:
            if current_value in ordinal_order:
                current_idx = ordinal_order.index(current_value)
            else:
                current_idx = int(float(current_value))
                if current_idx not in ordinal_order:
                    # Find closest
                    current_idx = min(range(len(ordinal_order)),
                                     key=lambda i: abs(ordinal_order[i] - current_idx)
                                     if isinstance(ordinal_order[i], (int, float)) else 0)
                else:
                    current_idx = ordinal_order.index(current_idx)

            max_idx = len(ordinal_order) - 1
            if feature_meta.get("direction") == "reduce":
                headroom = current_idx / max_idx if max_idx > 0 else 0
            else:
                headroom = (max_idx - current_idx) / max_idx if max_idx > 0 else 0
        except (ValueError, IndexError):
            headroom = 0.5

    score = abs(shap_value) * (0.3 + 0.7 * headroom)
    return score


def get_suggested_value(current_value, ordinal_order, direction):
    """Suggest one-two steps improvement in the appropriate direction."""
    if not ordinal_order:
        return "Improve"

    try:
        if current_value in ordinal_order:
            current_idx = ordinal_order.index(current_value)
        else:
            cv = int(float(current_value)) if current_value is not None else 0
            if cv in ordinal_order:
                current_idx = ordinal_order.index(cv)
            else:
                return str(ordinal_order[-1]) if direction != "reduce" else str(ordinal_order[0])

        if direction == "reduce":
            target_idx = max(0, current_idx - 1)
        else:
            target_idx = min(len(ordinal_order) - 1, current_idx + 1)

        return str(ordinal_order[target_idx])
    except (ValueError, IndexError):
        return "Improve"


def generate_narrative(rec):
    """Produce human-readable recommendation text."""
    if rec.direction == "activate":
        return (
            f"Consider engaging in {rec.display_name.lower()}. "
            f"This factor has a SHAP impact of {rec.shap_impact:+.4f} on your "
            f"{rec.objective} prediction, suggesting activation could improve outcomes."
        )
    elif rec.direction == "reduce":
        return (
            f"Reducing {rec.display_name.lower()} (currently {rec.current_value}) "
            f"could positively impact your {rec.objective} prediction "
            f"(SHAP impact: {rec.shap_impact:+.4f})."
        )
    else:
        return (
            f"Increasing {rec.display_name.lower()} from {rec.current_value} "
            f"to {rec.suggested_value} could improve your {rec.objective} prediction "
            f"(SHAP impact: {rec.shap_impact:+.4f})."
        )


def check_shap_uniformity(shap_values_modifiable, threshold_cv=0.3):
    """Check if SHAP values are nearly uniform (hard to differentiate)."""
    if len(shap_values_modifiable) < 2:
        return True
    abs_vals = np.abs(shap_values_modifiable)
    if abs_vals.max() == 0:
        return True
    cv = abs_vals.std() / abs_vals.mean() if abs_vals.mean() > 0 else 0
    return cv < threshold_cv


def generate_recommendations(
    student_features, shap_values, feature_names, feature_meta_dict,
    objective, top_k=3
):
    """Generate personalized recommendations for a single student.

    Args:
        student_features: 1D array of encoded feature values.
        shap_values: 1D array of SHAP values for this student's prediction.
            For multi-class, pass the SHAP for the predicted/worst class.
        feature_names: List of feature names.
        feature_meta_dict: Dict of modifiable feature metadata (from config).
        objective: 'cgpa_category', 'at_risk', or 'career_ready'.
        top_k: Number of recommendations to generate.

    Returns:
        List of Recommendation objects.
    """
    if isinstance(shap_values, list):
        # Multi-class: use the class with most negative impact
        shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[0]
    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()

    # Build feature-value-shap tuples for modifiable features
    modifiable_data = []
    for i, fname in enumerate(feature_names):
        if fname in feature_meta_dict:
            modifiable_data.append({
                "feature_name": fname,
                "meta": feature_meta_dict[fname],
                "current_value": student_features[i],
                "shap_value": shap_values[i],
            })

    if not modifiable_data:
        logger.warning("No modifiable features found for recommendations")
        return []

    # Check SHAP uniformity edge case
    shap_vals_mod = np.array([d["shap_value"] for d in modifiable_data])
    is_uniform = check_shap_uniformity(shap_vals_mod)

    if is_uniform:
        logger.info("  SHAP values nearly uniform — using global importance fallback")
        # Fallback: rank by headroom alone
        for d in modifiable_data:
            d["ranking_score"] = compute_ranking_score(
                d["feature_name"], -0.1, d["current_value"], d["meta"]
            )
    else:
        # Filter to features with negative SHAP (hurting prediction)
        neg_modifiable = [d for d in modifiable_data if d["shap_value"] < 0]
        if neg_modifiable:
            modifiable_data = neg_modifiable

        # Compute ranking scores
        for d in modifiable_data:
            d["ranking_score"] = compute_ranking_score(
                d["feature_name"], d["shap_value"], d["current_value"], d["meta"]
            )

    # Sort by ranking score descending
    modifiable_data.sort(key=lambda d: d["ranking_score"], reverse=True)

    # Generate top-k recommendations
    recommendations = []
    for rank, d in enumerate(modifiable_data[:top_k], start=1):
        meta = d["meta"]
        current_val = d["current_value"]
        suggested_val = get_suggested_value(current_val, meta.get("ordinal_order", []), meta["direction"])

        # Determine confidence
        shap_75th = np.percentile(np.abs(shap_vals_mod), 75) if len(shap_vals_mod) > 0 else 0
        confidence = "High" if abs(d["shap_value"]) > shap_75th else "Moderate"

        rec = Recommendation(
            rank=rank,
            feature_name=d["feature_name"],
            display_name=meta["display_name"],
            category=meta["category"],
            current_value=str(current_val),
            suggested_value=str(suggested_val),
            shap_impact=float(d["shap_value"]),
            ranking_score=float(d["ranking_score"]),
            direction=meta["direction"],
            narrative="",
            objective=objective,
            confidence=confidence,
        )
        rec.narrative = generate_narrative(rec)
        recommendations.append(rec)

    # If fewer than top_k, try contextual features
    if len(recommendations) < top_k:
        remaining = top_k - len(recommendations)
        for fname in CONTEXTUAL_FEATURES:
            if remaining <= 0:
                break
            if fname in feature_names:
                idx = feature_names.index(fname)
                ctx_meta = CONTEXTUAL_FEATURES[fname]
                rec = Recommendation(
                    rank=len(recommendations) + 1,
                    feature_name=fname,
                    display_name=ctx_meta["display_name"],
                    category=ctx_meta["category"],
                    current_value=str(student_features[idx]),
                    suggested_value="Improve",
                    shap_impact=float(shap_values[idx]),
                    ranking_score=0.0,
                    direction=ctx_meta["direction"],
                    narrative=f"Consider improving {ctx_meta['display_name'].lower()} as a supplementary action.",
                    objective=objective,
                    confidence="Moderate",
                )
                recommendations.append(rec)
                remaining -= 1

    return recommendations


def _format_recommendation_report(student_idx, student_features, feature_names, recs, objective):
    """Format a Markdown recommendation report for one student."""
    lines = [
        f"# Recommendation Report: Student {student_idx}",
        f"**Objective**: {objective}",
        f"**Recommendations**: {len(recs)}",
        "",
    ]

    for rec in recs:
        lines.extend([
            f"## Recommendation {rec.rank}: {rec.display_name}",
            f"- **Category**: {rec.category}",
            f"- **Current Value**: {rec.current_value}",
            f"- **Suggested Value**: {rec.suggested_value}",
            f"- **Direction**: {rec.direction}",
            f"- **SHAP Impact**: {rec.shap_impact:+.4f}",
            f"- **Ranking Score**: {rec.ranking_score:.4f}",
            f"- **Confidence**: {rec.confidence}",
            f"- **Narrative**: {rec.narrative}",
            "",
        ])

    return "\n".join(lines)


def run_recommendations(X, y_cgpa, y_at_risk, y_career_ready, feature_names, encoders):
    """Generate recommendations for sample students across objectives."""
    objectives = {"cgpa_category": y_cgpa, "at_risk": y_at_risk, "career_ready": y_career_ready}

    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)

    for obj_name, y in objectives.items():
        logger.info("Generating recommendations for %s...", obj_name)

        # Load SHAP values for TabNet (primary model)
        shap_path = os.path.join(SHAP_DIR, f"local_shap_TabNet_{obj_name}_fold0.npy")
        if not os.path.exists(shap_path):
            # Fallback to any available model
            import glob
            shap_files = glob.glob(os.path.join(SHAP_DIR, f"local_shap_*_{obj_name}_fold0.npy"))
            if shap_files:
                shap_path = shap_files[0]
            else:
                logger.warning("No SHAP values found for %s, skipping", obj_name)
                continue

        shap_values_all = np.load(shap_path, allow_pickle=True)

        # Normalize SHAP shape to (n_samples, n_features)
        # Possible shapes: (n_samples, n_features), (n_samples, n_features, n_classes),
        # (n_classes, n_samples, n_features) — average across classes
        if shap_values_all.ndim == 3:
            if shap_values_all.shape[0] <= 10:  # likely (n_classes, n_samples, n_features)
                shap_values_all = np.abs(shap_values_all).mean(axis=0)  # → (n_samples, n_features)
            else:  # likely (n_samples, n_features, n_classes)
                shap_values_all = np.abs(shap_values_all).mean(axis=2)  # → (n_samples, n_features)

        # Determine number of samples in the SHAP array
        n_shap_samples = shap_values_all.shape[0] if shap_values_all.ndim >= 2 else 1

        # SHAP values correspond to the first n_shap_samples rows of X
        # (since we used X_test[:n] in the SHAP computation which comes from fold-0 split)
        # For recommendations, we pick diverse students FROM the SHAP-available range
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = list(cv.split(X, y))
        _, test_idx = splits[0]
        # Only the first n_shap_samples of the test set were SHAP-explained
        available_test_idx = test_idx[:n_shap_samples]

        # Select one student per target class from available test indices
        sample_pairs = []  # (shap_array_idx, global_dataset_idx)
        unique_classes = np.unique(y)
        for cls in unique_classes:
            for shap_i, global_i in enumerate(available_test_idx):
                if y[global_i] == cls:
                    sample_pairs.append((shap_i, global_i))
                    break

        # Fallback: if some class not found, pick any remaining
        if not sample_pairs:
            sample_pairs = [(0, available_test_idx[0])]

        for shap_idx, student_idx in sample_pairs:
            # Get SHAP values for this student (already normalized to 2D)
            if shap_values_all.ndim == 2:
                student_shap = shap_values_all[shap_idx, :]
            else:
                student_shap = shap_values_all.flatten() if shap_values_all.ndim == 1 else shap_values_all[shap_idx]

            student_features = X[student_idx]

            # Generate recommendations
            recs = generate_recommendations(
                student_features, student_shap, feature_names,
                MODIFIABLE_FEATURES, obj_name, top_k=5,
            )

            # Validate: no immutable features
            for rec in recs:
                if rec.feature_name in IMMUTABLE_FEATURES:
                    logger.error("VIOLATION: Recommendation targets immutable feature: %s", rec.feature_name)

            # Save report
            report = _format_recommendation_report(
                student_idx, student_features, feature_names, recs, obj_name
            )
            report_path = os.path.join(
                RECOMMENDATIONS_DIR, f"student_{student_idx}_{obj_name}.md"
            )
            with open(report_path, "w") as f:
                f.write(report)

            logger.info(
                "  Student %d (%s): %d recommendations generated",
                student_idx, obj_name, len(recs),
            )

    logger.info("Recommendation generation complete.")
