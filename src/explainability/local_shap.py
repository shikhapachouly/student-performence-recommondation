"""Local per-student SHAP explanations."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_local_shap(model, explainer, student_record, feature_names, objective):
    """Compute per-feature SHAP values for a single student.

    Args:
        model: Trained model.
        explainer: Pre-built SHAP explainer.
        student_record: 1D or 2D array of feature values for one student.
        feature_names: List of feature names.
        objective: 'cgpa_category', 'at_risk', or 'career_ready'.

    Returns:
        shap_values: Per-feature SHAP values (1D array or list for multi-class).
    """
    if student_record.ndim == 1:
        student_record = student_record.reshape(1, -1)

    shap_values = explainer.shap_values(student_record)
    return shap_values


def generate_explanation_text(shap_values, feature_names, student_data, objective):
    """Produce a human-readable explanation string for a single student's prediction.

    Args:
        shap_values: Per-feature SHAP values for this student (1D array).
            For multi-class, pass the SHAP values for the predicted class.
        feature_names: List of feature names.
        student_data: Dict or Series of raw feature values for this student.
        objective: 'cgpa_category', 'at_risk', or 'career_ready'.

    Returns:
        explanation: Human-readable explanation string.
    """
    if isinstance(shap_values, list):
        # Multi-class: use the class with highest absolute sum
        abs_sums = [np.abs(sv).sum() for sv in shap_values]
        shap_values = shap_values[np.argmax(abs_sums)]

    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()

    # Pair features with their SHAP values
    feature_shap = list(zip(feature_names, shap_values))

    # Sort by absolute SHAP value (most impactful first)
    feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    # Build explanation
    objective_labels = {"cgpa_category": "CGPA category", "at_risk": "at-risk status", "career_ready": "career readiness"}
    objective_label = objective_labels.get(objective, objective)
    parts = [f"Key factors influencing predicted {objective_label}:\n"]

    for i, (feat, val) in enumerate(feature_shap[:5]):
        direction = "positively" if val > 0 else "negatively"
        current_val = student_data.get(feat, "N/A") if isinstance(student_data, dict) else "N/A"
        parts.append(
            f"  {i + 1}. **{feat}** (current: {current_val}) — "
            f"contributes {direction} (SHAP: {val:+.4f})"
        )

    # Identify top negative contributors for narrative
    neg_features = [(f, v) for f, v in feature_shap if v < 0][:3]
    if neg_features:
        neg_names = [f for f, _ in neg_features]
        parts.append(
            f"\nPrimary negative factors: {', '.join(neg_names)} "
            f"are reducing the predicted {objective_label}."
        )

    return "\n".join(parts)
