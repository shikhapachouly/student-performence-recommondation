import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba=None, objective="cgpa"):
    """Compute classification metrics and return a BenchmarkResult dict.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for AUC-ROC).
        objective: 'cgpa_category' (3-class), 'at_risk' (binary), or 'career_ready' (binary).

    Returns:
        dict with all metrics.
    """
    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # AUC-ROC (binary only, or multi-class with probabilities)
    if y_proba is not None:
        try:
            if is_binary:
                # Use probability of the positive class
                if y_proba.ndim == 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba)
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            result["auc_roc"] = float(auc)
        except (ValueError, IndexError) as e:
            logger.warning("Could not compute AUC-ROC: %s", e)
            result["auc_roc"] = None
    else:
        result["auc_roc"] = None

    return result


def aggregate_fold_results(fold_results):
    """Aggregate per-fold metrics into mean +/- std.

    Args:
        fold_results: list of BenchmarkResult dicts from compute_metrics.

    Returns:
        dict with keys like 'accuracy_mean', 'accuracy_std', etc.
    """
    if not fold_results:
        return {}

    scalar_keys = [
        "accuracy", "f1_macro", "f1_weighted", "balanced_accuracy", "auc_roc"
    ]

    aggregated = {}
    for key in scalar_keys:
        values = [r[key] for r in fold_results if r.get(key) is not None]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        else:
            aggregated[f"{key}_mean"] = None
            aggregated[f"{key}_std"] = None

    # Aggregate per-class metrics
    for key in ["precision_per_class", "recall_per_class", "f1_per_class"]:
        arrays = [np.array(r[key]) for r in fold_results if r.get(key) is not None]
        if arrays:
            stacked = np.stack(arrays)
            aggregated[f"{key}_mean"] = np.mean(stacked, axis=0).tolist()
            aggregated[f"{key}_std"] = np.std(stacked, axis=0).tolist()

    return aggregated
