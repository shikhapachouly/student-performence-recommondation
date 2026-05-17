"""Top-level orchestrator for the IJIES revision modules.

Usage (from the project root)::

    python recent-review-comments/revision/code/run_all.py

The script puts both the project root *and* the ``recent-review-comments``
directory on ``sys.path``. That makes:

* ``import src.config`` work (so revision modules can read the canonical
  ``src/config.py`` and ``src/data/preprocessor.py``);
* ``import revision.code.<module>`` work, even though the parent directory
  ``recent-review-comments`` contains a hyphen (not a valid identifier).

Each stage is wrapped in a try/except — a missing optional dependency
(LightGBM, PyTorch, etc.) disables only the affected table.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Callable

# --- Path bootstrap --------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REVISION_ROOT = os.path.abspath(os.path.join(_HERE, ".."))                  # …/recent-review-comments/revision
_RRC_DIR = os.path.abspath(os.path.join(_REVISION_ROOT, ".."))               # …/recent-review-comments
_PROJECT_ROOT = os.path.abspath(os.path.join(_RRC_DIR, ".."))                # …/student-recommondation

for p in (_PROJECT_ROOT, _RRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from revision.code.common import OUTPUT_DIR, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _run(name: str, fn: Callable[[], object]) -> bool:
    logger.info("=" * 78)
    logger.info("[run_all] Stage: %s", name)
    logger.info("=" * 78)
    try:
        fn()
        logger.info("[run_all] Stage %s: OK", name)
        return True
    except Exception as exc:  # noqa: BLE001 — orchestrator: log and continue
        logger.error("[run_all] Stage %s FAILED: %s", name, exc)
        traceback.print_exc()
        return False


def main() -> None:
    configure_logging()
    logger.info("Output directory: %s", OUTPUT_DIR)

    results = {}

    # 1. Replication tables (config dump — fast, no model training).
    from revision.code.replication_tables import run_replication_tables
    results["replication_tables"] = _run("replication_tables", run_replication_tables)

    # 2. Feature-count audit (reads X.npy + feature_names.json).
    from revision.code.feature_count_audit import run_feature_audit
    results["feature_audit"] = _run("feature_audit", run_feature_audit)

    # 3. Leakage-free ablation (XGBoost × 2 configs × 3 objectives).
    from revision.code.leakage_ablation import run_leakage_ablation
    results["leakage_ablation"] = _run("leakage_ablation", run_leakage_ablation)

    # 4. Fairness audit (out-of-fold predictions + subgroup metrics).
    from revision.code.fairness import run_fairness_analysis
    results["fairness"] = _run("fairness", run_fairness_analysis)

    # 5. Propensity-score matching causal-plausibility check.
    from revision.code.causal_psm import run_causal_psm
    results["causal_psm"] = _run("causal_psm", run_causal_psm)

    # 6. SOTA comparison (most expensive — last, so earlier stages always finish).
    from revision.code.sota_baselines import run_sota_comparison
    results["sota_comparison"] = _run("sota_comparison", run_sota_comparison)

    # 7. Fill « fill » placeholders in manuscript_patches/*.md and produce .docx outputs.
    from revision.code.fill_patches import main as fill_main
    results["fill_patches"] = _run("fill_patches", fill_main)

    logger.info("=" * 78)
    logger.info("Summary:")
    for k, ok in results.items():
        logger.info("  %-25s %s", k, "OK" if ok else "FAILED")
    logger.info("=" * 78)
    logger.info("All artefacts under: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
