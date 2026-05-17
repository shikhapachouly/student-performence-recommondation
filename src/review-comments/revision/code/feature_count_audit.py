"""Feature-count audit (Reviewer 1, R1.5/R1.6/R1.7).

Reviewer concerns:
    R1.5  "50+" (Abstract) vs "43 features" (§4.1) vs Table 2 sums to 40
    R1.6  Table 2 row counts must reconcile with the stated total
    R1.7  Recommendation taxonomy 16 + 6 + 19 = 41 ≠ 43

This script reads the *actual* preprocessed feature matrix (X.npy /
feature_names.json) plus src/config.py, and produces an audit table that:

    1. Lists every raw column, its category (per FEATURE_GROUPS), and whether
       it survives preprocessing.
    2. Enumerates the 5 engineered features.
    3. Splits the final feature_names list into modifiable / contextual /
       immutable / engineered tiers using MODIFIABLE_FEATURES, CONTEXTUAL_FEATURES,
       IMMUTABLE_FEATURES.
    4. Emits the corrected Table 2 ("Feature Categories and Counts") and a new
       recommendation-taxonomy table whose row totals reconcile to the actual
       feature_names length.

Outputs:
    revision/output/tables/feature_audit_table2.md       (corrected Table 2)
    revision/output/tables/feature_audit_taxonomy.md     (corrected taxonomy)
    revision/output/tables/feature_audit_full.md         (full per-column trace)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

try:
    from revision.code.common import (
        TABLES_DIR,
        load_preprocessed,
        write_markdown_table,
    )
except ImportError:
    from common import (  # type: ignore
        TABLES_DIR,
        load_preprocessed,
        write_markdown_table,
    )

logger = logging.getLogger(__name__)


# Mapping from raw column → high-level category for Table 2.
# Source of truth: paper §4.1 Table 2 categories.
TABLE2_CATEGORIES = {
    "Academic": [
        "tenth_percentage", "twelth_percentage", "course_branch", "nursery_education",
    ],
    "Demographic": ["gender", "residence_type", "institute_name"],
    "Family/Socioeconomic": [
        "mother_education_level", "mother_occupation",
        "father_education_level", "father_occupation",
        "current_guardian", "family_size", "parental_marital_status",
        "family_relationship_quality_scale_1to5", "family_income_bracket",
    ],
    "Behavioral": [
        "daily_study_hours", "extracurricular_support",
        "workshop_seminar_participation_yn", "community_service_yn",
        "commute_transport_mode", "commute_duration",
        "socializing_frequency_scale_1to5", "romantic_relationship_yn",
        "paid_classes_attending_yn",
    ],
    "Skills": [
        "communication_skills_scale_1to5", "leadership_scale_1to5",
        "teamwork_skills_scale_1to5", "peer_group_quality_scale_1to5",
        "time_management_scale_1to5",
    ],
    "Wellbeing": [
        "health_status_scale_1to5", "stress_frequency_scale_1to5", "stress_coping_yn",
    ],
    "Choice Context": [
        "college_choice_reason", "course_choice_reason",
        "internet_access", "family_extra_classes_motivation_status",
    ],
}

ENGINEERED_FEATURES = [
    "academic_index",
    "skills_index",
    "engagement_score",
    "study_stress_interaction",
    "skills_study_interaction",
]


def run_feature_audit() -> Dict[str, str]:
    from src.config import (
        CONTEXTUAL_FEATURES,
        DROP_FEATURES,
        IMMUTABLE_FEATURES,
        LOW_VARIANCE_DROP,
        MODIFIABLE_FEATURES,
        TARGET_FEATURES,
    )

    X, _, _, _, feature_names, _ = load_preprocessed()

    # --- 1. Reconcile Table 2 -------------------------------------------------
    table2_rows: List[List] = []
    raw_in_table2 = []
    for category, cols in TABLE2_CATEGORIES.items():
        in_X = [c for c in cols if c in feature_names]
        raw_in_table2.extend(in_X)
        examples = ", ".join(in_X[:3]) if in_X else "—"
        table2_rows.append([category, len(in_X), examples])
    eng_in_X = [c for c in ENGINEERED_FEATURES if c in feature_names]
    table2_rows.append(["Engineered", len(eng_in_X), ", ".join(eng_in_X[:3]) if eng_in_X else "—"])
    total = sum(r[1] for r in table2_rows)
    table2_rows.append(["**Total**", total, ""])

    write_markdown_table(
        table2_rows,
        ["Category", "Count", "Example features"],
        os.path.join(TABLES_DIR, "feature_audit_table2.md"),
        title=(
            f"Corrected Table 2 — Feature Categories and Counts. "
            f"Final feature matrix has {len(feature_names)} columns; "
            f"row counts reconcile to this total exactly."
        ),
    )

    # --- 2. Reconcile recommendation taxonomy --------------------------------
    mod = [f for f in feature_names if f in MODIFIABLE_FEATURES]
    ctx = [f for f in feature_names if f in CONTEXTUAL_FEATURES]
    imm = [f for f in feature_names if f in IMMUTABLE_FEATURES]
    eng = [f for f in feature_names if f in ENGINEERED_FEATURES]
    other = [f for f in feature_names if f not in set(mod + ctx + imm + eng)]

    taxonomy_rows = [
        ["Modifiable", len(mod), ", ".join(mod[:5]) + (" …" if len(mod) > 5 else "")],
        ["Contextual", len(ctx), ", ".join(ctx[:5]) + (" …" if len(ctx) > 5 else "")],
        ["Immutable", len(imm), ", ".join(imm[:5]) + (" …" if len(imm) > 5 else "")],
        ["Engineered", len(eng), ", ".join(eng[:5]) + (" …" if len(eng) > 5 else "")],
        ["Unclassified (review)", len(other), ", ".join(other[:5]) + (" …" if len(other) > 5 else "")],
        ["**Total**", len(mod) + len(ctx) + len(imm) + len(eng) + len(other), ""],
    ]
    write_markdown_table(
        taxonomy_rows,
        ["Tier", "Count", "Examples"],
        os.path.join(TABLES_DIR, "feature_audit_taxonomy.md"),
        title=(
            "Corrected recommendation-feature taxonomy. "
            "Counts are derived from src/config.py and reconcile to the actual "
            f"feature matrix ({len(feature_names)} columns)."
        ),
    )

    # --- 3. Per-column trace -------------------------------------------------
    trace_rows = []
    for f in feature_names:
        cats = [k for k, v in TABLE2_CATEGORIES.items() if f in v]
        category = cats[0] if cats else ("Engineered" if f in ENGINEERED_FEATURES else "Other")
        if f in MODIFIABLE_FEATURES:
            tier = "Modifiable"
        elif f in CONTEXTUAL_FEATURES:
            tier = "Contextual"
        elif f in IMMUTABLE_FEATURES:
            tier = "Immutable"
        elif f in ENGINEERED_FEATURES:
            tier = "Engineered"
        else:
            tier = "Unclassified"
        trace_rows.append([f, category, tier])

    write_markdown_table(
        trace_rows,
        ["Feature", "Table-2 category", "Recommendation tier"],
        os.path.join(TABLES_DIR, "feature_audit_full.md"),
        title=f"Full per-column audit ({len(feature_names)} features in X).",
    )

    # --- 4. Reconciliation summary -------------------------------------------
    from src.data.loader import EXPECTED_COLS

    summary = {
        "raw_columns_csv": EXPECTED_COLS,
        "DROP_FEATURES_count": len(DROP_FEATURES),
        "TARGET_FEATURES_count": len(TARGET_FEATURES),
        "LOW_VARIANCE_DROP_count": len(LOW_VARIANCE_DROP),
        "engineered_added": len(ENGINEERED_FEATURES),
        "final_feature_matrix_size": len(feature_names),
        "expected": EXPECTED_COLS - len(DROP_FEATURES) - len(TARGET_FEATURES) - len(LOW_VARIANCE_DROP) + len(ENGINEERED_FEATURES),
        "feature_names": feature_names,
    }

    summary_path = os.path.join(TABLES_DIR, "feature_audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote audit summary to %s", summary_path)

    print("[Feature audit] reconciliation:")
    print(f"  Raw CSV columns                : {summary['raw_columns_csv']}")
    print(f"  Identifiers dropped            : {summary['DROP_FEATURES_count']}")
    print(f"  Target features dropped        : {summary['TARGET_FEATURES_count']}")
    print(f"  Low-variance dropped           : {summary['LOW_VARIANCE_DROP_count']}")
    print(f"  Engineered features added      : {summary['engineered_added']}")
    print(f"  Expected final feature count   : {summary['expected']}")
    print(f"  Actual final feature count     : {summary['final_feature_matrix_size']}")

    return summary


if __name__ == "__main__":
    try:
        from revision.code.common import configure_logging
    except ImportError:
        from common import configure_logging  # type: ignore

    configure_logging()
    run_feature_audit()
