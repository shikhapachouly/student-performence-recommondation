"""Feature engineering: composite scores, interaction features, low-variance removal."""

import logging

import numpy as np
import pandas as pd

from src.config import LOW_VARIANCE_DROP, STUDY_HOURS_ORDINAL

logger = logging.getLogger(__name__)


def engineer_features(df):
    """Create engineered features from cleaned DataFrame.

    Creates:
        1. academic_index: mean of min-max normalized 10th%, 12th%
           NOTE: current_cgpa is EXCLUDED because it defines the cgpa_category
           and at_risk targets — including it would cause data leakage.
        2. skills_index: mean of communication, leadership, teamwork, time_management, peer_group scales
        3. engagement_score: mean of binary participation features
        4. study_stress_interaction: ordinal(study_hours) x stress_frequency
        5. skills_study_interaction: skills_index x ordinal(study_hours)

    Drops low-variance columns per config.LOW_VARIANCE_DROP.

    Args:
        df: Cleaned DataFrame (after clean_data, before encoding).

    Returns:
        DataFrame with engineered features added and low-variance columns removed.
    """
    df = df.copy()

    # 1. academic_index: mean of min-max normalized 10th%, 12th%
    # NOTE: current_cgpa is EXCLUDED to prevent data leakage into cgpa_category
    # and at_risk prediction tasks (both targets are derived from current_cgpa).
    # Only prior academic performance (10th, 12th) is used.
    for col in ["tenth_percentage", "twelth_percentage"]:
        if col in df.columns:
            cmin, cmax = df[col].min(), df[col].max()
            if cmax > cmin:
                df[f"_norm_{col}"] = (df[col] - cmin) / (cmax - cmin)
            else:
                df[f"_norm_{col}"] = 0.0
    norm_cols = [c for c in df.columns if c.startswith("_norm_")]
    if norm_cols:
        df["academic_index"] = df[norm_cols].mean(axis=1)
        df.drop(columns=norm_cols, inplace=True)
        logger.info("Created academic_index (mean of normalized 10th%%, 12th%% — CGPA excluded to prevent leakage)")

    # 2. skills_index: mean of 5 skill scales
    skill_cols = [
        "communication_skills_scale_1to5",
        "leadership_scale_1to5",
        "teamwork_skills_scale_1to5",
        "time_management_scale_1to5",
        "peer_group_quality_scale_1to5",
    ]
    present_skill_cols = [c for c in skill_cols if c in df.columns]
    if present_skill_cols:
        df["skills_index"] = df[present_skill_cols].mean(axis=1) / 5.0
        logger.info("Created skills_index from %d skill columns", len(present_skill_cols))

    # 3. engagement_score: mean of binary participation features
    # NOTE: internship_completion and academic_projects_completed_yn are EXCLUDED
    # because they define the career_ready target — including them would cause data leakage.
    # engagement_score captures extracurricular, workshop, and community service participation only.
    engagement_cols = [
        "extracurricular_support",
        "workshop_seminar_participation_yn",
        "community_service_yn",
    ]
    present_eng_cols = [c for c in engagement_cols if c in df.columns]
    if present_eng_cols:
        # Convert Yes/No to 1/0 for computation (handles Arrow-backed string dtypes)
        eng_df = df[present_eng_cols].copy()
        for col in present_eng_cols:
            eng_df[col] = eng_df[col].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0}).fillna(0).astype(float)
        df["engagement_score"] = eng_df.mean(axis=1)
        logger.info("Created engagement_score from %d engagement columns", len(present_eng_cols))

    # 4. study_stress_interaction: ordinal(study_hours) x stress_frequency
    if "daily_study_hours" in df.columns and "stress_frequency_scale_1to5" in df.columns:
        study_ord = df["daily_study_hours"].map(STUDY_HOURS_ORDINAL).fillna(2)
        stress = df["stress_frequency_scale_1to5"]
        df["study_stress_interaction"] = study_ord * stress
        logger.info("Created study_stress_interaction")

    # 5. skills_study_interaction: skills_index x ordinal(study_hours)
    if "skills_index" in df.columns and "daily_study_hours" in df.columns:
        study_ord = df["daily_study_hours"].map(STUDY_HOURS_ORDINAL).fillna(2)
        df["skills_study_interaction"] = df["skills_index"] * study_ord
        logger.info("Created skills_study_interaction")

    # 6. Drop low-variance columns
    dropped = []
    for col in LOW_VARIANCE_DROP:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
    if dropped:
        logger.info("Dropped low-variance columns: %s", dropped)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df
