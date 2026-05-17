import logging
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    BINARY_COLUMNS,
    CGPA_HIGH_THRESHOLD,
    CGPA_LOW_THRESHOLD,
    DATASET_PATH,
    DROP_FEATURES,
    ORDINAL_ENCODINGS,
    ORDINAL_SCALE_COLUMNS,
    PREPROCESSED_DIR,
    RANDOM_SEED,
    SUSPICIOUS_12TH,
    SUSPICIOUS_CGPA,
    TARGET_FEATURES,
)
from src.data.loader import load_dataset

logger = logging.getLogger(__name__)


def clean_data(df):
    """Clean raw DataFrame: drop identifiers, normalize casing, handle suspicious values."""
    df = df.copy()

    # 1. Drop identifier and high-cardinality columns
    cols_to_drop = [c for c in DROP_FEATURES if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    logger.info("Dropped columns: %s", cols_to_drop)

    # 2. Normalize casing for binary columns
    binary_cols_in_df = [c for c in BINARY_COLUMNS if c in df.columns]
    for col in binary_cols_in_df:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"YES": "Yes", "NO": "No", "yes": "Yes", "no": "No"})

    # 3. Normalize education labels
    for col in ["mother_education_level", "father_education_level"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace({
                "Highschool": "High School",
                "Not Educated": "No Education",
            })

    # 4. Replace suspicious imputed values with NaN
    if "current_cgpa" in df.columns:
        suspicious_cgpa_count = (df["current_cgpa"] == SUSPICIOUS_CGPA).sum()
        df.loc[df["current_cgpa"] == SUSPICIOUS_CGPA, "current_cgpa"] = np.nan
        logger.info("Replaced %d suspicious CGPA values with NaN", suspicious_cgpa_count)

    if "twelth_percentage" in df.columns:
        suspicious_12th_count = (df["twelth_percentage"] == SUSPICIOUS_12TH).sum()
        df.loc[df["twelth_percentage"] == SUSPICIOUS_12TH, "twelth_percentage"] = np.nan
        logger.info("Replaced %d suspicious 12th%% values with NaN", suspicious_12th_count)

    # 5. Convert tenth_percentage to numeric
    if "tenth_percentage" in df.columns:
        df["tenth_percentage"] = pd.to_numeric(df["tenth_percentage"], errors="coerce")

    # 6. Impute NaN values: median for numeric, mode for categorical
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info("Imputed %s with median=%.4f", col, median_val)
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info("Imputed %s with mode=%s", col, mode_val)

    logger.info("Data cleaning complete. Shape: %s", df.shape)
    return df


def create_targets(df):
    """Create 3 data-driven target arrays from cleaned DataFrame.
    
    Objectives:
    1. cgpa_category: Low(<6.5)=0, Medium(6.5-8.0)=1, High(>8.0)=2
    2. at_risk: 1 if CGPA < 7.0 OR backlog_status = YES, else 0
    3. career_ready: 1 if internship_completion = Yes AND academic_projects_completed = Yes, else 0
    
    NOTE: Placement prediction dropped (7.1% imbalance, no predictive signal).
    """
    from src.config import AT_RISK_CGPA_THRESHOLD

    # CGPA categorization: Low(<6.5)=0, Medium(6.5-8.0)=1, High(>8.0)=2
    y_cgpa = np.where(
        df["current_cgpa"] < CGPA_LOW_THRESHOLD, 0,
        np.where(df["current_cgpa"] <= CGPA_HIGH_THRESHOLD, 1, 2)
    ).astype(np.int64)

    # At-risk: CGPA < 7.0 OR backlog_status = YES
    backlog_yes = df["backlog_status"].astype(str).str.strip().str.upper().isin(["YES", "1"])
    low_cgpa = df["current_cgpa"] < AT_RISK_CGPA_THRESHOLD
    y_at_risk = (low_cgpa | backlog_yes).astype(np.int64).values

    # Career readiness: internship_completion = Yes AND academic_projects_completed = Yes
    has_internship = df["internship_completion"].astype(str).str.strip().str.lower().isin(["yes", "1"])
    has_projects = df["academic_projects_completed_yn"].astype(str).str.strip().str.lower().isin(["yes", "1"])
    y_career_ready = (has_internship & has_projects).astype(np.int64).values

    # Log class distributions
    unique_cgpa, counts_cgpa = np.unique(y_cgpa, return_counts=True)
    logger.info("CGPA target distribution: %s", dict(zip(unique_cgpa.tolist(), counts_cgpa.tolist())))

    unique_risk, counts_risk = np.unique(y_at_risk, return_counts=True)
    logger.info("At-risk target distribution: %s", dict(zip(unique_risk.tolist(), counts_risk.tolist())))

    unique_career, counts_career = np.unique(y_career_ready, return_counts=True)
    logger.info("Career-ready target distribution: %s", dict(zip(unique_career.tolist(), counts_career.tolist())))

    return y_cgpa, y_at_risk, y_career_ready


def encode_features(df):
    """Encode features: ordinal, label, binary, scale. Returns X array, encoders dict, feature names."""
    df = df.copy()
    encoders = {}

    # Drop target columns from features
    target_cols = [c for c in TARGET_FEATURES if c in df.columns]
    df.drop(columns=target_cols, inplace=True)

    # 1. OrdinalEncoder for ordinal features with explicit category ordering
    for col, categories in ORDINAL_ENCODINGS.items():
        if col in df.columns:
            # Ensure all values are in the known categories; map unknown to most common
            known = set(categories)
            unknown_mask = ~df[col].isin(known)
            if unknown_mask.any():
                mode_val = df.loc[~unknown_mask, col].mode()[0] if (~unknown_mask).any() else categories[0]
                df.loc[unknown_mask, col] = mode_val
                logger.info("Mapped %d unknown values in %s to %s", unknown_mask.sum(), col, mode_val)

            enc = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1)
            df[col] = enc.fit_transform(df[[col]]).astype(np.float32)
            encoders[f"ordinal_{col}"] = enc

    # 2. Binary 0/1 mapping for Yes/No columns (that remain in features)
    binary_feature_cols = [c for c in BINARY_COLUMNS if c in df.columns]
    for col in binary_feature_cols:
        df[col] = df[col].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0, "SOMETIMES": 1, "1": 1, "0": 0}).fillna(0).astype(np.float32)

    # 3. Ordinal 1-5 scale features: keep as-is
    for col in ORDINAL_SCALE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # 4. LabelEncoder for remaining categorical (nominal) features
    nominal_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            nominal_cols.append(col)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[f"label_{col}"] = le

    logger.info("Label-encoded nominal columns: %s", nominal_cols)

    # 5. StandardScaler on numerical features (tenth_percentage, twelth_percentage)
    numerical_cols = ["tenth_percentage", "twelth_percentage"]
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    if numerical_cols:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols]).astype(np.float32)
        encoders["scaler"] = scaler
        logger.info("StandardScaler applied to: %s", numerical_cols)

    # Convert all to float32
    feature_names = list(df.columns)
    X = df.values.astype(np.float32)

    logger.info("Encoded features shape: %s, feature count: %d", X.shape, len(feature_names))
    return X, encoders, feature_names


def preprocess_pipeline(dataset_path=None):
    """Full preprocessing pipeline: load -> clean -> engineer -> create_targets -> encode -> save.

    Leakage control:
        - Raw target-defining columns (current_cgpa, backlog_status, internship_completion,
          academic_projects_completed_yn) are dropped from the feature set in encode_features().
        - Engineered features are constructed WITHOUT target-defining columns:
          * academic_index uses only tenth_percentage and twelth_percentage (current_cgpa excluded).
          * engagement_score uses only extracurricular_support, workshop_seminar_participation_yn,
            and community_service_yn (internship/project columns excluded).
        - All three tasks share the same leakage-free feature matrix X.
    """
    # Set all random seeds
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
    except (ImportError, OSError):
        pass

    if dataset_path is None:
        dataset_path = DATASET_PATH

    # Load
    df = load_dataset(dataset_path)

    # Clean
    df_clean = clean_data(df)

    # Feature engineering (before encoding, after cleaning)
    from src.data.feature_engineering import engineer_features
    df_clean = engineer_features(df_clean)

    # Create targets (before dropping target columns)
    y_cgpa, y_at_risk, y_career_ready = create_targets(df_clean)

    # Encode features (drops target columns internally)
    X, encoders, feature_names = encode_features(df_clean)

    # Save outputs
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PREPROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(PREPROCESSED_DIR, "y_cgpa.npy"), y_cgpa)
    np.save(os.path.join(PREPROCESSED_DIR, "y_at_risk.npy"), y_at_risk)
    np.save(os.path.join(PREPROCESSED_DIR, "y_career_ready.npy"), y_career_ready)

    with open(os.path.join(PREPROCESSED_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(PREPROCESSED_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    logger.info("Preprocessed data saved to %s", PREPROCESSED_DIR)
    logger.info("X shape: %s, y_cgpa: %s, y_at_risk: %s, y_career_ready: %s",
                X.shape, y_cgpa.shape, y_at_risk.shape, y_career_ready.shape)

    return X, y_cgpa, y_at_risk, y_career_ready, feature_names, encoders
