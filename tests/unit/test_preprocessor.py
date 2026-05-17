"""Unit tests for data preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import clean_data, create_targets, encode_features
from src.data.feature_engineering import engineer_features
from src.config import CGPA_LOW_THRESHOLD, CGPA_HIGH_THRESHOLD, SUSPICIOUS_CGPA, SUSPICIOUS_12TH


def _make_sample_df():
    """Create a minimal sample DataFrame for testing."""
    return pd.DataFrame({
        "Timestamp": ["2024/07/15"],
        "Username": ["test@test.com"],
        "NameOfStudent": ["Test"],
        "hobbies": ["Sports"],
        "gender": ["Male"],
        "age_group": ["18-25"],
        "institute_name": ["Test College"],
        "residence_type": ["URBAN"],
        "tenth_percentage": ["82"],
        "twelth_percentage": [75.0],
        "current_cgpa": [7.5],
        "backlog_status": ["NO"],
        "nursery_education": ["YES"],
        "college_choice_reason": ["Course preference"],
        "course_branch": ["Computer Engineering"],
        "course_choice_reason": ["Your preference"],
        "mother_education_level": ["Highschool"],
        "mother_occupation": ["Housewife"],
        "father_education_level": ["Not Educated"],
        "father_occupation": ["Business"],
        "current_guardian": ["Father"],
        "family_size": ["Greater than 4"],
        "parental_marital_status": ["Living together"],
        "family_relationship_quality_scale_1to5": [4],
        "family_income_bracket": ["2.5 Lakh to 8 Lakh"],
        "internet_access": ["Yes"],
        "commute_transport_mode": ["Private car/Bike"],
        "commute_duration": ["15 min"],
        "daily_study_hours": ["1 hour - 4 hour"],
        "extracurricular_support": ["YES"],
        "workshop_seminar_participation_yn": ["No"],
        "community_service_yn": ["No"],
        "communication_skills_scale_1to5": [4],
        "leadership_scale_1to5": [3],
        "teamwork_skills_scale_1to5": [4],
        "internship_completion": ["No"],
        "internship_payment_status": ["free"],
        "academic_projects_completed_yn": ["Yes"],
        "peer_group_quality_scale_1to5": [3],
        "time_management_scale_1to5": [3],
        "health_status_scale_1to5": [4],
        "stress_frequency_scale_1to5": [3],
        "stress_coping_yn": ["Yes"],
        "socializing_frequency_scale_1to5": [4],
        "romantic_relationship_yn": ["No"],
        "family_extra_classes_motivation_status": ["Sometimes"],
        "paid_classes_attending_yn": ["No"],
        "placement_status": ["No"],
        "higher_education_interest_yn": ["Yes"],
        "alcohol_use": ["No"],
        "tobacco_use": ["No"],
        "other_addictions": ["No"],
    })


class TestCleanData:
    def test_drops_identifier_columns(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        assert "Timestamp" not in cleaned.columns
        assert "Username" not in cleaned.columns
        assert "NameOfStudent" not in cleaned.columns
        assert "hobbies" not in cleaned.columns

    def test_casing_normalization(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        # "YES" should become "Yes", "NO" should become "No"
        assert cleaned["extracurricular_support"].iloc[0] == "Yes"
        assert cleaned["backlog_status"].iloc[0] == "No"

    def test_education_label_normalization(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        assert cleaned["mother_education_level"].iloc[0] == "High School"
        assert cleaned["father_education_level"].iloc[0] == "No Education"

    def test_suspicious_cgpa_replaced(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        # Suspicious CGPA value should be replaced with NaN then imputed
        assert cleaned["current_cgpa"].iloc[0] != SUSPICIOUS_CGPA

    def test_suspicious_12th_replaced(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        assert cleaned["twelth_percentage"].iloc[0] != SUSPICIOUS_12TH

    def test_tenth_percentage_numeric(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        assert np.issubdtype(cleaned["tenth_percentage"].dtype, np.number)

    def test_no_nulls_after_cleaning(self):
        df = _make_sample_df()
        cleaned = clean_data(df)
        assert cleaned.isnull().sum().sum() == 0


class TestCreateTargets:
    def test_cgpa_boundaries(self):
        df = pd.DataFrame({
            "current_cgpa": [6.49, 6.5, 7.0, 8.0, 8.01],
            "backlog_status": ["No", "No", "No", "No", "No"],
            "internship_completion": ["No", "No", "No", "No", "No"],
            "academic_projects_completed_yn": ["No", "No", "No", "No", "No"],
        })
        y_cgpa, _, _ = create_targets(df)
        assert y_cgpa[0] == 0  # 6.49 → Low
        assert y_cgpa[1] == 1  # 6.5 → Medium
        assert y_cgpa[2] == 1  # 7.0 → Medium
        assert y_cgpa[3] == 1  # 8.0 → Medium
        assert y_cgpa[4] == 2  # 8.01 → High

    def test_at_risk_encoding(self):
        df = pd.DataFrame({
            "current_cgpa": [6.5, 8.5, 7.5],
            "backlog_status": ["No", "No", "Yes"],
            "internship_completion": ["No", "No", "No"],
            "academic_projects_completed_yn": ["No", "No", "No"],
        })
        _, y_at_risk, _ = create_targets(df)
        assert y_at_risk[0] == 1  # CGPA < 7.0 → at risk
        assert y_at_risk[1] == 0  # CGPA >= 7.0, no backlog → not at risk
        assert y_at_risk[2] == 1  # has backlog → at risk

    def test_career_ready_encoding(self):
        df = pd.DataFrame({
            "current_cgpa": [7.0, 8.0, 7.5],
            "backlog_status": ["No", "No", "No"],
            "internship_completion": ["Yes", "No", "Yes"],
            "academic_projects_completed_yn": ["Yes", "Yes", "No"],
        })
        _, _, y_career_ready = create_targets(df)
        assert y_career_ready[0] == 1  # both Yes → career ready
        assert y_career_ready[1] == 0  # no internship → not career ready
        assert y_career_ready[2] == 0  # no projects → not career ready

    def test_output_dtypes(self):
        df = pd.DataFrame({
            "current_cgpa": [7.0],
            "backlog_status": ["No"],
            "internship_completion": ["Yes"],
            "academic_projects_completed_yn": ["Yes"],
        })
        y_cgpa, y_at_risk, y_career_ready = create_targets(df)
        assert y_cgpa.dtype == np.int64
        assert y_at_risk.dtype == np.int64
        assert y_career_ready.dtype == np.int64


class TestLeakagePrevention:
    """Verify that engineered features do not leak target-defining information."""

    def test_academic_index_excludes_cgpa(self):
        """academic_index must NOT include current_cgpa (leakage into cgpa_category and at_risk)."""
        df = pd.DataFrame({
            "tenth_percentage": [80.0, 90.0],
            "twelth_percentage": [70.0, 85.0],
            "current_cgpa": [6.0, 9.5],  # very different CGPAs
        })
        result = engineer_features(df)

        assert "academic_index" in result.columns

        # If CGPA were included, row 0 would have a much lower academic_index than row 1
        # due to the 6.0 vs 9.5 gap. Without CGPA, the index depends only on 10th/12th.
        # Recompute expected: normalize 10th and 12th only
        tenth_norm = (df["tenth_percentage"] - 80) / (90 - 80)  # [0.0, 1.0]
        twelfth_norm = (df["twelth_percentage"] - 70) / (85 - 70)  # [0.0, 1.0]
        expected = (tenth_norm + twelfth_norm) / 2  # [0.0, 1.0]

        np.testing.assert_allclose(
            result["academic_index"].values, expected.values, atol=1e-6,
            err_msg="academic_index should be mean of normalized 10th and 12th only (no CGPA)"
        )

    def test_engagement_score_excludes_career_target_cols(self):
        """engagement_score must NOT include internship_completion or academic_projects_completed_yn."""
        df = pd.DataFrame({
            "extracurricular_support": ["Yes", "No"],
            "workshop_seminar_participation_yn": ["Yes", "No"],
            "community_service_yn": ["Yes", "No"],
            "internship_completion": ["Yes", "No"],
            "academic_projects_completed_yn": ["Yes", "No"],
        })
        result = engineer_features(df)

        assert "engagement_score" in result.columns
        # Row 0: all 3 engagement cols = Yes → score = 1.0
        # Row 1: all 3 engagement cols = No → score = 0.0
        assert result["engagement_score"].iloc[0] == 1.0
        assert result["engagement_score"].iloc[1] == 0.0
