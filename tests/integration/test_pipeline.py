"""Integration test for end-to-end pipeline reproducibility."""

import os
import json

import numpy as np
import pytest

from src.config import PREPROCESSED_DIR, RANDOM_SEED


class TestPipelineReproducibility:
    """Test that the preprocessing pipeline produces identical results with the same seed."""

    def test_preprocessing_reproducibility(self):
        """Run preprocessing twice with the same seed and verify output arrays are identical (SC-006)."""
        from src.data.preprocessor import preprocess_pipeline

        # First run
        X1, y_cgpa1, y_at_risk1, y_career_ready1, feat1, _ = preprocess_pipeline()

        # Second run
        X2, y_cgpa2, y_at_risk2, y_career_ready2, feat2, _ = preprocess_pipeline()

        np.testing.assert_array_equal(X1, X2, err_msg="Feature matrices differ between runs")
        np.testing.assert_array_equal(y_cgpa1, y_cgpa2, err_msg="CGPA targets differ between runs")
        np.testing.assert_array_equal(y_at_risk1, y_at_risk2, err_msg="At-risk targets differ between runs")
        np.testing.assert_array_equal(y_career_ready1, y_career_ready2, err_msg="Career-ready targets differ between runs")
        assert feat1 == feat2, "Feature names differ between runs"

    def test_preprocessed_files_exist(self):
        """Verify that all expected preprocessed files exist after a pipeline run."""
        from src.data.preprocessor import preprocess_pipeline
        preprocess_pipeline()

        expected_files = [
            "X.npy", "y_cgpa.npy", "y_at_risk.npy", "y_career_ready.npy",
            "feature_names.json", "encoders.pkl",
        ]
        for fname in expected_files:
            path = os.path.join(PREPROCESSED_DIR, fname)
            assert os.path.exists(path), f"Missing preprocessed file: {fname}"

    def test_preprocessed_shapes(self):
        """Verify preprocessed data shapes are reasonable."""
        from src.data.preprocessor import preprocess_pipeline
        X, y_cgpa, y_at_risk, y_career_ready, feature_names, _ = preprocess_pipeline()

        assert X.shape[0] == 1378, f"Expected 1378 rows, got {X.shape[0]}"
        assert X.shape[1] == len(feature_names), "Feature count mismatch"
        assert len(y_cgpa) == 1378
        assert len(y_at_risk) == 1378
        assert len(y_career_ready) == 1378
        assert X.shape[1] > 30, f"Too few features: {X.shape[1]}"

    def test_no_leakage_in_features(self):
        """Verify that target-defining columns and CGPA-derived engineered features are excluded."""
        from src.data.preprocessor import preprocess_pipeline
        _, _, _, _, feature_names, _ = preprocess_pipeline()

        # Raw target-defining columns must not be in feature set
        leakage_columns = [
            "current_cgpa", "backlog_status",
            "internship_completion", "academic_projects_completed_yn",
            "internship_payment_status", "placement_status",
            "higher_education_interest_yn",
        ]
        for col in leakage_columns:
            assert col not in feature_names, f"Target-defining column found in features: {col}"

        # academic_index should be present (it's leakage-free: only 10th + 12th)
        assert "academic_index" in feature_names, "academic_index missing from features"
