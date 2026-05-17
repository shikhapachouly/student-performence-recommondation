"""Unit tests for the recommendation engine."""

import numpy as np
import pytest

from src.recommendations.engine import (
    Recommendation,
    check_shap_uniformity,
    compute_ranking_score,
    generate_recommendations,
    get_suggested_value,
)
from src.config import IMMUTABLE_FEATURES, MODIFIABLE_FEATURES


class TestModifiableFilter:
    def test_no_immutable_features_in_output(self):
        """Recommendations must never target immutable features."""
        n_features = 10
        feature_names = list(MODIFIABLE_FEATURES.keys())[:5] + IMMUTABLE_FEATURES[:5]
        student_features = np.random.rand(n_features).astype(np.float32)
        shap_values = np.random.randn(n_features).astype(np.float32)
        shap_values[:5] = -0.5  # Make modifiable features negative

        recs = generate_recommendations(
            student_features, shap_values, feature_names,
            MODIFIABLE_FEATURES, "cgpa", top_k=3,
        )

        for rec in recs:
            assert rec.feature_name not in IMMUTABLE_FEATURES, (
                f"Recommendation targets immutable feature: {rec.feature_name}"
            )


class TestRankingScore:
    def test_higher_shap_gives_higher_score(self):
        meta = {"ordinal_order": [1, 2, 3, 4, 5], "direction": "increase"}
        score_high = compute_ranking_score("test", -0.5, 2, meta)
        score_low = compute_ranking_score("test", -0.1, 2, meta)
        assert score_high > score_low

    def test_more_headroom_gives_higher_score(self):
        meta = {"ordinal_order": [1, 2, 3, 4, 5], "direction": "increase"}
        score_low_val = compute_ranking_score("test", -0.3, 1, meta)  # More headroom
        score_high_val = compute_ranking_score("test", -0.3, 4, meta)  # Less headroom
        assert score_low_val > score_high_val


class TestMinimumRecommendations:
    def test_minimum_3_recommendations(self):
        """Must generate at least 3 recommendations per student."""
        feature_names = list(MODIFIABLE_FEATURES.keys())[:8]
        n = len(feature_names)
        student_features = np.ones(n, dtype=np.float32) * 2
        shap_values = np.random.randn(n).astype(np.float32) * -0.3

        recs = generate_recommendations(
            student_features, shap_values, feature_names,
            MODIFIABLE_FEATURES, "cgpa", top_k=3,
        )

        assert len(recs) >= 3, f"Expected >= 3 recommendations, got {len(recs)}"


class TestUniformShapFallback:
    def test_uniform_detected(self):
        vals = np.array([0.1, 0.1, 0.1, 0.1])
        assert check_shap_uniformity(vals, threshold_cv=0.3)

    def test_non_uniform_not_detected(self):
        vals = np.array([0.01, 0.05, 0.3, 0.8])
        assert not check_shap_uniformity(vals, threshold_cv=0.3)

    def test_uniform_still_generates_recommendations(self):
        feature_names = list(MODIFIABLE_FEATURES.keys())[:5]
        n = len(feature_names)
        student_features = np.ones(n, dtype=np.float32) * 2
        # Nearly uniform SHAP values
        shap_values = np.ones(n, dtype=np.float32) * -0.1

        recs = generate_recommendations(
            student_features, shap_values, feature_names,
            MODIFIABLE_FEATURES, "cgpa", top_k=3,
        )

        assert len(recs) >= 3


class TestSuggestedValue:
    def test_increase_direction(self):
        order = [1, 2, 3, 4, 5]
        result = get_suggested_value(2, order, "increase")
        assert result == "3"

    def test_reduce_direction(self):
        order = [1, 2, 3, 4, 5]
        result = get_suggested_value(3, order, "reduce")
        assert result == "2"

    def test_already_at_max(self):
        order = [1, 2, 3, 4, 5]
        result = get_suggested_value(5, order, "increase")
        assert result == "5"
