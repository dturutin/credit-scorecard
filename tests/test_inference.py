"""Tests for scorecard.inference — scoring new applicants from raw input."""

import numpy as np
import pandas as pd

from scorecard.inference import score_new_applicant


class FakeModel:
    """Minimal stand-in for a fitted LogisticRegression."""

    def __init__(self, intercept, coefs):
        self.intercept_ = np.array([intercept])
        self.coef_ = np.array([coefs])

    def predict_proba(self, X):
        # simple logistic function over the linear combination,
        # just enough to be deterministic and checkable
        z = self.intercept_[0] + (X.values @ self.coef_[0])
        p_bad = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p_bad, p_bad])


def test_score_new_applicant_respects_feature_order():
    """
    Regression test for the feature-order bug: binning_artifacts'
    internal key order must NOT affect the order columns are fed to
    the model — only the explicit feature_order argument should.
    """
    # two features, WOE tables trivially simple
    woe_tables = {
        "feat_a": pd.DataFrame({"woe": {"low": 1.0, "high": -1.0}}),
        "feat_b": pd.DataFrame({"woe": {"low": 0.5, "high": -0.5}}),
    }

    # binning_artifacts keys deliberately in the OPPOSITE order to feature_order
    binning_artifacts = {
        "feat_b": {
            "type": "categorical",
            "rare_levels": [],
            "known_levels": ["low", "high"],
        },
        "feat_a": {
            "type": "categorical",
            "rare_levels": [],
            "known_levels": ["low", "high"],
        },
    }

    # model "knows" feat_a's coefficient is 2.0, feat_b's is 1.0 — order-sensitive
    model = FakeModel(intercept=0.0, coefs=[2.0, 1.0])
    feature_order = ["feat_a", "feat_b"]  # the order the model was "trained" on

    raw_applicant = {"feat_a": "low", "feat_b": "low"}

    scorecard_table = pd.DataFrame(
        [
            {"feature": "feat_a", "bin": "low", "points": 10},
            {"feature": "feat_a", "bin": "high", "points": -10},
            {"feature": "feat_b", "bin": "low", "points": 5},
            {"feature": "feat_b", "bin": "high", "points": -5},
        ]
    )

    result = score_new_applicant(
        raw_applicant,
        binning_artifacts,
        woe_tables,
        scorecard_table,
        model,
        feature_order,
    )

    # manually compute the expected PD using the CORRECT order (feat_a=1.0, feat_b=0.5)
    z = 0.0 + (2.0 * 1.0) + (1.0 * 0.5)
    expected_pd = 1 / (1 + np.exp(-z))

    assert abs(result["pd"] - expected_pd) < 1e-6
    assert result["points"] == 15  # 10 + 5


def test_score_new_applicant_handles_unseen_category():
    woe_tables = {
        "feat_a": pd.DataFrame({"woe": {"low": 1.0, "other": 0.0}}),
    }
    binning_artifacts = {
        "feat_a": {
            "type": "categorical",
            "rare_levels": [],
            "known_levels": ["low", "high"],
        },
    }
    model = FakeModel(intercept=0.0, coefs=[1.0])
    scorecard_table = pd.DataFrame(
        [
            {"feature": "feat_a", "bin": "low", "points": 10},
            {"feature": "feat_a", "bin": "other", "points": 0},
        ]
    )

    # "mystery" was never seen in training at all
    raw_applicant = {"feat_a": "mystery"}

    result = score_new_applicant(
        raw_applicant, binning_artifacts, woe_tables, scorecard_table, model, ["feat_a"]
    )

    assert result["points"] == 0  # unseen category correctly mapped to "other"
