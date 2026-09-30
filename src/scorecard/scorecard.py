"""Scorecard points conversion from a fitted WOE logistic regression"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_scorecard(
    model,
    woe_tables: dict[str, pd.DataFrame],
    feature_names: list[str],
    offset: float = 600,
    pdo: float = 20,
) -> pd.DataFrame:
    """
    Convert a fitted logistic regression on WOE features into a
    points scorecard (feature -> bin -> points).

    Formula: points = -(beta_i * WOE_i,b) * Factor - (beta_0 * Factor / n) + (Offset / n)
    where Factor = PDO / ln(2), n = number of features (offset split evenly).
    """

    factor = pdo / np.log(2)
    beta_0 = model.intercept_[0]
    n = len(feature_names)

    rows = []
    for i, feature in enumerate(feature_names):
        beta_i = model.coef_[0][i]
        for bin_label, woe_val in woe_tables[feature]["woe"].items():
            points = -beta_i * woe_val * factor - (beta_0 * factor / n) + (offset / n)
            rows.append(
                {"feature": feature, "bin": str(bin_label), "points": round(points, 2)}
            )

    return pd.DataFrame(rows)


def score_applicant(aplicant_binned: dict[str, str], scorecard: pd.DataFrame) -> float:
    """Sum points across all features for one applicant's binned values."""
    total = 0.0
    for feature, bin_label in aplicant_binned.items():
        match = scorecard[
            (scorecard["feature"] == feature) & (scorecard["bin"] == str(bin_label))
        ]
        if match.empty:
            raise ValueError(f"No scorecard entry for {feature}={bin_label}")
        total += match["points"].iloc[0]
    return total
