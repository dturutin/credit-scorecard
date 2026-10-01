"""Serving-time scoring: take a raw applicant's feature values and produce a predicted PD and scorecard points, using artifacts fitted at training time."""

from __future__ import annotations

import pandas as pd

from scorecard.scorecard import score_applicant


def bin_new_value(value, artifact: dict):
    """
    Bin a single raw feature value using training-time artifacts.

    Categorical: known-but-rare and genuinely-unseen values both map
    to "other". Continuous: pd.cut against the saved (already
    -inf/+inf-extended) edges handles out-of-range values automatically.
    """
    if artifact["type"] == "categorical":
        if value in artifact["rare_levels"]:
            return "other"
        if value not in artifact["known_levels"]:
            return "other"  # genuinely unseen category
        return value
    else:
        return pd.cut([value], bins=artifact["edges"], include_lowest=True)[0]


def score_new_applicant(
    raw_features: dict,
    binning_artifacts: dict,
    woe_tables: dict,
    scorecard_table: pd.DataFrame,
    model,
    feature_order: list[str],
) -> dict[str, float]:
    """
    Score a new applicant from raw feature values (e.g. an API request body).

    Returns
    -------
    dict with keys "pd" (predicted probability of default) and
    "points" (total scorecard points).
    """
    woe_values = {}
    binned_values = {}

    for feature in feature_order:
        artifact = binning_artifacts[feature]
        raw_value = raw_features[feature]
        binned = bin_new_value(raw_value, artifact)
        binned_values[feature] = binned
        # 0.0 = neutral WOE fallback if a bin somehow has no lookup entry
        woe_values[feature] = woe_tables[feature]["woe"].to_dict().get(binned, 0.0)

    woe_row = pd.DataFrame(
        [[woe_values[f] for f in feature_order]], columns=feature_order
    )
    pd_pred = float(model.predict_proba(woe_row)[0, 1])
    points = score_applicant(binned_values, scorecard_table)

    return {"pd": pd_pred, "points": points}
