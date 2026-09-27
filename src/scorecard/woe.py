"""WOE and Information Value computation."""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_woe_iv(
    binned_col: pd.Series, target: pd.Series
) -> tuple[pd.DataFrame, float]:
    """
    Compute WOE and IV for a single already-binned feature.

    Uses +0.5 Laplace smoothing to avoid log(0) when a bin has zero
    goods or zero bads.

    Returns
    -------
    (summary_df, total_iv) — summary_df has one row per bin with WOE,
    IV contribution, and raw counts; total_iv is the sum across bins.
    """
    grouped = pd.DataFrame({"bin": binned_col, "target": target})
    summary = grouped.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    summary.columns = ["total", "bad"]
    summary["good"] = summary["total"] - summary["bad"]

    total_good = summary["good"].sum()
    total_bad = summary["bad"].sum()

    summary["pct_good"] = (summary["good"] + 0.5) / (total_good + 0.5)
    summary["pct_bad"] = (summary["bad"] + 0.5) / (total_bad + 0.5)

    summary["woe"] = np.log(summary["pct_good"] / summary["pct_bad"])
    summary["iv_contribution"] = (summary["pct_good"] - summary["pct_bad"]) * summary["woe"]

    return summary, summary["iv_contribution"].sum()


def apply_woe(binned_col: pd.Series, woe_table: pd.DataFrame) -> pd.Series:
    """Map a binned column to its WOE values using a fitted woe_table."""
    return binned_col.map(woe_table["woe"].to_dict())