"""Feature binning utilities for WOE-based scorecard modeling."""

from __future__ import annotations
import numpy as np
import pandas as pd


def bin_categorical(
    train_col: pd.Series,
    other_col: pd.Series,
    rare_threshold: int = 20,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """
    Bin a categorical feature by merging rare levels into "other".

    Rare levels are determined from train_col only. The same merge
    is applied to other_col (typically the test set) to avoid leakage.

    Parameters
    ----------
    train_col : categorical feature values from the training set
    other_col : same feature from another set (e.g. test) to apply
                the same merge to
    rare_threshold : levels with fewer than this many train observations
                      are merged into "other"

    Returns
    -------
    (train_binned, other_binned, rare_levels)
    """
    counts = train_col.value_counts()
    rare_levels = counts[counts < rare_threshold].index.tolist()
    merge_map = {lvl: "other" for lvl in rare_levels}

    train_binned = train_col.replace(merge_map)
    other_binned = other_col.replace(merge_map)

    # any level in other_col that never appeared in train is also unknown —
    # merge it into "other" too, since WOE has no value for it otherwise
    unseen = set(other_binned.unique()) - set(train_binned.unique())
    if unseen:
        other_binned = other_binned.replace({lvl: "other" for lvl in unseen})

    return train_binned, other_binned, rare_levels


def bin_continuous(
    train_col: pd.Series,
    other_col: pd.Series,
    q: int = 5,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """
    Bin a continuous feature into quantile bins, with the outer edges
    extended to +/-inf so out-of-range values at inference time still
    map to the nearest bin instead of producing NaN.

    Bins are learned from train_col only; other_col is cut using the
    same edges to avoid leakage.

    Returns
    -------
    (train_binned, other_binned, edges)
    """
    _, edges = pd.qcut(train_col, q=q, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_binned = pd.cut(train_col, bins=edges.tolist(), include_lowest=True)
    other_binned = pd.cut(other_col, bins=edges.tolist(), include_lowest=True)

    return train_binned, other_binned, edges