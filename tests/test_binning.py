# tests/test_binning.py
import numpy as np
import pandas as pd

from scorecard.binning import bin_categorical, bin_continuous


def test_bin_continuous_handles_out_of_range_values():
    train = pd.Series([10, 20, 30, 40, 50] * 20)
    test = pd.Series([5, 100])  # below min and above max of train
    train_binned, test_binned, edges = bin_continuous(train, test, q=5)
    assert not test_binned.isna().any()
    assert not train_binned.isna().any()
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_bin_categorical_merges_rare_levels():
    train = pd.Series(["A"] * 50 + ["B"] * 50 + ["C"] * 5)
    test = pd.Series(["A", "C", "D"])  # "D" unseen entirely
    train_binned, test_binned, rare = bin_categorical(train, test, rare_threshold=10)

    assert "C" in rare
    assert test_binned.iloc[2] == "other"  # unseen "D" got merged too
    assert (
        "C" not in train_binned.unique()
    )  # rare "C" was merged in train too, not just test
    assert "other" in train_binned.unique()
