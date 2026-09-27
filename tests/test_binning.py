# tests/test_binning.py
import pandas as pd
import numpy as np
from scorecard.binning import bin_continuous, bin_categorical

def test_bin_continuous_handles_out_of_range_values():
    train = pd.Series([10, 20, 30, 40, 50] * 20)
    test = pd.Series([5, 100])  # below min and above max of train
    train_binned, test_binned, edges = bin_continuous(train, test, q=5)
    assert not test_binned.isna().any()  # this is the duration=72 bug, caught by a test now

def test_bin_categorical_merges_rare_levels():
    train = pd.Series(["A"] * 50 + ["B"] * 50 + ["C"] * 5)
    test = pd.Series(["A", "C", "D"])  # "D" unseen entirely
    train_binned, test_binned, rare = bin_categorical(train, test, rare_threshold=10)
    assert "C" in rare
    assert test_binned.iloc[2] == "other"  # unseen "D" got merged too