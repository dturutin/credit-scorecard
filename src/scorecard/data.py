"""Data loading utilities for the German Credit scorecard project."""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "target",
]


def load_german_credit(path: str | Path) -> pd.DataFrame:
    """
    Load the raw German Credit dataset and apply column names + target remap.

    Parameters
    ----------
    path : path to german.data

    Returns
    -------
    DataFrame with named columns and target remapped to 0=good, 1=bad.
    """
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    df["target"] = df["target"].map({1: 0, 2: 1})

    if df["target"].isna().any():
        raise ValueError("Unexpected value in target column after remapping")

    return df
