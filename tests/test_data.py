from scorecard.data import load_german_credit

def test_load_german_credit_shape():
    df = load_german_credit("data/raw/german_credit/german.data")
    assert df.shape == (1000, 21)

def test_target_is_binary():
    df = load_german_credit("data/raw/german_credit/german.data")
    assert set(df["target"].unique()) == {0, 1}