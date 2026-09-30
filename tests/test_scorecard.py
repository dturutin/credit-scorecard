import numpy as np
import pandas as pd

from scorecard.scorecard import build_scorecard


class FakeModel:
    """Minimal stand-in for a fitted LogisticRegression, for isolated testing."""

    def __init__(self, intercept, coefs):
        self.intercept_ = np.array([intercept])
        self.coef_ = np.array([coefs])


def test_build_scorecard_matches_hand_calc():
    """
    Regression test for a single-feature scorecard (n=1).
    Expected values derived from the same formula, computed independently
    here rather than copied from the notebook's 15-feature run (where n=15
    changes the offset/intercept split).
    """
    import numpy as np

    woe_a14 = 1.1326506651072012
    woe_a11 = -0.7642452828882257
    beta_0 = -0.8914000679351611
    beta_checking = -0.7701314495804317
    offset = 600
    pdo = 20
    n = 1  # only one feature passed to build_scorecard in this test

    factor = pdo / np.log(2)
    expected_a14 = (
        -beta_checking * woe_a14 * factor - (beta_0 * factor / n) + (offset / n)
    )
    expected_a11 = (
        -beta_checking * woe_a11 * factor - (beta_0 * factor / n) + (offset / n)
    )

    woe_tables = {
        "checking_status": pd.DataFrame({"woe": {"A14": woe_a14, "A11": woe_a11}})
    }
    model = FakeModel(intercept=beta_0, coefs=[beta_checking])
    scorecard = build_scorecard(
        model, woe_tables, ["checking_status"], offset=offset, pdo=pdo
    )

    a14_points = scorecard[scorecard["bin"] == "A14"]["points"].iloc[0]
    a11_points = scorecard[scorecard["bin"] == "A11"]["points"].iloc[0]

    assert abs(a14_points - round(expected_a14, 2)) < 0.01
    assert abs(a11_points - round(expected_a11, 2)) < 0.01

    # directional sanity check independent of the exact numbers:
    # A14 (lowest-risk bin) must score more points than A11 (highest-risk bin)
    assert a14_points > a11_points
