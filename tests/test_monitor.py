import numpy as np
import pandas as pd

from score_drift.monitor import first_alert, monthly_drift, retrain_flags


def _sample(n_per_month: int, shift: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    months = ["2015-01-15", "2015-02-15", "2015-03-15"]
    dates, goods, scores = [], [], []
    for i, m in enumerate(months):
        s = rng.normal(0.0 + (shift if i == 2 else 0.0), 1.0, n_per_month)
        p = 1 / (1 + np.exp(-s))
        y = (rng.random(n_per_month) < p).astype(int)
        dates += [m] * n_per_month
        goods.append(y)
        scores.append(s)
    return pd.Series(dates), pd.Series(np.concatenate(goods)), np.concatenate(scores)


def test_stable_months_have_low_psi_and_no_alert():
    dates, y, s = _sample(3000, shift=0.0)
    ref = np.random.default_rng(1).normal(0, 1, 20000)
    table = monthly_drift(dates, y, s, ref, min_n=1000)
    assert (table["psi"] < 0.1).all()
    assert first_alert(retrain_flags(table)) is None


def test_shifted_month_triggers_retrain_flag():
    dates, y, s = _sample(3000, shift=1.5)
    ref = np.random.default_rng(1).normal(0, 1, 20000)
    table = monthly_drift(dates, y, s, ref, min_n=1000)
    flagged = retrain_flags(table)
    assert flagged.loc["2015-03", "psi_alert"]
    assert first_alert(flagged) == "2015-03"


def test_small_months_get_nan_metrics():
    dates, y, s = _sample(200, shift=0.0)
    ref = np.random.default_rng(1).normal(0, 1, 20000)
    table = monthly_drift(dates, y, s, ref, min_n=1000)
    assert table["psi"].isna().all()
    assert table["gini"].isna().all()


def test_gini_is_in_expected_range_for_informative_scores():
    dates, y, s = _sample(3000, shift=0.0)
    ref = np.random.default_rng(1).normal(0, 1, 20000)
    table = monthly_drift(dates, y, s, ref, min_n=1000)
    assert table["gini"].between(0.2, 0.9).all()
