"""Monthly score drift monitoring: PSI against a reference, KS and Gini per month, retrain flag."""

import numpy as np
import pandas as pd

from credit_core.metrics import gini, ks_statistic, psi

PSI_RETRAIN = 0.25


def monthly_drift(
    issue_date: pd.Series,
    y_good: pd.Series,
    score: np.ndarray,
    reference_score: np.ndarray,
    min_n: int = 1000,
) -> pd.DataFrame:
    """One row per month: volume, bad rate, PSI vs reference, KS, and Gini.

    `y_good` is 1 for good. Months with fewer than `min_n` loans, or with a single outcome class,
    get NaN for the ranking metrics, so small months do not produce noisy alerts.
    """
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(np.asarray(issue_date)).to_period("M").astype(str),
            "good": np.asarray(y_good).astype(int),
            "score": np.asarray(score, dtype=float),
        }
    )
    rows = []
    for month, sub in df.groupby("month"):
        n = len(sub)
        ranked_ok = n >= min_n and sub["good"].nunique() == 2
        rows.append(
            {
                "month": month,
                "n": n,
                "bad_rate": 1.0 - sub["good"].mean(),
                "mean_score": sub["score"].mean(),
                "psi": psi(reference_score, sub["score"].to_numpy()) if n >= min_n else np.nan,
                "ks": ks_statistic(sub["good"], sub["score"]) if ranked_ok else np.nan,
                "gini": gini(sub["good"], sub["score"]) if ranked_ok else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("month")


def retrain_flags(table: pd.DataFrame, threshold: float = PSI_RETRAIN) -> pd.DataFrame:
    """Flag months whose PSI exceeds the retrain threshold."""
    out = table.copy()
    out["psi_alert"] = out["psi"] > threshold
    return out


def first_alert(table: pd.DataFrame) -> str | None:
    alerts = table.index[table["psi_alert"]]
    return alerts[0] if len(alerts) else None
