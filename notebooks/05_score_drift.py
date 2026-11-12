# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CR05 - Score drift monitoring (LendingClub)
#
# Goal: decide when the score population has drifted enough to retrain. The CR02 monotone
# LightGBM challenger is fit on loans issued 2007 to 2014. Its score distribution on 2015 to 2016
# loans is compared month by month with a reference (training scores). We track PSI, KS, and Gini
# per month, and flag months with PSI above 0.25.
#
# Data: LendingClub accepted loans (Kaggle), resolved outcomes only, same split as CR02.
# Target: good = Fully Paid. Test window ends 2016-12 (2017 and 2018 outcomes are censored).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_core.metrics import gini
from gbm_monotone.pipeline import challenger_scores, fit_challenger, load_oot
from score_drift.monitor import PSI_RETRAIN, first_alert, monthly_drift, retrain_flags

FIG = Path("../reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

# %% [markdown]
# ## 1. Fit the challenger on 2007 to 2014 and score the reference

# %%
d = load_oot()
model = fit_challenger(d)
print(f"Train n = {len(d.y_train):,}, test n = {len(d.y_test):,}")

ref_idx = RNG.choice(len(d.X_train), size=min(200_000, len(d.X_train)), replace=False)
reference = challenger_scores(model, d, d.X_train.iloc[ref_idx])
score_test = challenger_scores(model, d, d.X_test)

overall_gini = gini(d.y_test, score_test)
print(f"Overall test Gini (2015 to 2016): {overall_gini:.3f}")

# %% [markdown]
# ## 2. Monthly monitoring table

# %%
table = retrain_flags(
    monthly_drift(d.issue_test, d.y_test, score_test, reference, min_n=5000),
    threshold=PSI_RETRAIN,
)
table.round(4)

# %%
print(f"First month above PSI {PSI_RETRAIN}: {first_alert(table)}")
print(f"Months flagged: {int(table['psi_alert'].sum())} of {len(table)}")

# %% [markdown]
# ## 3. Plots

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
x = table.index

axes[0].plot(x, table["psi"], marker="o", label="PSI vs reference")
axes[0].axhline(PSI_RETRAIN, color="red", ls="--", lw=0.8, label=f"retrain threshold {PSI_RETRAIN}")
axes[0].set(ylabel="PSI", title="Monthly score PSI vs training reference")
axes[0].legend()

axes[1].plot(x, table["gini"], marker="o", label="monthly Gini")
axes[1].axhline(overall_gini, color="gray", ls=":", lw=0.8, label=f"overall test Gini {overall_gini:.3f}")
axes[1].set(ylabel="Gini", title="Monthly Gini (ranking)")
axes[1].legend()

axes[2].plot(x, table["ks"], marker="o", color="tab:green", label="monthly KS")
axes[2].plot(x, table["bad_rate"], marker="s", color="tab:orange", label="bad rate")
axes[2].set(ylabel="value", title="Monthly KS and bad rate")
axes[2].tick_params(axis="x", rotation=45)
axes[2].legend()

fig.tight_layout()
fig.savefig(FIG / "cr05_monthly_monitoring.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Summary
#
# - PSI stays below 0.03 in every month (no retrain trigger). The input score distribution has
#   not moved, so PSI alone would say the model is fine.
# - Monthly Gini falls from 0.43 to 0.48 in 2015 to 0.36 to 0.37 in late 2016. The bad rate
#   rises from about 0.19 to 0.21 in 2015 to about 0.25 to 0.27 in mid and late 2016, while the mean
#   score barely changes. This is performance drift, which PSI does not detect. Gini and bad-rate
#   monitoring must be tracked alongside PSI.
# - Caveat: the late-2016 drop is partly a resolved-only effect. Loans issued in late 2016 that
#   have a final status are mostly fast resolutions (early payoffs or defaults), not a random
#   sample. The drop may be part selection bias, and it cannot be separated from real drift here.
# - Months with fewer than 5,000 loans get no PSI, and months with one outcome class get no
#   Gini or KS.

# %%
table.round(3)
