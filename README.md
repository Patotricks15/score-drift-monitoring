# CR05: Score drift monitoring

Monitors a credit score over time. It checks whether the score distribution shifts (PSI, KS) and
whether the score still ranks risk (Gini), month by month, on LendingClub loans.

## Setup

- Data: LendingClub accepted loans. See [data/README.md](data/README.md).
- Target: `y = 1` means good (Fully Paid). Charged Off and Default are bad. Only resolved loans are
  kept.
- Reference model: the LightGBM monotone challenger from
  [gbm-monotone-challenger](../gbm-monotone-challenger), trained on loans issued 2007 to 2014
  (n = 453,809).
- Monitoring window: loans issued 2015 to 2016 (n = 668,651). Overall test Gini is 0.430.
- Each month is scored against the training-period score distribution.

## Method

- **PSI** per month against the reference distribution. Above 0.25 triggers a retrain flag
  (`PSI_RETRAIN`).
- **KS** between the reference and monthly score distributions.
- **Gini** and **bad rate** per month, computed only for months with enough resolved loans
  (`min_n` guard).
- `retrain_flags` marks the months that trigger a retrain. `first_alert` returns the first one.

## Results

- PSI is below 0.03 in all 24 months, so the PSI trigger never fires.
- Monthly Gini falls from 0.43 to 0.48 in 2015 to 0.36 to 0.37 in late 2016.
- The bad rate rises from about 0.19 to 0.21 in 2015 to about 0.25 to 0.27 in mid and late 2016,
  while the mean score barely moves.

The score distribution stays stable, but its ranking power and the observed default rate drift.
PSI alone would not have raised an alert. Monitoring must track Gini and bad rate alongside PSI.

See [notebooks/05_score_drift.py](notebooks/05_score_drift.py) and
[reports/figures/cr05_monthly_monitoring.png](reports/figures/cr05_monthly_monitoring.png).

## Limitations

- **Resolved-only selection:** late-2016 loans that have not finished are excluded. Those are
  more likely to default, so the late-2016 Gini and bad rate may be biased. The drop may be partly
  a selection effect.
- **Censoring:** 2017 and later are excluded for the same reason.
- **In-sample reference:** the reference scores come from the challenger's training data. Out-of-
  sample scores may spread more, which would affect PSI.
- **Issue date as scoring date:** the month key is the issue date. In production, the scoring date
  is what matters.

## Reproduce

```bash
make install
make test
make notebook
```

Notebooks run with the `credit-venv` kernel. The notebook takes about 1 minute.
