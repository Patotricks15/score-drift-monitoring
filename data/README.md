# Data

LendingClub accepted loans (2007 to 2018 Q4) is not committed to this repo. It is loaded by
`credit_core.datasets.load_lendingclub()` from the local copy at
`~/.local/share/credit-data/lendingclub/accepted_2007_to_2018Q4.csv.gz` (about 392 MB).

Source: the public LendingClub accepted-loans file, distributed on Kaggle. Download it with the
Kaggle CLI and place it at that path before running the notebook.

The notebook uses only loans issued 2007 to 2014 for training the challenger, and 2015 to 2016 for
monitoring. Loans from 2017 onward are excluded because their resolved outcomes are censored.
