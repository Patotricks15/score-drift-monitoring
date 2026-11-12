.PHONY: install test notebook clean

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/05_score_drift.py -o notebooks/05_score_drift.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=credit-venv --ExecutePreprocessor.timeout=1200 notebooks/05_score_drift.ipynb

clean:
	rm -rf .pytest_cache __pycache__ src/*.egg-info
