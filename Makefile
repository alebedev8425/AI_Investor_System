.PHONY: venv install torch-cpu torch-mps torch-cuda lint test run clean

venv:
	python3.11 -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && \
	pip install -r requirements/base.txt -r requirements/developer.txt

torch-cpu:
	. .venv/bin/activate && pip install -r requirements/torch_cpu.txt

torch-mps:
	. .venv/bin/activate && pip install -r requirements/torch_mps_macos.txt

torch-cuda:
	. .venv/bin/activate && pip install -r requirements/torch_cuda118.txt

lint:
	. .venv/bin/activate && ruff check src && black --check src && mypy src

test:
	. .venv/bin/activate && pytest -q

run:
	. .venv/bin/activate && PYTHONPATH=src python -m aiis.cli run --config experiments/exp_baseline.yaml

clean:
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
