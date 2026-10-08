PYTHON ?= python3
CONFIG ?= configs/tiny_cpu.yaml
SMOKE_CONFIG ?= configs/smoke_mock.yaml
OUTPUT ?= reports

.PHONY: install test lint dataset activations directions validate behavior report all smoke

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

dataset:
	pvsteer make-dataset --config $(CONFIG) --output-dir $(OUTPUT)

activations:
	pvsteer extract-activations --config $(CONFIG) --output-dir $(OUTPUT)

directions:
	pvsteer fit-directions --config $(CONFIG) --output-dir $(OUTPUT)

validate:
	pvsteer validate --config $(CONFIG) --output-dir $(OUTPUT)

behavior:
	pvsteer behavioral-benchmark --config $(CONFIG) --output-dir $(OUTPUT)

report:
	pvsteer report --config $(CONFIG) --output-dir $(OUTPUT)

all:
	pvsteer run-all --config $(CONFIG) --output-dir $(OUTPUT)

smoke:
	pvsteer run-all --config $(SMOKE_CONFIG) --output-dir artifacts/smoke
