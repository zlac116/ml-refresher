# ml-revision — regenerate the three revision notebooks end-to-end.
#
# Each notebook goes through three stages:
#   1. build    — per-folder builder script emits the raw prompt→code→details structure
#   2. transform — executes the notebook in a kernel, captures outputs, stubs the
#                  answer cell, inserts a rendered "Expected output" markdown cell
#   3. hints     — inserts a "Before you start" markdown block after each section header
#   4. validate  — asserts every exercise follows the 4-cell pattern
#
# `make all` runs every stage for every notebook. Full regeneration takes 30–60 min
# because the transform stage executes every code cell end-to-end (Optuna studies,
# SHAP, HW fits, etc).

VENV          := .venv
PY            := $(VENV)/bin/python
PYTHON        ?= python3.12

CLASSIFICATION := classification/classification.ipynb
REGRESSION     := regression/regression.ipynb
TIMESERIES     := time-series/time_series.ipynb
NOTEBOOKS      := $(CLASSIFICATION) $(REGRESSION) $(TIMESERIES)

TRANSFORM := scripts/transform_exercises.py
INJECT    := scripts/inject_hints.py
VALIDATE  := scripts/validate_exercises.py
HINTS     := scripts/hints_data.py
PIPELINE  := $(TRANSFORM) $(INJECT) $(VALIDATE) $(HINTS)

# absolute path so the builder scripts (invoked from their own folder) still find the venv
ABSPY := $(abspath $(PY))

.PHONY: help venv lab notebook all classification regression time-series validate clean realclean

help:
	@echo "ml-revision notebook pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make venv             — create .venv and install requirements (no-op if .venv exists)"
	@echo "  make lab              — launch Jupyter Lab in the venv (ctrl-c to stop)"
	@echo "  make notebook         — launch classic Jupyter Notebook in the venv"
	@echo ""
	@echo "Regeneration targets (long-running, execute notebooks end-to-end):"
	@echo "  make all              — rebuild all three notebooks"
	@echo "  make classification   — rebuild classification notebook only"
	@echo "  make regression       — rebuild regression notebook only"
	@echo "  make time-series      — rebuild time-series notebook only"
	@echo ""
	@echo "Fast targets:"
	@echo "  make validate         — check every exercise follows the 4-cell pattern"
	@echo "  make clean            — remove caches, checkpoints, and .bak files"
	@echo "  make realclean        — also remove generated artifacts/ dirs"

# Create the virtualenv only if it isn't already there.
# `$(PY)` is the sentinel file — if it exists, Make treats the target as up-to-date.
venv: $(PY)

$(PY): requirements.txt
	@if [ ! -x "$(PY)" ]; then \
		echo "Creating $(VENV) with $(PYTHON)..."; \
		$(PYTHON) -m venv $(VENV); \
		$(PY) -m pip install --upgrade pip; \
		$(PY) -m pip install -r requirements.txt; \
	else \
		echo "$(VENV) already exists — skipping creation."; \
	fi
	@touch $(PY)

all: $(NOTEBOOKS)

classification: $(CLASSIFICATION)
regression:     $(REGRESSION)
time-series:    $(TIMESERIES)

# Each notebook target depends on its builder + the shared pipeline scripts.
# Builders use hardcoded absolute out-paths so we invoke them from repo root.
$(CLASSIFICATION): classification/build_notebook.py $(PIPELINE)
	$(PY) classification/build_notebook.py
	$(PY) $(TRANSFORM) $@
	$(PY) $(INJECT) $@
	$(PY) $(VALIDATE) $@

$(REGRESSION): regression/build_nb.py $(PIPELINE)
	$(PY) regression/build_nb.py
	$(PY) $(TRANSFORM) $@
	$(PY) $(INJECT) $@
	$(PY) $(VALIDATE) $@

$(TIMESERIES): time-series/build_notebook.py $(PIPELINE)
	$(PY) time-series/build_notebook.py
	$(PY) $(TRANSFORM) $@
	$(PY) $(INJECT) $@
	$(PY) $(VALIDATE) $@

validate:
	$(PY) $(VALIDATE) $(NOTEBOOKS)

lab: $(PY)
	$(VENV)/bin/jupyter lab

notebook: $(PY)
	$(VENV)/bin/jupyter notebook

clean:
	rm -rf scripts/__pycache__
	rm -rf classification/.ipynb_checkpoints regression/.ipynb_checkpoints time-series/.ipynb_checkpoints
	rm -f classification/*.ipynb.bak regression/*.ipynb.bak time-series/*.ipynb.bak

realclean: clean
	rm -rf classification/artifacts regression/artifacts time-series/artifacts
