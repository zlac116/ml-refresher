# ml-revision — minimal Makefile.
#
# The notebooks ARE the deliverable; there are no build scripts to regenerate them.
# This file exists to streamline setup and launch.

VENV   := .venv
PY     := $(VENV)/bin/python
PYTHON ?= python3.12

.PHONY: help venv lab notebook clean

help:
	@echo "Setup:"
	@echo "  make venv      — create .venv and install requirements (no-op if .venv exists)"
	@echo "  make lab       — launch Jupyter Lab"
	@echo "  make notebook  — launch classic Jupyter Notebook"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean     — remove caches, checkpoints, .bak files"

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

lab: $(PY)
	$(VENV)/bin/jupyter lab

notebook: $(PY)
	$(VENV)/bin/jupyter notebook

clean:
	find . -type d -name __pycache__ -not -path './.venv*' -not -path './.git*' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -not -path './.venv*' -not -path './.git*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.ipynb.bak' -not -path './.venv*' -not -path './.git*' -delete 2>/dev/null || true
