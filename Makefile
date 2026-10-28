# Common entry points. GPU stages (generation, scoring, activations) run on Colab via scripts/colab_gen.sh;
# everything below runs on a laptop CPU from the result files in runs/.
PY := .venv/bin/python
# modules run from src/ so that relative runs/ paths in arguments and configs resolve as recorded
RUN := cd src && ../$(PY) -m refusal_parity

.PHONY: setup data lint format test numbers figures paper report verify

setup:            ## create the environment from pyproject.toml (exact versions)
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r pyproject.toml --extra dev

data:             ## released data -> data/, runs/ (GATED=1 adds responses, activations, arms; FROM=hf/release for a local copy)
	$(PY) scripts/fetch_data.py $(if $(GATED),--gated) $(if $(FROM),--from "$(FROM)")

lint:             ## static checks (ruff)
	.venv/bin/ruff check && .venv/bin/ruff format --check

format:           ## apply the formatter
	.venv/bin/ruff format && .venv/bin/ruff check --fix

test:             ## unit tests (CPU, under a minute)
	.venv/bin/pytest

report:           ## pre-registered comparisons -> runs/arms_report.json, runs/controls_report.json
	$(RUN).analysis.arms_report
	$(RUN).analysis.controls_report

verify:           ## rerun the analysis; fail unless every paper number is reproduced byte for byte (needs GATED=1 data)
	$(PY) scripts/verify_reproduction.py

numbers:          ## every number in the paper, generated from result files -> paper/numbers.tex
	$(RUN).reporting.paper_numbers

figures:          ## every figure in the paper -> paper/figs/ (pdf, png, provenance json)
	$(RUN).reporting.figures_results all
	$(RUN).reporting.figures_diagnostics mech ../runs/mech_global
	$(RUN).reporting.figures_diagnostics gates ../runs/mech_global
	$(RUN).reporting.figures_diagnostics cast ../runs/mech_global
	$(RUN).reporting.figures_diagnostics outcomes B0 --judge q3g4b
	$(RUN).reporting.figures_diagnostics outcomes B0 --judge gem
	$(RUN).reporting.figures_diagnostics drift B0 E0

paper: numbers figures  ## build the PDF (tectonic)
	cd paper && tectonic -X compile main.tex

