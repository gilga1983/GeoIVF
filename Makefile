PYTHON ?= python3
PAPER_OUTPUT ?= artifacts/paper

.PHONY: all test review paper-data figures lint format format-check
all: test review

# Reviewer checks use only Python's standard library and never run SSD work.
test:
	$(PYTHON) -m unittest discover -s tests -v

review:
	$(PYTHON) repro/run.py review --bundle repro/navhints-reviewer-evidence.zip

paper-data:
	$(PYTHON) repro/export_paper_results.py --output "$(PAPER_OUTPUT)" --overwrite

# Optional: requires pdfLaTeX, TikZ/PGFPlots and the documented TeX packages.
figures: paper-data
	cd "$(PAPER_OUTPUT)" && pdflatex -interaction=nonstopmode -halt-on-error figures.tex
	cd "$(PAPER_OUTPUT)" && pdflatex -interaction=nonstopmode -halt-on-error figures.tex

# Optional contributor tools from requirements-dev.txt; reviewer checks above
# deliberately retain their standard-library-only Python dependencies.
lint:
	$(PYTHON) -m ruff check scripts repro/*.py tests

format:
	$(PYTHON) -m black scripts repro/*.py tests

format-check:
	$(PYTHON) -m black --check scripts repro/*.py tests
