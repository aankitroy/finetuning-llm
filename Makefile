# LLM fine-tuning course — common commands. Run `make help` for a list.

FTCOURSE_ROOT ?= $(CURDIR)
MODULE_TESTS = $(wildcard modules/m$(M)_*/tests)

.PHONY: help doctor check progress test

help:
	@echo "make doctor              Check your environment (completes M0)"
	@echo "make check M=NN          Run module NN's checkpoint and record progress"
	@echo "make check M=NN LABEL=x  ...tagging this attempt (e.g. LABEL=\"r=8\")"
	@echo "make progress            Open the progress dashboard (TEXT=1 for terminal)"
	@echo "make test                Run the project's own test suite"

doctor:
	@uv run python scripts/doctor.py
	@$(MAKE) -s check M=00

check:
	@if [ -z "$(M)" ]; then \
		echo "usage: make check M=NN   (e.g. make check M=06)"; exit 2; fi
	@if [ -z "$(MODULE_TESTS)" ]; then \
		echo "No checkpoint tests found for m$(M) (looked in modules/m$(M)_*/tests)"; exit 2; fi
	@FTCOURSE_RECORD=$(M) FTCOURSE_ROOT="$(FTCOURSE_ROOT)" FTCOURSE_LABEL="$(LABEL)" \
		uv run pytest -p common.progress_plugin $(MODULE_TESTS)

progress:
	@uv run python scripts/progress_server.py $(if $(TEXT),--text,)

test:
	@uv run pytest
