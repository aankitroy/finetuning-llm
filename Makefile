# LLM fine-tuning course — common commands. Run `make help` for a list.

FTCOURSE_ROOT ?= $(CURDIR)
MODULE_TESTS = $(wildcard modules/m$(M)_*/tests)

.PHONY: help doctor check progress test lint fmt hint solution reset answer-hashes narration

help:
	@echo "make doctor              Check your environment (completes M0)"
	@echo "make check M=NN          Run module NN's checkpoint and record progress"
	@echo "make check M=NN LABEL=x  ...tagging this attempt (e.g. LABEL=\"r=8\")"
	@echo "make progress            Open the progress dashboard (TEXT=1 for terminal)"
	@echo "make hint M=NN EX=3      A nudge for one exercise (logged on the dashboard)"
	@echo "make solution M=NN EX=3  The reference answer for one exercise (logged)"
	@echo "make reset [M=NN]        Start the course (or one module) over; asks first"
	@echo "make test                Run the project's own test suite"
	@echo "make lint / make fmt     Check / fix code style"
	@echo "make narration M=NN      (course authors) Make a module video's voice with Kokoro"

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

lint:
	@uv run ruff check . && uv run ruff format --check .

fmt:
	@uv run ruff check --fix . && uv run ruff format .

COURSE = uv run python scripts/course.py --root "$(FTCOURSE_ROOT)"

hint solution:
	@if [ -z "$(M)" ] || [ -z "$(EX)" ]; then \
		echo "usage: make $@ M=NN EX=<exercise>   (e.g. make $@ M=01 EX=3)"; exit 2; fi
	@$(COURSE) $@ $(M) $(EX)

reset:
	@$(COURSE) reset $(M)

answer-hashes:
	@if [ -z "$(M)" ]; then echo "usage: make answer-hashes M=NN"; exit 2; fi
	@$(COURSE) answer-hashes $(M)

narration:
	@if [ -z "$(M)" ]; then echo "usage: make narration M=NN [FORCE=1]"; exit 2; fi
	@uv run --group media python scripts/narrate.py $(M) $(if $(FORCE),--force,)
