# Contributing

Thanks for helping make this course better. This guide covers setting up, the checks your change must pass, and how to add a module.

## Set up

```bash
uv sync                      # install dependencies
uv run pre-commit install    # lint and format on every commit
```

## Checks

```bash
make test    # the tooling's own test suite
make lint    # ruff lint + format check (make fmt fixes most issues)
```

CI runs both on macOS (Apple Silicon) and Linux with Python 3.11 and 3.12.

`make test` never writes to your progress history. Only `make check M=NN` does.

## Adding a module

1. Copy the template: `cp -r modules/_template modules/mNN_your_slug`.
2. Fill in `module.toml`. Every field is required except `[artifacts]` and `[[metrics]]`:
   - `id`, `title`, `part`, `order`, `summary`, `est_minutes`
   - `[next]`: `start` (how to begin), `finish` (`make check M=NN`), and `why` (what the checkpoint proves)
   - `[artifacts] any`: globs under `outputs/mNN/` that show the learner has started
   - `[[metrics]]`: each metric your checkpoint reports, with `key`, `label` and `better` (`"lower"` or `"higher"`)
3. Write the lesson: `README.md` and `lesson.ipynb`. Execute the notebook end to end before committing.
4. Write the learner's files and keep an identical copy in `starter/` (that's what `make reset` restores):
   - `exercises.py`: functions with docstrings and `raise NotImplementedError("Exercise N: ...")`
   - `answers.toml`: 3–6 understanding questions with `answer = ""`
5. Write the reference answers:
   - `solutions/exercises.py`
   - `solutions/answers.toml`, with a `kind` (`shape`, `number`, `choice` or `text`), the `answer`, and a `why`
   - then run `make answer-hashes M=NN` to write `tests/answer_hashes.json`
6. Write `hints.toml`: one `[[exercise]]` per exercise and question, with `id`, `hint`, the checkpoint `test` that checks it (e.g. `test_mse_loss` or `test_understanding[q2]`), and for code exercises the `name` of the function or class.
7. Add `[[steps]]` to `module.toml` for the dashboard's step-by-step guide (see the template). Kinds are `lesson`, `exercises`, `answers`, `train` and `checkpoint`; each one's state is detected from files, so keep `file` and `metric` accurate.
8. Write checkpoint tests in `tests/`. Load the learner's code with `common.checkpoint.load_exercise` and check answers with `common.checkpoint.check_answer`.
9. Add the module to `MODULES` in `tests/test_module_solutions.py`, so CI proves the checkpoint passes with the reference solutions.

### Rules for checkpoints

- **Real learning first.** A module must require the learner to write code *and* answer questions about why it works. Never ship a checkpoint that passes on code alone.
- **Checkpoints verify. They don't retrain.** Training scripts write `outputs/mNN/run_summary.json` (honouring `FTCOURSE_OUTPUTS`), including `used_solutions`, and the checkpoint reads it.
- Report numbers with `common.progress.report_metric("key", value)` so they show up on the dashboard.
- Modules that train a model must report the shared metrics `peak_mem_gb`, `tokens_per_sec` and `trainable_pct`.
- If a test depends on something optional, mark it `@pytest.mark.optional_checkpoint` and skip it with a clear message. Any other skip fails the checkpoint.
- Every lesson's `--tiny` setting must finish in under 2 minutes on an M-series Mac.
- Failure messages should tell the learner what to do next, without giving the answer away.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- Fill in the pull request checklist.

## Licensing of contributions

By contributing, you agree that your code is licensed under [MIT](LICENSE) and your course content under [CC BY 4.0](LICENSE-CONTENT).
