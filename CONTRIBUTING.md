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
3. Write the lesson: `README.md`, then `lesson.ipynb` and `exercises.ipynb`.
4. Write checkpoint tests in `tests/test_checkpoint.py`.

### Rules for checkpoints

- **Checkpoints verify. They don't retrain.** Training scripts write `outputs/mNN/run_summary.json`, and the checkpoint reads it and asserts on it.
- Report numbers with `common.progress.report_metric("key", value)` so they show up on the dashboard.
- Modules that train a model must report the shared metrics `peak_mem_gb`, `tokens_per_sec` and `trainable_pct`. The dashboard compares techniques using these.
- If a test depends on something optional (such as a package installed in a later module), mark it `@pytest.mark.optional_checkpoint` and skip it with a clear message. Any other skip fails the checkpoint.
- Every lesson's `--tiny` setting must finish in under 2 minutes on an M-series Mac.
- Failure messages should tell the learner what to do next.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- Fill in the pull request checklist.

## Licensing of contributions

By contributing, you agree that your code is licensed under [MIT](LICENSE) and your course content under [CC BY 4.0](LICENSE-CONTENT).
