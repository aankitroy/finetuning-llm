# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Step-by-step module guide on the dashboard (and in `make progress TEXT=1`): each step's state is detected from your files and your last check, with per-exercise results, failure messages and hint commands.
- Checkpoint runs record which tests failed and why.
- Understanding checks: each module's `answers.toml` is checked against stored hashes.
- `make hint` and `make solution` for one exercise at a time, logged and shown on the dashboard.
- `make reset` / `make reset M=NN` to start over, with a typed confirmation.
- M1 · Tensors, autograd & PyTorch essentials: lesson notebook, six exercises, a two-moons training script and a checkpoint at 95% accuracy.
- Course dependencies (PyTorch, NumPy, matplotlib, JupyterLab) in a `course` dependency group.
- Course structure for 17 modules (`COURSE.md`) with a manifest per module.
- Progress dashboard (`make progress`): next-step guidance, the course route, and results charts over time. Updates live as you pass checkpoints.
- Terminal progress view (`make progress TEXT=1`).
- Strict checkpoint recording through `make check M=NN`. Runs against the reference solutions never count.
- M0 · Environment & Apple Silicon setup (`make doctor`).
- Open-source project setup: licences, contributing guide, code of conduct, issue and PR templates, and CI.
