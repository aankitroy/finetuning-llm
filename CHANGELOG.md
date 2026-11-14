# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- M1 · Tensors, autograd & PyTorch essentials: lesson notebook, six exercises, a two-moons training script and a checkpoint at 95% accuracy.
- Course dependencies (PyTorch, NumPy, matplotlib, JupyterLab) in a `course` dependency group.
- Course structure for 17 modules (`COURSE.md`) with a manifest per module.
- Progress dashboard (`make progress`): next-step guidance, the course route, and results charts over time. Updates live as you pass checkpoints.
- Terminal progress view (`make progress TEXT=1`).
- Strict checkpoint recording through `make check M=NN`.
- M0 · Environment & Apple Silicon setup (`make doctor`).
- Open-source project setup: licences, contributing guide, code of conduct, issue and PR templates, and CI.
