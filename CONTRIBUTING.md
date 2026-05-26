# Contributing to rag-pipelines

Thanks for your interest in contributing to the rag-pipelines!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Setup Environment

Clone the repository and create the python virtual environment:

```bash
uv sync --all-groups
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [Google format](https://google.github.io/styleguide/pyguide.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which is then checked using
[ty](https://docs.astral.sh/ty/) (Astral's Rust-based type checker).

## Development Commands

The project uses a `Makefile` for common development tasks. Run `make help` to see all available targets:

**Setup:**

- `make sync` - Create/sync development environment (installs all dependencies).

**Testing:**

- `make test` - Run unit tests (excludes integration tests).
- `make test-cov` - Run tests with coverage collection.
- `make test-ci` - Run tests with coverage + XML/junit output (used in CI).
- `make cov` - Run tests and generate coverage reports (html + xml).
- `make cov-report` - Generate coverage reports (html + xml) from existing coverage data.

**Code Quality:**

- `make lint-all` - Run all code quality checks and formatting (lint + type check + typos).
- `make lint-fmt` - Format code and apply auto-fixes (ruff).
- `make lint-check` - Check formatting and lint without modifying files.
- `make lint-style` - Lint with ruff (check only).
- `make lint-typing` - Type check only (ty).
- `make lint-typos` - Check for typos.

**Security:**

- `make security-bandit` - Run Bandit security scan.
- `make security-audit` - Run pip-audit for dependency vulnerabilities.
- `make security` - Run all security scans.

## How to Contribute

We welcome contributions that extend the functionality of this project. Here are a few ways you can contribute:

### Adding a New Dataset

1. **Create a new directory** in `src/rag_pipelines` (e.g., `src/rag_pipelines/new_dataset`).
2. **Create the necessary files**, following the structure of the existing dataset modules:
    - An indexing script (`new_dataset_indexing.py`).
    - A RAG module (`new_dataset_rag.py`).
    - Configuration files for indexing and RAG evaluation.
3. **Implement the pipeline logic** in your RAG module, composing the shared utilities from `src/rag_pipelines/utils` as needed.

### Adding a New Utility

1. **Create a new module** in the `src/rag_pipelines/utils` directory.
2. **Ensure it has a clear and well-defined responsibility**.
3. **Add docstrings and type hints**.
4. **Integrate the new module** into one or more RAG pipelines to demonstrate its usage.

### Experimenting with Different Models

To experiment with different language or embedding models, simply change the model names and API keys in the `.yml` configuration files for the relevant dataset.
