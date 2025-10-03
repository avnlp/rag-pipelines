# Contributing to rag-pipelines

Thanks for your interest in contributing to the rag-pipelines!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Setup Environment

Clone the repository and create the python virtual environment:

```bash
uv sync --all-extras --dev
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
[mypy](https://mypy.readthedocs.io/en/stable/).
