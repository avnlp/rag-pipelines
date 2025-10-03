# RAG Pipelines

----------------------------------------------------------------------------------------

[![code checks](https://github.com/avnlp/rag-pipelines/actions/workflows/code_checks.yml/badge.svg)](https://github.com/avnlp/rag-pipelines/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/avnlp/rag-pipelines/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/avnlp/rag-pipelines/actions/workflows/integration_tests.yml)
[![docs](https://github.com/avnlp/rag-pipelines/actions/workflows/docs.yml/badge.svg)](https://github.com/avnlp/rag-pipelines/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/avnlp/rag-pipelines/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/avnlp/rag-pipelines)
![GitHub License](https://img.shields.io/github/license/avnlp/rag-pipelines)

Advanced RAG Pipelines and Evaluation: Self-Reflective RAG, Corrective RAG, Adaptive RAG, Sub-Query Generation and Routing, DeepEval.

## Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```
