# RAG Pipelines

![GitHub License](https://img.shields.io/github/license/avnlp/rag-pipelines)

## Datasets

We evaluate the RAG pipelines on the following datasets:

| Dataset | Description |
| :--- | :--- |
| **HealthBench** | A comprehensive benchmark for evaluating medical AI, featuring multi-turn conversations and expert assessments. |
| **MedCaseReasoning** | A collection of medical case studies that include detailed, step-by-step reasoning processes. |
| **MetaMedQA** | A medical question-answering dataset where contexts are sourced from USMLE textbooks. |
| **PubMedQA** | A biomedical question-answering dataset derived from abstracts in PubMed articles. |


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
