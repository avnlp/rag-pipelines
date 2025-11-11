<h1 align="center"> <a href="https://github.com/avnlp/rag-pipelines"> RAG Pipelines </a> </h1>

<div align="center">

[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/avnlp/rag-pipelines)
[![code checks](https://github.com/avnlp/rag-pipelines/actions/workflows/code_checks.yml/badge.svg)](https://github.com/avnlp/rag-pipelines/actions/workflows/code_checks.yml)
[![tests](https://github.com/avnlp/rag-pipelines/actions/workflows/tests.yml/badge.svg)](https://github.com/avnlp/rag-pipelines/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/avnlp/rag-pipelines/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/avnlp/rag-pipelines)
[![License](https://img.shields.io/github/license/avnlp/rag-pipelines?color=green)](https://github.com/avnlp/rag-pipelines/blob/main/LICENSE)

</div>

This repository contains advanced Retrieval-Augmented Generation (RAG) pipelines specifically designed for domain-specific tasks.

The RAG pipelines follow a standardized architecture:

- [**LangGraph**](https://www.langchain.com/langgraph) for workflow orchestration.
- [**Unstructured**](https://unstructured.io/) for document processing.
- [**Milvus**](https://milvus.io/) vector database for hybrid search and retrieval.
- [**DeepEval**](https://deepeval.com/) for comprehensive evaluation metrics.
- [**Confident AI**](https://www.confident-ai.com/) for tracing and debugging.

Each pipeline is configured through YAML files that allow for flexible customization of document processing, retrieval strategies, and generation parameters.

## Datasets

The project includes several domain-specific datasets:

- [**HealthBench**](https://openai.com/index/healthbench/): A comprehensive benchmark for evaluating medical AI systems with multi-turn conversations and expert rubric evaluations.
- [**MedCaseReasoning**](https://github.com/kevinwu23/Stanford-MedCaseReasoning): Dataset containing medical case studies with detailed reasoning processes.
- [**MetaMedQA**](https://github.com/maximegmd/MetaMedQA-benchmark): Medical question answering dataset based on USMLE textbook content.
- [**PubMedQA**](https://pubmedqa.github.io/): Biomedical question answering dataset based on PubMed articles.
- [**FinanceBench**](https://github.com/patronus-ai/financebench): FinanceBench is a question answering dataset that comprises of questions from public filings including 10Ks, 10Qs, 8Ks, and Earnings Calls.
- [**Earnings Calls**](https://huggingface.co/datasets/lamini/earnings-calls-qa): Financial question answering dataset based on Earnings Call Transcripts of over 2800 companies.

## Pipeline Architecture

Each pipeline follows a consistent architecture with the following nodes:

- **Indexing**: Processes raw documents from the dataset, chunks them using unstructured.io data processors, extracts metadata using LLM-powered extraction, and stores them in a Milvus vector database with BM25 hybrid search capability. The indexing process also applies metadata schemas to ensure consistent metadata across documents.

- **Metadata Extraction**: Uses an LLM-powered extractor to parse the input question and produce a structured filter dictionary based on a predefined JSON schema. This output is used to constrain document retrieval to relevant subsets (e.g., by publication year, study type, etc.) to improve retrieval precision.

- **Document Retrieval**: Retrieves relevant documents using a configured retriever (typically from Milvus vector store) based on the input question and optional metadata filter from the previous step. The retrieved documents are converted into both raw Document objects and plain text for downstream use.

- **Document Reranking**: Reranks the retrieved documents based on their relevance to the query using a specialized contextual reranker model. This step improves the relevance of documents used for answer generation by reordering them according to their contextual similarity to the query.

- **Answer Generation**: Generates an answer using an LLM conditioned on the retrieved context. This node uses a pre-defined prompt template that injects the context and question into the LLM call, producing a raw string response that forms the final answer.

- **Evaluation**: Evaluates the generated response against the ground truth using DeepEval metrics. This node constructs an LLMTestCase from the ground truth, generated answer, and retrieved context, then runs a suite of pre-configured metrics including contextual recall, precision, relevancy, answer relevancy, and faithfulness.

## Components

### Contextual Ranker

The ContextualReranker uses the reranker models by [Contextual AI](https://contextual.ai/blog/introducing-instruction-following-reranker) to reorder documents based on their relevance to a given query.

- Uses the contextual-rerank models from HuggingFace for reranking.
- Supports custom instructions to refine query context during reranking.
- Uses model logits for scoring document relevance.
- Preserves document metadata during reranking.

### Metadata Extractor

The MetadataExtractor extracts structured metadata from text using a language model and a user specified JSON schema.

- Uses LLMs with structured-output generation for metadata extraction.
- Dynamically converts JSON schema into Pydantic models for type safety and validation.
- Only includes successfully extracted (non-null) fields in results.
- Supports string, number, and boolean field types with optional enums.

### Unstructured Document Loaders and Chunker

**UnstructuredAPIDocumentLoader**: Loads and transforms documents using the Unstructured API. It supports extracting text, tables, and images from various document formats.

**UnstructuredDocumentLoader**: Loads and transforms PDF documents using the Unstructured API with various processing strategies.

**UnstructuredChunker**: Chunks documents using different strategies from the `unstructured` library, supporting "basic" and "by_title" chunking approaches.

- Support for multiple document formats (PDF, DOCX, PPTX, etc.).
- Various processing strategies (hi_res, auto, fast).
- Configurable chunking with overlap and size parameters.
- Metadata preservation during document processing.
- Recursive directory processing for batch document loading.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. First, ensure uv is installed:

```bash
# Install uv (if not already installed)
pip install uv
```

Then install the project dependencies:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Usage

### Environment Setup

Create a `.env` file in the project root with the required environment variables:

```env
GROQ_API_KEY=your_groq_api_key
MILVUS_URI=your_milvus_uri
MILVUS_TOKEN=your_milvus_token
UNSTRUCTURED_API_KEY=your_unstructured_api_key
```

### Indexing

Each dataset module includes an indexing script to process and store documents in the vector database:

Example for HealthBench:

```bash
cd src/rag_pipelines/healthbench
python healthbench_indexing.py
```

### RAG Evaluation

Each dataset module includes a RAG evaluation script to test the pipeline performance:

Example for HealthBench:

```bash
cd src/rag_pipelines/healthbench
python healthbench_rag.py
```

## Contributing

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
