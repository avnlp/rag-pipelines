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

- [**LangGraph**](https://www.langchain.com/langgraph) for async workflow orchestration.
- [**BAML**](https://www.boundaryml.com/) for robust, structured outputs generation with multi-provider fallback.
- [**Unstructured**](https://unstructured.io/) for document processing and chunking.
- [**Milvus**](https://milvus.io/) vector database with hybrid search (dense + BM25) and Reciprocal Rank Fusion.
- [**Contextual AI**](https://contextual.ai/) instruction-following reranker models for neural document reranking.
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

Each pipeline follows a consistent architecture split into two stages:

### Indexing Pipeline

The indexing pipeline is an offline process run once per dataset to prepare the vector store:

1. Load the dataset from Hugging Face Hub (or from PDF files for financial documents).
2. Chunk documents using configurable strategies from the Unstructured library.
3. Enrich each chunk with all three metadata layers (structural, dynamic, and fixed) via the Metadata Enricher.
4. Store the enriched chunks in a Milvus vector database with both dense and sparse (BM25) indexing for hybrid search.

### RAG Evaluation Pipeline

The RAG evaluation pipeline is orchestrated by LangGraph and consists of the following nodes:

- **Metadata Enrichment**: Parses the input question to extract structured metadata and generate a filter expression for the vector database. This narrows retrieval to relevant document subsets (e.g., by clinical specialty, company, year).

- **Document Retrieval**: Performs hybrid search combining dense (semantic) and sparse (BM25) retrieval, merged via Reciprocal Rank Fusion (RRF). Optionally applies metadata filters from the enrichment step to improve retrieval precision.

- **Document Reranking**: Reranks retrieved documents using a neural reranker model, with domain-specific instructions to prioritize the most relevant content for the query.

- **Answer Generation**: Generates a structured answer using a BAML-defined LLM function. Each domain pipeline has its own prompt template and typed output schema, ensuring the response includes a chain-of-thought explanation alongside the final answer.

- **Evaluation**: Scores the generated response against ground truth using a suite of metrics including contextual recall, contextual precision, contextual relevancy, answer relevancy, and faithfulness.

## Components

### Contextual Ranker

The Contextual Ranker uses instruction-following reranker models by [Contextual AI](https://contextual.ai/blog/introducing-instruction-following-reranker) to reorder documents based on their relevance to a given query.

- Uses instruction-following reranker models from HuggingFace for neural reranking.
- Supports per-domain custom instructions to guide the reranker (e.g., prioritizing medical articles or financial documents).
- Automatic GPU detection with optimized precision settings for efficient inference.
- Uses model logits for scoring document relevance.
- Preserves all document metadata through the reranking process.

### Metadata Enricher

The Metadata Enricher automatically enriches documents and queries with structured metadata using a three-layer architecture designed for cost/quality tradeoffs.

**Three-layer enrichment:**

1. **Structural (Layer 1)**: Rule-based extraction with zero LLM cost - content hashing, word/character counts, language detection, page numbers, section titles, and heading hierarchy.
2. **Dynamic (Layer 2)**: User-defined fields extracted via a language model. Supports string, number, boolean, and enum field types. The schema is specified per-pipeline in the YAML configuration.
3. **Fixed (Layer 3)**: RAG-optimized fields automatically generated by the LLM - potential questions the chunk answers, a concise summary, keywords, content type classification, and a descriptive header.

**Key capabilities:**

- Three enrichment modes (minimal, dynamic, full) allowing pipelines to balance LLM cost against metadata richness.
- Multi-level caching using content hashes to avoid redundant LLM calls, with concurrent request deduplication.
- Query-time metadata extraction that parses the input question and produces both a metadata dictionary and a vector database filter expression for constrained retrieval.
- Batch async processing with configurable batch sizes for parallel document enrichment.

### Unstructured Document Loaders and Chunker

The project includes document loading and chunking utilities built on the [Unstructured](https://unstructured.io/) library:

- **API Document Loader**: Loads and transforms documents using the Unstructured API. Supports extracting text, tables, and images from various document formats.
- **PDF Document Loader**: Loads and transforms PDF documents with various processing strategies.
- **Document Chunker**: Chunks documents using configurable strategies, with section-aware chunking as the default for most pipelines.

Key features:

- Support for multiple document formats (PDF, DOCX, PPTX, etc.).
- Various processing strategies (hi_res, auto, fast).
- Configurable chunking with overlap and size parameters.
- Metadata preservation during document processing.
- Recursive directory processing for batch document loading.

### BAML

[BAML](https://www.boundaryml.com/) is a domain-specific language for defining LLM interactions. All LLM logic in this project — prompts, input/output schemas, client configurations, and test cases — is written in `.baml` files, fully separated from the Python application code. A Rust-based compiler then generates a typed Python client from these definitions, bridging the two layers.

- **All LLM calls are BAML functions**: Every LLM interaction — answer generation across all six domain pipelines and metadata extraction in the enrichment system — is defined as a typed BAML function with declared inputs, output schema, and prompt template.
- **Structured output with automatic parsing**: BAML uses Schema-Aligned Parsing to transform raw LLM text into typed objects matching the declared schema. Handles malformed JSON, missing fields, and other common output issues without manual parsing code, working with any model without requiring tool-calling APIs.
- **Domain-specific prompt templates**: Each pipeline defines its own function with a tailored system prompt and output schema. Medical pipelines emphasize clinical reasoning and safety guidelines; financial pipelines focus on analytical rigor. Prompts use Jinja-like templating with role markers and automatic schema injection.
- **Multi-provider fallback chain**: Three LLM providers (Groq, Cerebras, SambaNova) are chained with automatic failover - if the primary provider fails, the system transparently retries with the next.
- **Retry policy**: Built-in exponential backoff with configurable retries for transient failures.
- **Dynamic type generation at runtime**: The metadata enricher converts user-defined JSON schemas into typed BAML definitions at runtime, so each pipeline can define its own metadata schema in a YAML config.
- **Co-located test cases**: Each BAML function includes test cases alongside its definition, enabling prompt iteration and regression testing without modifying Python test files.

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
