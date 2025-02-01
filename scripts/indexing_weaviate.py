import argparse
import logging

import weave
from dataloaders.langchain import FinanceBenchDataloader
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipelines.unstructured.unstructured_chunker import UnstructuredChunker
from rag_pipelines.unstructured.unstructured_pdf_loader import UnstructuredDocumentLoader
from rag_pipelines.utils.logging import LoggerFactory
from rag_pipelines.vectordb.weaviate import (
    WeaviateVectorDB,
)  # Assumes the WeaviateVectorDB class is defined as shown above

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the FinanceBench pipeline to load, process, chunk, embed, and index documents in Weaviate."
    )

    # FinanceBench dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PatronusAI/financebench",
        help="Name of the FinanceBench dataset to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:1]",
        help="Dataset split to use (e.g., 'train[:1]').",
    )

    # PDF directory for unstructured document loader
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="pdfs/",
        help="Directory path containing PDF files.",
    )

    # UnstructuredDocumentLoader parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="fast",
        help="Processing strategy for the unstructured document loader.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="elements",
        help="Extraction mode for the unstructured document loader.",
    )

    # Weaviate connection parameters
    parser.add_argument(
        "--cluster_url",
        type=str,
        required=True,
        help="URL of the Weaviate cluster.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for Weaviate authentication.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="financebench",
        help="Name of the Weaviate collection to create/use.",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field name that contains document text in Weaviate.",
    )

    # Dense embedding model parameters
    parser.add_argument(
        "--dense_model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model name.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the FinanceBench document processing pipeline using Weaviate.

    The pipeline performs the following steps:
      1. Initializes Weave tracing.
      2. Loads a subset of the FinanceBench dataset.
      3. Retrieves PDF documents from the specified directory.
      4. Processes PDFs using the UnstructuredDocumentLoader.
      5. Chunks documents using the UnstructuredChunker.
      6. Generates dense embeddings.
      7. Sets up a Weaviate vector database and indexes the documents.
    """
    args = parse_arguments()

    # Initialize Weave tracing
    weave.init("financebench_test")

    # Load FinanceBench dataset and retrieve corpus PDFs
    dataloader = FinanceBenchDataloader(
        dataset_name=args.dataset_name,
        split=args.split,
    )
    dataloader.get_corpus_pdfs()

    # Load and transform PDF documents from the specified directory
    unstructured_document_loader = UnstructuredDocumentLoader(
        strategy=args.strategy,
        mode=args.mode,
    )
    documents = unstructured_document_loader.transform_documents(args.pdf_dir)
    logger.info("Loaded Documents:")
    logger.info(documents)

    # Chunk the documents using the UnstructuredChunker
    chunker = UnstructuredChunker()
    chunked_documents = chunker.transform_documents(documents)
    logger.info("Chunked Documents:")
    logger.info(chunked_documents)

    # Initialize the dense embedding model
    embeddings = HuggingFaceEmbeddings(model_name=args.dense_model_name)

    # Initialize the Weaviate vector database client
    weaviate_vector_db = WeaviateVectorDB(
        cluster_url=args.cluster_url,
        api_key=args.api_key,
        collection_name=args.collection_name,
        text_field=args.text_field,
        dense_embedding_model=embeddings,
    )

    # Index the chunked documents in Weaviate using the dense embeddings
    weaviate_vector_db.add_documents(documents=chunked_documents)
    logger.info("Documents have been indexed successfully in Weaviate.")


if __name__ == "__main__":
    main()
