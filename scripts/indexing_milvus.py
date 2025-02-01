import argparse

import weave
from dataloaders.langchain import FinanceBenchDataloader
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import CollectionSchema, DataType, FieldSchema

from rag_pipelines.embeddings.sparse_fastembed_milvus import SparseEmbeddings
from rag_pipelines.unstructured.unstructured_chunker import UnstructuredChunker
from rag_pipelines.unstructured.unstructured_pdf_loader import UnstructuredDocumentLoader
from rag_pipelines.vectordb.milvus import MilvusVectorDB


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the FinanceBench pipeline to load, process, chunk, embed, and index documents."
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

    # Milvus connection parameters
    parser.add_argument(
        "--milvus_uri",
        type=str,
        default="https://in03-e006e6b0b452c6c.serverless.gcp-us-west1.cloud.zilliz.com",
        help="URI for the Milvus server.",
    )
    parser.add_argument(
        "--milvus_token",
        type=str,
        default="e86d8d91dbcd3088b0ddcaf2baae5c762338d782f335ebeaf9438fd0ab29b10c13fa53c8c711ac3e7f1bd2ff35bf68e63014f9ce",
        help="Authentication token for Milvus.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="financebench",
        help="Name of the Milvus collection to create/use.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the FinanceBench document processing pipeline.

    This function performs the following steps:
    1. Initializes Weave tracing.
    2. Loads a subset of the FinanceBench dataset.
    3. Retrieves PDF documents from the specified directory.
    4. Processes PDFs using the UnstructuredDocumentLoader.
    5. Chunks documents using the UnstructuredChunker.
    6. Generates dense and sparse embeddings.
    7. Sets up a Milvus vector database and indexes the documents.
    """
    args = parse_arguments()

    # Initialize Weave tracing
    weave.init("financebench_test")

    # Initialize FinanceBench dataloader and load the corpus PDFs
    dataloader = FinanceBenchDataloader(
        dataset_name=args.dataset_name,
        split=args.split,
    )
    dataloader.get_corpus_pdfs()

    # Load and transform PDF documents from the provided directory
    unstructured_document_loader = UnstructuredDocumentLoader(
        strategy=args.strategy,
        mode=args.mode,
    )
    documents = unstructured_document_loader.transform_documents(args.pdf_dir)
    print("Loaded Documents:")
    print(documents)

    # Chunk the documents using the UnstructuredChunker
    chunker = UnstructuredChunker()
    chunked_documents = chunker.transform_documents(documents)
    print("Chunked Documents:")
    print(chunked_documents)

    # Initialize dense and sparse embedding models
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    sparse_embeddings = SparseEmbeddings(model_name="prithvida/Splade_PP_en_v1")

    # Define Milvus collection fields and schema
    pk_field = "doc_id"
    dense_field = "dense_vector"
    sparse_field = "sparse_vector"
    text_field = "text"
    metadata_field = "metadata"
    fields = [
        FieldSchema(
            name=pk_field,
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        ),
        FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
        FieldSchema(name=metadata_field, dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

    # Initialize the Milvus vector database client
    milvus_vector_db = MilvusVectorDB(
        uri=args.milvus_uri,
        token=args.milvus_token,
        collection_name=args.collection_name,
        collection_schema=schema,
        dense_field=dense_field,
        sparse_field=sparse_field,
        text_field=text_field,
        metadata_field=metadata_field,
        dense_index_params={"index_type": "FLAT", "metric_type": "IP"},
        sparse_index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        create_new_collection=True,
    )

    # Add documents to the Milvus vector database with both dense and sparse embeddings
    milvus_vector_db.add_documents(
        documents=chunked_documents,
        dense_embedding_model=embeddings,
        sparse_embedding_model=sparse_embeddings,
    )
    print("Documents have been indexed successfully in Milvus.")


if __name__ == "__main__":
    main()
