import argparse

import weave
from dataloaders.langchain import FinanceBenchDataloader

from rag_pipelines.embeddings.dense import DenseEmbeddings
from rag_pipelines.embeddings.sparse_pinecone_text import SparseEmbeddings
from rag_pipelines.vectordb.pinecone_hybrid_index import PineconeHybridVectorDB


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the FinanceBench pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process FinanceBench data, generate embeddings, and add processed documents to a Pinecone hybrid index."
    )

    # Weave tracing project name
    parser.add_argument(
        "--project_name",
        required=True,
        help="Weave project name to initialize tracing.",
    )

    # FinanceBench dataloader arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the FinanceBench dataset (e.g., 'PatronusAI/financebench').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:1]",
        help="Dataset split to use (e.g., 'train[:1]').",
    )

    # Dense Embeddings arguments
    parser.add_argument(
        "--dense_model_name",
        type=str,
        required=True,
        help="Dense embedding model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2').",
    )
    parser.add_argument(
        "--dense_device",
        type=str,
        default="cpu",
        help="Device to run the dense embedding model (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Flag to normalize embeddings during encoding.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Flag to show progress during embedding generation.",
    )

    # Sparse Embeddings arguments
    parser.add_argument(
        "--sparse_max_seq_length",
        type=int,
        required=True,
        help="Maximum sequence length for sparse embeddings.",
    )

    # Semantic Chunking arguments (if applicable in your downstream pipeline)
    parser.add_argument(
        "--chunking_threshold_type",
        type=str,
        default="percentile",
        help="Threshold type for semantic chunking (e.g., 'percentile' or 'absolute').",
    )

    # Pinecone configuration arguments
    parser.add_argument(
        "--pinecone_api_key",
        type=str,
        required=True,
        help="API key for the Pinecone vector database.",
    )
    parser.add_argument(
        "--pinecone_index_name",
        type=str,
        required=True,
        help="Name of the Pinecone index.",
    )
    parser.add_argument(
        "--pinecone_dimension",
        type=int,
        required=True,
        help="Vector dimension in the Pinecone index.",
    )
    parser.add_argument(
        "--pinecone_metric",
        type=str,
        required=True,
        help="Similarity metric for the Pinecone index (e.g., 'dotproduct' or 'cosine').",
    )
    parser.add_argument(
        "--pinecone_region",
        type=str,
        required=True,
        help="Pinecone region (e.g., 'us-east-1').",
    )
    parser.add_argument(
        "--pinecone_cloud",
        type=str,
        required=True,
        help="Pinecone cloud provider (e.g., 'aws').",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="Namespace for document storage in Pinecone.",
    )

    return parser.parse_args()


def main() -> None:
    """Load FinanceBench data, generate dense and sparse embeddings, add processed documents to a Pinecone index.

    The pipeline performs the following steps:
    1. Initialize Weave tracing.
    2. Load FinanceBench documents.
    3. Generate dense and sparse embeddings for the documents.
    4. Initialize and configure the Pinecone hybrid vector database.
    5. Index the processed documents in Pinecone.
    """
    args = parse_arguments()

    # Initialize Weave tracing
    weave.init(args.project_name)

    # Load FinanceBench dataset using FinanceBenchDataloader
    data_loader = FinanceBenchDataloader(
        dataset_name=args.dataset_name,
        split=args.split,
    )
    # Download and prepare PDF documents from the dataset (if not already cached)
    data_loader.get_corpus_pdfs()
    # Create structured documents from the downloaded PDFs
    documents = data_loader.create_documents()
    print("Loaded Documents:")
    print(documents)

    # Initialize dense embedding model
    dense_embeddings = DenseEmbeddings(
        model_name=args.dense_model_name,
        model_kwargs={"device": args.dense_device},
        encode_kwargs={"normalize_embeddings": args.normalize_embeddings},
        show_progress=args.show_progress,
    )

    # Initialize sparse embedding model
    sparse_embeddings = SparseEmbeddings(model_kwargs={"max_seq_length": args.sparse_max_seq_length})

    # Extract embedding models for use in the Pinecone vector database
    dense_embedding_model = dense_embeddings.embedding_model
    sparse_embedding_model = sparse_embeddings.sparse_embedding_model

    # Initialize PineconeHybridVectorDB with specified configuration
    pinecone_vector_db = PineconeHybridVectorDB(
        api_key=args.pinecone_api_key,
        index_name=args.pinecone_index_name,
        dimension=args.pinecone_dimension,
        metric=args.pinecone_metric,
        region=args.pinecone_region,
        cloud=args.pinecone_cloud,
    )

    # Add the processed documents to the Pinecone hybrid index using both dense and sparse embeddings
    pinecone_vector_db.add_documents(
        documents=documents,
        dense_embedding_model=dense_embedding_model,
        sparse_embedding_model=sparse_embedding_model,
        namespace=args.namespace,
    )

    print("Documents have been indexed successfully in Pinecone.")


if __name__ == "__main__":
    main()
