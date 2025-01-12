import argparse

import weave
from data_loaders import EdgarDataLoader

from rag_pipelines.embeddings.dense import DenseEmbeddings
from rag_pipelines.embeddings.sparse import SparseEmbeddings
from rag_pipelines.llms.groq import ChatGroqGenerator
from rag_pipelines.vectordb.pinecone_hybrid_index import PineconeHybridVectorDB


def main():
    """Main function to load SEC filings data, generate embeddings, perform semantic chunking,
    and add processed documents to a Pinecone hybrid index.

    All configurations, such as model selection, Pinecone settings, and chunking options,
    are configurable via command-line arguments.
    """
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Process EDGAR data, chunk documents, and add to Pinecone index.")

    # Weave Project Name
    parser.add_argument(
        "--project_name",
        required=True,
        help="Weave Project Name",
    )

    # Arguments for EDGAR data loader
    parser.add_argument(
        "--companies",
        nargs="+",
        required=True,
        help="List of company symbols to retrieve SEC filings for (e.g., AAPL MSFT GOOG)",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        required=True,
        help="List of SEC forms to retrieve (e.g., 10-K, 10-Q)",
    )

    # Arguments for ChatGroqGenerator (LLM)
    parser.add_argument("--llm_model", required=True, help="Model name for ChatGroqGenerator")
    parser.add_argument(
        "--llm_temperature",
        type=float,
        required=True,
        help="Temperature setting for LLM",
    )
    parser.add_argument("--llm_api_key", required=True, help="API key for ChatGroqGenerator")

    # Arguments for Dense Embeddings
    parser.add_argument(
        "--dense_model_name",
        required=True,
        help="Dense embedding model name (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--dense_device",
        default="cpu",
        help="Device to run dense embedding model (e.g., cpu, cuda)",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Flag to normalize embeddings during encoding",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Flag to show progress during embedding generation",
    )

    # Arguments for Sparse Embeddings
    parser.add_argument(
        "--sparse_max_seq_length",
        type=int,
        required=True,
        help="Maximum sequence length for sparse embeddings",
    )

    # Arguments for Semantic Chunking
    parser.add_argument(
        "--chunking_threshold_type",
        default="percentile",
        help="Threshold type for semantic chunking (e.g., percentile, absolute)",
    )

    # Arguments for Pinecone configuration
    parser.add_argument("--pinecone_api_key", required=True, help="API key for Pinecone vector database")
    parser.add_argument("--pinecone_index_name", required=True, help="Pinecone index name")
    parser.add_argument(
        "--pinecone_dimension",
        type=int,
        required=True,
        help="Vector dimension in Pinecone index",
    )
    parser.add_argument(
        "--pinecone_metric",
        required=True,
        help="Similarity metric for Pinecone index (e.g., dotproduct, cosine)",
    )
    parser.add_argument("--pinecone_region", required=True, help="Pinecone region (e.g., us-east-1)")
    parser.add_argument("--pinecone_cloud", required=True, help="Pinecone cloud provider (e.g., aws)")
    parser.add_argument("--namespace", required=True, help="Namespace for document storage in Pinecone")

    # Parse arguments
    args = parser.parse_args()

    # Initialize Weave tracing
    weave.init(args.project_name)

    # Initialize LLM (ChatGroqGenerator) for generating descriptions for SEC filings
    llm = ChatGroqGenerator(
        model=args.llm_model,
        api_key=args.llm_api_key,
        llm_params={"temperature": args.llm_temperature},
    )

    # Initialize EDGAR data loader with provided companies and forms
    data_loader = EdgarDataLoader(
        companies=args.companies,
        forms=args.forms,
        image_description_generator=llm,
    )
    documents = data_loader.create_documents()

    # Initialize DenseEmbeddings with specified model and settings
    dense_embeddings = DenseEmbeddings(
        model_name=args.dense_model_name,
        model_kwargs={"device": args.dense_device},
        encode_kwargs={"normalize_embeddings": args.normalize_embeddings},
        show_progress=args.show_progress,
    )

    # Initialize SparseEmbeddings with specified sequence length
    sparse_embeddings = SparseEmbeddings(model_kwargs={"max_seq_length": args.sparse_max_seq_length})

    # Store the embedding models for later use in Pinecone
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

    # Add the processed documents to Pinecone using both dense and sparse embedding models
    pinecone_vector_db.add_documents(
        documents=documents,
        dense_embedding_model=dense_embedding_model,
        sparse_embedding_model=sparse_embedding_model,
        namespace=args.namespace,
    )


if __name__ == "__main__":
    main()
