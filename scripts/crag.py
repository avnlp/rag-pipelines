import argparse

from rag_pipelines.embeddings.dense import DenseEmbeddings
from rag_pipelines.embeddings.sparse import SparseEmbeddings
from rag_pipelines.llms.groq import ChatGroqGenerator
from rag_pipelines.pipelines.crag import CorrectiveRAGPipeline
from rag_pipelines.retrieval_evaluator.document_grader import DocumentGrader
from rag_pipelines.retrieval_evaluator.retrieval_evaluator import RetrievalEvaluator
from rag_pipelines.vectordb.pinecone_hybrid_index import PineconeHybridVectorDB
from rag_pipelines.vectordb.pinecone_hybrid_retriever import PineconeHybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Run the Corrective RAG pipeline.")

    # Dense embeddings arguments
    parser.add_argument(
        "--dense_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Dense embedding model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the dense embedding model.",
    )

    # Sparse embeddings arguments
    parser.add_argument(
        "--sparse_max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for sparse embeddings.",
    )

    # Pinecone arguments
    parser.add_argument("--pinecone_api_key", type=str, required=True, help="Pinecone API key.")
    parser.add_argument("--index_name", type=str, default="edgar", help="Pinecone index name.")
    parser.add_argument("--dimension", type=int, default=384, help="Dimension of embeddings.")
    parser.add_argument("--metric", type=str, default="dotproduct", help="Metric for similarity search.")
    parser.add_argument("--region", type=str, default="us-east-1", help="Pinecone region.")
    parser.add_argument(
        "--namespace",
        type=str,
        default="edgar-all",
        help="Namespace for Pinecone retriever.",
    )

    # Retriever arguments
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for hybrid retriever.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top documents to retrieve.")

    # LLM arguments
    parser.add_argument(
        "--llm_model",
        type=str,
        default="llama-3.2-90b-vision-preview",
        help="Language model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for the language model.",
    )
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for the language model.")

    # Retrieval Evaluator and Document Grader arguments
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=0.7,
        help="Relevance threshold for document grading.",
    )

    # Query
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to run through the Corrective RAG pipeline.",
    )

    args = parser.parse_args()

    # Initialize embeddings
    dense_embeddings = DenseEmbeddings(
        model_name=args.dense_model_name,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
    sparse_embeddings = SparseEmbeddings(model_kwargs={"max_seq_length": args.sparse_max_seq_length})

    dense_embedding_model = dense_embeddings.embedding_model
    sparse_embedding_model = sparse_embeddings.sparse_embedding_model

    # Initialize Pinecone vector DB
    pinecone_vector_db = PineconeHybridVectorDB(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        region=args.region,
    )

    # Initialize Pinecone retriever
    pinecone_retriever = PineconeHybridRetriever(
        index=pinecone_vector_db.index,
        dense_embedding_model=dense_embedding_model,
        sparse_embedding_model=sparse_embedding_model,
        alpha=args.alpha,
        top_k=args.top_k,
        namespace=args.namespace,
    )

    # Initialize RetrievalEvaluator and DocumentGrader
    retrieval_evaluator = RetrievalEvaluator(
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        temperature=args.temperature,
    )
    document_grader = DocumentGrader(
        evaluator=retrieval_evaluator,
        threshold=args.relevance_threshold,
    )

    # Load the prompt and initialize the generator
    generator = ChatGroqGenerator(
        model=args.llm_model,
        api_key=args.llm_api_key,
        llm_params={"temperature": args.temperature},
    )
    llm = generator.llm

    # Initialize the Corrective RAG pipeline
    corrective_rag = CorrectiveRAGPipeline(
        retriever=pinecone_retriever.hybrid_retriever,
        prompt=retrieval_evaluator.prompt_template,
        llm=llm,
        document_grader=document_grader,
        tracing_project_name="sec_corrective_rag",
    )

    # Run the pipeline
    output = corrective_rag.run(args.query)
    print(output)


if __name__ == "__main__":
    main()
