import argparse

from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag_pipelines.pipelines.self_rag import SelfRAGPipeline
from rag_pipelines.query_transformer.query_transformer import QueryTransformer
from rag_pipelines.retrieval_evaluator.document_grader import DocumentGrader
from rag_pipelines.retrieval_evaluator.retrieval_evaluator import RetrievalEvaluator
from rag_pipelines.websearch.web_search import WebSearch


def main():
    parser = argparse.ArgumentParser(description="Run the Self-RAG pipeline.")

    # Pinecone retriever arguments
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

    # Query Transformer arguments
    parser.add_argument(
        "--query_transformer_model",
        type=str,
        default="t5-small",
        help="Model used for query transformation.",
    )

    # Retrieval Evaluator arguments
    parser.add_argument(
        "--llm_model",
        type=str,
        default="llama-3.2-90b-vision-preview",
        help="Language model name for retrieval evaluator.",
    )
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for the language model.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for the language model.",
    )
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=0.7,
        help="Relevance threshold for document grading.",
    )

    # Web Search arguments
    parser.add_argument("--web_search_api_key", type=str, required=True, help="API key for web search.")

    # Prompt arguments
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        required=True,
        help="Path to the prompt template for LLM.",
    )

    # Query
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to run through the Self-RAG pipeline.",
    )

    args = parser.parse_args()

    # Initialize Pinecone retriever
    retriever = PineconeHybridSearchRetriever(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        region=args.region,
        namespace=args.namespace,
    )

    # Initialize Query Transformer
    query_transformer = QueryTransformer(model_name=args.query_transformer_model)

    # Initialize Retrieval Evaluator and Document Grader
    retrieval_evaluator = RetrievalEvaluator(
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        temperature=args.temperature,
    )
    document_grader = DocumentGrader(
        evaluator=retrieval_evaluator,
        threshold=args.relevance_threshold,
    )

    # Initialize Web Search
    web_search = WebSearch(api_key=args.web_search_api_key)

    # Load the prompt template
    with open(args.prompt_template_path) as file:
        prompt_template_str = file.read()
    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    # Initialize the LLM
    llm = ChatGroq(
        model=args.llm_model,
        api_key=args.llm_api_key,
        llm_params={"temperature": args.temperature},
    )

    # Initialize Self-RAG Pipeline
    self_rag_pipeline = SelfRAGPipeline(
        retriever=retriever,
        query_transformer=query_transformer,
        retrieval_evaluator=retrieval_evaluator,
        document_grader=document_grader,
        web_search=web_search,
        prompt=prompt,
        llm=llm,
    )

    # Run the pipeline
    output = self_rag_pipeline.run(args.query)
    print(output)


if __name__ == "__main__":
    main()
