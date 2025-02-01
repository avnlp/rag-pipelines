import argparse

import dspy
from datasets import load_dataset
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

from rag_pipelines.dspy.dspy_evaluator import DSPyEvaluator
from rag_pipelines.dspy.dspy_rag import DSPyRAG
from rag_pipelines.vectordb.weaviate import WeaviateVectorDB


def main(args):
    """Runs the DSPy RAG optimization pipeline.

    This function:
    1. Loads the earnings calls dataset.
    2. Splits the dataset into training, development, and test sets.
    3. Initializes a Weaviate vector database and an LLM.
    4. Evaluates an unoptimized RAG pipeline.
    5. Optimizes the RAG pipeline using BootstrapFewShot.
    6. Evaluates the optimized RAG pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments for configuring the pipeline.
    """
    # Load the dataset (Earnings Calls QA dataset)
    earnings_calls_data = load_dataset("lamini/earnings-calls-qa", split="train[:50]")
    questions = earnings_calls_data["question"]

    # Split the dataset into training (20), development (10), and test sets
    trainset = [dspy.Example(question=q).with_inputs("question") for q in questions[:20]]
    devset = [dspy.Example(question=q).with_inputs("question") for q in questions[20:30]]
    [dspy.Example(question=q).with_inputs("question") for q in questions[30:]]  # Test set (not used in this script)

    # Initialize Weaviate VectorDB for storing and retrieving embeddings
    weaviate_db = WeaviateVectorDB(
        cluster_url=args.cluster_url,  # URL of the Weaviate cluster
        api_key=args.api_key,  # API key for authentication
        index_name=args.index_name,  # Name of the Weaviate index
        model_name=args.embedding_model,  # Embedding model to use for vector storage
    )

    # Initialize LLM with DSPy
    llm = dspy.LM(args.llm_model, api_key=args.llm_api_key, num_retries=args.num_retries)
    dspy.configure(lm=llm)  # Set DSPy's global configuration for LLM usage

    # Initialize the unoptimized RAG pipeline
    uncompiled_rag = DSPyRAG(weaviate_db)

    # Evaluate the unoptimized RAG pipeline using the development set
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(uncompiled_rag, metric=DSPyEvaluator.llm_metric())

    # Optimize the RAG pipeline using BootstrapFewShot
    optimizer = BootstrapFewShot(metric=DSPyEvaluator.llm_metric())

    # Compile an optimized version of the RAG model using the training set
    optimized_compiled_rag = optimizer.compile(uncompiled_rag, trainset=trainset)

    # Evaluate the optimized RAG pipeline
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(optimized_compiled_rag)


if __name__ == "__main__":
    """
    Parses command-line arguments and runs the main function.
    """

    parser = argparse.ArgumentParser(description="DSPy RAG Optimization Pipeline")

    # Weaviate parameters (for vector storage and retrieval)
    parser.add_argument("--cluster_url", type=str, required=True, help="Weaviate cluster URL.")
    parser.add_argument("--api_key", type=str, required=True, help="Weaviate API key.")
    parser.add_argument("--index_name", type=str, required=True, help="Weaviate index name.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="jinaai/jina-embeddings-v3",
        help="Embedding model used for document vectorization.",
    )

    # LLM parameters (for DSPy-based language model inference)
    parser.add_argument("--llm_model", type=str, default="groq/llama-3.3-70b-versatile", help="LLM model name.")
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for accessing the LLM service.")
    parser.add_argument("--num_retries", type=int, default=120, help="Number of retries for LLM API calls.")

    # Parse command-line arguments and execute the pipeline
    args = parser.parse_args()
    main(args)
