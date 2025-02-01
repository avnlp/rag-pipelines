import argparse

import dspy
from datasets import load_dataset
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from rag_pipelines.dspy.dspy_evaluator import DSPyEvaluator
from rag_pipelines.dspy.dspy_rag import DSPyRAG
from rag_pipelines.vectordb.weaviate import WeaviateVectorDB


def main(args):
    """Main function to run the DSPy RAG optimization pipeline.

    This function loads a dataset, initializes a Weaviate vector database and an LLM,
    evaluates an unoptimized RAG pipeline, optimizes it using BootstrapFewShotWithRandomSearch,
    and then evaluates the optimized pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments for configuring the pipeline.
    """
    # Load dataset (Earnings Calls QA)
    earnings_calls_data = load_dataset("lamini/earnings-calls-qa", split="train[:50]")
    questions = earnings_calls_data["question"]

    # Split dataset into training, development, and test sets
    trainset = [dspy.Example(question=q).with_inputs("question") for q in questions[:20]]
    devset = [dspy.Example(question=q).with_inputs("question") for q in questions[20:30]]
    [dspy.Example(question=q).with_inputs("question") for q in questions[30:]]  # Test set (not used here)

    # Initialize Weaviate Vector Database
    weaviate_db = WeaviateVectorDB(
        cluster_url=args.cluster_url,
        api_key=args.api_key,
        index_name=args.index_name,
        model_name=args.embedding_model,
    )

    # Initialize the LLM
    llm = dspy.LM(args.llm_model, api_key=args.llm_api_key, num_retries=args.num_retries)
    dspy.configure(lm=llm)  # Set DSPy's global LLM configuration

    # Initialize the unoptimized RAG pipeline
    uncompiled_rag = DSPyRAG(weaviate_db)

    # Evaluate the unoptimized RAG model
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(uncompiled_rag, metric=DSPyEvaluator.llm_metric())

    # Optimize RAG using BootstrapFewShotWithRandomSearch
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=DSPyEvaluator.llm_metric(),
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        max_rounds=args.max_rounds,
        num_candidate_programs=args.num_candidate_programs,
        num_threads=args.num_threads,
    )

    # Compile an optimized version of the RAG model
    optimized_compiled_rag = optimizer.compile(uncompiled_rag, trainset=trainset)

    # Evaluate the optimized RAG model
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(optimized_compiled_rag)


if __name__ == "__main__":
    """
    Parses command-line arguments and runs the main function.
    """

    parser = argparse.ArgumentParser(description="DSPy RAG Optimization Pipeline")

    # Weaviate parameters
    parser.add_argument("--cluster_url", type=str, required=True, help="Weaviate cluster URL.")
    parser.add_argument("--api_key", type=str, required=True, help="Weaviate API key.")
    parser.add_argument("--index_name", type=str, required=True, help="Weaviate index name.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="jinaai/jina-embeddings-v3",
        help="Embedding model to use for vector retrieval.",
    )

    # LLM parameters
    parser.add_argument("--llm_model", type=str, default="groq/llama-3.3-70b-versatile", help="LLM model name.")
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for accessing the LLM.")
    parser.add_argument("--num_retries", type=int, default=120, help="Number of retries for LLM calls.")

    # Optimization parameters
    parser.add_argument("--max_bootstrapped_demos", type=int, default=4, help="Max bootstrapped demonstrations.")
    parser.add_argument("--max_labeled_demos", type=int, default=4, help="Max labeled demonstrations.")
    parser.add_argument("--max_rounds", type=int, default=1, help="Max optimization rounds.")
    parser.add_argument("--num_candidate_programs", type=int, default=2, help="Number of candidate programs.")
    parser.add_argument("--num_threads", type=int, default=2, help="Number of threads for optimization.")

    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)
