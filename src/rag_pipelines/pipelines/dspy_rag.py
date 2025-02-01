"""Main entry point for running the DSPy RAG pipeline with configurable parameters.

This script performs the following steps:
    1. Loads a dataset of questions using the HuggingFace datasets library.
    2. Splits the dataset into train, development (dev), and test subsets.
    3. Initializes a Weaviate vector database for document retrieval.
    4. Configures a language model (LLM) for DSPy.
    5. Creates a DSPy Retrieval-Augmented Generation (RAG) module that retrieves context from Weaviate and generates answers.
    6. Evaluates the generated answers using an LLM-based metric.

Usage:
    python main.py --weaviate_url <URL> --weaviate_api_key <KEY> --index_name <NAME> --llm_api_key <KEY>
"""

import argparse

import dspy
from datasets import load_dataset

from rag_pipelines.dspy.dspy_evaluator import DSPyEvaluator
from rag_pipelines.dspy.dspy_rag import DSPyRAG
from rag_pipelines.vectordb.weaviate import WeaviateVectorDB


def main(args):
    """Run the DSPy RAG pipeline.

    Steps:
        1. Load a dataset of questions.
        2. Split the dataset into subsets (train/dev/test).
        3. Initialize the Weaviate vector database.
        4. Configure the language model (LLM) for DSPy.
        5. Initialize the DSPy RAG module.
        6. Evaluate the RAG module on the dev set using an LLM-based metric.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration parameters.
    """
    # Load the dataset specified by the --dataset_name argument.
    # The dataset is loaded from the HuggingFace datasets library.
    dataset = load_dataset(args.dataset_name, split=f"train[:{args.dataset_size}]")
    questions = dataset["question"]

    # Create examples from the questions.
    # The first 20 questions are used for training (not stored in a variable here).
    [dspy.Example(question=q).with_inputs("question") for q in questions[:20]]
    # The next 10 questions (indices 20 to 29) form the development set.
    devset = [dspy.Example(question=q).with_inputs("question") for q in questions[20:30]]
    # Remaining questions (from index 30 onward) could be used as the test set.
    [dspy.Example(question=q).with_inputs("question") for q in questions[30:]]

    # Initialize the Weaviate vector database for document retrieval.
    # The vector database is configured with the provided Weaviate URL, API key, index name, and embedding model.
    weaviate_db = WeaviateVectorDB(
        cluster_url=args.weaviate_url,
        api_key=args.weaviate_api_key,
        index_name=args.index_name,
        model_name=args.embedding_model,
    )

    # Initialize the language model (LLM) for DSPy.
    # The LLM is configured with the provided model name and API key.
    llm = dspy.LM(args.llm_model, api_key=args.llm_api_key)
    # Configure DSPy to use the initialized LLM.
    dspy.configure(lm=llm)

    # Initialize the DSPy RAG module using the Weaviate vector database.
    rag = DSPyRAG(weaviate_db)

    # Initialize the DSPy evaluator, which uses an LLM-based metric for evaluation.
    evaluator = DSPyEvaluator()

    # Set up the evaluation process on the development set.
    # The evaluation uses one thread and displays progress and a summary table.
    evaluate = dspy.Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    # Run the evaluation on the RAG module with the specified metric.
    evaluate(rag, metric=evaluator.llm_metric)


if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Run DSPy RAG pipeline with configurable parameters.")

    # Dataset configuration arguments.
    parser.add_argument(
        "--dataset_name", type=str, default="lamini/earnings-calls-qa", help="Name of the dataset to use."
    )
    parser.add_argument("--dataset_size", type=str, default="50", help="Size of dataset to load (e.g., '50').")

    # Weaviate configuration arguments.
    parser.add_argument("--weaviate_url", type=str, required=True, help="Weaviate cloud cluster URL.")
    parser.add_argument("--weaviate_api_key", type=str, required=True, help="API key for Weaviate.")
    parser.add_argument("--index_name", type=str, required=True, help="Index name in Weaviate.")
    parser.add_argument(
        "--embedding_model", type=str, default="jinaai/jina-embeddings-v3", help="Embedding model for Weaviate."
    )

    # LLM configuration arguments.
    parser.add_argument("--llm_model", type=str, default="groq/llama-3.3-70b-versatile", help="LLM model to use.")
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for LLM.")

    # Parse command-line arguments.
    args = parser.parse_args()

    # Run the main function with the parsed arguments.
    main(args)
