import argparse

import dspy
from datasets import load_dataset

from rag_pipelines.dspy.dspy_evaluator import DSPyEvaluator
from rag_pipelines.dspy.dspy_rag import DSPyRAG
from rag_pipelines.vectordb.weaviate import WeaviateVectorDB


def main(cluster_url, api_key, index_name, model_name, llm_model, llm_api_key):
    """Executes the DSPy-based Retrieval-Augmented Generation (RAG) pipeline.

    This function:
    1. Loads a dataset of earnings call Q&A pairs.
    2. Prepares development (dev) and test datasets for evaluation.
    3. Initializes a Weaviate vector database for storing and retrieving embeddings.
    4. Configures a Large Language Model (LLM) with DSPy.
    5. Instantiates and evaluates the RAG pipeline before optimization.

    Args:
        cluster_url (str): URL of the Weaviate vector database cluster.
        api_key (str): API key for authenticating access to Weaviate.
        index_name (str): Name of the Weaviate index for storing vectors.
        model_name (str): Embedding model name for vectorization.
        llm_model (str): Name of the LLM used for inference.
        llm_api_key (str): API key for accessing the LLM.
    """
    # Load the earnings calls Q&A dataset (first 50 samples)
    earnings_calls_data = load_dataset("lamini/earnings-calls-qa", split="train[:50]")
    questions = earnings_calls_data["question"]

    # Prepare dataset splits:
    # - The first 20 questions are used for training (not explicitly utilized here).
    # - The next 10 questions are used as the development set (devset) for evaluation.
    # - The remaining questions are part of the test set (not used in this script).
    devset = [dspy.Example(question=q).with_inputs("question") for q in questions[20:30]]

    # Initialize Weaviate VectorDB for embedding storage and retrieval
    weaviate_db = WeaviateVectorDB(
        cluster_url=cluster_url,  # Weaviate cluster URL
        api_key=api_key,  # API key for authentication
        index_name=index_name,  # Name of the index for vector storage
        model_name=model_name,  # Embedding model used for vectorization
    )

    # Initialize the LLM with DSPy
    llm = dspy.LM(llm_model, api_key=llm_api_key, num_retries=120)
    dspy.configure(lm=llm)  # Set DSPy’s global LLM configuration

    # Instantiate the RAG pipeline
    rag = DSPyRAG(weaviate_db)

    # Initialize evaluator for measuring LLM-based retrieval performance
    evaluator = DSPyEvaluator()

    # Evaluate the RAG pipeline before optimization
    evaluate = dspy.Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(rag, metric=evaluator.llm_metric)


if __name__ == "__main__":
    """
    Parses command-line arguments and runs the DSPy-based RAG pipeline.
    """

    parser = argparse.ArgumentParser(description="Run DSPy-based RAG pipeline")

    # Weaviate configuration parameters
    parser.add_argument("--cluster_url", type=str, required=True, help="Weaviate cluster URL.")
    parser.add_argument("--api_key", type=str, required=True, help="Weaviate API key.")
    parser.add_argument("--index_name", type=str, required=True, help="Weaviate index name.")
    parser.add_argument("--model_name", type=str, required=True, help="Embedding model name for vectorization.")

    # LLM configuration parameters
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model name.")
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for LLM access.")

    # Parse command-line arguments and execute the pipeline
    args = parser.parse_args()
    main(args.cluster_url, args.api_key, args.index_name, args.model_name, args.llm_model, args.llm_api_key)
