import argparse

import dspy
import weaviate
from datasets import load_dataset
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import COPRO
from langchain_huggingface import HuggingFaceEmbeddings
from weaviate.classes.init import Auth

from rag_pipelines.dspy.dspy_evaluator import DSPyEvaluator
from rag_pipelines.dspy.dspy_rag import DSPyRAG
from rag_pipelines.vectordb.weaviate import WeaviateVectorStore


def parse_args():
    """Parse command-line arguments for the DSPy RAG pipeline with Weaviate and LLM evaluation.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run DSPy RAG pipeline with Weaviate and LLM evaluation.")

    # Dataset Arguments
    parser.add_argument(
        "--dataset_name", type=str, default="lamini/earnings-calls-qa", help="Name of the dataset to use."
    )
    parser.add_argument("--dataset_size", type=int, default=50, help="Number of examples to load from the dataset.")

    # Weaviate Configuration
    parser.add_argument("--weaviate_url", type=str, required=True, help="Weaviate cloud cluster URL.")
    parser.add_argument("--weaviate_api_key", type=str, required=True, help="API key for Weaviate.")
    parser.add_argument("--index_name", type=str, required=True, help="Index name in Weaviate.")
    parser.add_argument(
        "--embedding_model", type=str, default="jinaai/jina-embeddings-v3", help="Embedding model for Weaviate."
    )

    # LLM Configuration
    parser.add_argument("--llm_model", type=str, default="groq/llama-3.3-70b-versatile", help="LLM model to use.")
    parser.add_argument("--llm_api_key", type=str, required=True, help="API key for LLM.")

    return parser.parse_args()


def main():
    """Run the DSPy RAG pipeline, including dataset loading, embedding initialization, Weaviate connection, LLM evaluation, and model optimization.

    This function orchestrates the entire pipeline from loading data, preparing datasets,
    connecting to Weaviate, initializing embeddings, evaluating the model, and optimizing the RAG pipeline.
    """
    # Parse command-line arguments
    args = parse_args()

    # Load dataset from Hugging Face and extract questions
    dataset = load_dataset(args.dataset_name, split=f"train[:{args.dataset_size}]")
    questions = dataset["question"]

    # Create DSPy datasets for training and evaluation
    trainset = [dspy.Example(question=q).with_inputs("question") for q in questions[:20]]
    devset = [dspy.Example(question=q).with_inputs("question") for q in questions[20:30]]
    [dspy.Example(question=q).with_inputs("question") for q in questions[30:]]

    # Initialize HuggingFace embeddings for retrieval tasks
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"task": "retrieval.query", "prompt_name": "retrieval.query"}
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Connect to Weaviate using the provided URL and API key
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=args.weaviate_url,
        auth_credentials=Auth.api_key(args.weaviate_api_key),
    )

    # Initialize Weaviate vector store with the embeddings and client connection
    WeaviateVectorStore(
        index_name=args.index_name,
        embedding=embeddings,
        client=weaviate_client,
        text_key="text",
    )

    # Initialize the LLM (Language Model) with the specified model and API key
    llm = dspy.LM(args.llm_model, api_key=args.llm_api_key, num_retries=120)
    dspy.configure(lm=llm)

    # Evaluate the initial RAG pipeline
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(DSPyRAG(), metric=DSPyEvaluator.llm_metric())

    # Optimize the RAG model using COPRO (Collaborative Prompt Optimization)
    optimizer = COPRO(
        prompt_model=dspy.settings.lm,
        metric=DSPyEvaluator.llm_metric(),
        breadth=3,
        depth=2,
        init_temperature=0.25,
        verbose=False,
    )

    # Compile the optimized RAG model with the training set
    optimized_compiled_rag = optimizer.compile(
        DSPyRAG(),
        trainset=trainset,
        eval_kwargs={"num_threads": 1, "display_progress": True, "display_table": 0},
    )

    # Evaluate the optimized model on the development set
    evaluate = Evaluate(
        metric=DSPyEvaluator.llm_metric(),
        devset=devset,
        num_threads=1,
        display_progress=True,
        display_table=5,
    )
    evaluate(optimized_compiled_rag)


if __name__ == "__main__":
    main()
