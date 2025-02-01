import argparse

import dspy
import weaviate
from datasets import load_dataset
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BayesianSignatureOptimizer, BootstrapFewShotWithRandomSearch
from langchain_huggingface import HuggingFaceEmbeddings
from weaviate.classes.init import Auth

from rag_pipelines.dspy.dspy_evaluator import DSPyEvaluator
from rag_pipelines.dspy.dspy_rag import DSPyRAG
from rag_pipelines.vectordb.weaviate import WeaviateVectorStore


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimize and evaluate RAG pipeline with DSPy.")

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

    # Optimization Method
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["bootstrap", "bayesian"],
        default="bootstrap",
        help="Choose the optimization method.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=f"train[:{args.dataset_size}]")
    questions = dataset["question"]

    # Create DSPy datasets
    trainset = [dspy.Example(question=q).with_inputs("question") for q in questions[:20]]
    devset = [dspy.Example(question=q).with_inputs("question") for q in questions[20:30]]
    testset = [dspy.Example(question=q).with_inputs("question") for q in questions[30:]]

    # Initialize embeddings
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"task": "retrieval.query", "prompt_name": "retrieval.query"}
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Connect to Weaviate
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=args.weaviate_url,
        auth_credentials=Auth.api_key(args.weaviate_api_key),
    )
    WeaviateVectorStore(
        index_name=args.index_name,
        embedding=embeddings,
        client=weaviate_client,
        text_key="text",
    )

    # Configure LLM
    llm = dspy.LM(args.llm_model, api_key=args.llm_api_key, num_retries=120)
    dspy.configure(lm=llm)

    # Evaluate before optimization
    evaluate = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(DSPyRAG(), metric=DSPyEvaluator.llm_metric())

    # Select Optimizer
    if args.optimizer == "bootstrap":
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=DSPyEvaluator.llm_metric(),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            max_rounds=1,
            num_candidate_programs=2,
            num_threads=2,
        )
    else:
        optimizer = BayesianSignatureOptimizer(
            task_model=dspy.settings.lm,
            metric=DSPyEvaluator.llm_metric(),
            prompt_model=dspy.settings.lm,
            n=5,
            verbose=False,
        )

    # Compile optimized RAG
    optimized_compiled_rag = optimizer.compile(DSPyRAG(), testset=testset, trainset=trainset)

    # Evaluate optimized RAG
    evaluate = Evaluate(
        metric=DSPyEvaluator.llm_metric(), devset=devset, num_threads=1, display_progress=True, display_table=5
    )
    evaluate(optimized_compiled_rag)


if __name__ == "__main__":
    main()
