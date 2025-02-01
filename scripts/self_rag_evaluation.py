import argparse

from dataloaders.langchain import (
    ARCDataloader,
    EdgarDataLoader,
    FactScoreDataloader,
    PopQADataloader,
    TriviaQADataloader,
)
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag_pipelines.evaluation import (
    AnswerRelevancyScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    ContextualRelevancyScorer,
    Evaluator,
    FaithfulnessScorer,
    HallucinationScorer,
    SummarizationScorer,
)
from rag_pipelines.pipelines.self_rag import SelfRAGPipeline
from rag_pipelines.query_transformer.query_transformer import QueryTransformer
from rag_pipelines.retrieval_evaluator.document_grader import DocumentGrader
from rag_pipelines.retrieval_evaluator.retrieval_evaluator import RetrievalEvaluator
from rag_pipelines.websearch.web_search import WebSearch

SUPPORTED_DATASETS = {
    "arc": ARCDataloader,
    "edgar": EdgarDataLoader,
    "popqa": PopQADataloader,
    "factscore": FactScoreDataloader,
    "triviaqa": TriviaQADataloader,
}

SCORER_CLASSES = {
    "contextual_precision": ContextualPrecisionScorer,
    "contextual_recall": ContextualRecallScorer,
    "contextual_relevancy": ContextualRelevancyScorer,
    "answer_relevancy": AnswerRelevancyScorer,
    "faithfulness": FaithfulnessScorer,
    "summarization": SummarizationScorer,
    "hallucination": HallucinationScorer,
}


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

    # Load evaluation data
    parser = argparse.ArgumentParser(description="Load evaluation dataset and initialize the dataloader.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="edgar",
        choices=SUPPORTED_DATASETS.keys(),
        help="Dataset to load from. Options: arc, edgar, popqa, factscore, triviaqa.",
    )
    parser.add_argument(
        "--hf_dataset_path",
        type=str,
        default="lamini/earnings-calls-qa",
        help="Path to the HuggingFace dataset.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Split of the dataset to use (e.g., train, validation, test).",
    )

    # Scorer arguments
    parser.add_argument(
        "--scorer",
        type=str,
        default="contextual_precision",
        choices=[
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "answer_relevancy",
            "faithfullness",
            "summarization",
            "hallucination",
        ],
        help="Scorer to use.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--evaluation_name",
        type=str,
        default="hybrid_rag",
        help="Name of the evaluation.",
    )

    # Add argument for selecting scorers
    parser.add_argument(
        "--scorers",
        type=str,
        nargs="+",
        choices=SCORER_CLASSES.keys(),
        required=True,
        help="List of scorers to use. Options: contextual_precision, contextual_recall, contextual_relevancy, "
        "answer_relevancy, faithfulness, summarization, hallucination.",
    )

    # Add shared arguments for scorer parameters
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for evaluation.")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model to use for scoring.")
    parser.add_argument("--include_reason", action="store_true", help="Include reasons in scoring.")
    parser.add_argument(
        "--assessment_questions",
        type=str,
        nargs="*",
        help="List of assessment questions for scoring.",
    )
    parser.add_argument("--strict_mode", action="store_true", help="Enable strict mode for scoring.")
    parser.add_argument("--async_mode", action="store_true", help="Enable asynchronous processing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument(
        "--truths_extraction_limit",
        type=int,
        default=None,
        help="Limit for truth extraction in scoring.",
    )

    args = parser.parse_args()

    # Initialize dataloader based on the dataset type
    try:
        DataLoaderClass = SUPPORTED_DATASETS[args.dataset_type]
        dataloader = DataLoaderClass(dataset_name=args.hf_dataset_path, split=args.dataset_split)
    except KeyError:
        msg = (
            f"Dataset '{args.dataset_type}' is not supported. "
            f"Supported options are: {', '.join(SUPPORTED_DATASETS.keys())}."
        )
        raise ValueError(msg)

    eval_dataset = dataloader.get_eval_data()

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

    # Initialize the scorers with the provided arguments
    scorers = []
    for scorer_name in args.scorers:
        if scorer_name in SCORER_CLASSES:
            scorer_class = SCORER_CLASSES[scorer_name]
            scorer = scorer_class(
                threshold=args.threshold,
                model=args.model,
                include_reason=args.include_reason,
                assessment_questions=args.assessment_questions,
                strict_mode=args.strict_mode,
                async_mode=args.async_mode,
                verbose=args.verbose,
                truths_extraction_limit=args.truths_extraction_limit,
            )
            scorers.append(scorer)
        else:
            msg = f"Scorer '{scorer_name}' is not supported."
            raise ValueError(msg)

    # Run the pipeline
    evaluator = Evaluator(
        evaluation_name=args.evaluation_name,
        pipeline=self_rag_pipeline,
        dataset=eval_dataset,
        scorers=[scorers],
    )

    evaluation_results = evaluator.evaluate()
    print(evaluation_results)


if __name__ == "__main__":
    main()
