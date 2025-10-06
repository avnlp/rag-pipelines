"""Evaluation of RAG pipeline on HealthBench dataset."""

import logging
import os
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from deepeval import evaluate
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langgraph.graph import END, StateGraph
from tqdm import tqdm
from typing_extensions import TypedDict

from rag_pipelines.utils import ContextualReranker, MetadataExtractor


logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State schema for the RAG pipeline graph.

    Attributes:
        question: The input question from the dataset.
        metadata_filter: Structured metadata extracted from the question for filtering.
        retrieved_docs: List of Document objects retrieved from the vector store.
        retrieved_context: List of raw text strings from retrieved documents.
        context: Concatenated string of retrieved context, separated by newlines.
        response: Generated answer from the LLM.
        answer: Ground truth answer from the dataset.
        evaluation_scores: Retrieval and Response evaluation scores.
    """

    question: str
    metadata_filter: dict[str, Any]
    retrieved_docs: list[Document]
    retrieved_context: list[str]
    context: str
    response: str
    answer: str
    evaluation_scores: dict[str, Any]


class MetadataExtractionNode:
    """Node responsible for extracting structured metadata from the input question.

    This node uses an LLM-powered extractor to parse the question and produce
    a structured filter dictionary based on a predefined schema. The output is
    used to constrain document retrieval to relevant subsets (e.g., by publication year,
    study type, etc.).
    """

    def __init__(self, extractor: MetadataExtractor, schema: dict[str, Any]) -> None:
        """Initialize the metadata extraction node.

        Args:
            extractor: An instance of MetadataExtractor capable of structured output.
            schema: JSON schema defining expected metadata fields and types.
        """
        self.extractor = extractor
        self.schema = schema

    def __call__(self, state: RAGState) -> RAGState:
        """Invoke metadata extraction on the current question.

        Args:
            state: Current RAG state containing at least the 'question' key.

        Returns:
            Updated state with 'metadata_filter' populated.
        """
        question = state["question"]
        metadata_filter = self.extractor.invoke(question, self.schema)
        return {**state, "metadata_filter": metadata_filter}


class DocumentRetrievalNode:
    """Node that retrieves relevant documents using a configured retriever.

    Applies optional metadata filtering (from prior node) during retrieval.
    Converts retrieved Document objects into both raw Document list and
    plain text list for downstream use.
    """

    def __init__(self, retriever: BaseRetriever) -> None:
        """Initialize the document retrieval node.

        Args:
            retriever: A LangChain retriever instance (e.g., from Milvus).
        """
        self.retriever = retriever

    def __call__(self, state: RAGState) -> RAGState:
        """Retrieve documents based on question and optional metadata filter.

        Args:
            state: Current RAG state containing 'question' and optionally
                'metadata_filter'.

        Returns:
            Updated state with 'retrieved_docs', 'retrieved_context', and 'context'.
        """
        question = state["question"]
        metadata_filter = state.get("metadata_filter", {})
        retrieved_docs = self.retriever.invoke(question, filter=metadata_filter)
        retrieved_context = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n".join(retrieved_context)
        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "retrieved_context": retrieved_context,
            "context": context_text,
        }


class DocumentRerankerNode:
    """Node that reranks retrieved documents based on their relevance to the query."""

    def __init__(self, reranker: ContextualReranker) -> None:
        """Initialize the document reranker node.

        Args:
            reranker: A ContextualReranker instance.
        """
        self.reranker = reranker

    def __call__(self, state: RAGState) -> RAGState:
        """Rerank documents based on the query and retrieved context.

        Args:
            state: Current RAG state containing 'question' and 'retrieved_docs'.

        Returns:
            Updated state with 'retrieved_docs' sorted by relevance.
        """
        question = state["question"]
        retrieved_docs = state["retrieved_docs"]
        reranked_docs = self.reranker.rerank(question, retrieved_docs)
        return {**state, "retrieved_docs": reranked_docs}


class AnswerGenerationNode:
    """Node that generates an answer using an LLM conditioned on retrieved context.

    Uses a pre-defined prompt template that injects the context and question
    into the LLM call. The output is the raw string response from the model.
    """

    def __init__(self, llm: BaseChatModel, prompt_template: ChatPromptTemplate) -> None:
        """Initialize the answer generation node.

        Args:
            llm: A LangChain chat model (e.g., ChatGroq).
            prompt_template: A ChatPromptTemplate with placeholders for 'context'
                            and 'question'.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.chain = prompt_template | llm

    def __call__(self, state: RAGState) -> RAGState:
        """Generate an answer using the LLM and retrieved context.

        Args:
            state: Current RAG state containing 'question' and 'context'.

        Returns:
            Updated state with 'response' populated from the LLM output.
        """
        question = state["question"]
        context_text = state.get("context", "")
        result = self.chain.invoke({"context": context_text, "question": question})
        response = str(result.content)
        return {**state, "response": response}


class EvaluationNode:
    """Node that evaluates the RAG pipeline output using DeepEval metrics.

    Constructs an LLMTestCase from the ground truth, generated answer, and
    retrieved context, then runs a suite of pre-configured metrics.
    """

    def __init__(self, metrics: list[BaseMetric]) -> None:
        """Initialize the evaluation node.

        Args:
            metrics: List of DeepEval metric instances to evaluate the response.
        """
        self.metrics = metrics

    def __call__(self, state: RAGState) -> RAGState:
        """Evaluate the generated response against ground truth and context.

        Args:
            state: Current RAG state containing 'question', 'answer', 'response',
                    and 'retrieved_context'.

        Returns:
            Updated state with 'evaluation_scores' containing DeepEval results.
        """
        question = state["question"]
        answer = state["answer"]
        response = state["response"]
        retrieved_context = state["retrieved_context"]

        evaluation_test_case = LLMTestCase(
            input=question,
            expected_output=answer,
            actual_output=response,
            retrieval_context=retrieved_context,
        )
        evaluation_result = evaluate(
            test_cases=[evaluation_test_case], metrics=self.metrics
        )
        # Convert EvaluationResult to dictionary format
        evaluation_scores = {
            "test_results": [
                test_result.__dict__ for test_result in evaluation_result.test_results
            ],
            "confident_link": evaluation_result.confident_link,
        }
        return {**state, "evaluation_scores": evaluation_scores}


def main() -> None:
    """Run HealthBench RAG pipeline evaluation.

    Loads configuration, initializes components (LLMs, embeddings, vector store,
    retriever, metrics, prompt), builds a LangGraph workflow, and evaluates
    the pipeline on the HealthBench dataset.

    The pipeline consists of four sequential nodes:
        1. Metadata extraction (for filtered retrieval)
        2. Document retrieval (with optional metadata filtering)
        3. Answer generation (using retrieved context)
        4. Evaluation (using DeepEval metrics and HealthBench-aligned scoring)

    Results are printed for each sample; evaluation scores are captured in the state.
    """
    # Load environment variables
    load_dotenv()

    # Load YAML configuration
    with open("healthbench_rag_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Extract config sections
    dataset_config = config["dataset"]
    llm_config = config["llm"]
    embedding_config = config["embedding"]
    vectorstore_config = config["vectorstore"]
    retriever_config = config["retriever"]
    prompt_config = config["prompt"]
    metrics_config = config["metrics"]
    metadata_schema = config["metadata_schema"]

    # Load HealthBench dataset
    logger.info(f"Loading dataset: {dataset_config['path']}")
    dataset: Dataset = load_dataset(
        dataset_config["path"],
        name=dataset_config["split_name"],
        split=dataset_config["split"],
    )

    extractor_llm = ChatGroq(
        model=llm_config["extractor"]["model"],
        temperature=llm_config["extractor"]["temperature"],
        max_tokens=llm_config["extractor"]["max_tokens"],
        max_retries=llm_config["extractor"]["max_retries"],
    )
    metadata_extractor = MetadataExtractor(llm=extractor_llm)

    response_llm = ChatGroq(
        model=llm_config["response"]["model"],
        temperature=llm_config["response"]["temperature"],
        max_tokens=llm_config["response"]["max_tokens"],
        max_retries=llm_config["response"]["max_retries"],
        reasoning_format=llm_config["response"]["reasoning_format"],
    )

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_config["model_name"])

    # Initialize Milvus vector store with BM25 hybrid search
    vectorstore = Milvus(
        embedding_function=embeddings,
        builtin_function=BM25BuiltInFunction(),
        vector_field=vectorstore_config["vector_fields"],
        collection_name=vectorstore_config["collection_name"],
        connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
        },
        consistency_level=vectorstore_config["consistency_level"],
        drop_old=vectorstore_config["drop_old"],
    )

    # Create retriever on vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_config["k"]})

    # Initialize Contextual Reranker
    reranker = ContextualReranker(
        model_path=llm_config["reranker"]["model"],
        instruction=llm_config["reranker"]["instruction"],
    )

    # Initialize Evaluation metrics
    contextual_recall = ContextualRecallMetric(**metrics_config["contextual_recall"])
    contextual_precision = ContextualPrecisionMetric(
        **metrics_config["contextual_precision"]
    )
    contextual_relevancy = ContextualRelevancyMetric(
        **metrics_config["contextual_relevancy"]
    )
    answer_relevancy = AnswerRelevancyMetric(**metrics_config["answer_relevancy"])
    faithfulness = FaithfulnessMetric(**metrics_config["faithfulness"])
    g_eval = GEval(**metrics_config["g_eval"])

    # Construct RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_config["system_message"].strip()),
            ("human", prompt_config["human_message"]),
        ]
    )

    # Build LangGraph workflow
    workflow = StateGraph(RAGState)
    workflow.add_node(
        "extract_metadata", MetadataExtractionNode(metadata_extractor, metadata_schema)
    )
    workflow.add_node("retrieve_documents", DocumentRetrievalNode(retriever))
    workflow.add_node("document_reranker", DocumentRerankerNode(reranker))
    workflow.add_node("generate_answer", AnswerGenerationNode(response_llm, rag_prompt))
    workflow.add_node(
        "evaluate",
        EvaluationNode(
            [
                contextual_recall,
                contextual_precision,
                contextual_relevancy,
                answer_relevancy,
                faithfulness,
                g_eval,
            ]
        ),
    )

    # Define execution flow
    workflow.set_entry_point("extract_metadata")
    workflow.add_edge("extract_metadata", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "document_reranker")
    workflow.add_edge("document_reranker", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate")
    workflow.add_edge("evaluate", END)

    rag_pipeline = workflow.compile()

    # Extract questions and answers from dataset
    questions = dataset[dataset_config["question_field"]]
    answers = dataset[dataset_config["answer_field"]]

    # Evaluate pipeline
    for question, answer in tqdm(zip(questions, answers)):
        initial_state: RAGState = {
            "question": question[0]["content"],
            "answer": answer["ideal_completion"],
            "metadata_filter": {},
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "evaluation_scores": {},
        }
        result = rag_pipeline.invoke(  # type: ignore[arg-type]
            initial_state,
            config={"callbacks": [CallbackHandler()]},
        )
        print(result)


if __name__ == "__main__":
    main()
