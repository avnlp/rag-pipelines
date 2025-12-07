"""Evaluation of RAG pipeline on PubMedQA dataset."""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

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
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langgraph.graph import END, StateGraph
from tqdm import tqdm
from typing_extensions import TypedDict

from rag_pipelines.baml_client import b
from rag_pipelines.utils import (
    ContextualReranker,
    EnrichmentConfig,
    EnrichmentMode,
    MetadataEnricher,
)


logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State schema for the RAG pipeline graph.

    Attributes:
        question: The input question from the dataset.
        metadata: Extracted metadata dictionary from the query.
        metadata_filter: Milvus filter expression string for metadata-based filtering.
        retrieved_docs: List of Document objects retrieved from the vector store.
        retrieved_context: List of raw text strings from retrieved documents.
        context: Concatenated string of retrieved context, separated by newlines.
        response: Generated answer from the LLM.
        answer: Ground truth answer from the dataset.
        evaluation_scores: Retrieval and Response evaluation scores.
    """

    question: str
    metadata: dict[str, Any]
    metadata_filter: Optional[str]
    retrieved_docs: list[Document]
    retrieved_context: list[str]
    context: str
    response: str
    answer: str
    evaluation_scores: dict[str, Any]


class MetadataEnrichmentNode:
    """Node responsible for extracting structured metadata from the query.

    Uses MetadataEnricher for fast query-time metadata extraction with DYNAMIC
    enrichment mode (structural + user-defined schema extraction).

    Converts extracted metadata into Milvus filter expressions for semantic search
    refinement. This enables filtering retrieved documents by user-specified fields
    (diseases, biological_entities, study_type, etc) extracted from the query.
    """

    def __init__(self, enricher: MetadataEnricher, schema: Dict[str, Any]) -> None:
        """Initialize the metadata enrichment node.

        Args:
            enricher: An instance of MetadataEnricher for dynamic schema
                extraction.
            schema: The JSON schema dictionary defining the metadata to extract.
        """
        self.enricher = enricher
        self.schema = schema

    async def __call__(self, state: RAGState) -> RAGState:
        """Extract metadata from the query using fast DYNAMIC mode.

        Args:
            state: Current RAG state.

        Returns:
            Updated state with metadata and metadata_filter.
        """
        question = state["question"]
        # Use DYNAMIC mode for user-specified schema extraction
        # Returns tuple of (metadata_dict, milvus_filter_expr)
        metadata_dict, milvus_filter = await self.enricher.extract_query_metadata(
            query=question, user_schema=self.schema
        )
        return {**state, "metadata": metadata_dict, "metadata_filter": milvus_filter}


class DocumentRetrievalNode:
    """Node that retrieves relevant documents using hybrid search with reranking.

    Applies optional metadata filtering (from prior node) during retrieval.
    Uses weighted hybrid search combining dense (semantic) and sparse (BM25
    keyword) retrieval strategies, then reranks results. Converts retrieved
    Document objects into both raw Document list and plain text list.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        vectorstore: Any = None,
        k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the document retrieval node.

        Args:
            retriever: A LangChain retriever instance (e.g., from Milvus).
            vectorstore: The underlying Milvus vectorstore (optional, for
                hybrid search with metadata filters and RRF reranking).
            k: Number of documents to retrieve. Default is 5.
            rrf_k: RRF (Reciprocal Rank Fusion) parameter. Default is 60.
        """
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.k = k
        self.rrf_k = rrf_k

    def __call__(self, state: RAGState) -> RAGState:
        """Retrieve documents using hybrid search with RRF reranking.

        Uses hybrid search combining dense (semantic) and sparse
        (keyword-based BM25) retrieval. Applies optional metadata filtering
        and reranks results using RRF (Reciprocal Rank Fusion).

        Args:
            state: Current RAG state containing 'question' and optionally
                'metadata_filter' (Milvus filter expression).

        Returns:
            Updated state with 'retrieved_docs', 'retrieved_context', and 'context'.
        """
        question = state["question"]
        metadata_filter = state.get("metadata_filter")

        if self.vectorstore is not None:
            # Use vectorstore directly for hybrid search with weighted reranking
            # Combines dense (semantic) and sparse (BM25 keyword) search
            retrieved_docs = self.vectorstore.similarity_search(
                question,
                k=self.k,
                expr=metadata_filter if metadata_filter else None,
                ranker_type="rrf",
                ranker_params={"rrf_k": self.rrf_k},
            )
        else:
            # Fall back to configured retriever (no hybrid search)
            retrieved_docs = self.retriever.invoke(question)

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
    """Node that generates a structured answer using a BAML function.

    This node is responsible for calling the BAML-defined `GeneratePubMedAnswer`
    function, which encapsulates the LLM prompt, client, and output parsing.
    """

    async def __call__(self, state: RAGState) -> RAGState:
        """Generate answer by invoking the BAML `GeneratePubMedAnswer` function.

        This method retrieves the question and context from the state,
        calls the BAML function, and extracts the final answer from the
        structured response.

        Args:
            state: The current state of the RAG pipeline.

        Returns:
            The updated state with the generated `response`.
        """
        question = state["question"]
        context_text = state.get("context", "")

        baml_response = await b.GeneratePubMedAnswer(
            context=context_text, question=question
        )

        # The response from BAML is a structured object (`PubMedAnswer` class),
        # The final answer is stored in the answer field
        response = baml_response.answer
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


async def main() -> None:
    """Run PubMedQA RAG pipeline with MetadataEnricher.

    Loads configuration, initializes components (LLMs, embeddings, vector
    store, retriever, metrics), builds a LangGraph workflow, and evaluates
    the pipeline on a sample of the PubMedQA dataset.

    The pipeline consists of five sequential nodes:
        1. Metadata enrichment (fast query-time extraction for filtering)
        2. Document retrieval (with optional metadata filtering)
        3. Document reranking (based on relevance to query)
        4. Answer generation (using retrieved context)
        5. Evaluation (using DeepEval metrics)

    The pipeline is traced using Confident AI.
    """
    # Load environment variables
    load_dotenv()

    # Load YAML configuration
    with open("pubmedqa_rag_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Extract config sections
    dataset_config = config["dataset"]
    embedding_config = config["embedding"]
    vectorstore_config = config["vectorstore"]
    retriever_config = config["retriever"]
    metrics_config = config["metrics"]

    # Extract the JSON schema definition from config
    metadata_schema = config.get("metadata_schema", {})

    # Load PubMedQA dataset
    logger.info(f"Loading dataset: {dataset_config['path']}")
    dataset: Dataset = load_dataset(
        dataset_config["path"],
        name=dataset_config["split_name"],
        split=dataset_config["split"],
    )

    # Initialize MetadataEnricher
    enricher_cfg = config.get("metadata_enricher", {})
    enrichment_config = EnrichmentConfig(
        mode=EnrichmentMode(enricher_cfg.get("mode", "full")),
        batch_size=enricher_cfg.get("batch_size", 10),
    )
    metadata_enricher = MetadataEnricher(enrichment_config)

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
        model_path=config["reranker"]["model"],
        instruction=config["reranker"]["instruction"],
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

    # Build LangGraph workflow
    workflow = StateGraph(RAGState)
    workflow.add_node(
        "enrich_metadata",
        MetadataEnrichmentNode(metadata_enricher, metadata_schema),
    )
    k = retriever_config.get("k", 5)
    rrf_k = retriever_config.get("rrf_k", 60)
    workflow.add_node(
        "retrieve_documents",
        DocumentRetrievalNode(retriever, vectorstore, k, rrf_k),
    )
    workflow.add_node("document_reranker", DocumentRerankerNode(reranker))
    workflow.add_node("generate_answer", AnswerGenerationNode())
    workflow.add_node(
        "evaluate",
        EvaluationNode(
            [
                contextual_recall,
                contextual_precision,
                contextual_relevancy,
                answer_relevancy,
                faithfulness,
            ]
        ),
    )

    # Define execution flow
    workflow.set_entry_point("enrich_metadata")
    workflow.add_edge("enrich_metadata", "retrieve_documents")
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
            "question": question,
            "answer": answer,
            "metadata": {},
            "metadata_filter": None,
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "evaluation_scores": {},
        }
        result = await rag_pipeline.ainvoke(
            initial_state,
            config={"callbacks": [CallbackHandler()]},
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
