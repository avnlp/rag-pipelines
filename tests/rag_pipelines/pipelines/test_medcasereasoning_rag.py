"""Test the MedCaseReasoning RAG module."""

import os
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest
from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_pipelines.baml_client.types import MedCaseReasoningAnswer
from rag_pipelines.medcasereasoning.medcasereasoning_rag import (
    AnswerGenerationNode,
    DocumentRerankerNode,
    DocumentRetrievalNode,
    MetadataEnrichmentNode,
    RAGState,
    main,
)


class TestMedCaseReasoningRAG:
    """Test the MedCaseReasoning RAG module components."""

    def test_rag_state_structure(self):
        """Test that RAGState has the correct structure."""
        state: RAGState = {
            "question": "test question",
            "metadata": {},
            "metadata_filter": None,
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "test context",
            "response": "test response",
            "answer": "test answer",
            "evaluation_scores": {},
        }

        assert state["question"] == "test question"
        assert state["metadata"] == {}
        assert state["metadata_filter"] is None
        assert state["retrieved_docs"] == []
        assert state["retrieved_context"] == []
        assert state["context"] == "test context"
        assert state["response"] == "test response"
        assert state["answer"] == "test answer"
        assert state["evaluation_scores"] == {}

    @pytest.mark.asyncio
    async def test_metadata_enrichment_node(self):
        """Test the MetadataEnrichmentNode functionality."""
        mock_enricher = AsyncMock()
        # Mock the async extract_query_metadata method to return a tuple
        mock_enricher.extract_query_metadata.return_value = (
            {"test_field": "test_value"},
            "milvus_filter_expr",
        )
        metadata_schema = {"properties": {"test_field": {"type": "string"}}}

        node = MetadataEnrichmentNode(mock_enricher, metadata_schema)

        initial_state = {
            "question": "test question",
            "metadata": {},
            "metadata_filter": None,
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = await node(initial_state)

        # Check that the enricher was called with the question and schema
        mock_enricher.extract_query_metadata.assert_called_once_with(
            query="test question", user_schema=metadata_schema
        )

        # Check that the result includes the metadata and filter
        assert result["metadata"] == {"test_field": "test_value"}
        assert result["metadata_filter"] == "milvus_filter_expr"

        # Check that other fields remain unchanged
        assert result["question"] == "test question"
        assert result["response"] == ""

    def test_document_retrieval_node(self):
        """Test the DocumentRetrievalNode functionality with hybrid search."""
        mock_retriever = Mock()
        mock_retriever.search_kwargs = {"k": 5}
        mock_retriever.invoke.return_value = [
            Document(page_content="doc1 content"),
            Document(page_content="doc2 content"),
        ]

        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="doc1 content"),
            Document(page_content="doc2 content"),
        ]

        k = 5
        rrf_k = 60
        node = DocumentRetrievalNode(mock_retriever, mock_vectorstore, k, rrf_k)

        initial_state = {
            "question": "test question",
            "metadata": {},
            "metadata_filter": "year >= 2020",
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = node(initial_state)

        # Check that vectorstore similarity_search was called with RRF reranking
        mock_vectorstore.similarity_search.assert_called_once_with(
            "test question",
            k=k,
            expr="year >= 2020",
            ranker_type="rrf",
            ranker_params={"rrf_k": rrf_k},
        )

        # Check that documents are properly retrieved and processed
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_context"] == ["doc1 content", "doc2 content"]
        assert result["context"] == "doc1 content\n\ndoc2 content"

    def test_document_reranker_node(self):
        """Test the DocumentRerankerNode functionality."""
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            Document(page_content="doc2 content"),
            Document(page_content="doc1 content"),
        ]

        node = DocumentRerankerNode(mock_reranker)

        initial_state = {
            "question": "test question",
            "metadata": {},
            "metadata_filter": None,
            "retrieved_docs": [
                Document(page_content="doc1 content"),
                Document(page_content="doc2 content"),
            ],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = node(initial_state)

        # Check that reranker was called with question and documents
        mock_reranker.rerank.assert_called_once_with(
            "test question",
            [
                Document(page_content="doc1 content"),
                Document(page_content="doc2 content"),
            ],
        )

        # Check that the documents are reranked
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0].page_content == "doc2 content"
        assert result["retrieved_docs"][1].page_content == "doc1 content"

    def test_document_retrieval_fallback_without_vectorstore(self):
        """Verify retrieval falls back to retriever when vectorstore is unavailable.

        Tests that the retriever node uses retriever.invoke() instead of
        vectorstore.similarity_search() when vectorstore=None, enabling
        operation without hybrid search infrastructure.
        """
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Document(page_content="fallback doc content")
        ]

        node = DocumentRetrievalNode(mock_retriever, vectorstore=None, k=5, rrf_k=60)

        initial_state = {
            "question": "test question",
            "metadata": {},
            "metadata_filter": None,
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = node(initial_state)

        mock_retriever.invoke.assert_called_once_with("test question")

        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_context"] == ["fallback doc content"]
        assert result["context"] == "fallback doc content"

    @pytest.mark.asyncio
    async def test_answer_generation_node(self):
        """Test the async `AnswerGenerationNode` with a mocked BAML client.

        This test ensures that the node correctly calls the BAML
        `GenerateMedCaseReasoningAnswer` function and processes the structured output.
        """
        # We mock the expected output type for the BAML function
        mock_baml_response = MedCaseReasoningAnswer(
            diagnosis="This is the mocked BAML answer.",
            chain_of_thought="mocked reasoning",
        )

        with patch(
            "rag_pipelines.medcasereasoning.medcasereasoning_rag.b.GenerateMedCaseReasoningAnswer",
            new_callable=AsyncMock,
        ) as mock_generate_answer:
            mock_generate_answer.return_value = mock_baml_response

            node = AnswerGenerationNode()

            initial_state = {
                "question": "test question",
                "metadata": {},
                "metadata_filter": None,
                "retrieved_docs": [],
                "retrieved_context": [],
                "context": "test context",
                "response": "",
                "answer": "",
                "evaluation_scores": {},
            }

            result = await node(initial_state)

            # The assertion checks if the `response` in the final state matches
            # the `diagnosis` from our mock BAML response.
            assert result["response"] == "This is the mocked BAML answer."
            # Verify that the BAML function was called exactly once with initial state
            mock_generate_answer.assert_awaited_once_with(
                context="test context", question="test question"
            )

    @pytest.mark.asyncio
    @patch(
        "rag_pipelines.medcasereasoning.medcasereasoning_rag.b.GenerateMedCaseReasoningAnswer"
    )
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.evaluate")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.load_dataset")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.HuggingFaceEmbeddings")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.Milvus")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.ContextualReranker")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.MetadataEnricher")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.ContextualRecallMetric")
    @patch(
        "rag_pipelines.medcasereasoning.medcasereasoning_rag.ContextualPrecisionMetric"
    )
    @patch(
        "rag_pipelines.medcasereasoning.medcasereasoning_rag.ContextualRelevancyMetric"
    )
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.AnswerRelevancyMetric")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.FaithfulnessMetric")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "dataset": {
                "path": "test/path",
                "split": "test",
                "question_field": "prompt",
                "answer_field": "ideal_completion",
            },
            "embedding": {"model_name": "test-embedding-model"},
            "vectorstore": {
                "vector_fields": ["vector"],
                "collection_name": "test_collection",
                "consistency_level": "Strong",
                "drop_old": False,
            },
            "retriever": {"k": 5, "rrf_k": 60},
            "reranker": {
                "model": "test-model",
                "instruction": "test-instruction",
            },
            "metrics": {
                "contextual_recall": {"threshold": 0.5},
                "contextual_precision": {"threshold": 0.5},
                "contextual_relevancy": {"threshold": 0.5},
                "answer_relevancy": {"threshold": 0.5},
                "faithfulness": {"threshold": 0.5},
            },
            "metadata_enricher": {
                "mode": "dynamic",
                "batch_size": 10,
            },
            "metadata_schema": {"properties": {"test_field": {"type": "string"}}},
        },
    )
    async def test_main_with_mocked_dependencies(
        self,
        mock_yaml_load,  # 'yaml.safe_load'
        mock_open_file,  # 'builtins.open'
        mock_load_dotenv,  # 'load_dotenv'
        mock_faithfulness,  # 'FaithfulnessMetric'
        mock_answer_relevancy,  # 'AnswerRelevancyMetric'
        mock_contextual_relevancy,  # 'ContextualRelevancyMetric'
        mock_contextual_precision,  # 'ContextualPrecisionMetric'
        mock_contextual_recall,  # 'ContextualRecallMetric'
        mock_metadata_enricher,  # 'MetadataEnricher'
        mock_reranker,  # 'ContextualReranker'
        mock_milvus,  # 'Milvus'
        mock_embeddings,  # 'HuggingFaceEmbeddings'
        mock_dataset,  # 'load_dataset'
        mock_evaluate,  # 'evaluate'
        mock_generate_answer,  # 'b.GenerateMedCaseReasoningAnswer'
    ):
        """Test the main function with mocked dependencies to ensure it executes."""
        mock_generate_answer.return_value = MedCaseReasoningAnswer(
            diagnosis="Test diagnosis", chain_of_thought="Test reasoning"
        )

        # Mock dataset
        mock_dataset.return_value = Dataset.from_dict(
            {"prompt": ["test question"], "ideal_completion": ["test answer"]}
        )

        # Mock embeddings instance
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        # Mock Milvus instance
        milvus_instance = Mock()
        retriever_instance = Mock(spec=BaseRetriever)
        retriever_instance.search_kwargs = {"k": 5}
        retriever_instance.invoke.return_value = [Document(page_content="test doc")]
        milvus_instance.as_retriever.return_value = retriever_instance
        milvus_instance.similarity_search.return_value = [
            Document(page_content="test doc")
        ]
        mock_milvus.return_value = milvus_instance

        # Mock reranker instance
        reranker_instance = Mock()
        mock_reranker.return_value = reranker_instance

        # Mock metadata enricher instance
        metadata_enricher_instance = AsyncMock()
        metadata_enricher_instance.extract_query_metadata.return_value = (
            {"test_field": "test_value"},
            None,
        )
        mock_metadata_enricher.return_value = metadata_enricher_instance

        # Mock metric instances
        mock_contextual_recall_instance = Mock()
        mock_contextual_precision_instance = Mock()
        mock_contextual_relevancy_instance = Mock()
        mock_answer_relevancy_instance = Mock()
        mock_faithfulness_instance = Mock()

        mock_contextual_recall.return_value = mock_contextual_recall_instance
        mock_contextual_precision.return_value = mock_contextual_precision_instance
        mock_contextual_relevancy.return_value = mock_contextual_relevancy_instance
        mock_answer_relevancy.return_value = mock_answer_relevancy_instance
        mock_faithfulness.return_value = mock_faithfulness_instance

        # Mock evaluate to avoid validation errors
        mock_evaluation_result = Mock()
        mock_evaluation_result.test_results = [Mock()]
        mock_evaluation_result.confident_link = "test_link"
        mock_evaluate.return_value = mock_evaluation_result

        # Set up environment variables
        os.environ["MILVUS_URI"] = "test_uri"
        os.environ["MILVUS_TOKEN"] = "test_token"

        # The main function should execute without errors with all dependencies mocked

        # Call main in a fully mocked context to ensure it executes properly
        await main()

        # Verify that essential functions were called
        mock_load_dotenv.assert_called_once()

        # Reset environment variables
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        # Simply importing to verify no syntax errors in main function
        assert callable(main)
