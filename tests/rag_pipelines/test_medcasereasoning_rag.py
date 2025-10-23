"""Test the MedCaseReasoning RAG module."""

import os
from unittest.mock import Mock, mock_open, patch

from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from rag_pipelines.medcasereasoning.medcasereasoning_rag import (
    AnswerGenerationNode,
    DocumentRerankerNode,
    DocumentRetrievalNode,
    MetadataExtractionNode,
    RAGState,
    main,
)


class TestMedcasereasoningRAG:
    """Test the MedCaseReasoning RAG module components."""

    def test_rag_state_structure(self):
        """Test that RAGState has the correct structure."""
        state: RAGState = {
            "question": "test question",
            "metadata_filter": {},
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "test context",
            "response": "test response",
            "answer": "test answer",
            "evaluation_scores": {},
        }

        assert state["question"] == "test question"
        assert state["metadata_filter"] == {}
        assert state["retrieved_docs"] == []
        assert state["retrieved_context"] == []
        assert state["context"] == "test context"
        assert state["response"] == "test response"
        assert state["answer"] == "test answer"
        assert state["evaluation_scores"] == {}

    def test_metadata_extraction_node(self):
        """Test the MetadataExtractionNode functionality."""
        mock_extractor = Mock()
        mock_extractor.invoke.return_value = {"test_field": "test_value"}
        schema = {"properties": {"test_field": {"type": "string"}}}

        node = MetadataExtractionNode(mock_extractor, schema)

        initial_state = {
            "question": "test question",
            "metadata_filter": {},
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = node(initial_state)

        # Check that the extractor was called with the question and schema
        mock_extractor.invoke.assert_called_once_with("test question", schema)

        # Check that the result includes the metadata filter
        assert result["metadata_filter"] == {"test_field": "test_value"}

        # Check that other fields remain unchanged
        assert result["question"] == "test question"
        assert result["response"] == ""

    def test_document_retrieval_node(self):
        """Test the DocumentRetrievalNode functionality."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Document(page_content="doc1 content"),
            Document(page_content="doc2 content"),
        ]

        node = DocumentRetrievalNode(mock_retriever)

        initial_state = {
            "question": "test question",
            "metadata_filter": {"year": 2020},
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = node(initial_state)

        # Check that the retriever was called with question and filter
        mock_retriever.invoke.assert_called_once_with(
            "test question", filter={"year": 2020}
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
            "metadata_filter": {},
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

    def test_answer_generation_node(self):
        """Test the AnswerGenerationNode functionality."""
        mock_llm = Mock(spec=BaseChatModel)
        mock_result = Mock()
        mock_result.content = "generated answer"
        mock_llm.invoke.return_value = mock_result

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("human", "{question}")]
        )

        # Create node and mock the chain separately
        node = AnswerGenerationNode(mock_llm, prompt_template)

        # Replace the chain with a mock that directly returns the expected result
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        node.chain = mock_chain

        initial_state = {
            "question": "test question",
            "metadata_filter": {},
            "retrieved_docs": [],
            "retrieved_context": [],
            "context": "test context",
            "response": "",
            "answer": "",
            "evaluation_scores": {},
        }

        result = node(initial_state)

        # Check that the chain was invoked with the correct parameters
        mock_chain.invoke.assert_called_once_with(
            {"context": "test context", "question": "test question"}
        )

        # Check that the response was updated correctly
        assert result["response"] == "generated answer"

    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.evaluate")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.load_dataset")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.ChatGroq")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.HuggingFaceEmbeddings")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.Milvus")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.ContextualReranker")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_rag.MetadataExtractor")
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
                "split_name": "test_split",
                "split": "test",
                "question_field": "prompt",
                "answer_field": "ideal_completion",
            },
            "llm": {
                "extractor": {
                    "model": "test_model",
                    "temperature": 0.1,
                    "max_tokens": 100,
                    "max_retries": 3,
                },
                "response": {
                    "model": "test_model",
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "max_retries": 3,
                    "reasoning_format": "test_format",
                },
                "reranker": {"model": "test_model", "instruction": "test instruction"},
            },
            "embedding": {"model_name": "test-embedding-model"},
            "vectorstore": {
                "vector_fields": ["vector"],
                "collection_name": "test_collection",
                "consistency_level": "Strong",
                "drop_old": False,
            },
            "retriever": {"k": 5},
            "prompt": {
                "system_message": "You are a helpful assistant.",
                "human_message": "{context}\n\nQuestion: {question}",
            },
            "metrics": {
                "contextual_recall": {"threshold": 0.5},
                "contextual_precision": {"threshold": 0.5},
                "contextual_relevancy": {"threshold": 0.5},
                "answer_relevancy": {"threshold": 0.5},
                "faithfulness": {"threshold": 0.5},
                "g_eval": {"threshold": 0.5},
            },
            "metadata_schema": {"properties": {"test_field": {"type": "string"}}},
        },
    )
    def test_main_with_mocked_dependencies(
        self,
        mock_yaml_load,  # 'yaml.safe_load'
        mock_open_file,  # 'builtins.open'
        mock_load_dotenv,  # 'load_dotenv'
        mock_faithfulness,  # 'FaithfulnessMetric'
        mock_answer_relevancy,  # 'AnswerRelevancyMetric'
        mock_contextual_relevancy,  # 'ContextualRelevancyMetric'
        mock_contextual_precision,  # 'ContextualPrecisionMetric'
        mock_contextual_recall,  # 'ContextualRecallMetric'
        mock_metadata_extractor,  # 'MetadataExtractor'
        mock_reranker,  # 'ContextualReranker'
        mock_milvus,  # 'Milvus'
        mock_embeddings,  # 'HuggingFaceEmbeddings'
        mock_llm,  # 'ChatGroq'
        mock_dataset,  # 'load_dataset'
        mock_evaluate,  # 'evaluate'
    ):
        """Test the main function with mocked dependencies to ensure it executes."""
        # Mock dataset
        mock_dataset.return_value = Dataset.from_dict(
            {"prompt": ["test question"], "ideal_completion": ["test answer"]}
        )

        # Mock LLM instance
        llm_instance = Mock()
        mock_llm.return_value = llm_instance

        # Mock embeddings instance
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        # Mock Milvus instance
        milvus_instance = Mock()
        retriever_instance = Mock(spec=BaseRetriever)
        retriever_instance.invoke.return_value = [Document(page_content="test doc")]
        milvus_instance.as_retriever.return_value = retriever_instance
        mock_milvus.return_value = milvus_instance

        # Mock reranker instance
        reranker_instance = Mock()
        mock_reranker.return_value = reranker_instance

        # Mock metadata extractor instance
        metadata_extractor_instance = Mock()
        mock_metadata_extractor.return_value = metadata_extractor_instance

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
        main()

        # Verify that essential functions were called
        mock_load_dotenv.assert_called_once()

        # Reset environment variables
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        # Simply importing to verify no syntax errors in main function
        assert callable(main)
