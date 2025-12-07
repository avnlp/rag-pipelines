"""Test the MetaMedQA indexing module."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, mock_open, patch

from datasets import Dataset
from langchain_core.documents import Document

from rag_pipelines.metamedqa.metamedqa_indexing import main


class TestMetaMedQAIndexing:
    """Test the MetaMedQA indexing module."""

    @patch("rag_pipelines.metamedqa.metamedqa_indexing.load_dataset")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.Milvus.from_documents")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.MetadataEnricher")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.UnstructuredChunker")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "metadata_schema": {
                "properties": {
                    "medical_domain": {
                        "type": "string",
                        "description": "Medical domain",
                    },
                    "specialty": {"type": "string", "description": "Medical specialty"},
                }
            },
            "embedding": {"model_name": "test-embedding-model"},
            "dataset": {
                "path": "test/path",
                "split_name": "test_split",
                "split": "test",
            },
            "chunking": {
                "chunking_strategy": "test",
                "max_characters": 1000,
                "new_after_n_chars": 500,
                "overlap": 100,
                "overlap_all": True,
                "combine_text_under_n_chars": 200,
                "include_orig_elements": True,
                "multipage_sections": True,
            },
            "metadata_enricher": {
                "mode": "full",
                "batch_size": 10,
            },
            "metadata_defaults": {"default_value": "default"},
            "vectorstore": {
                "vector_fields": ["vector"],
                "collection_name": "test_collection",
                "consistency_level": "Strong",
                "drop_old": False,
            },
        },
    )
    def test_main_with_mocked_dependencies(
        self,
        mock_yaml_load,
        mock_open_file,
        mock_load_dotenv,
        mock_chunker,
        mock_metadata_enricher,
        mock_milvus_from_docs,
        mock_embeddings,
        mock_dataset,
    ):
        """Test the main function with mocked dependencies to ensure it executes."""
        mock_dataset.return_value = Dataset.from_dict(
            {"context": [{"contexts": ["Context 1 text", "Context 2 text"]}]}
        )

        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        chunker_instance = Mock()
        chunker_instance.transform_documents.return_value = [
            Document(page_content="test content", metadata={"test_field": "value"})
        ]
        mock_chunker.return_value = chunker_instance

        metadata_enricher_instance = Mock()
        metadata_enricher_instance.atransform_documents = AsyncMock(
            return_value=[
                Document(page_content="test content", metadata={"test_field": "value"})
            ]
        )
        mock_metadata_enricher.return_value = metadata_enricher_instance

        os.environ["MILVUS_URI"] = "test_uri"
        os.environ["MILVUS_TOKEN"] = "test_token"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()

        mock_load_dotenv.assert_called_once()
        mock_open_file.assert_called_with("metamedqa_indexing_config.yml", "r")
        mock_yaml_load.assert_called_once()

        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        assert callable(main)

    @patch("rag_pipelines.metamedqa.metamedqa_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.Milvus.from_documents")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.MetadataEnricher")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.UnstructuredChunker")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.load_dataset")
    @patch("rag_pipelines.metamedqa.metamedqa_indexing.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "metadata_schema": {"properties": {"test_field": {"type": "string"}}},
            "embedding": {"model_name": "test-embedding-model"},
            "dataset": {
                "path": "test/path",
                "split_name": "test_split",
                "split": "test",
            },
            "chunking": {
                "chunking_strategy": "test",
                "max_characters": 1000,
                "new_after_n_chars": 500,
                "overlap": 100,
                "overlap_all": True,
                "combine_text_under_n_chars": 200,
                "include_orig_elements": True,
                "multipage_sections": True,
            },
            "metadata_enricher": {"mode": "full", "batch_size": 10},
            "metadata_defaults": {"default_value": "MISSING"},
            "vectorstore": {
                "vector_fields": ["vector"],
                "collection_name": "test_collection",
                "consistency_level": "Strong",
                "drop_old": False,
            },
        },
    )
    def test_applies_metadata_defaults(
        self,
        mock_yaml_load,
        mock_open_file,
        mock_load_dotenv,
        mock_load_dataset,
        mock_chunker,
        mock_metadata_enricher,
        mock_milvus_from_docs,
        mock_embeddings,
    ):
        """Verify missing metadata fields receive default values when configured.

        Tests that lines ~107-115 apply default_value to all documents
        for schema fields that are missing from the enriched metadata.
        """
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        dataset = Dataset.from_dict({"context": [{"contexts": ["Context 1"]}]})
        mock_load_dataset.return_value = dataset

        chunker_instance = Mock()
        chunker_instance.transform_documents.return_value = [
            Document(page_content="test content", metadata={})
        ]
        mock_chunker.return_value = chunker_instance

        metadata_enricher_instance = Mock()
        metadata_enricher_instance.atransform_documents = AsyncMock(
            return_value=[Document(page_content="test content", metadata={})]
        )
        mock_metadata_enricher.return_value = metadata_enricher_instance

        os.environ["MILVUS_URI"] = "test_uri"
        os.environ["MILVUS_TOKEN"] = "test_token"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()

        mock_milvus_from_docs.assert_called_once()
        call_args = mock_milvus_from_docs.call_args
        docs = call_args.kwargs["documents"]
        assert any("MISSING" in str(doc.metadata.values()) for doc in docs)

        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)
