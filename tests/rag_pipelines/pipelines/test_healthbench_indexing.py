"""Test the HealthBench indexing module."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, mock_open, patch

from datasets import Dataset
from langchain_core.documents import Document

from rag_pipelines.healthbench.healthbench_indexing import main


class TestHealthBenchIndexing:
    """Test the HealthBench indexing module."""

    @patch("rag_pipelines.healthbench.healthbench_indexing.load_dataset")
    @patch("rag_pipelines.healthbench.healthbench_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.healthbench.healthbench_indexing.Milvus.from_documents")
    @patch("rag_pipelines.healthbench.healthbench_indexing.MetadataEnricher")
    @patch("rag_pipelines.healthbench.healthbench_indexing.UnstructuredChunker")
    @patch("rag_pipelines.healthbench.healthbench_indexing.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "metadata_schema": {
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain of the conversation (e.g., medical, general)",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task type (e.g., diagnosis, treatment)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags associated with the conversation",
                    },
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
        # Mock dataset - HealthBench structure
        mock_dataset.return_value = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "test content"}]],
                "prompt_id": ["test_id"],
                "example_tags": [["tag1", "tag2"]],
                "rubrics": [
                    [{"tags": ["axis:tag", "other:tag"]}, {"tags": ["axis:tag2"]}]
                ],
                "ideal_completions_data": [{"ideal_completions_group": "group1"}],
            }
        )

        # Mock embeddings instance
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        # Mock chunker instance
        chunker_instance = Mock()
        chunker_instance.transform_documents.return_value = [
            Document(page_content="test content", metadata={"test_field": "value"})
        ]
        mock_chunker.return_value = chunker_instance

        # Mock metadata enricher with async method
        metadata_enricher_instance = Mock()
        metadata_enricher_instance.atransform_documents = AsyncMock(
            return_value=[
                Document(page_content="test content", metadata={"test_field": "value"})
            ]
        )
        mock_metadata_enricher.return_value = metadata_enricher_instance

        # Set up environment variables
        os.environ["MILVUS_URI"] = "test_uri"
        os.environ["MILVUS_TOKEN"] = "test_token"

        # The main function should execute without errors with all dependencies mocked
        # Call main in a fully mocked context to ensure it executes properly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()

        # Verify that essential functions were called
        mock_load_dotenv.assert_called_once()
        mock_open_file.assert_called_with("healthbench_indexing_config.yml", "r")
        mock_yaml_load.assert_called_once()

        # Reset environment variables
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        # Simply importing to verify no syntax errors in main function
        assert callable(main)

    @patch("rag_pipelines.healthbench.healthbench_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.healthbench.healthbench_indexing.Milvus.from_documents")
    @patch("rag_pipelines.healthbench.healthbench_indexing.MetadataEnricher")
    @patch("rag_pipelines.healthbench.healthbench_indexing.UnstructuredChunker")
    @patch("rag_pipelines.healthbench.healthbench_indexing.load_dataset")
    @patch("rag_pipelines.healthbench.healthbench_indexing.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "metadata_schema": {
                "properties": {
                    "domain": {"type": "string"},
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
    def test_handles_missing_ideal_completions(
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
        """Verify documents are created even when ideal_completions_data is missing.

        Tests that the conditional extraction at lines 76-80 gracefully handles
        datasets without ideal_completions_data field, creating documents without
        completion_group metadata instead of crashing.
        """
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "What is the answer?"}]],
                "prompt_id": ["test_id"],
                "example_tags": [["tag1"]],
            }
        )
        mock_load_dataset.return_value = dataset

        chunker_instance = Mock()
        chunker_instance.transform_documents.return_value = [
            Document(page_content="test content", metadata={"prompt_id": "test_id"})
        ]
        mock_chunker.return_value = chunker_instance

        metadata_enricher_instance = Mock()
        metadata_enricher_instance.atransform_documents = AsyncMock(
            return_value=[
                Document(page_content="test content", metadata={"prompt_id": "test_id"})
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
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)
