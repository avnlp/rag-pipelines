"""Test the MedCaseReasoning indexing module."""

import os
from unittest.mock import Mock, mock_open, patch

from datasets import Dataset
from langchain_core.documents import Document

from rag_pipelines.medcasereasoning.medcasereasoning_indexing import main


class TestMedcasereasoningIndexing:
    """Test the MedCaseReasoning indexing module."""

    @patch("rag_pipelines.medcasereasoning.medcasereasoning_indexing.load_dataset")
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_indexing.ChatGroq")
    @patch(
        "rag_pipelines.medcasereasoning.medcasereasoning_indexing.HuggingFaceEmbeddings"
    )
    @patch(
        "rag_pipelines.medcasereasoning.medcasereasoning_indexing.Milvus.from_documents"
    )
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_indexing.MetadataExtractor")
    @patch(
        "rag_pipelines.medcasereasoning.medcasereasoning_indexing.UnstructuredChunker"
    )
    @patch("rag_pipelines.medcasereasoning.medcasereasoning_indexing.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "llm": {
                "model": "test_model",
                "temperature": 0.1,
                "max_tokens": 100,
                "max_retries": 3,
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
            "metadata_schema": {"properties": {"test_field": {"type": "string"}}},
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
        mock_metadata_extractor,
        mock_milvus_from_docs,
        mock_embeddings,
        mock_llm,
        mock_dataset,
    ):
        """Test the main function with mocked dependencies to ensure it executes."""
        # Mock dataset - MedCaseReasoning structure has 'title', 'text', 'pmcid', etc.
        mock_dataset.return_value = Dataset.from_dict(
            {
                "title": ["Test Title"],
                "text": ["Test article text content"],
                "pmcid": ["PMC123456"],
                "journal": ["Test Journal"],
                "article_link": ["http://example.com"],
                "publication_date": ["2023-01-01"],
            }
        )

        # Mock LLM instance
        llm_instance = Mock()
        mock_llm.return_value = llm_instance

        # Mock embeddings instance
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        # Mock chunker instance
        chunker_instance = Mock()
        chunker_instance.transform_documents.return_value = [
            Document(page_content="test content", metadata={"test_field": "value"})
        ]
        mock_chunker.return_value = chunker_instance

        # Mock metadata extractor
        metadata_extractor_instance = Mock()
        metadata_extractor_instance.transform_documents.return_value = [
            Document(page_content="test content", metadata={"test_field": "value"})
        ]
        mock_metadata_extractor.return_value = metadata_extractor_instance

        # Set up environment variables
        os.environ["MILVUS_URI"] = "test_uri"
        os.environ["MILVUS_TOKEN"] = "test_token"

        # The main function should execute without errors with all dependencies mocked

        # Call main in a fully mocked context to ensure it executes properly
        main()

        # Verify that essential functions were called
        mock_load_dotenv.assert_called_once()
        mock_open_file.assert_called_with("medcasereasoning_indexing_config.yml", "r")
        mock_yaml_load.assert_called_once()

        # Reset environment variables
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        # Simply importing to verify no syntax errors in main function
        assert callable(main)
