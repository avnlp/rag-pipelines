"""Test the FinanceBench indexing module."""

import os
from unittest.mock import Mock, mock_open, patch

from langchain_core.documents import Document

from rag_pipelines.financebench.financebench_indexing import main


class TestFinancebenchIndexing:
    """Test the FinanceBench indexing module."""

    @patch("rag_pipelines.financebench.financebench_indexing.fsspec.filesystem")
    @patch("rag_pipelines.financebench.financebench_indexing.load_dotenv")
    @patch("rag_pipelines.financebench.financebench_indexing.ChatGroq")
    @patch("rag_pipelines.financebench.financebench_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.financebench.financebench_indexing.Milvus.from_documents")
    @patch("rag_pipelines.financebench.financebench_indexing.MetadataExtractor")
    @patch("rag_pipelines.financebench.financebench_indexing.UnstructuredChunker")
    @patch(
        "rag_pipelines.financebench.financebench_indexing.UnstructuredDocumentLoader"
    )
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
                "org": "test-org",
                "repo": "test-repo",
            },
            "unstructured_loader": {
                "strategy": "test",
                "mode": "test",
                "include_page_breaks": True,
                "infer_table_structure": True,
                "ocr_languages": ["eng"],
                "languages": ["eng"],
                "extract_images_in_pdf": False,
                "extract_forms": True,
                "form_extraction_skip_tables": True,
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
        mock_loader,
        mock_chunker,
        mock_metadata_extractor,
        mock_milvus_from_docs,
        mock_embeddings,
        mock_llm,
        mock_load_dotenv,
        mock_filesystem,
    ):
        """Test the main function with mocked dependencies to ensure it executes."""
        # Mock filesystem instance
        fs_instance = Mock()
        fs_instance.ls.return_value = ["test1.pdf", "test2.pdf"]
        fs_instance.get = Mock()
        mock_filesystem.return_value = fs_instance

        # Mock LLM instance
        llm_instance = Mock()
        mock_llm.return_value = llm_instance

        # Mock embeddings instance
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        # Mock document loader instance
        loader_instance = Mock()
        loader_instance.transform_documents.return_value = [
            Document(
                page_content="test document content", metadata={"test_field": "value"}
            )
        ]
        mock_loader.return_value = loader_instance

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
        mock_open_file.assert_called_with("financebench_indexing_config.yml", "r")
        mock_yaml_load.assert_called_once()
        mock_filesystem.assert_called_once_with(
            "github", org="test-org", repo="test-repo"
        )

        # Reset environment variables
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        # Simply importing to verify no syntax errors in main function
        assert callable(main)
