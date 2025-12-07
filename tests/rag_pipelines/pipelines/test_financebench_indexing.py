"""Test the FinanceBench indexing module."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, mock_open, patch

from langchain_core.documents import Document

from rag_pipelines.financebench.financebench_indexing import main


class TestFinanceBenchIndexing:
    """Test the FinanceBench indexing module."""

    @patch("rag_pipelines.financebench.financebench_indexing.fsspec")
    @patch("rag_pipelines.financebench.financebench_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.financebench.financebench_indexing.Milvus.from_documents")
    @patch("rag_pipelines.financebench.financebench_indexing.MetadataEnricher")
    @patch("rag_pipelines.financebench.financebench_indexing.UnstructuredChunker")
    @patch(
        "rag_pipelines.financebench.financebench_indexing.UnstructuredDocumentLoader"
    )
    @patch("rag_pipelines.financebench.financebench_indexing.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data="test_config_content")
    @patch(
        "yaml.safe_load",
        return_value={
            "metadata_schema": {
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain of the financial document",
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Type of document",
                    },
                }
            },
            "embedding": {"model_name": "test-embedding-model"},
            "dataset": {
                "org": "test_org",
                "repo": "test_repo",
            },
            "unstructured_loader": {
                "strategy": "test",
                "mode": "test",
                "include_page_breaks": True,
                "infer_table_structure": True,
                "ocr_languages": [],
                "languages": [],
                "extract_images_in_pdf": False,
                "extract_forms": False,
                "form_extraction_skip_tables": False,
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
        mock_loader,
        mock_chunker,
        mock_metadata_enricher,
        mock_milvus_from_docs,
        mock_embeddings,
        mock_fsspec,
    ):
        """Test the main function with mocked dependencies to ensure it executes."""
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        fs_mock = Mock()
        fs_mock.ls.return_value = ["test.pdf"]
        mock_fsspec.filesystem.return_value = fs_mock

        loader_instance = Mock()
        loader_instance.transform_documents.return_value = [
            Document(page_content="test content")
        ]
        mock_loader.return_value = loader_instance

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
        mock_open_file.assert_called_with("financebench_indexing_config.yml", "r")
        mock_yaml_load.assert_called_once()

        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    def test_main_execution_structure(self):
        """Test to verify that main function has proper structure and imports."""
        assert callable(main)

    @patch("rag_pipelines.financebench.financebench_indexing.fsspec.filesystem")
    @patch("rag_pipelines.financebench.financebench_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.financebench.financebench_indexing.Milvus.from_documents")
    @patch("rag_pipelines.financebench.financebench_indexing.MetadataEnricher")
    @patch("rag_pipelines.financebench.financebench_indexing.UnstructuredChunker")
    @patch(
        "rag_pipelines.financebench.financebench_indexing.UnstructuredDocumentLoader"
    )
    @patch("rag_pipelines.financebench.financebench_indexing.load_dotenv")
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
            "dataset": {"org": "test_org", "repo": "test_repo"},
            "unstructured_loader": {
                "strategy": "test",
                "mode": "test",
                "include_page_breaks": True,
                "infer_table_structure": True,
                "ocr_languages": [],
                "languages": [],
                "extract_images_in_pdf": False,
                "extract_forms": False,
                "form_extraction_skip_tables": False,
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
            "metadata_defaults": {"default_value": "default"},
            "vectorstore": {
                "vector_fields": ["vector"],
                "collection_name": "test_collection",
                "consistency_level": "Strong",
                "drop_old": False,
            },
        },
    )
    def test_handles_pdf_download_failure(
        self,
        mock_yaml_load,
        mock_open_file,
        mock_load_dotenv,
        mock_loader,
        mock_chunker,
        mock_metadata_enricher,
        mock_milvus_from_docs,
        mock_embeddings,
        mock_fsspec,
    ):
        """Verify indexing continues gracefully when PDF download fails.

        Tests that the exception handler at lines 60-61 gracefully handles
        failed PDF downloads from fsspec, allowing the pipeline to process
        successfully downloaded files instead of crashing entirely.
        """
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        fs_mock = Mock()
        fs_mock.ls.return_value = ["file1.pdf", "file2.pdf"]
        fs_mock.get.side_effect = [Exception("Download failed"), None]
        mock_fsspec.return_value = fs_mock

        loader_instance = Mock()
        loader_instance.transform_documents.return_value = [
            Document(page_content="test content")
        ]
        mock_loader.return_value = loader_instance

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

        assert mock_fsspec.call_count >= 1
        os.environ.pop("MILVUS_URI", None)
        os.environ.pop("MILVUS_TOKEN", None)

    @patch("rag_pipelines.financebench.financebench_indexing.HuggingFaceEmbeddings")
    @patch("rag_pipelines.financebench.financebench_indexing.Milvus.from_documents")
    @patch("rag_pipelines.financebench.financebench_indexing.MetadataEnricher")
    @patch("rag_pipelines.financebench.financebench_indexing.UnstructuredChunker")
    @patch(
        "rag_pipelines.financebench.financebench_indexing.UnstructuredDocumentLoader"
    )
    @patch("rag_pipelines.financebench.financebench_indexing.fsspec.filesystem")
    @patch("rag_pipelines.financebench.financebench_indexing.load_dotenv")
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
            "dataset": {"org": "test_org", "repo": "test_repo"},
            "unstructured_loader": {
                "strategy": "test",
                "mode": "test",
                "include_page_breaks": True,
                "infer_table_structure": True,
                "ocr_languages": [],
                "languages": [],
                "extract_images_in_pdf": False,
                "extract_forms": False,
                "form_extraction_skip_tables": False,
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
        mock_fsspec,
        mock_loader,
        mock_chunker,
        mock_metadata_enricher,
        mock_milvus_from_docs,
        mock_embeddings,
    ):
        """Verify missing metadata fields receive default values.

        Tests that when metadata_defaults.default_value is configured,
        all documents have missing schema fields populated with the default
        value before indexing to ensure consistent metadata coverage.
        """
        embeddings_instance = Mock()
        mock_embeddings.return_value = embeddings_instance

        fs_mock = Mock()
        fs_mock.ls.return_value = ["test.pdf"]
        mock_fsspec.return_value = fs_mock

        loader_instance = Mock()
        loader_instance.transform_documents.return_value = [
            Document(page_content="test content")
        ]
        mock_loader.return_value = loader_instance

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
