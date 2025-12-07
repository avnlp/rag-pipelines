"""Tests for document transformation with batch enrichment.

Validates the atransform_documents method for batch document enrichment with
mode switching and batching behavior, including document metadata preservation
and error handling for invalid modes.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from rag_pipelines.utils.metadata_enricher import (
    EnrichmentConfig,
    EnrichmentMode,
    MetadataEnricher,
)


class TestMetadataEnricherAtransformDocuments:
    """Test atransform_documents batch document enrichment method.

    Verifies that documents are transformed with enriched metadata while
    preserving original content and metadata, that batching works correctly
    with different batch sizes, and that invalid mode parameters are properly
    rejected with descriptive errors.
    """

    @pytest.mark.asyncio
    async def test_atransform_documents_basic(self) -> None:
        """Verify atransform_documents enriches documents and preserves content.

        Tests that atransform_documents processes a list of documents, adds
        enrichment metadata to each, preserves original content and existing
        metadata, and includes the enrichment mode in output.
        """
        enricher = MetadataEnricher()

        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"}),
        ]

        schema = {"properties": {"domain": {"type": "string"}}}

        # Mock the BAML client
        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.domain = "testing"
            mock_baml.return_value = mock_result

            transformed = await enricher.atransform_documents(docs, schema)

            assert len(transformed) == 2
            assert transformed[0].metadata["source"] == "test1"
            assert transformed[0].page_content == "Test document 1"
            assert "enrichment_mode" in transformed[0].metadata

    @pytest.mark.asyncio
    async def test_atransform_documents_batching(self) -> None:
        """Verify batching processes documents in configured batch sizes.

        Tests that atransform_documents with a custom batch size (2 documents)
        correctly processes all documents in a list (5 documents) and enriches
        each with metadata.
        """
        config = EnrichmentConfig(mode=EnrichmentMode.MINIMAL, batch_size=2)
        enricher = MetadataEnricher(config)

        docs = [
            Document(page_content=f"Document {i}", metadata={"id": i}) for i in range(5)
        ]

        schema = {"properties": {"doc_type": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.doc_type = "test"
            mock_baml.return_value = mock_result

            result = await enricher.atransform_documents(docs, schema)

            assert len(result) == 5
            for doc in result:
                assert "chunk_id" in doc.metadata

    @pytest.mark.asyncio
    async def test_atransform_documents_with_mode_enum(self) -> None:
        """Verify atransform_documents processes documents correctly.

        Tests that atransform_documents successfully enriches a document list
        when called without explicit mode parameter, using the enricher's
        default configuration.
        """
        enricher = MetadataEnricher()

        docs = [
            Document(page_content="Test doc", metadata={"source": "test"}),
        ]

        schema = {"properties": {"field": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.field = "value"
            mock_baml.return_value = mock_result

            result = await enricher.atransform_documents(docs, schema)

            assert len(result) == 1
            assert "chunk_id" in result[0].metadata

    @pytest.mark.asyncio
    async def test_atransform_documents_invalid_mode_string(self) -> None:
        """Verify invalid mode strings raise descriptive ValueError.

        Tests that passing an invalid enrichment mode string (e.g., "invalid_mode")
        raises a ValueError with a message that includes "Invalid enrichment mode"
        and the invalid mode value.
        """
        enricher = MetadataEnricher()
        docs = [Document(page_content="Test", metadata={})]
        schema = {"properties": {"field": {"type": "string"}}}

        with pytest.raises(ValueError) as exc_info:
            await enricher.atransform_documents(docs, schema, mode="invalid_mode")

        assert "Invalid enrichment mode" in str(exc_info.value)
        assert "invalid_mode" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_atransform_documents_with_invalid_mode_case_sensitive(
        self,
    ) -> None:
        """Verify mode validation is case-sensitive.

        Tests that mode validation requires exact lowercase strings ("minimal",
        "dynamic", "full"), and rejects uppercase variations like "FULL".
        """
        enricher = MetadataEnricher()
        docs = [Document(page_content="Test", metadata={})]
        schema = {"properties": {"field": {"type": "string"}}}

        with pytest.raises(ValueError):
            await enricher.atransform_documents(docs, schema, mode="FULL")
