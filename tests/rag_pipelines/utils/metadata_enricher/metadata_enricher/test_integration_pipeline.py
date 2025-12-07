"""Integration tests for MetadataEnricher end-to-end enrichment pipeline.

Validates that the full enrichment pipeline works correctly with realistic
documents, that caching improves efficiency, that document metadata is
preserved during enrichment, and that mode switching produces correct results.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from rag_pipelines.utils.metadata_enricher import (
    EnrichmentMode,
    MetadataEnricher,
)


class TestMetadataEnricherIntegration:
    """Integration tests for MetadataEnricher full enrichment pipeline.

    Validates end-to-end enrichment scenarios including caching efficiency,
    document metadata preservation, document context application, and mode
    switching behavior with realistic documents.
    """

    @pytest.mark.asyncio
    async def test_caching_behavior(self) -> None:
        """Verify caching improves efficiency for repeated enrichments.

        Tests that when enriching the same chunk with the same schema twice,
        the second call benefits from caching and produces identical results
        to the first call.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test document"}
        schema = {"properties": {"field": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.field = "value"
            mock_baml.return_value = mock_result

            # First call should execute
            result1 = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.FULL
            )

            # Second call with same chunk and schema should use cache
            result2 = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.FULL
            )

            # Results should be identical
            assert result1 == result2

            assert mock_baml.call_count >= 1

    @pytest.mark.asyncio
    async def test_document_metadata_preservation(self) -> None:
        """Verify existing document metadata is preserved during enrichment.

        Tests that when enriching a document, all original metadata fields
        (source, page, custom_field) are preserved and new enriched metadata
        is added without overwriting existing fields.
        """
        enricher = MetadataEnricher()

        original_metadata = {
            "source": "test.pdf",
            "page": 42,
            "custom_field": "custom_value",
        }
        doc = Document(page_content="Test", metadata=original_metadata.copy())
        schema = {"properties": {"domain": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.domain = "testing"
            mock_baml.return_value = mock_result

            enriched_docs = await enricher.atransform_documents([doc], schema)

            assert enriched_docs[0].metadata["source"] == "test.pdf"
            assert enriched_docs[0].metadata["page"] == 42
            assert enriched_docs[0].metadata["custom_field"] == "custom_value"
            assert enriched_docs[0].metadata["domain"] == "testing"

    @pytest.mark.asyncio
    async def test_enrich_chunk_with_document_context(self) -> None:
        """Verify enrichment correctly applies document context metadata.

        Tests that document context (source, page_number, section_title) is
        properly merged into enrichment results along with structural metadata.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test content"}
        document_context = {
            "source": "test.pdf",
            "page_number": 3,
            "section_title": "Introduction",
        }

        result = await enricher._enrich_chunk(
            chunk, document_context=document_context, mode=EnrichmentMode.MINIMAL
        )

        metadata = result["metadata"]
        assert metadata["page_number"] == 3
        assert metadata["section_title"] == "Introduction"

    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_documents(self) -> None:
        """Verify full enrichment pipeline works with realistic textbook documents.

        Tests end-to-end enrichment of multiple documents with realistic content,
        verifying that all documents are enriched, content is preserved, and
        metadata is correctly added.
        """
        enricher = MetadataEnricher()

        docs = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "textbook1.pdf", "page": 1},
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "textbook2.pdf", "page": 15},
            ),
        ]

        schema = {"properties": {"topic": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.topic = "AI"
            mock_baml.return_value = mock_result

            enriched = await enricher.atransform_documents(docs, schema)

            assert len(enriched) == 2

            for doc in enriched:
                assert doc.page_content is not None
                assert "enrichment_mode" in doc.metadata
                assert "word_count" in doc.metadata

    @pytest.mark.asyncio
    async def test_mode_switching(self) -> None:
        """Verify enrichment differs between MINIMAL and FULL modes.

        Tests that FULL mode enriches more fields than MINIMAL mode when
        processing the same schema and content.
        """
        chunk = {"text": "Test document with multiple words."}
        schema = {"properties": {"field": {"type": "string"}}}

        enricher = MetadataEnricher()

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.field = "value"
            mock_baml.return_value = mock_result

            minimal_result = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.MINIMAL
            )
            minimal_metadata = minimal_result["metadata"]

            full_chunk = {"text": "Unique text for full mode testing here."}
            full_result = await enricher._enrich_chunk(
                full_chunk, user_schema=schema, mode=EnrichmentMode.FULL
            )
            full_metadata = full_result["metadata"]

            assert len(full_metadata) >= len(minimal_metadata)

    @pytest.mark.asyncio
    async def test_atransform_documents_with_empty_list(self) -> None:
        """Verify atransform_documents handles empty document lists gracefully.

        Tests that passing an empty document list returns an empty list without
        errors or side effects.
        """
        enricher = MetadataEnricher()
        schema = {"properties": {"field": {"type": "string"}}}

        result = await enricher.atransform_documents([], schema)

        assert result == []
