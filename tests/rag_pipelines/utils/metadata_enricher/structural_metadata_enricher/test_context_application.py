"""Tests for structural metadata document context handling.

Validates that document context (source, page number, section) is correctly
applied to enriched metadata, that caching works consistently with context,
and that non-standard text inputs are gracefully handled.
"""

import pytest

from rag_pipelines.utils.metadata_enricher.structural_metadata_enricher import (
    StructuralMetadataEnricher,
)


class TestStructuralMetadataEnricherDocumentContext:
    """Test document context application and preservation.

    Verifies that document context (source, page number, section) is correctly
    applied to enriched metadata, that caching works consistently with context,
    and that non-standard text inputs are gracefully handled.
    """

    @pytest.mark.asyncio
    async def test_structural_cache_key_with_document_context(self) -> None:
        """Test cache key generation with document context."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "Test with context"}
        context = {"page_number": 5, "section": "intro"}

        result1 = await enricher.extract(chunk, context)
        result2 = await enricher.extract(chunk, context)

        # Both should have same page_number from context
        assert result1["page_number"] == 5
        assert result2["page_number"] == 5
        # Results should be identical (cached)
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_structural_enricher_context_none_branch(self) -> None:
        """Test structural enricher without document context."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "Simple test"}

        # Explicitly pass None context to trigger "none" hash branch
        result = await enricher.extract(chunk, document_context=None)

        assert "chunk_id" in result
        assert result["page_number"] == 1  # Default page number
        assert result["section_title"] == ""  # Default empty section

    @pytest.mark.asyncio
    async def test_structural_enricher_with_document_context_dict(self) -> None:
        """Test structural enricher document context application."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "Test content"}
        # Provide a document context that doesn't have "none" hash
        document_context = {"page_number": 5, "section_title": "Methods"}

        result1 = await enricher.extract(chunk, document_context)

        # Verify context is applied
        assert result1["page_number"] == 5
        assert result1["section_title"] == "Methods"

    @pytest.mark.asyncio
    async def test_structural_enricher_handles_nonstring_text(self) -> None:
        """Test structural enricher with non-string text values."""
        enricher = StructuralMetadataEnricher()

        # Test with None value in chunk
        result_none = await enricher.extract({"text": None})
        assert result_none["word_count"] == 0
        assert result_none["char_count"] == 0

        # Test with numeric value in chunk
        result_num = await enricher.extract({"text": 12345})
        assert isinstance(result_num["word_count"], int)
        assert isinstance(result_num["char_count"], int)
