"""Tests for structural metadata extraction and language detection.

Validates that structural metadata (word count, character count, language,
content hash) is accurately computed from text input, that language detection
correctly identifies document language, and that the enricher handles None/empty
text gracefully.
"""

import pytest

from rag_pipelines.utils.metadata_enricher.structural_metadata_enricher import (
    StructuralMetadataEnricher,
)


class TestStructuralMetadataEnricherExtraction:
    """Test structural metadata extraction and language detection.

    Validates that word count, character count, language identification, and
    content hashing are correctly computed, that document context is properly
    applied, and edge cases like None/empty text are handled gracefully.
    """

    @pytest.mark.asyncio
    async def test_extract_structural_metadata(self) -> None:
        """Verify structural metadata extraction with document context.

        Tests that extract correctly computes word count (8 words), language
        (English), character count (37 chars), content hash, and preserves
        document context fields like page_number.
        """
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "This is a test chunk with some words."}
        document_context = {"source": "test.pdf", "page_number": 1}

        result = await enricher.extract(chunk, document_context)

        assert "chunk_id" in result
        assert "content_hash" in result
        assert result["word_count"] == 8
        assert result["language"] == "en"
        assert result["char_count"] == 37
        assert result["page_number"] == 1

    @pytest.mark.asyncio
    async def test_structural_without_document_context(self) -> None:
        """Verify structural extraction works without document context.

        Tests that extract can process chunks without context, using default
        values (e.g., page_number defaults to 1).
        """
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "Simple test"}

        result = await enricher.extract(chunk)

        assert "chunk_id" in result
        assert result["word_count"] == 2
        assert result["page_number"] == 1

    @pytest.mark.asyncio
    async def test_structural_with_none_text(self) -> None:
        """Verify structural extraction handles None text gracefully.

        Tests that when text is None, word_count and char_count are both 0
        and extraction still succeeds with a chunk_id.
        """
        enricher = StructuralMetadataEnricher()
        chunk = {"text": None}

        result = await enricher.extract(chunk)

        assert "chunk_id" in result
        assert result["word_count"] == 0
        assert result["char_count"] == 0

    @pytest.mark.asyncio
    async def test_language_detection_english(self) -> None:
        """Verify language detection correctly identifies English text.

        Tests that the language detection component correctly identifies
        English text and returns "en" language code.
        """
        enricher = StructuralMetadataEnricher()
        result = await enricher.extract({"text": "Hello world"}, None)
        assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_language_detection_with_short_text_graceful_fallback(self) -> None:
        """Verify language detection gracefully handles very short text.

        Tests that language detection returns a valid string language code
        even with very short text ("a") that might be difficult to classify.
        """
        enricher = StructuralMetadataEnricher()
        result = await enricher.extract({"text": "a"})

        assert "chunk_id" in result
        assert isinstance(result["language"], str)
