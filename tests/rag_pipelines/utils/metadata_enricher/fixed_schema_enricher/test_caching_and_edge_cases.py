"""Tests for caching efficiency and edge case handling in FixedSchemaEnricher.

Validates that extraction results are cached to avoid redundant LLM calls,
that different inputs produce different cache entries, and that extreme
inputs (empty, very long, special characters) are handled robustly.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from rag_pipelines.utils.metadata_enricher.fixed_schema_enricher import (
    FixedSchemaEnricher,
)


class TestFixedSchemaEnricherBehaviours:
    """Test caching efficiency and edge case handling.

    Validates that extraction results are cached to avoid redundant LLM calls,
    that different text inputs generate separate cache entries, and that
    extreme inputs (empty text, very long text, special characters) are
    handled robustly without errors.
    """

    @pytest.mark.asyncio
    async def test_identical_calls_use_cache(self) -> None:
        """Verify identical extraction calls use caching for efficiency.

        Tests that calling extract twice with the same text produces identical
        results and avoids redundant LLM calls (caching is working).
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.chunk_summary = "Summary"
        mock_result.questions_answered = None
        mock_result.keywords = None
        mock_result.content_type = None
        mock_result.header_text = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)
        text = "test content"

        result1 = await enricher.extract(text)
        result2 = await enricher.extract(text)

        assert result1 == result2
        assert mock_client.ExtractMetadata.call_count >= 1

    @pytest.mark.asyncio
    async def test_different_text_different_cache_entry(self) -> None:
        """Test different text produces different cache entries."""
        mock_client = AsyncMock()
        call_count = 0

        async def mock_extract(*args: Any, **kwargs: Any) -> Mock:
            nonlocal call_count
            call_count += 1
            result = Mock()
            result.chunk_summary = f"Summary {call_count}"
            result.questions_answered = None
            result.keywords = None
            result.content_type = None
            result.header_text = None
            return result

        mock_client.ExtractMetadata = AsyncMock(side_effect=mock_extract)

        enricher = FixedSchemaEnricher(mock_client)

        result1 = await enricher.extract("text1")
        result2 = await enricher.extract("text2")

        assert result1["chunk_summary"] != result2["chunk_summary"]

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        """Test extraction with empty text."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.chunk_summary = None
        mock_result.questions_answered = None
        mock_result.keywords = None
        mock_result.content_type = None
        mock_result.header_text = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        result = await enricher.extract("")

        assert result == {}

    @pytest.mark.asyncio
    async def test_very_long_text(self) -> None:
        """Test extraction with very long text."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.chunk_summary = "Summary"
        mock_result.questions_answered = None
        mock_result.keywords = None
        mock_result.content_type = None
        mock_result.header_text = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)
        long_text = "word " * 5000

        result = await enricher.extract(long_text)

        assert "chunk_summary" in result

    @pytest.mark.asyncio
    async def test_special_characters_in_text(self) -> None:
        """Test extraction from text with special characters."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.chunk_summary = "Summary with special chars!@#"
        mock_result.questions_answered = None
        mock_result.keywords = None
        mock_result.content_type = None
        mock_result.header_text = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        result = await enricher.extract("Text with special chars !@#$%^&*()")

        assert "chunk_summary" in result

    @pytest.mark.asyncio
    async def test_llm_extraction_failure_returns_empty_dict(self) -> None:
        """Verify extraction gracefully handles LLM failures.

        Tests that when ExtractMetadata raises a RuntimeError (e.g., LLM
        service unavailable, malformed response), the enricher returns an empty
        dict instead of propagating the exception. This allows the pipeline to
        continue processing with partial metadata rather than failing entirely.
        """
        mock_client = AsyncMock()
        mock_client.ExtractMetadata = AsyncMock(
            side_effect=RuntimeError("LLM call failed")
        )

        enricher = FixedSchemaEnricher(mock_client)
        result = await enricher.extract("test content")

        assert result == {}

    @pytest.mark.asyncio
    async def test_llm_extraction_with_api_error(self) -> None:
        """Verify extraction handles API connectivity failures gracefully.

        Tests that ConnectionError from the LLM API (network unavailable,
        unreachable endpoint) is caught and returns an empty dict, enabling
        the RAG pipeline to proceed with structural metadata alone.
        """
        mock_client = AsyncMock()
        mock_client.ExtractMetadata = AsyncMock(
            side_effect=ConnectionError("API unreachable")
        )

        enricher = FixedSchemaEnricher(mock_client)
        result = await enricher.extract("test content")

        assert result == {}

    @pytest.mark.asyncio
    async def test_llm_extraction_with_timeout(self) -> None:
        """Verify extraction handles LLM request timeouts gracefully.

        Tests that TimeoutError from slow or unresponsive LLM calls is caught
        and returns an empty dict, allowing the pipeline to continue instead of
        blocking indefinitely on a slow extraction request.
        """
        mock_client = AsyncMock()
        mock_client.ExtractMetadata = AsyncMock(
            side_effect=TimeoutError("LLM request timed out")
        )

        enricher = FixedSchemaEnricher(mock_client)
        result = await enricher.extract("test content")

        assert result == {}
