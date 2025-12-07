"""Tests for FixedSchemaEnricher initialization, extraction, and field validation.

Validates that fixed schema extraction correctly transforms LLM output into
structured metadata fields (summary, keywords, questions, content_type,
header_text), that None and empty string fields are properly excluded, and
that initialization with BAML client works correctly.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from rag_pipelines.utils.metadata_enricher.fixed_schema_enricher import (
    FixedSchemaEnricher,
)


class TestFixedSchemaEnricherExtraction:
    """Test FixedSchemaEnricher initialization, extraction, and field validation.

    Validates that the fixed schema extraction correctly transforms LLM output
    into structured metadata fields, that None and empty string values are
    properly excluded, and that initialization with BAML client works.
    """

    def test_initialization_with_baml_client(self) -> None:
        """Verify FixedSchemaEnricher initializes with BAML client.

        Tests that FixedSchemaEnricher correctly stores the provided BAML
        client for use in LLM-based metadata extraction.
        """
        mock_client = Mock()
        enricher = FixedSchemaEnricher(mock_client)

        assert enricher.baml_client == mock_client

    @pytest.mark.asyncio
    async def test_extract_returns_dict(self) -> None:
        """Verify extract returns a dictionary with extracted fields.

        Tests that extract() returns a dict containing all extracted metadata
        fields (summary, keywords, questions, content_type, header_text).
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.chunk_summary = "A summary"
        mock_result.keywords = "key1, key2"
        mock_result.questions_answered = "What?"
        mock_result.content_type = "Definition"
        mock_result.header_text = "Title"
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        result = await enricher.extract("test content")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_extract_field_values(self) -> None:
        """Verify extracted field values are correctly stored in result.

        Tests that all fixed schema fields (questions_answered, chunk_summary,
        keywords, content_type, header_text) are correctly extracted and
        preserved in the returned dictionary.
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.questions_answered = "How does this work?"
        mock_result.chunk_summary = "This section explains"
        mock_result.keywords = "keyword1, keyword2, keyword3"
        mock_result.content_type = "Example"
        mock_result.header_text = "Section Title"
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        result = await enricher.extract("content")

        assert result["questions_answered"] == "How does this work?"
        assert result["chunk_summary"] == "This section explains"
        assert result["keywords"] == "keyword1, keyword2, keyword3"
        assert result["content_type"] == "Example"
        assert result["header_text"] == "Section Title"

    @pytest.mark.asyncio
    async def test_realistic_document_chunk(self) -> None:
        """Verify realistic multi-field extraction from technical document.

        Tests end-to-end extraction from a realistic ML document chunk,
        verifying that all five fixed schema fields (summary, keywords,
        questions, content_type, header) are correctly extracted.
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.questions_answered = (
            "What is transformer architecture? How does self-attention work?"
        )
        mock_result.chunk_summary = "Transformers use self-attention mechanism to process sequential data in parallel."
        mock_result.keywords = "transformer, self-attention, neural networks, NLP"
        mock_result.content_type = "Conceptual"
        mock_result.header_text = "Transformer Architecture Overview"
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        document_text = """
        Transformers are neural network architectures that use self-attention mechanisms.
        Self-attention allows the model to weight different positions in the input sequence.
        This enables parallel processing of sequences, unlike RNNs which are sequential.
        """

        result = await enricher.extract(document_text)

        assert len(result) == 5
        assert "questions_answered" in result
        assert "chunk_summary" in result
        assert "keywords" in result
        assert "content_type" in result
        assert "header_text" in result

    @pytest.mark.asyncio
    async def test_none_field_excluded(self) -> None:
        """Verify None field values are excluded from extraction results.

        Tests that when LLM extraction returns None for optional fields,
        those fields are not included in the result dict, only non-None
        fields like chunk_summary are included.
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.questions_answered = None
        mock_result.chunk_summary = "Summary"
        mock_result.keywords = None
        mock_result.content_type = None
        mock_result.header_text = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        result = await enricher.extract("content")

        assert "questions_answered" not in result
        assert result["chunk_summary"] == "Summary"
        assert "keywords" not in result
        assert "content_type" not in result
        assert "header_text" not in result

    @pytest.mark.asyncio
    async def test_empty_string_excluded(self) -> None:
        """Verify empty string field values are excluded from extraction results.

        Tests that empty string values are treated similarly to None,
        being excluded from the result dict to provide clean metadata.
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.questions_answered = ""
        mock_result.chunk_summary = "Summary"
        mock_result.keywords = ""
        mock_result.content_type = ""
        mock_result.header_text = ""
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = FixedSchemaEnricher(mock_client)

        result = await enricher.extract("content")

        assert "questions_answered" not in result
        assert result["chunk_summary"] == "Summary"
        assert "keywords" not in result
        assert "content_type" not in result
        assert "header_text" not in result
