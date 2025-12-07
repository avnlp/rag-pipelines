"""Tests for query metadata extraction and filter expression generation.

Validates that query metadata is correctly extracted from query text,
transformed into filter expressions for vector DB filtering, that different
field types (string, number, boolean, object) are handled appropriately,
and that structural fields are excluded from filter expressions.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rag_pipelines.utils.metadata_enricher import MetadataEnricher


class TestMetadataEnricherExtractQueryMetadata:
    """Test extract_query_metadata query enrichment and filter generation method.

    Validates that query metadata is correctly extracted from query text,
    transformed into filter expressions for vector DB filtering, that different
    field types (string, number, boolean, object) are handled appropriately,
    and that structural fields are excluded from filter expressions.
    """

    @pytest.mark.asyncio
    async def test_extract_query_metadata_returns_tuple(self) -> None:
        """Verify extract_query_metadata returns (metadata dict, filter expression).

        Tests that extract_query_metadata returns a tuple containing metadata
        dict with structural fields (word_count, chunk_id, etc.) and a filter
        expression (string or None) for query filtering.
        """
        enricher = MetadataEnricher()
        query = "What is machine learning?"
        schema = {"properties": {"topic": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.topic = "machine_learning"
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            assert isinstance(metadata, dict)
            assert "word_count" in metadata
            assert filter_expr is None or isinstance(filter_expr, str)

    @pytest.mark.asyncio
    async def test_extract_query_metadata_with_multiple_types(self) -> None:
        """Verify filter expression generation handles multiple field types.

        Tests that extract_query_metadata correctly processes schemas with
        different field types (string, integer, array) and generates appropriate
        filter expressions for retrieval.
        """
        enricher = MetadataEnricher()
        query = "Apple 2023 Python JavaScript"
        schema = {
            "properties": {
                "company": {"type": "string"},
                "year": {"type": "integer"},
                "languages": {"type": "array"},
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.company = "Apple"
            mock_result.year = 2023
            mock_result.languages = ["Python", "JavaScript"]
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            assert metadata["company"] == "Apple"
            assert metadata["year"] == 2023
            assert "like" in filter_expr or filter_expr is None
            assert "year" in filter_expr or filter_expr is None

    @pytest.mark.asyncio
    async def test_extract_query_metadata_with_boolean_type(self) -> None:
        """Test filter expression with boolean type."""
        enricher = MetadataEnricher()
        query = "Test with boolean field"
        schema = {
            "properties": {
                "is_verified": {"type": "boolean"},
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.is_verified = True
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            assert metadata["is_verified"] is True
            # Boolean should generate equality filter
            assert "is_verified == True" in filter_expr

    @pytest.mark.asyncio
    async def test_extract_query_metadata_filters_structural_fields(self) -> None:
        """Test that structural fields are filtered from filter expressions."""
        enricher = MetadataEnricher()
        query = "Test for filtering"
        schema = {
            "properties": {
                "domain": {"type": "string"},
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.domain = "testing"
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            # Should have structural fields in metadata
            assert "word_count" in metadata
            assert "chunk_id" in metadata
            assert "language" in metadata

            # But structural fields should NOT be in filter expression
            assert "chunk_id" not in filter_expr if filter_expr else True
            assert "word_count" not in filter_expr if filter_expr else True

    @pytest.mark.asyncio
    async def test_extract_query_metadata_with_float_type(self) -> None:
        """Test filter expression with float/number type."""
        enricher = MetadataEnricher()
        query = "Confidence score 0.95"
        schema = {
            "properties": {
                "confidence_score": {"type": "number"},
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.confidence_score = 0.95
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            assert metadata["confidence_score"] == 0.95
            # Float type should generate equality filter
            assert "confidence_score == 0.95" in filter_expr

    @pytest.mark.asyncio
    async def test_extract_query_metadata_with_object_type(self) -> None:
        """Test filter expression with object type."""
        enricher = MetadataEnricher()
        query = "Document contains metadata"
        schema = {
            "properties": {
                "tags": {"type": "object"},
            }
        }

        # Mock the dynamic enricher's extract to return object field

        original_extract = enricher.dynamic_enricher.extract

        async def mock_extract(
            text: str, user_schema: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Return the object field directly
            return {"tags": {"type": "article", "level": "beginner"}}

        enricher.dynamic_enricher.extract = mock_extract

        metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

        # The object should be in metadata
        assert "tags" in metadata
        assert metadata["tags"] == {"type": "article", "level": "beginner"}
        # Should use JSON_CONTAINS_ANY with keys for object type
        assert "JSON_CONTAINS_ANY" in filter_expr if filter_expr else False

        enricher.dynamic_enricher.extract = original_extract

    @pytest.mark.asyncio
    async def test_extract_query_metadata_with_object_empty_dict(self) -> None:
        """Test filter expression with empty object value."""
        enricher = MetadataEnricher()
        query = "Test query"
        schema = {
            "properties": {
                "metadata": {"type": "object"},
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.metadata = {}  # Empty object
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            # Empty object should not generate filter (no keys)
            if filter_expr:
                # If filter exists, it should not contain the empty metadata
                assert (
                    "metadata" not in filter_expr
                    or "JSON_CONTAINS_ANY" not in filter_expr
                )

    @pytest.mark.asyncio
    async def test_extract_query_metadata_with_null_extracted_values(self) -> None:
        """Test that None values are filtered from filter expressions."""
        enricher = MetadataEnricher()
        query = "Test query"
        schema = {
            "properties": {
                "company": {"type": "string"},
                "year": {"type": "integer"},
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.company = None  # Not extracted
            mock_result.year = 2023  # Extracted
            mock_baml.return_value = mock_result

            metadata, filter_expr = await enricher.extract_query_metadata(query, schema)

            # None values should be filtered out
            assert "company" not in metadata
            assert metadata["year"] == 2023
            # Filter should only have year condition
            assert "year == 2023" in filter_expr
            assert "company" not in filter_expr

    @pytest.mark.asyncio
    async def test_extract_query_metadata_basic_test_query(self) -> None:
        """Test extract_query_metadata with basic test query and empty schema.

        This test validates that extract_query_metadata can handle a simple
        query with an empty schema and returns proper tuple structure.
        """
        enricher = MetadataEnricher()
        metadata, filter_expr = await enricher.extract_query_metadata("test query", {})

        assert isinstance(metadata, dict)
        assert "word_count" in metadata
