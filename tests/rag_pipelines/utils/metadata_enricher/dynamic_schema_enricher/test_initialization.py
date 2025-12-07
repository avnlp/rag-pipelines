"""Tests for DynamicSchemaEnricher initialization and cache key consistency.

Validates that the enricher correctly initializes with BAML client and that
realistic multi-field document schemas are properly handled, with cache key
consistency verification.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from rag_pipelines.utils.metadata_enricher.dynamic_schema_enricher import (
    DynamicSchemaEnricher,
    _make_schema_key,
)


class TestDynamicSchemaEnricherInitialization:
    """Test initialization of DynamicSchemaEnricher.

    Validates that DynamicSchemaEnricher correctly initializes with BAML client
    and handles realistic complex schemas with multiple field types.
    """

    def test_initialization_with_baml_client(self) -> None:
        """Verify DynamicSchemaEnricher initializes with BAML client.

        Tests that DynamicSchemaEnricher correctly stores the provided BAML
        client for use in LLM-based metadata extraction.
        """
        mock_client = Mock()
        enricher = DynamicSchemaEnricher(mock_client)

        assert enricher.baml_client == mock_client

    @pytest.mark.asyncio
    async def test_real_world_document_schema(self) -> None:
        """Verify extraction works with realistic multi-field document schemas.

        Tests extraction from a realistic document with company name, year,
        category (enum), and boolean flag, ensuring all field types are
        correctly extracted and stored.
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.company = "Acme Corp"
        mock_result.year = 2023
        mock_result.category = "Finance"
        mock_result.has_tables = True
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "company": {"type": "string", "description": "Company name"},
                "year": {"type": "integer", "description": "Year of document"},
                "category": {
                    "type": "string",
                    "enum": ["finance", "technical", "marketing"],
                },
                "has_tables": {"type": "boolean"},
            }
        }

        result = await enricher.extract("Acme Corp 2023 financial report", schema)

        assert result["company"] == "Acme Corp"
        assert result["year"] == 2023
        assert result["category"] == "Finance"
        assert result["has_tables"] is True

    @pytest.mark.asyncio
    async def test_large_schema_10_fields(self) -> None:
        """Verify extraction scales to large schemas with many fields.

        Tests extraction from a schema with 10 string fields, ensuring that
        the enricher correctly handles realistic large schemas without errors.
        """
        mock_client = AsyncMock()
        mock_result = Mock()
        for i in range(10):
            setattr(mock_result, f"field{i}", f"value{i}")
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {f"field{i}": {"type": "string"} for i in range(10)}}

        result = await enricher.extract("content", schema)

        assert len(result) == 10
        for i in range(10):
            assert f"field{i}" in result

    def test_cache_key_generation_with_initialization(self) -> None:
        """Verify cache key generation works with positional and keyword arguments.

        Tests that _make_schema_key produces identical keys whether arguments
        are passed positionally, as keywords, or mixed, ensuring consistent
        caching behavior during enricher initialization and operation.
        """
        text = "test content"
        schema = {"properties": {"field": {"type": "string"}}}

        # Verify both calling styles work
        key1 = _make_schema_key(text, schema=schema)  # Positional + keyword
        key2 = _make_schema_key(text=text, schema=schema)  # Keyword only

        # Verify they produce identical results
        assert key1 == key2
        assert isinstance(key1, tuple)
        assert len(key1) == 2
        assert isinstance(key1[0], str) and isinstance(key1[1], str)
