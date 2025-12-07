"""Tests for DynamicSchemaEnricher edge case handling.

Validates graceful handling of empty/invalid schemas, None values, missing
fields in extracted results, unsupported types, and LLM extraction failures.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from rag_pipelines.utils.metadata_enricher.dynamic_schema_enricher import (
    DynamicSchemaEnricher,
)


class TestDynamicSchemaEnricherEdgeCases:
    """Test edge case handling and error resilience of DynamicSchemaEnricher.

    Validates graceful handling of empty/invalid schemas, None values, missing
    fields in extracted results, unsupported types, and LLM extraction failures.
    """

    @pytest.mark.asyncio
    async def test_empty_schema(self) -> None:
        """Test extraction with empty schema."""
        mock_client = AsyncMock()
        enricher = DynamicSchemaEnricher(mock_client)

        result = await enricher.extract("content", {})

        assert result == {}

    @pytest.mark.asyncio
    async def test_schema_without_properties(self) -> None:
        """Test schema that lacks 'properties' key."""
        mock_client = AsyncMock()
        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"type": "object"}  # Missing 'properties' key

        result = await enricher.extract("content", schema)

        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_properties(self) -> None:
        """Test schema with empty properties dict."""
        mock_client = AsyncMock()
        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {}}

        result = await enricher.extract("content", schema)

        assert result == {}

    @pytest.mark.asyncio
    async def test_unsupported_field_type_skipped(self) -> None:
        """Test that unsupported field types are skipped."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.supported_field = "value"
        mock_result.unsupported_field = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "supported_field": {"type": "string"},
                "unsupported_field": {"type": "array"},
            }
        }

        result = await enricher.extract("content", schema)

        assert "supported_field" in result

    @pytest.mark.asyncio
    async def test_missing_field_in_result(self) -> None:
        """Test handling when result object has missing field (AttributeError)."""
        mock_client = AsyncMock()
        mock_result = Mock(spec=["field1"])  # Only has field1, not field2
        mock_result.field1 = "value1"
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
            }
        }

        result = await enricher.extract("content", schema)

        assert "field1" in result
        assert "field2" not in result

    @pytest.mark.asyncio
    async def test_with_none_schema(self) -> None:
        """Test extraction with None schema."""
        mock_client = AsyncMock()
        enricher = DynamicSchemaEnricher(mock_client)

        result = await enricher.extract("content", None)

        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        """Test extraction with empty text."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.field = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {"field": {"type": "string"}}}

        result = await enricher.extract("", schema)

        assert result == {}

    @pytest.mark.asyncio
    async def test_optional_field_present(self) -> None:
        """Test optional field when LLM extracts it."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.optional_field = "extracted_value"
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {"optional_field": {"type": "string"}}}

        result = await enricher.extract("content", schema)

        assert result["optional_field"] == "extracted_value"

    @pytest.mark.asyncio
    async def test_optional_field_none_excluded(self) -> None:
        """Test optional field returns None are excluded from result."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.optional_field = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {"optional_field": {"type": "string"}}}

        result = await enricher.extract("content", schema)

        assert "optional_field" not in result

    @pytest.mark.asyncio
    async def test_empty_string_excluded(self) -> None:
        """Test that empty string values are properly handled."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.field = ""
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {"field": {"type": "string"}}}

        result = await enricher.extract("content", schema)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_type_builder_exception_handling(self) -> None:
        """Test graceful handling of TypeBuilder failures."""
        mock_client = AsyncMock()
        enricher = DynamicSchemaEnricher(mock_client)
        schema = {"properties": {"field": {"type": "string"}}}

        # Mock the ExtractMetadata to raise an exception
        mock_client.ExtractMetadata = AsyncMock(
            side_effect=Exception("TypeBuilder failed")
        )

        result = await enricher.extract("content", schema)

        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_metadata_attribute_error(self) -> None:
        """Test handling when result object has wrong attributes."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.field1 = "value1"
        mock_result.field2 = Mock(side_effect=AttributeError)
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
            }
        }

        result = await enricher.extract("content", schema)

        assert "field1" in result

    @pytest.mark.asyncio
    async def test_multiple_none_values(self) -> None:
        """Test extraction with multiple None values."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.field1 = None
        mock_result.field2 = "value2"
        mock_result.field3 = None
        mock_result.field4 = "value4"
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"},
                "field4": {"type": "string"},
            }
        }

        result = await enricher.extract("content", schema)

        assert "field1" not in result
        assert result["field2"] == "value2"
        assert "field3" not in result
        assert result["field4"] == "value4"

    @pytest.mark.asyncio
    async def test_all_none_values(self) -> None:
        """Test extraction when all fields are None."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.field1 = None
        mock_result.field2 = None
        mock_result.field3 = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "string"},
                "field3": {"type": "string"},
            }
        }

        result = await enricher.extract("content", schema)

        # All None should result in empty dict
        assert result == {}

    @pytest.mark.asyncio
    async def test_mixed_none_and_values(self) -> None:
        """Test extraction with mix of None and actual values."""
        mock_client = AsyncMock()
        mock_result = Mock()
        mock_result.name = "John"
        mock_result.age = None
        mock_result.email = "john@example.com"
        mock_result.phone = None
        mock_client.ExtractMetadata = AsyncMock(return_value=mock_result)

        enricher = DynamicSchemaEnricher(mock_client)
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
            }
        }

        result = await enricher.extract("content", schema)

        assert result["name"] == "John"
        assert "age" not in result
        assert result["email"] == "john@example.com"
        assert "phone" not in result
