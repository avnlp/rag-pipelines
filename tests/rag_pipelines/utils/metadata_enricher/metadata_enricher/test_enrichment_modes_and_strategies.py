"""Tests for enrichment modes and strategies.

Validates that enrichment strategy selection (MINIMAL, DYNAMIC, FULL modes)
produces correct layer-wise metadata application (structural, dynamic, fixed
layers), that dynamic user-defined schemas are correctly processed, and that
LLM failures gracefully degrade to structural metadata only.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from rag_pipelines.utils.metadata_enricher import (
    EnrichmentMode,
    MetadataEnricher,
)
from rag_pipelines.utils.metadata_enricher.dynamic_schema_enricher import (
    _make_schema_key,
)


class TestEnrichmentStrategies:
    """Test enrichment strategies, three-layer enrichment, and LLM failure handling.

    Verifies that the three-layer enrichment pipeline (structural, dynamic, fixed)
    is correctly applied based on enrichment mode, that each mode includes only its
    designated layers, that LLM failures gracefully degrade to structural metadata,
    and that mode selection affects output correctly.
    """

    @pytest.mark.asyncio
    async def test_three_layer_enrichment_full_mode(self) -> None:
        """Verify FULL mode applies all three metadata layers correctly.

        Tests that in FULL enrichment mode, structural metadata (Layer 1),
        user-defined schema metadata (Layer 2), and fixed schema metadata
        (Layer 3) are all present in the output.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Machine learning transforms healthcare."}
        schema = {"properties": {"domain": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_baml.return_value = type(
                "obj",
                (),
                {
                    "domain": "healthcare",
                    "questions_answered": ["How does ML transform healthcare?"],
                    "keywords": ["machine learning", "healthcare"],
                    "chunk_summary": "ML transforms healthcare.",
                    "content_type": "analytical",
                    "header_text": "ML in Healthcare",
                },
            )()

            result = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.FULL
            )
            metadata = result["metadata"]

            # Layer 1: Structural
            assert "chunk_id" in metadata
            assert "word_count" in metadata

            # Layer 2: User schema
            assert metadata.get("domain") == "healthcare"

            # Layer 3: Fixed schema
            assert "questions_answered" in metadata
            assert "keywords" in metadata
            assert "chunk_summary" in metadata

    @pytest.mark.asyncio
    async def test_minimal_mode_excludes_fixed_schema(self) -> None:
        """Verify MINIMAL mode excludes fixed schema Layer 3.

        Tests that MINIMAL mode includes only Layer 1 (structural metadata)
        and excludes Layer 3 (fixed schema metadata like questions_answered,
        keywords).
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Simple test document"}

        result = await enricher._enrich_chunk(chunk, mode=EnrichmentMode.MINIMAL)
        metadata = result["metadata"]

        assert "chunk_id" in metadata
        assert "questions_answered" not in metadata
        assert "keywords" not in metadata

    @pytest.mark.asyncio
    async def test_dynamic_mode(self) -> None:
        """Verify DYNAMIC mode includes Layers 1-2 but excludes Layer 3.

        Tests that DYNAMIC mode includes structural metadata (Layer 1) and
        user-defined schema metadata (Layer 2), but excludes fixed schema
        metadata (Layer 3).
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test document for dynamic mode"}
        schema = {"properties": {"field1": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_baml.return_value = type(
                "obj",
                (),
                {
                    "field1": "value1",
                    "questions_answered": None,
                },
            )()

            result = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.DYNAMIC
            )
            metadata = result["metadata"]

            assert "chunk_id" in metadata
            assert "field1" in metadata
            assert "questions_answered" not in metadata

    @pytest.mark.asyncio
    async def test_dynamic_enricher_exception_handling(self) -> None:
        """Verify DYNAMIC mode gracefully handles LLM failures.

        Tests that when an LLM failure occurs during dynamic schema extraction,
        the enricher degrades gracefully and still returns structural metadata
        (Layer 1) instead of failing completely.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test document"}
        schema = {"properties": {"field": {"type": "string"}}}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_baml.side_effect = Exception("LLM service unavailable")

            result = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.DYNAMIC
            )

            assert "chunk_id" in result["metadata"]
            assert "word_count" in result["metadata"]

    @pytest.mark.asyncio
    async def test_fixed_enricher_exception_handling(self) -> None:
        """Verify FULL mode gracefully handles LLM failures in Layer 3.

        Tests that when an LLM failure occurs during fixed schema extraction,
        the enricher degrades to Layers 1-2 (structural + dynamic) and does not
        include Layer 3 (fixed schema) fields in output.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test document"}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_baml.side_effect = Exception("LLM timeout")

            result = await enricher._enrich_chunk(chunk, mode=EnrichmentMode.FULL)

            assert "chunk_id" in result["metadata"]
            assert "word_count" in result["metadata"]
            assert "questions_answered" not in result["metadata"]

    @pytest.mark.asyncio
    async def test_fixed_enricher_with_nondict_spec(self) -> None:
        """Verify FULL mode handles internal type specification edge cases.

        Tests that fixed schema enricher correctly processes chunks even when
        internal type builders handle non-dict type specifications, completing
        successfully without errors.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test content"}

        result = await enricher._enrich_chunk(chunk, mode=EnrichmentMode.FULL)

        assert "chunk_id" in result["metadata"]
        assert result["enrichment_mode"] == "full"


class TestMetadataEnricherDynamicSchema:
    """Test dynamic user-defined schema handling during enrichment.

    Validates that user-defined schemas with various field types (string,
    integer, enum, unsupported) are correctly processed, that invalid/empty
    schemas are handled gracefully, and that enrichment mode is properly
    tracked in results.
    """

    @pytest.mark.asyncio
    async def test_enrich_chunk_returns_mode_value(self) -> None:
        """Verify _enrich_chunk includes enrichment_mode in result.

        Tests that the enrichment result always includes a string representation
        of the enrichment mode that was used (e.g., "minimal", "dynamic", "full").
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test"}

        result = await enricher._enrich_chunk(chunk, mode=EnrichmentMode.MINIMAL)

        assert "enrichment_mode" in result
        assert result["enrichment_mode"] == "minimal"

    @pytest.mark.asyncio
    async def test_dynamic_schema_with_enum_values(self) -> None:
        """Verify dynamic schema correctly extracts enum-constrained fields.

        Tests that when a user schema defines a field with enum constraints,
        the extracted value matches one of the allowed enum values.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "This is analytical content"}
        schema = {
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": ["definition", "example", "procedure"],
                    "description": "Type of content",
                }
            }
        }

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_result.content_type = "example"
            mock_baml.return_value = mock_result

            result = await enricher._enrich_chunk(
                chunk, user_schema=schema, mode=EnrichmentMode.DYNAMIC
            )

            assert result["metadata"]["content_type"] == "example"

    @pytest.mark.asyncio
    async def test_dynamic_enricher_with_unsupported_types(self) -> None:
        """Verify unsupported field types are skipped without error.

        Tests that when a user schema contains only unsupported types (e.g.,
        "array", "unsupported_type"), the enricher gracefully skips them and
        returns only structural metadata.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test document"}
        schema = {
            "properties": {
                "field1": {"type": "unsupported_type"},
                "field2": {"type": "array"},
            }
        }

        result = await enricher._enrich_chunk(
            chunk, user_schema=schema, mode=EnrichmentMode.DYNAMIC
        )

        assert "chunk_id" in result["metadata"]
        assert "field1" not in result["metadata"]
        assert "field2" not in result["metadata"]

    @pytest.mark.asyncio
    async def test_dynamic_schema_with_invalid_schema(self) -> None:
        """Verify invalid schema (missing 'properties' key) is handled gracefully.

        Tests that when a user schema lacks the required 'properties' key,
        the enricher returns structural metadata without error.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test document"}
        invalid_schema = {}

        with patch.object(
            enricher.baml_client, "ExtractMetadata", new_callable=AsyncMock
        ) as mock_baml:
            mock_result = Mock()
            mock_baml.return_value = mock_result

            result = await enricher._enrich_chunk(
                chunk, user_schema=invalid_schema, mode=EnrichmentMode.DYNAMIC
            )

            assert "chunk_id" in result["metadata"]

    @pytest.mark.asyncio
    async def test_dynamic_schema_with_none_schema(self) -> None:
        """Verify None schema is handled gracefully without error.

        Tests that passing None as the user schema does not cause errors and
        still returns structural metadata.
        """
        enricher = MetadataEnricher()
        chunk = {"text": "Test content"}

        result = await enricher._enrich_chunk(
            chunk, user_schema=None, mode=EnrichmentMode.DYNAMIC
        )

        assert "chunk_id" in result["metadata"]

    @pytest.mark.asyncio
    async def test_dynamic_schema_cache_key_generation(self) -> None:
        """Verify cache keys are consistent for identical text and schema.

        Tests that _make_schema_key produces identical keys when called with
        the same text and schema, enabling proper caching behavior.
        """
        text = "Sample text for caching"
        schema = {"properties": {"field": {"type": "string"}}}

        key1 = _make_schema_key(text=text, schema=schema)
        key2 = _make_schema_key(text=text, schema=schema)

        assert key1 == key2
        assert len(key1) == 2
