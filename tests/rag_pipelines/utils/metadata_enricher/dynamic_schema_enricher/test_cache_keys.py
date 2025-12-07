"""Tests for DynamicSchemaEnricher cache key generation.

Validates that _make_schema_key produces consistent, deterministic keys
from text and schema inputs, supporting caching behavior.
"""

import hashlib

from rag_pipelines.utils.metadata_enricher.dynamic_schema_enricher import (
    _make_schema_key,
)


class TestDynamicSchemaEnricherCacheKeys:
    """Test cache key generation consistency and determinism.

    Validates that _make_schema_key produces consistent, deterministic keys
    from text and schema inputs, and that different schemas/text produce
    different keys while identical inputs produce identical keys.
    """

    def test_key_consistency_same_text_and_schema(self) -> None:
        """Verify cache keys are deterministic for identical text and schema.

        Tests that calling _make_schema_key multiple times with the same text
        and schema produces the exact same key, required for caching to work.
        """
        text = "Same content"
        schema = {"properties": {"field": {"type": "string"}}}

        key1 = _make_schema_key(text, schema=schema)
        key2 = _make_schema_key(text, schema=schema)

        assert key1 == key2

    def test_key_difference_different_text(self) -> None:
        """Verify different text produces different text hashes.

        Tests that when text content differs, the text hash component of the
        cache key differs while the schema hash component remains the same.
        """
        schema = {"properties": {"field": {"type": "string"}}}

        key1 = _make_schema_key("Text A", schema=schema)
        key2 = _make_schema_key("Text B", schema=schema)

        assert key1[0] != key2[0]
        assert key1[1] == key2[1]

    def test_key_difference_different_schema(self) -> None:
        """Verify different schemas produce different schema hashes.

        Tests that when user schema differs, the schema hash component of the
        cache key differs while the text hash component remains the same.
        """
        text = "same text"
        schema1 = {"properties": {"field1": {"type": "string"}}}
        schema2 = {"properties": {"field2": {"type": "integer"}}}

        key1 = _make_schema_key(text, schema=schema1)
        key2 = _make_schema_key(text, schema=schema2)

        assert key1[0] == key2[0]
        assert key1[1] != key2[1]

    def test_key_with_none_schema(self) -> None:
        """Verify None schema produces 'none' schema hash.

        Tests that when schema is None, the schema hash component is the
        literal string "none", enabling cache key generation.
        """
        text = "test"

        key = _make_schema_key(text, schema=None)

        assert key[0] != "none"
        assert key[1] == "none"

    def test_key_with_none_text(self) -> None:
        """Verify None text is treated as empty string for hashing.

        Tests that when text is None, it is hashed as an empty string,
        producing the standard empty SHA256 hash.
        """
        schema = {"properties": {"field": {"type": "string"}}}

        key = _make_schema_key(text=None, schema=schema)

        expected_hash = hashlib.sha256(b"").hexdigest()
        assert key[0] == expected_hash
        assert key[1] != "none"

    def test_key_with_complex_schema(self) -> None:
        """Verify cache keys work with realistic multi-field schemas.

        Tests that _make_schema_key correctly hashes complex schemas with
        multiple fields, various types, enums, and descriptions.
        """
        text = "content"
        schema = {
            "properties": {
                "field1": {"type": "string", "description": "A field"},
                "field2": {"type": "integer"},
                "field3": {
                    "type": "string",
                    "enum": ["option1", "option2", "option3"],
                },
            }
        }

        key = _make_schema_key(text, schema=schema)

        assert isinstance(key[1], str)
        assert key[1] != "none"

    def test_key_order_independence_in_schema(self) -> None:
        """Verify schema property order doesn't affect cache key hash.

        Tests that schemas with identical properties in different order
        produce the same schema hash, ensuring proper cache matching.
        """
        text = "content"
        schema1 = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        schema2 = {"properties": {"b": {"type": "integer"}, "a": {"type": "string"}}}

        key1 = _make_schema_key(text, schema=schema1)
        key2 = _make_schema_key(text, schema=schema2)

        assert key1[1] == key2[1]
