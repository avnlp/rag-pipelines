"""Tests for structural metadata cache key generation.

Validates that _make_structural_key produces consistent, deterministic keys
from chunk and context inputs, supporting caching behavior.
"""

from rag_pipelines.utils.metadata_enricher.structural_metadata_enricher import (
    _make_structural_key,
)


class TestStructuralMetadataEnricherCacheKeys:
    """Test cache key generation consistency and determinism.

    Validates that _make_structural_key produces consistent, deterministic keys
    from chunk and context inputs, that context field order doesn't affect the
    hash, and that different chunks/contexts produce different keys.
    """

    def test_key_with_all_args_positional(self) -> None:
        """Verify cache key generation works with positional arguments.

        Tests that _make_structural_key produces a tuple of two string hashes
        when called with positional arguments (chunk, context).
        """
        chunk = {"text": "test content"}
        context = {"page_number": 1, "section": "intro"}

        key = _make_structural_key(chunk, context)

        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)

    def test_key_with_kwargs(self) -> None:
        """Verify cache key generation works with keyword arguments.

        Tests that _make_structural_key produces identical key tuple structure
        when called with keyword arguments.
        """
        chunk = {"text": "test content"}
        context = {"page_number": 1}

        key = _make_structural_key(chunk=chunk, document_context=context)

        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_key_consistency_same_content(self) -> None:
        """Verify cache keys are deterministic for identical chunk and context.

        Tests that calling _make_structural_key multiple times with the same
        chunk and context produces identical keys.
        """
        chunk1 = {"text": "Same content"}
        chunk2 = {"text": "Same content"}
        context = {"page": 1}

        key1 = _make_structural_key(chunk1, context)
        key2 = _make_structural_key(chunk2, context)

        assert key1 == key2

    def test_key_difference_different_content(self) -> None:
        """Different chunks should produce different cache keys."""
        chunk1 = {"text": "Content A"}
        chunk2 = {"text": "Content B"}

        key1 = _make_structural_key(chunk1)
        key2 = _make_structural_key(chunk2)

        assert key1 != key2

    def test_key_difference_different_context(self) -> None:
        """Same chunk, different context should produce different context hash."""
        chunk = {"text": "Same content"}
        context1 = {"page": 1}
        context2 = {"page": 2}

        key1 = _make_structural_key(chunk, context1)
        key2 = _make_structural_key(chunk, context2)

        assert key1[0] == key2[0]  # Text hash should be same
        assert key1[1] != key2[1]  # Context hash should differ

    def test_key_with_none_context(self) -> None:
        """Test key generation when context is None."""
        chunk = {"text": "test"}

        key = _make_structural_key(chunk, None)

        assert key[0] != "none"
        assert key[1] == "none"

    def test_key_with_complex_context(self) -> None:
        """Test key generation with complex nested context."""
        chunk = {"text": "test"}
        context = {
            "page": 1,
            "section": "intro",
            "heading_hierarchy": ["Chapter 1", "Section 1.1"],
            "metadata": {"author": "John", "date": "2024"},
        }

        key = _make_structural_key(chunk, context)

        assert isinstance(key[1], str)
        assert key[1] != "none"

    def test_key_order_independence_in_dict_context(self) -> None:
        """Same context dict in different order should produce same hash."""
        chunk = {"text": "test"}
        context1 = {"page": 1, "section": "a", "title": "b"}
        context2 = {"title": "b", "page": 1, "section": "a"}

        key1 = _make_structural_key(chunk, context1)
        key2 = _make_structural_key(chunk, context2)

        assert key1[1] == key2[1]
