"""Shared fixtures for metadata enricher tests."""

import pytest

from rag_pipelines.utils.metadata_enricher.dynamic_schema_enricher import (
    _dynamic_cache,
)
from rag_pipelines.utils.metadata_enricher.fixed_schema_enricher import (
    _fixed_cache,
)
from rag_pipelines.utils.metadata_enricher.metadata_enricher import _chunk_cache
from rag_pipelines.utils.metadata_enricher.structural_metadata_enricher import (
    _structural_cache,
)


@pytest.fixture(autouse=True)
def clear_enricher_caches() -> None:
    """Clear all enricher caches before each test.

    Ensures test isolation by clearing global caches that persist across tests.
    Without this, tests with similar inputs hit cached results from previous tests,
    bypassing mock setup. This is especially important for extract_query_metadata
    tests which use the _chunk_cache.
    """
    _chunk_cache.clear()
    _structural_cache.clear()
    _dynamic_cache.clear()
    _fixed_cache.clear()
    yield
    _chunk_cache.clear()
    _structural_cache.clear()
    _dynamic_cache.clear()
    _fixed_cache.clear()
