"""Metadata Enricher for RAG pipelines.

This package provides enhanced metadata extraction with three-layer automatic
enrichment while maintaining zero breaking changes to existing code.
"""

from rag_pipelines.utils.metadata_enricher.metadata_enricher import (
    EnrichmentConfig,
    EnrichmentMode,
    MetadataEnricher,
)


__all__ = [
    "MetadataEnricher",
    "EnrichmentMode",
    "EnrichmentConfig",
]
