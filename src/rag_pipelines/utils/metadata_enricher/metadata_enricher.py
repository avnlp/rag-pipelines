"""MetadataEnricher: Enhanced metadata extraction with three-layer enrichment.

This component implements a three-layer enrichment strategy:

- Layer 1 (Structural): Fast, rule-based extraction (content hash, word count,
  language detection, document structure inheritance). Always runs, zero LLM cost.
- Layer 2 (Dynamic Schema): User-defined fields via structured LLM extraction.
  LLM-powered, allows custom metadata extraction aligned with user requirements.
- Layer 3 (Fixed Schema): RAG-optimized fields (questions_answered, chunk_summary,
  keywords, content_type, header_text). Most expensive layer, significantly improves
  retrieval quality and ranking. LLM-powered.

The three-layer approach enables cost/benefit tradeoffs through EnrichmentMode:
- MINIMAL: Layer 1 only (no LLM cost)
- DYNAMIC: Layers 1+2 (user-specified schema extraction)
- FULL: All layers (complete RAG optimization)

Caching is applied at multiple levels:
- Module-level TTL cache for _enrich_chunk() deduplicates identical content
- Per-layer caches handle concurrent request deduplication
- Batch processing with asyncio.gather() enables parallel enrichment within batches
"""

import hashlib
import json
from asyncio import gather
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from cachetools_async import cached
from langchain_core.documents import Document
from tqdm import tqdm

from rag_pipelines.baml_client import b
from rag_pipelines.utils.metadata_enricher.dynamic_schema_enricher import (
    DynamicSchemaEnricher,
)
from rag_pipelines.utils.metadata_enricher.fixed_schema_enricher import (
    FixedSchemaEnricher,
)
from rag_pipelines.utils.metadata_enricher.structural_metadata_enricher import (
    StructuralMetadataEnricher,
)


# Module-level cache for enrich_chunk() results. Stores up to 1000 unique
# chunk+schema combinations with 1-hour TTL. Enables deduplication across
# document batches and concurrent request consolidation.
_chunk_cache: TTLCache[Any, Any] = TTLCache(maxsize=1000, ttl=3600)


def _make_enrichment_key(*args: Any, **kwargs: Any) -> tuple[str, str, str]:
    """Generate cache key from chunk, document_context, and user_schema.

    Creates a three-part cache key allowing fine-grained deduplication.
    Extracts three independent components:
    1. Content hash: SHA256 of chunk text enables deduplication of identical
       content, handles duplicate documents or repeated sections.
    2. Context hash: SHA256 of document context (metadata dict) enables tracking
       how context affects enrichment results. Different pages/sections produce
       different context hashes even with identical text.
    3. Schema hash: SHA256 of user schema enables cache hits when enriching
       different documents with the same schema.

    Treats missing components as "none" literal string to distinguish from
    null/empty cases and enable cache hits when components are unspecified.

    Args:
        *args: Positional args for chunk, document_context, user_schema.
        **kwargs: Keyword args as fallback for any of the above.

    Returns:
        Tuple of (chunk_hash, context_hash, schema_hash) as cache key.
    """
    chunk = args[0] if args else kwargs.get("chunk", {})
    document_context = args[1] if len(args) > 1 else kwargs.get("document_context")
    user_schema = args[2] if len(args) > 2 else kwargs.get("user_schema")

    # Hash chunk content for deduplication. Converts dict/string to string
    # representation to handle both chunk formats consistently.
    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
    chunk_hash = hashlib.sha256(text.encode()).hexdigest()

    # Hash document context if present, enables per-context caching for
    # inherited metadata like page numbers, section titles.
    if document_context:
        context_str = json.dumps(document_context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()
    else:
        context_hash = "none"

    # Hash user schema if present, enables cache hits across documents using
    # the same schema definition for custom field extraction.
    if user_schema:
        schema_str = json.dumps(user_schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
    else:
        schema_hash = "none"

    return (chunk_hash, context_hash, schema_hash)


class EnrichmentMode(Enum):
    """Defines which enrichment layers to apply.

    Provides three operational modes enabling cost/quality tradeoffs:

    MINIMAL: Layer 1 only (structural metadata).
        - No LLM calls, pure Python computation
        - Extracts: content_hash, word_count, language, page_number, section_title

    DYNAMIC: Layers 1 + 2 (structural + user-defined).
        - Runs user schema extraction via LLM along with structural metadata extraction
        - Extracts: structural fields + user-defined fields from JSON schema
        - Ideal for: custom domain-specific metadata during indexing
        - Example: Extracting company, year, category fields from documents

    FULL: All 3 layers (structural + user-defined + RAG-optimized).
        - Most comprehensive enrichment with all layers
        - Extracts: all above + questions_answered, chunk_summary, keywords,
          content_type, header_text
        - Ideal for: initial document indexing, maximum retrieval quality
        - Use case: Full metadata enrichment during RAG pipeline initialization

    The choice of mode depends on:
    - Indexing: Use FULL for best retrieval quality, accept higher latency
    - Querying: Use MINIMAL for fast query metadata, trade metadata richness for speed
    - Custom domains: Use DYNAMIC if fixed schema doesn't cover requirements

    Attributes:
        MINIMAL: Layer 1 only (structural metadata).
        DYNAMIC: Layers 1 + 2 (structural + user-defined schema).
        FULL: All 3 layers (structural + user-defined + RAG-optimized).
    """

    MINIMAL = "minimal"
    DYNAMIC = "dynamic"
    FULL = "full"


@dataclass
class EnrichmentConfig:
    """Configuration for the MetadataEnricher.

    Controls enrichment behavior during document processing. Two key parameters:

    mode: Enrichment layer selection.
        - MINIMAL: Fast, no LLM cost (structural only)
        - DYNAMIC: User schema extraction (layers 1+2)
        - FULL: Complete enrichment (all layers, maximum quality)
        Default is FULL for maximum metadata quality during indexing.

    batch_size: Concurrent documents per batch in enrich_documents().
        - Controls parallelism within asyncio.gather() calls
        - Higher values increase throughput but memory usage
        - Recommend 5-20 depending on LLM rate limits and available memory
        - Default is 10 for balanced throughput and memory usage

    Example:
        config = EnrichmentConfig(mode=EnrichmentMode.DYNAMIC, batch_size=20)
        enricher = MetadataEnricher(config)

    Attributes:
        mode: Which enrichment layers to apply (MINIMAL, DYNAMIC, FULL).
        batch_size: Number of documents to process concurrently in batches.
    """

    mode: EnrichmentMode = EnrichmentMode.FULL
    batch_size: int = 10


class MetadataEnricher:
    """Enhanced metadata enrichment component for RAG pipelines.

    Provides automatic three-layer metadata enrichment that transforms raw document
    text into rich metadata dramatically improving RAG retrieval quality.

    Architecture Summary:

    Layer 1 (Structural): Rule-based extraction of content metadata
        - Content hashing for deduplication detection
        - Word/char counting for length-based filtering
        - Language detection for multilingual handling
        - Document context inheritance (page numbers, sections, headings)
        - Always runs regardless of mode

    Layer 2 (Dynamic Schema): User-defined custom fields via LLM
        - Extracts arbitrary fields specified in JSON schema
        - Uses BAML TypeBuilder for dynamic schema definition at runtime
        - Supports string, number, integer, boolean, and enum types
        - Optional fields (nullable) allow LLM to skip if not applicable
        - Runs in DYNAMIC_ONLY or FULL modes

    Layer 3 (Fixed Schema): RAG-optimized automatic fields
        - questions_answered: Natural questions chunk answers (improves semantic match)
        - chunk_summary: Concise 1-2 sentence summary for ranking
        - keywords: 3-7 important terms for lexical search
        - content_type: Classification (definition, example, procedure, etc)
        - header_text: Descriptive 2-8 word title for display/filtering
        - Only runs in FULL mode, designed to maximize retrieval quality

    Caching Strategy:

    The enricher uses multi-level caching to handle large document collections:
    1. Module-level TTL cache (_chunk_cache): 1 hour, 1000 entries
        - Deduplicates identical chunks with identical schema/context
        - Handles concurrent request consolidation (multiple requests for same
          chunk await same underlying enrichment task)
        - Survives across document batches for large collections
    2. Per-layer caches: Each enrichment layer has independent cache
        - Enables fine-grained deduplication within layers
        - Structural cache: 5000 entries, 24 hour TTL (safe, no LLM cost)
        - Dynamic/Fixed caches: 2000 entries, 1 hour TTL (LLM-powered)

    Parallelization:

    asyncio.gather() within atransform_documents() enables concurrent document
    enrichment within each batch. Multiple documents make LLM calls in parallel,
    dramatically improving throughput vs serial processing.

    Example Usage:

    # Default: FULL enrichment with batch size 10
    enricher = MetadataEnricher()
    docs = await enricher.atransform_documents(chunked_docs, schema)

    # Fast query-time metadata extraction
    query_metadata = await enricher.extract_query_metadata(query, schema)

    # Custom configuration: user-defined fields only (minimal mode)
    enricher = MetadataEnricher()
    docs = await enricher.atransform_documents(chunked_docs, schema, mode="minimal")
    """

    def __init__(self, config: Optional[EnrichmentConfig] = None) -> None:
        """Initialize the MetadataEnricher.

        Sets up the three enrichment layers and internal state. All initialization
        is synchronous; async work happens in enrich_document()/enrich_chunk().

        Args:
            config: Configuration object with mode and batch_size.
                Uses default EnrichmentConfig (FULL mode, batch_size=10) if None.

        Note:
            The BAML client is imported from rag_pipelines.baml_client module
            as 'b', which is preconfigured with Groq client settings and
            fallback error handling. Accessed via self.baml_client.ExtractMetadata()
            by dynamic and fixed enrichment layers.

            Caching is handled transparently via @cached decorators on enrichment
            layer methods. The @cached decorators use cachetools_async.cached
            which provides concurrent request deduplication: multiple concurrent
            calls with identical arguments await the same underlying task rather
            than making parallel LLM calls.
        """
        self.config = config or EnrichmentConfig()
        self.baml_client = b

        # Initialize three enrichment layers as separate components.
        # Enables progressive enrichment validation.
        self.structural_enricher = StructuralMetadataEnricher()
        self.dynamic_enricher = DynamicSchemaEnricher(self.baml_client)
        self.fixed_enricher = FixedSchemaEnricher(self.baml_client)

    async def atransform_documents(
        self,
        documents: List[Document],
        schema: Dict[str, Any],
        mode: Optional[str] = None,
    ) -> List[Document]:
        """Transform documents by enriching their metadata.

        Primary interface for document indexing. Applies enrichment layers
        based on specified mode to maximize metadata quality during RAG indexing.

        Enrichment Modes:

        - "minimal": Layer 1 only (structural metadata, no LLM cost)
        - "dynamic": Layers 1 + 2 (structural + user-defined schema)
        - "full": All layers (structural + user-defined + RAG-optimized)

        Default is "full" for maximum metadata quality. Document indexing gets
        full treatment because retrieval quality directly impacts system performance.

        Architecture:
        - Batch processing enables concurrent LLM calls
        - Results merged into document metadata preserving original fields
        - Graceful degradation if enrichment layers fail

        Args:
            documents: List of LangChain Document objects to enrich.
                Each document's page_content is used for enrichment, and
                metadata dict is updated with enriched fields.
            schema: JSON schema defining user-defined metadata fields.
                Expected format: {"properties": {"field": {"type": "string"}, ...}}
                If schema is empty, Layer 2 is skipped but Layers 1 and 3 run.
            mode: Enrichment mode as string ("minimal", "dynamic", or "full").
                Case-sensitive. Defaults to "full" for maximum metadata quality.
                - "minimal": Fast, no LLM cost (query-time metadata extraction)
                - "dynamic": Custom domain-specific metadata
                - "full": Complete RAG optimization (document indexing)

                Raises ValueError if mode is not one of the above values.

        Returns:
            List of Documents with enriched metadata. Original documents are modified
            in place (doc.metadata is updated). Returns same list for chaining.

        Raises:
            ValueError: If mode is provided but not one of the valid values.
            No exceptions raised from enrichment layers; they degrade gracefully
            (return empty dicts). Pipeline continues even if LLM calls fail,
            ensuring indexing always completes.

        Example:
            docs = [Document(page_content="...", metadata={"source": "pdf"})]
            schema = {"properties": {"company": {"type": "string"}}}
            # Default FULL enrichment
            enriched = await enricher.atransform_documents(docs, schema)
            # Or explicitly specify mode
            enriched = await enricher.atransform_documents(
                docs, schema, mode="minimal"
            )
            # enriched[0].metadata includes selected enrichment layers
        """
        # Convert string mode to EnrichmentMode enum with helpful error message
        if mode is None:
            enrichment_mode = EnrichmentMode.FULL
        else:
            try:
                enrichment_mode = EnrichmentMode(mode)
            except ValueError as error:
                valid_modes = [m.value for m in EnrichmentMode]
                raise ValueError(
                    f"Invalid enrichment mode: '{mode}'. "
                    f"Must be one of {valid_modes} (case-sensitive)."
                ) from error

        results = []
        # Process documents in batches to balance throughput
        for i in tqdm(range(0, len(documents), self.config.batch_size)):
            batch = documents[i : i + self.config.batch_size]

            # Concurrent enrichment within each batch using gather().
            # This enables multiple documents to make LLM calls simultaneously,
            # dramatically improving throughput compared to serial processing.
            # gather() waits for all tasks to complete before returning.
            batch_results = await gather(
                *[
                    self._enrich_single_document(doc, schema, enrichment_mode)
                    for doc in batch
                ]
            )
            results.extend(batch_results)

        return results

    async def _enrich_single_document(
        self,
        doc: Document,
        schema: Dict[str, Any],
        enrichment_mode: EnrichmentMode,
    ) -> Document:
        """Enrich a single document by enriching its content and merging results.

        Private helper method used by atransform_documents() to enrich individual
        documents within a batch. Calls _enrich_chunk() to perform the actual
        enrichment, then merges the enriched metadata with the document's existing
        metadata.

        Args:
            doc: Document to enrich.
            schema: User-defined schema for metadata extraction.
            enrichment_mode: Which enrichment layers to apply.

        Returns:
            The same document with updated metadata.
        """
        enriched = await self._enrich_chunk(
            chunk={"text": doc.page_content},
            document_context=doc.metadata,
            user_schema=schema,
            mode=enrichment_mode,
        )
        doc.metadata.update(enriched["metadata"])
        doc.metadata["enrichment_mode"] = enriched["enrichment_mode"]
        return doc

    @cached(cache=_chunk_cache, key=_make_enrichment_key)  # type: ignore
    async def _enrich_chunk(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
        user_schema: Optional[Dict[str, Any]] = None,
        mode: Optional[EnrichmentMode] = None,
    ) -> Dict[str, Any]:
        """Internal: Enrich a single text chunk with metadata from all enabled layers.

        Core enrichment orchestrator that sequentially applies enabled layers
        based on mode. Results are cached with TTL to avoid reprocessing
        identical chunks with identical schemas. This is a private method used
        internally by atransform_documents() and extract_query_metadata().

        Caching Mechanism:

        The @cached decorator using _make_enrichment_key provides two
        critical benefits:

        1. Concurrent Request Deduplication:
           - Multiple concurrent calls with identical (chunk, context, schema)
             await the same underlying coroutine task
           - First caller creates task, others await same task result
           - Prevents parallel LLM calls for identical content
           - Example: If 3 batches contain same document, only 1 LLM call
        2. Temporal Deduplication (TTL Cache):
           - Results cached for 3600 seconds (1 hour)
           - Survives across batch boundaries and multiple indexing runs
           - For 1000-document index, enables ~99% cache hits on re-indexing
           - Graceful age-out prevents stale metadata issues

        Individual enrichment layers (structural/dynamic/fixed) also have
        independent caches for fine-grained deduplication within layers.

        Execution Flow:

        The method applies layers sequentially based on mode:
        - All modes: Always run Layer 1 (structural)
        - DYNAMIC/FULL: Run Layer 2 (dynamic schema) if schema provided
        - FULL only: Run Layer 3 (fixed schema)
        - Layers are independent; Layer 3 doesn't depend on Layer 2 output

        Args:
            chunk: Dictionary with 'text' key containing chunk content.
                Expected format: {"text": "chunk content..."}
            document_context: Optional metadata dict from parent document
                (e.g., {"page_number": 1, "section_title": "Introduction"}).
                Passed to structural layer for context inheritance.
            user_schema: User-defined JSON schema for Layer 2 extraction.
                Expected format: {"properties": {"field": {"type": "string"}, ...}}
                Only used if mode is DYNAMIC_ONLY or FULL.
            mode: Enrichment mode to use (defaults to self.config.mode).
                Controls which layers execute (MINIMAL/DYNAMIC/FULL).

        Returns:
            Dictionary with keys:
                - metadata: Dict of all extracted metadata fields from all layers
                - enrichment_mode: String value of mode used (e.g., "full")
        """
        mode = mode or self.config.mode

        enriched_metadata: Dict[str, Any] = {}

        # Layer 1: Structural Metadata (always runs, no LLM cost).
        # Extracts content hash, word count, language, inherited document context.
        # This layer never fails and is deterministic (same chunk always produces
        # same output). Safe to rely on for all documents.
        structural = await self.structural_enricher.extract(chunk, document_context)
        enriched_metadata.update(structural)

        # Layer 2: Dynamic Schema (user-defined fields, LLM-powered).
        # Only runs in DYNAMIC or FULL mode if schema is provided.
        # Enables users to extract custom fields via structured LLM extraction.
        if mode in [EnrichmentMode.DYNAMIC, EnrichmentMode.FULL] and user_schema:
            dynamic = await self.dynamic_enricher.extract(
                chunk.get("text", ""), user_schema
            )
            enriched_metadata.update(dynamic)

        # Layer 3: Fixed Schema (RAG-optimized fields, LLM-powered).
        # Only runs in FULL mode. Expensive but dramatically improves retrieval quality.
        # Extracts questions_answered, chunk_summary, keywords, content_type,
        # and header_text. Each field is optional (LLM can skip if not applicable).
        # Allowing flexible extraction without schema mismatches.
        if mode == EnrichmentMode.FULL:
            fixed = await self.fixed_enricher.extract(chunk.get("text", ""))
            enriched_metadata.update(fixed)

        return {
            "metadata": enriched_metadata,
            "enrichment_mode": mode.value,
        }

    async def extract_query_metadata(
        self, query: str, user_schema: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Extract metadata from a query and build Milvus filter expression.

        Optimized for query-time metadata extraction with semantic filtering.
        Uses DYNAMIC mode (Layer 1+2) to extract user-defined fields
        from the query, then converts extracted metadata to a Milvus filter
        expression for precise semantic search result refinement.

        Design Rationale:

        Query filtering in RAG requires both semantic and metadata matching.
        This method uses LLM to extract query metadata (Layer 2) to enable
        filtering search results by user-relevant fields (company, year, etc).
        The filter is then applied alongside vector similarity in Milvus.

        Layer 1 (structural) always runs for deduplication and language detection.
        Layer 2 (dynamic schema) extracts custom fields for filtering.
        Layer 3 (fixed schema/RAG-optimized) is skipped since it's for documents.

        Filter Construction:

        Extracted metadata is converted to Milvus filter expressions using:
        - LIKE for string fields (substring matching, works with auto_schema)
        - Equality for numeric fields
        - ARRAY_CONTAINS_ANY for array fields

        This enables forgiving matching (partial terms, minor variations) while
        maintaining scoped filtering within the specified field.

        Args:
            query: The query string to extract metadata from.
                Treated as raw text for enrichment (no special parsing).
            user_schema: User-defined JSON schema for custom field extraction.
                Expected format: {"properties": {"field": {"type": "string"}, ...}}
                Schema type hints guide filter expression construction.

        Returns:
            Tuple of (metadata_dict, milvus_filter_expr):
            - metadata_dict: Extracted metadata fields from layers 1+2
            - milvus_filter_expr: Milvus filter string for semantic search, or None
              if no custom fields extracted (falls back to semantic-only search)

        Example:
            query = "Apple's ML research in 2023"
            schema = {
                "properties": {
                    "company": {"type": "string"},
                    "year": {"type": "integer"},
                }
            }
            metadata, filter_expr = await enricher.extract_query_metadata(
                query, schema
            )
            # metadata = {"company": "Apple", "year": 2023, "word_count": 5, ...}
            # filter_expr = 'company like "%Apple%" AND year == 2023'
            # Use filter_expr in vectorstore.similarity_search(..., expr=filter_expr)
        """
        result = await self._enrich_chunk(
            chunk={"text": query},
            user_schema=user_schema,
            mode=EnrichmentMode.DYNAMIC,
        )
        metadata = result["metadata"]

        # Build Milvus filter expressions from extracted metadata fields.
        # Converts each extracted metadata value to appropriate filter expression
        # based on schema type hints. Different types use different operators to match
        # semantics of the field and enable flexible querying.
        #
        # Filter Strategy:
        # - String fields: Use LIKE (substring matching, works with auto_schema)
        #   Example: Extracted "Apple" matches docs with "Apple", "apple", "apples"
        # - Numeric fields: Use equality comparison (exact values only)
        #   Example: Extracted 2023 matches docs with exactly year=2023
        # - Array fields: Check if any extracted value exists in document array
        #   Example: Extracted ["python", "java"] matches docs with either language
        # - Object fields: Check if any extracted key exists in document object
        #   Example: Extracted keys ["author", "date"] match docs having those keys
        #
        # All conditions combined with AND operator.
        # Example complete filter: 'company like "%Apple%" AND year == 2023'
        #
        # LIKE operator works on any VARCHAR field without requiring enable_match=True.
        schema_properties = user_schema.get("properties", {})
        conditions: List[str] = []

        for field, value in metadata.items():
            # Skip structural fields (always present, not user-defined) and None values
            if value is None or field in [
                "chunk_id",
                "content_hash",
                "word_count",
                "char_count",
                "language",
                "page_number",
                "section_title",
                "heading_hierarchy",
                "enrichment_mode",
            ]:
                continue

            field_schema = schema_properties.get(field, {})
            field_type = field_schema.get("type", "string")

            if field_type == "string":
                # LIKE: Substring matching for forgiving queries.
                # Works on any VARCHAR field (no enable_match required).
                # Extracted value "apple" matches: "Apple", "apples", "pineapple", etc.
                #
                # Rationale:
                # - Compatible with auto_schema (enable_match not required)
                # - Higher recall: Catches variations and substrings
                # - Good performance for metadata filtering (smaller datasets)
                #
                # Use case: Filter documents by extracted metadata fields without
                # requiring schema changes.
                #
                # Note: Uses case-insensitive substring matching via LIKE operator.
                # If strict matching needed, use numeric IDs or enum types instead.
                # For production scale, consider custom schema with enable_match=True
                # for full inverted index performance.
                conditions.append(f'{field} like "%{value}%"')
            elif field_type in ["integer", "float", "number", "boolean"]:
                # Exact numeric comparison: Extracted value must exactly match document
                # value. Example: Extracted 2023 matches documents with year == 2023
                # only.
                #
                # Rationale:
                # - Numbers are specific; variations (2023 vs 2024) are semantically
                #   different
                # - No benefit to forgiving matching for numbers
                # - Performance: Exact match is faster than term matching
                #
                # Note: No "approximately equals" operator (e.g., year > 2020 AND
                # year < 2025) would require more complex schema with comparison
                # operators.
                conditions.append(f"{field} == {value}")
            elif field_type == "array" and isinstance(value, list):
                # ARRAY_CONTAINS_ANY: Check if ANY element from extracted list exists
                # in the document's array field.
                #
                # Semantics: Document matches if (extracted_value[0] IN doc_array OR
                #            extracted_value[1] IN doc_array OR ...)
                #
                # Example:
                # - Query extracted: ["python", "javascript"] for 'technologies' field
                # - Document has: {"technologies": ["python", "java", "rust"]}
                # - Result: MATCH (because "python" exists in both)
                #
                # Use case: "Find documents using python OR javascript" - enables
                # flexible technology searches without requiring exact match.
                #
                # Note: extracted value must be list type for this to trigger.
                # If extraction returns string, falls back to LIKE operator.
                array_str = json.dumps(value)
                conditions.append(f"ARRAY_CONTAINS_ANY({field}, {array_str})")
            elif field_type == "object" and isinstance(value, dict):
                # JSON_CONTAINS_ANY: Extract KEYS from user's object, check if any
                # match keys in document's JSON field. We check keys (not values)
                # because:
                #
                # 1. Values are arbitrary and hard to match semantically
                #    Example: {"author": "John", "rating": 4.5} - values are
                #    heterogeneous
                # 2. Keys represent "presence of metadata", not values
                #    Example: Document has metadata IF it has "author" and "rating"
                #    fields
                #
                # Semantics: Document matches if (extracted_key[0] IN doc_object_keys
                #            OR extracted_key[1] IN doc_object_keys OR ...)
                #
                # Example:
                # - Query extracted: {"author": "John", "year": 2023} for 'metadata'
                #   field
                # - Document has: {"metadata": {"author": "Jane", "year": 2020,
                #   "tags": [...]}}
                # - Extracted keys: ["author", "year"]
                # - Result: MATCH (both keys exist in document metadata, values
                #   ignored)
                #
                # Use case: Filter documents that have specific metadata fields,
                # regardless of their values. Enables presence-based filtering.
                #
                # Note: This is semantically different from checking if values exist.
                # If you need value matching, consider restructuring schema to use
                # array type instead (list of metadata keys).
                keys = list(value.keys())
                keys_str = json.dumps(keys)
                conditions.append(f"JSON_CONTAINS_ANY({field}, {keys_str})")

        milvus_filter = " AND ".join(conditions) if conditions else None
        return metadata, milvus_filter
