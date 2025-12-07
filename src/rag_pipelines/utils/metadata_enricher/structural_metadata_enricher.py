"""Layer 1: Structural Metadata extraction (fast, rule-based)."""

import hashlib
import json
from typing import Any, Dict, Optional

from cachetools import TTLCache
from cachetools_async import cached
from lingua import LanguageDetectorBuilder


_structural_cache: TTLCache[Any, Any] = TTLCache(maxsize=5000, ttl=86400)


def _make_structural_key(*args: Any, **kwargs: Any) -> tuple[str, str]:
    """Generate cache key from chunk and document_context."""
    chunk = args[0] if args else kwargs.get("chunk", {})
    document_context = args[1] if len(args) > 1 else kwargs.get("document_context")

    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
    chunk_hash = hashlib.sha256(text.encode()).hexdigest()

    if document_context:
        context_str = json.dumps(document_context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()
    else:
        context_hash = "none"

    return (chunk_hash, context_hash)


class StructuralMetadataEnricher:
    """Layer 1: Structural Metadata extraction (fast, rule-based).

    Extracts metadata that doesn't require LLM calls: content hashing,
    word/character counts, language detection, and inherited document
    context (page numbers, sections, headings). O(n) complexity where
    n is text length. This layer always runs regardless of enrichment mode.
    """

    def __init__(self) -> None:
        """Initialize the structural enricher."""

    @cached(cache=_structural_cache, key=_make_structural_key)  # type: ignore
    async def extract(
        self, chunk: Dict[str, Any], document_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract structural metadata from a chunk.

        Args:
            chunk: Dictionary with 'text' key containing the chunk content.
            document_context: Optional metadata from the parent document
                (e.g., page number, section title, heading hierarchy).

        Returns:
            Dictionary of extracted structural metadata including:
                - chunk_id: Unique identifier for the chunk
                - content_hash: MD5 hash of chunk content for deduplication
                - word_count: Number of words in the chunk
                - char_count: Number of characters in the chunk
                - language: Detected language ISO code
                - page_number: Source page (defaults to 1 if not available)
                - section_title: Section heading if available
                - heading_hierarchy: Nested heading path as JSON string if available
        """
        text = chunk.get("text", "")
        document_context = document_context or {}

        # Ensure text is a string (handle None or non-string values)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Calculate content hash for deduplication and caching. MD5 provides
        # sufficient collision resistance for identifying identical chunks.
        content_hash = hashlib.md5(text.encode()).hexdigest()

        # Count words using simple whitespace splitting. Suitable for most
        # languages; for CJK languages, this counts characters within tokens.
        word_count = len(text.split())

        # Detect language to enable language-specific processing in downstream
        # components. Gracefully degrades to 'unknown' on failure.
        language = self._detect_language(text)

        # Extract inherited metadata from parent document context to maintain
        # document structure relationships through the chunking process.
        page_number = document_context.get("page_number", 1)
        section_title = document_context.get("section_title", "")
        heading_hierarchy = document_context.get("heading_hierarchy", [])

        return {
            "chunk_id": f"chunk_{content_hash[:12]}",
            "content_hash": content_hash,
            "word_count": word_count,
            "language": language,
            "page_number": page_number,
            "section_title": section_title,
            "heading_hierarchy": json.dumps(heading_hierarchy),
            "char_count": len(text),
        }

    def _detect_language(self, text: str) -> str:
        """Detect language using lingua-py library.

        Uses LanguageDetectorBuilder from lingua library which supports 75+ languages
        with high accuracy. Falls back gracefully to 'unknown' if detection
        fails or text is too short for reliable detection.

        Args:
            text: Text to detect language for.

        Returns:
            Detected language ISO 639-1 code (e.g., 'en', 'fr', 'zh')
            or 'unknown' if detection fails.
        """
        try:
            detector = LanguageDetectorBuilder.from_all_languages().build()
            detected = detector.detect_language_of(text)
            if detected:
                return detected.iso_code_639_1.name.lower()
            return "unknown"
        except Exception:
            # Graceful fallback for texts too short, corrupted, or in
            # unsupported languages/scripts
            return "unknown"
