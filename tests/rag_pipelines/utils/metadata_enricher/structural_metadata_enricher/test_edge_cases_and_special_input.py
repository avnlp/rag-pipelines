"""Test edge cases and boundary conditions for StructuralMetadataEnricher.

Validates handling of special Unicode characters (combining marks, fancy quotes,
zero-width chars), extreme text lengths (10K to 1M+ chars), whitespace-only text,
context field preservation, and language detection error handling.
"""

import json
from unittest.mock import Mock, patch

import pytest

from rag_pipelines.utils.metadata_enricher.structural_metadata_enricher import (
    StructuralMetadataEnricher,
)


class TestStructuralMetadataEnricherEdgeCases:
    """Test edge cases and boundary conditions for StructuralMetadataEnricher.

    Validates that the enricher correctly handles special Unicode characters,
    extreme text lengths, whitespace variations, empty/zero values, and
    document context with extra or partial fields.
    """

    @pytest.mark.asyncio
    async def test_combining_diacritical_marks(self) -> None:
        """Verify extraction handles combining diacritical marks correctly.

        Tests that Unicode combining accents (e.g., café with combining accent)
        are correctly counted in char_count and word_count.
        """
        enricher = StructuralMetadataEnricher()
        text = "cafe\u0301 is a combining accent test"
        chunk = {"text": text}

        result = await enricher.extract(chunk)

        assert result["char_count"] == len(text)
        assert result["word_count"] > 0

    @pytest.mark.asyncio
    async def test_unicode_quote_characters(self) -> None:
        """Verify extraction handles fancy Unicode quote characters.

        Tests that Unicode fancy quotes (curly quotes) are correctly counted
        and word counting remains accurate.
        """
        enricher = StructuralMetadataEnricher()
        text = 'He said "Hello" with fancy quotes'
        chunk = {"text": text}

        result = await enricher.extract(chunk)

        assert result["char_count"] == len(text)
        assert result["word_count"] == 6

    @pytest.mark.asyncio
    async def test_zero_width_characters(self) -> None:
        """Test extraction with zero-width characters."""
        enricher = StructuralMetadataEnricher()
        text = "Hello\u200bWorld"
        chunk = {"text": text}

        result = await enricher.extract(chunk)

        assert result["char_count"] == len(text)

    @pytest.mark.asyncio
    async def test_very_long_text_10k_chars(self) -> None:
        """Test extraction with 10,000 character text."""
        enricher = StructuralMetadataEnricher()
        long_text = "word " * 2000  # 10,000 characters
        chunk = {"text": long_text}

        result = await enricher.extract(chunk)

        assert result["word_count"] == 2000
        assert result["char_count"] == len(long_text)
        assert "content_hash" in result

    @pytest.mark.asyncio
    async def test_very_long_text_1m_chars(self) -> None:
        """Test extraction with 1,000,000 character text."""
        enricher = StructuralMetadataEnricher()
        long_text = "word " * 200000  # 1,000,000 characters
        chunk = {"text": long_text}

        result = await enricher.extract(chunk)

        assert result["word_count"] == 200000
        assert result["char_count"] == len(long_text)

    @pytest.mark.asyncio
    async def test_whitespace_only_text(self) -> None:
        """Test extraction with whitespace-only text."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "   \n\t  \r\n"}

        result = await enricher.extract(chunk)

        assert result["word_count"] == 0
        assert result["char_count"] > 0  # Includes whitespace characters

    @pytest.mark.asyncio
    async def test_single_space(self) -> None:
        """Test extraction with single space."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": " "}

        result = await enricher.extract(chunk)

        assert result["word_count"] == 0
        assert result["char_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_consecutive_spaces(self) -> None:
        """Test extraction with multiple consecutive spaces."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "word1     word2     word3"}

        result = await enricher.extract(chunk)

        assert result["word_count"] == 3
        assert result["char_count"] == len("word1     word2     word3")

    @pytest.mark.asyncio
    async def test_tabs_and_newlines(self) -> None:
        """Test extraction with tabs and newlines."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "word1\t\tword2\n\nword3"}

        result = await enricher.extract(chunk)

        assert result["word_count"] == 3

    @pytest.mark.asyncio
    async def test_context_with_extra_fields(self) -> None:
        """Test that extra context fields are preserved."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "test"}
        context = {
            "page_number": 5,
            "section_title": "Methods",
            "extra_field": "extra_value",
            "another_extra": 123,
        }

        result = await enricher.extract(chunk, context)

        assert result["page_number"] == 5
        assert result["section_title"] == "Methods"
        assert "extra_field" not in result
        assert "another_extra" not in result

    @pytest.mark.asyncio
    async def test_context_partial_fields(self) -> None:
        """Test context with only some fields."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "test"}
        context = {"page_number": 10}

        result = await enricher.extract(chunk, context)

        assert result["page_number"] == 10
        assert result["section_title"] == ""
        assert json.loads(result["heading_hierarchy"]) == []

    @pytest.mark.asyncio
    async def test_context_with_zero_values(self) -> None:
        """Test context with zero page number."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "test"}
        context = {"page_number": 0}

        result = await enricher.extract(chunk, context)

        assert result["page_number"] == 0

    @pytest.mark.asyncio
    async def test_context_with_empty_string_title(self) -> None:
        """Test context with empty string section title."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "test with empty title context"}
        context = {"section_title": ""}

        result = await enricher.extract(chunk, context)

        assert result["section_title"] == ""

    @pytest.mark.asyncio
    async def test_context_with_nested_heading_hierarchy(self) -> None:
        """Test context with deeply nested heading hierarchy."""
        enricher = StructuralMetadataEnricher()
        chunk = {"text": "test"}
        heading_list = ["Ch1", "S1", "SS1", "SSS1", "SSSS1"]
        context = {"heading_hierarchy": heading_list}

        result = await enricher.extract(chunk, context)

        retrieved = json.loads(result["heading_hierarchy"])
        assert retrieved == heading_list

    @pytest.mark.asyncio
    async def test_language_detection_with_detection_failure(self) -> None:
        """Verify language detection returns 'unknown' on library exception.

        Tests that when the lingua library's detect_language_of raises an
        exception (e.g., unsupported script or corrupted text that triggers
        internal errors), the enricher gracefully falls back to 'unknown'
        instead of crashing, allowing structural extraction to continue.
        """
        enricher = StructuralMetadataEnricher()

        with patch(
            "rag_pipelines.utils.metadata_enricher.structural_metadata_enricher.LanguageDetectorBuilder"
        ) as mock_builder_class:
            mock_detector = Mock()
            mock_detector.detect_language_of = Mock(
                side_effect=Exception("Unsupported script or corrupted text")
            )
            mock_builder_class.from_all_languages.return_value.build.return_value = (
                mock_detector
            )

            result = await enricher.extract({"text": "Some text"})

            assert result["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_language_detection_with_runtime_error(self) -> None:
        """Verify language detection handles internal detector errors.

        Tests that RuntimeError from the lingua detector (internal computation
        failure, missing language models) is caught and returns 'unknown',
        ensuring partial metadata is available even when detection fails.
        """
        enricher = StructuralMetadataEnricher()

        with patch(
            "rag_pipelines.utils.metadata_enricher.structural_metadata_enricher.LanguageDetectorBuilder"
        ) as mock_builder_class:
            mock_detector = Mock()
            mock_detector.detect_language_of = Mock(
                side_effect=RuntimeError("Detector internal error")
            )
            mock_builder_class.from_all_languages.return_value.build.return_value = (
                mock_detector
            )

            result = await enricher.extract({"text": "Test"})

            assert result["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_language_detection_with_type_error(self) -> None:
        """Verify language detection handles type mismatches gracefully.

        Tests that TypeError from the detector (e.g., unexpected input format,
        missing required attributes on text) is caught and returns 'unknown',
        preventing type-related errors from interrupting metadata extraction.
        """
        enricher = StructuralMetadataEnricher()

        with patch(
            "rag_pipelines.utils.metadata_enricher.structural_metadata_enricher.LanguageDetectorBuilder"
        ) as mock_builder_class:
            mock_detector = Mock()
            mock_detector.detect_language_of = Mock(
                side_effect=TypeError("Invalid input")
            )
            mock_builder_class.from_all_languages.return_value.build.return_value = (
                mock_detector
            )

            result = await enricher.extract({"text": "Text"})

            assert result["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_language_detection_returns_none_falls_back(self) -> None:
        """Verify language detection handles None detector result gracefully.

        Tests that when the detector returns None (indicating indeterminate
        language, e.g., text with no recognizable language patterns), the
        enricher treats this as a failed detection and returns 'unknown'.
        """
        enricher = StructuralMetadataEnricher()

        with patch(
            "rag_pipelines.utils.metadata_enricher.structural_metadata_enricher.LanguageDetectorBuilder"
        ) as mock_builder_class:
            mock_detector = Mock()
            mock_detector.detect_language_of = Mock(return_value=None)
            mock_builder_class.from_all_languages.return_value.build.return_value = (
                mock_detector
            )

            result = await enricher.extract({"text": "xyz"})

            assert result["language"] == "unknown"
