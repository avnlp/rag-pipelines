"""Test suite for UnstructuredChunker class."""

import logging
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from unstructured.documents.elements import ElementMetadata, NarrativeText

from rag_pipelines.utils.unstructured.unstructured_chunker import UnstructuredChunker


# Sample biomedical texts from PubMed-style abstracts
BIOMED_TEXT_1 = (
    "Alzheimer's disease (AD) is a progressive neurodegenerative disorder characterized by "
    "the accumulation of amyloid-beta plaques and neurofibrillary tangles composed of hyperphosphorylated tau protein. "
    "Current therapeutic strategies focus on modulating these pathological hallmarks, though clinical efficacy remains limited."
)

BIOMED_TEXT_2 = (
    "CRISPR-Cas9 has revolutionized genome editing by enabling precise modifications to DNA sequences. "
    "Recent advances in delivery mechanisms, such as lipid nanoparticles and adeno-associated viruses, "
    "have improved targeting efficiency and reduced off-target effects in vivo."
)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=BIOMED_TEXT_1, metadata={"source": "pubmed", "id": "12345"}
        ),
        Document(
            page_content=BIOMED_TEXT_2, metadata={"source": "pubmed", "id": "67890"}
        ),
    ]


class TestUnstructuredChunker:
    """Test suite for UnstructuredChunker class."""

    def test_init_defaults(self):
        """Test default initialization of UnstructuredChunker."""
        chunker = UnstructuredChunker()
        assert chunker.chunking_strategy == "basic"
        assert chunker.max_characters == 500
        assert chunker.new_after_n_chars == 500
        assert chunker.overlap == 0
        assert chunker.overlap_all is False
        assert chunker.combine_text_under_n_chars is None
        assert chunker.include_orig_elements is None
        assert chunker.multipage_sections is None

    def test_init_custom_params(self):
        """Test initialization of UnstructuredChunker with custom parameters."""
        chunker = UnstructuredChunker(
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=900,
            overlap=50,
            overlap_all=True,
            combine_text_under_n_chars=200,
            include_orig_elements=True,
            multipage_sections=True,
        )
        assert chunker.chunking_strategy == "by_title"
        assert chunker.max_characters == 1000
        assert chunker.new_after_n_chars == 900
        assert chunker.overlap == 50
        assert chunker.overlap_all is True
        assert chunker.combine_text_under_n_chars == 200
        assert chunker.include_orig_elements is True
        assert chunker.multipage_sections is True

    def test_convert_documents_to_elements(self, sample_documents):
        """Test conversion of documents to elements."""
        chunker = UnstructuredChunker()
        elements, metadatas = chunker._convert_documents_to_elements(sample_documents)

        assert len(elements) == 2
        assert len(metadatas) == 2
        assert elements[0].text == BIOMED_TEXT_1
        assert elements[1].text == BIOMED_TEXT_2
        assert metadatas[0] == {"source": "pubmed", "id": "12345"}
        assert metadatas[1] == {"source": "pubmed", "id": "67890"}

    def test_convert_chunked_elements_to_documents(self):
        """Test conversion of chunked elements to documents."""
        chunker = UnstructuredChunker()
        # Create elements without passing metadata to constructor to avoid the issue
        mock_element1 = NarrativeText(text="Chunk 1")
        # Create proper ElementMetadata object and assign it
        mock_element1.metadata = ElementMetadata()
        # Set the source in the metadata appropriately
        mock_element1.metadata.__dict__["source"] = "test"

        mock_element2 = NarrativeText(text="Chunk 2")
        mock_element2.metadata = ElementMetadata()
        mock_element2.metadata.__dict__["source"] = "test"

        documents = chunker._convert_chunked_elements_to_documents(
            [mock_element1, mock_element2]
        )

        assert len(documents) == 2
        assert documents[0].page_content == "Chunk 1"
        assert documents[0].metadata == {"source": "test"}
        assert documents[1].page_content == "Chunk 2"
        assert documents[1].metadata == {"source": "test"}

    @patch("rag_pipelines.utils.unstructured.unstructured_chunker.chunk_elements")
    def test_transform_documents_basic_strategy(
        self, mock_chunk_elements, sample_documents
    ):
        """Test transformation of documents using basic strategy."""
        # Mock the unstructured chunking function
        mock_chunk1 = NarrativeText(text="Chunk A")
        mock_chunk2 = NarrativeText(text="Chunk B")
        mock_chunk_elements.return_value = [mock_chunk1, mock_chunk2]

        chunker = UnstructuredChunker(chunking_strategy="basic", max_characters=200)
        result = chunker.transform_documents(sample_documents)

        # Should call chunk_elements once per document
        assert mock_chunk_elements.call_count == 2
        call_args = mock_chunk_elements.call_args_list
        assert call_args[0].kwargs["max_characters"] == 200
        assert call_args[0].kwargs["new_after_n_chars"] == 500  # default
        assert call_args[0].kwargs["overlap"] == 0

        # Result should have 4 documents (2 per input doc)
        assert len(result) == 4
        assert result[0].page_content == "Chunk A"
        assert result[0].metadata == {"source": "pubmed", "id": "12345"}
        assert result[2].page_content == "Chunk A"
        assert result[2].metadata == {"source": "pubmed", "id": "67890"}

    @patch("rag_pipelines.utils.unstructured.unstructured_chunker.chunk_by_title")
    def test_transform_documents_by_title_strategy(
        self, mock_chunk_by_title, sample_documents
    ):
        """Test transformation of documents using by_title strategy."""
        mock_chunk = NarrativeText(text="Title-based chunk")
        mock_chunk_by_title.return_value = [mock_chunk]

        chunker = UnstructuredChunker(
            chunking_strategy="by_title",
            max_characters=300,
            combine_text_under_n_chars=100,
            multipage_sections=True,
        )
        result = chunker.transform_documents(sample_documents)

        assert mock_chunk_by_title.call_count == 2
        call_args = mock_chunk_by_title.call_args_list
        assert call_args[0].kwargs["max_characters"] == 300
        assert call_args[0].kwargs["combine_text_under_n_chars"] == 100
        assert call_args[0].kwargs["multipage_sections"] is True

        assert len(result) == 2
        assert result[0].page_content == "Title-based chunk"
        assert result[0].metadata == {"source": "pubmed", "id": "12345"}

    def test_transform_documents_empty_input(self):
        """Test transformation of documents with empty input."""
        chunker = UnstructuredChunker()
        with pytest.raises(ValueError, match="No documents provided"):
            chunker.transform_documents([])

    def test_transform_documents_unsupported_strategy(self, sample_documents):
        """Test transformation of documents with unsupported strategy."""
        chunker = UnstructuredChunker(chunking_strategy="invalid")
        with pytest.raises(ValueError, match="Unsupported chunking strategy"):
            chunker.transform_documents(sample_documents)

    def test_logging_behavior(self, sample_documents, caplog):
        """Test logging behavior during document transformation."""
        chunker = UnstructuredChunker(chunking_strategy="basic", max_characters=100)
        with patch(
            "rag_pipelines.utils.unstructured.unstructured_chunker.chunk_elements"
        ) as mock_chunk:
            mock_chunk.return_value = [NarrativeText(text="Short chunk")]

            with caplog.at_level(logging.INFO):
                result = chunker.transform_documents(sample_documents)

        assert "Transforming 2 documents using strategy: basic" in caplog.text
        assert "Combined all chunked documents into 2 documents." in caplog.text
        assert len(result) == 2

    def test_metadata_preservation_across_chunking(self, sample_documents):
        """Test metadata preservation across chunking."""
        chunker = UnstructuredChunker(
            chunking_strategy="basic", max_characters=200, overlap=20
        )
        result = chunker.transform_documents(sample_documents)

        # Each original doc should produce multiple chunks, all with same metadata
        first_doc_chunks = [d for d in result if d.metadata["id"] == "12345"]
        second_doc_chunks = [d for d in result if d.metadata["id"] == "67890"]

        assert len(first_doc_chunks) >= 1
        assert len(second_doc_chunks) >= 1

        for chunk in first_doc_chunks:
            assert chunk.metadata == {"source": "pubmed", "id": "12345"}
        for chunk in second_doc_chunks:
            assert chunk.metadata == {"source": "pubmed", "id": "67890"}

        # Ensure content is actually chunked
        total_chars_original = len(BIOMED_TEXT_1) + len(BIOMED_TEXT_2)
        total_chars_chunked = sum(len(d.page_content) for d in result)
        # Overlap may cause slight increase, but should be roughly same
        assert (
            abs(total_chars_chunked - total_chars_original) < 100
        )  # tolerance for overlap
