"""Test suite for UnstructuredDocumentLoader class."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from rag_pipelines.utils.unstructured.unstructured_pdf_loader import (
    UnstructuredDocumentLoader,
)


class TestUnstructuredDocumentLoader:
    """Test suite for UnstructuredDocumentLoader class."""

    @pytest.fixture
    def sample_biomedical_texts(self):
        """Return sample biomedical texts from PubMed for testing."""
        return [
            "The role of TGF-β signaling in tumor progression and metastasis "
            "has been extensively studied in colorectal cancer patients.",
            "Recent advances in CRISPR-Cas9 gene editing have shown promising "
            "results in treating sickle cell anemia and β-thalassemia.",
            "The gut microbiome composition significantly affects the efficacy "
            "of immune checkpoint inhibitors in melanoma treatment.",
            "Alzheimer's disease progression is correlated with amyloid-β "
            "plaque accumulation and tau protein hyperphosphorylation.",
        ]

    @pytest.fixture
    def mock_documents(self, sample_biomedical_texts):
        """Create mock Document objects with biomedical content."""
        return [
            Document(page_content=text, metadata={"source": f"test_{i}.pdf", "page": 1})
            for i, text in enumerate(sample_biomedical_texts)
        ]

    @pytest.fixture
    def temp_directory(self, tmp_path):
        """Create a temporary directory with test PDF files."""
        # Create main directory
        pdf_dir = tmp_path / "test_pdfs"
        pdf_dir.mkdir()

        # Create some test PDF files
        (pdf_dir / "document1.pdf").write_text("Test PDF 1")
        (pdf_dir / "document2.pdf").write_text("Test PDF 2")

        # Create a subdirectory with more PDFs
        sub_dir = pdf_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "document3.pdf").write_text("Test PDF 3")

        return str(pdf_dir)

    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        loader = UnstructuredDocumentLoader()

        assert loader.strategy == "hi_res"
        assert loader.mode == "elements"
        assert loader.include_page_breaks is False
        assert loader.infer_table_structure is False
        assert loader.ocr_languages is None
        assert loader.languages is None
        assert loader.hi_res_model_name is None
        assert loader.extract_images_in_pdf is False
        assert loader.extract_image_block_types is None
        assert loader.extract_image_block_output_dir is None
        assert loader.extract_image_block_to_payload is False
        assert loader.starting_page_number == 1
        assert loader.extract_forms is False
        assert loader.form_extraction_skip_tables is True

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_params = {
            "strategy": "fast",
            "mode": "single",
            "include_page_breaks": True,
            "infer_table_structure": True,
            "ocr_languages": "eng+fra",
            "languages": ["eng", "fra"],
            "hi_res_model_name": "yolox",
            "extract_images_in_pdf": True,
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_output_dir": "/tmp/images",
            "extract_image_block_to_payload": True,
            "starting_page_number": 5,
            "extract_forms": True,
            "form_extraction_skip_tables": False,
        }

        loader = UnstructuredDocumentLoader(**custom_params)

        for key, value in custom_params.items():
            assert getattr(loader, key) == value

    def test_get_all_file_paths_from_directory_valid(self, temp_directory):
        """Test retrieving file paths from valid directory."""
        loader = UnstructuredDocumentLoader()

        file_paths = loader._get_all_file_paths_from_directory(temp_directory)

        assert len(file_paths) == 3
        assert all(path.endswith(".pdf") for path in file_paths)
        assert any("document1.pdf" in path for path in file_paths)
        assert any("document2.pdf" in path for path in file_paths)
        assert any("document3.pdf" in path for path in file_paths)

    def test_get_all_file_paths_from_directory_nonexistent(self):
        """Test retrieving file paths from nonexistent directory."""
        loader = UnstructuredDocumentLoader()

        with pytest.raises(ValueError, match="Directory does not exist:"):
            loader._get_all_file_paths_from_directory("/nonexistent/path")

    def test_get_all_file_paths_from_directory_file_path(self, tmp_path):
        """Test retrieving file paths when path is a file, not directory."""
        loader = UnstructuredDocumentLoader()

        file_path = tmp_path / "test_file.pdf"
        file_path.write_text("test content")

        with pytest.raises(ValueError, match="Path is not a directory:"):
            loader._get_all_file_paths_from_directory(str(file_path))

    def test_get_all_file_paths_from_directory_empty(self, tmp_path):
        """Test retrieving file paths from empty directory."""
        loader = UnstructuredDocumentLoader()

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        file_paths = loader._get_all_file_paths_from_directory(str(empty_dir))

        assert file_paths == []

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_transform_documents_success(
        self, mock_loader_class, temp_directory, mock_documents
    ):
        """Test successful document transformation with biomedical content."""
        # Mock the loader instance and its load method
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = mock_documents
        mock_loader_class.return_value = mock_loader_instance

        loader = UnstructuredDocumentLoader()
        documents = loader.transform_documents(temp_directory)

        # Verify UnstructuredPDFLoader was called for each file
        assert mock_loader_class.call_count == 3

        # Verify all documents were processed and returned
        # Since there are 3 PDF files in the temp directory and mock_documents
        # has 4 documents, each PDF will return the same 4 documents,
        # resulting in 12 total documents
        assert len(documents) == 12
        assert all(isinstance(doc, Document) for doc in documents)

        # Verify biomedical content is present
        biomedical_keywords = ["TGF-β", "CRISPR-Cas9", "microbiome", "Alzheimer's"]
        content = " ".join(doc.page_content for doc in documents)
        for keyword in biomedical_keywords:
            assert keyword in content

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_transform_documents_custom_parameters(
        self, mock_loader_class, temp_directory
    ):
        """Test document transformation with custom parameters."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="test", metadata={})
        ]
        mock_loader_class.return_value = mock_loader_instance

        custom_params = {
            "strategy": "fast",
            "mode": "single",
            "include_page_breaks": True,
            "infer_table_structure": True,
            "ocr_languages": "eng",
            "languages": ["eng"],
            "hi_res_model_name": "detectron2",
            "extract_images_in_pdf": True,
            "starting_page_number": 2,
            "extract_forms": True,
            "form_extraction_skip_tables": False,
        }

        loader = UnstructuredDocumentLoader(**custom_params)
        loader.transform_documents(temp_directory)

        # Verify UnstructuredPDFLoader was initialized with correct parameters
        for call_args in mock_loader_class.call_args_list:
            args, kwargs = call_args

            assert kwargs["strategy"] == "fast"
            assert kwargs["mode"] == "single"
            assert kwargs["include_page_breaks"] is True
            assert kwargs["infer_table_structure"] is True
            assert kwargs["ocr_languages"] == "eng"
            assert kwargs["languages"] == ["eng"]
            assert kwargs["hi_res_model_name"] == "detectron2"
            assert kwargs["extract_images_in_pdf"] is True
            assert kwargs["starting_page_number"] == 2
            assert kwargs["extract_forms"] is True
            assert kwargs["form_extraction_skip_tables"] is False

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_transform_documents_empty_directory(self, mock_loader_class, tmp_path):
        """Test document transformation with empty directory."""
        empty_dir = tmp_path / "empty_test_dir"
        empty_dir.mkdir()

        loader = UnstructuredDocumentLoader()
        documents = loader.transform_documents(str(empty_dir))

        # Verify no loader calls were made
        mock_loader_class.assert_not_called()
        assert documents == []

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_transform_documents_loader_exception(
        self, mock_loader_class, temp_directory
    ):
        """Test document transformation when loader raises an exception."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = Exception("PDF processing error")
        mock_loader_class.return_value = mock_loader_instance

        loader = UnstructuredDocumentLoader()

        with pytest.raises(Exception, match="PDF processing error"):
            loader.transform_documents(temp_directory)

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_transform_documents_mixed_success(
        self, mock_loader_class, temp_directory, mock_documents
    ):
        """Test document transformation with mixed success and failure."""
        mock_loader_instance = Mock()
        # First call succeeds, second fails, third succeeds
        mock_loader_instance.load.side_effect = [
            [mock_documents[0]],  # First file success
            Exception("Corrupted PDF"),  # Second file fails
            [mock_documents[2]],  # Third file success
        ]
        mock_loader_class.return_value = mock_loader_instance

        loader = UnstructuredDocumentLoader()

        with pytest.raises(Exception, match="Corrupted PDF"):
            loader.transform_documents(temp_directory)

    def test_document_structure_and_metadata(self, mock_documents):
        """Test that documents maintain proper structure and metadata."""
        for _i, doc in enumerate(mock_documents):
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert isinstance(doc.page_content, str)
            assert isinstance(doc.metadata, dict)
            assert "source" in doc.metadata
            assert "page" in doc.metadata

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_image_extraction_parameters(self, mock_loader_class, temp_directory):
        """Test document transformation with image extraction parameters."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="test", metadata={})
        ]
        mock_loader_class.return_value = mock_loader_instance

        image_params = {
            "extract_images_in_pdf": True,
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_output_dir": "/tmp/extracted",
            "extract_image_block_to_payload": True,
        }

        loader = UnstructuredDocumentLoader(**image_params)
        loader.transform_documents(temp_directory)

        # Verify image extraction parameters were passed correctly
        for call_args in mock_loader_class.call_args_list:
            _, kwargs = call_args

            assert kwargs["extract_images_in_pdf"] is True
            assert kwargs["extract_image_block_types"] == ["Image", "Table"]
            assert kwargs["extract_image_block_output_dir"] == "/tmp/extracted"
            assert kwargs["extract_image_block_to_payload"] is True

    @patch("rag_pipelines.utils.unstructured.unstructured_pdf_loader.Path")
    def test_path_resolution(self, mock_path_class, temp_directory):
        """Test that paths are properly resolved to absolute paths."""
        # Create a mock for the path instance that gets created inside the method
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_dir.return_value = True
        mock_path_instance.resolve.return_value = (
            mock_path_instance  # Return itself when resolve() is called
        )

        # Create mock file objects that can be iterated over
        mock_file1 = Mock()
        mock_file1.is_file.return_value = True
        mock_file1.__str__ = lambda self: "file1.pdf"
        mock_file2 = Mock()
        mock_file2.is_file.return_value = True
        mock_file2.__str__ = lambda self: "file2.pdf"

        mock_path_instance.rglob.return_value = [mock_file1, mock_file2]
        mock_path_class.return_value = mock_path_instance

        def path_constructor(path_str):
            # Return the same mock instance for any Path constructor call
            return mock_path_instance

        mock_path_class.side_effect = path_constructor

        loader = UnstructuredDocumentLoader()
        loader._get_all_file_paths_from_directory(temp_directory)

        # Verify Path was called with the directory path and resolve was called
        assert mock_path_class.called
        mock_path_instance.resolve.assert_called_once()

        # Verify the rglob method was called with "*"
        mock_path_instance.rglob.assert_called_once_with("*")

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_pdf_loader.UnstructuredPDFLoader"
    )
    def test_large_number_of_files(self, mock_loader_class, temp_directory):
        """Test processing with a large number of files."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="test", metadata={})
        ]
        mock_loader_class.return_value = mock_loader_instance

        # Create many files in the directory
        pdf_dir = Path(temp_directory)
        for i in range(10):
            (pdf_dir / f"large_batch_{i}.pdf").write_text(f"Content {i}")

        loader = UnstructuredDocumentLoader()
        documents = loader.transform_documents(temp_directory)

        # Verify all files were processed (original 3 + 10 new = 13)
        assert mock_loader_class.call_count == 13
        assert len(documents) == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
