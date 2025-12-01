"""Test suite for UnstructuredAPIDocumentLoader."""

from unittest.mock import Mock, call, patch

import pytest
from langchain_core.documents import Document

from rag_pipelines.utils.unstructured.unstructured_api_loader import (
    UnstructuredAPIDocumentLoader,
)


# Biomedical text examples from PubMed abstracts
BIOMED_TEXT_1 = (
    "The renin-angiotensin-aldosterone system (RAAS) plays a critical role in blood pressure regulation. "
    "Dysregulation of RAAS is implicated in hypertension, heart failure, and chronic kidney disease. "
    "Angiotensin-converting enzyme inhibitors (ACEIs) are first-line therapies for these conditions. "
    "Recent meta-analyses confirm a 22% reduction in major cardiovascular events with ACEI therapy (95% CI: 18-26%)."
)

BIOMED_TEXT_2 = (
    "Sodium-glucose cotransporter-2 (SGLT2) inhibitors demonstrate significant cardiorenal benefits "
    "in patients with type 2 diabetes. The EMPEROR-Preserved trial showed empagliflozin reduced the risk "
    "of cardiovascular death or hospitalization for heart failure by 21% (HR 0.79; 95% CI 0.69-0.90) "
    "in heart failure patients with preserved ejection fraction."
)

BIOMED_TEXT_3 = (
    "CRISPR-Cas9-mediated gene editing of BCL11A enhancer in autologous CD34+ hematopoietic stem cells "
    "resulted in durable fetal hemoglobin induction in patients with sickle cell disease. "
    "In a phase 1 trial, 94% of edited cells engrafted successfully, with complete resolution of vaso-occlusive "
    "crises in all 6 treated patients at 18-month follow-up (NEJM 2021;384:252-260)."
)

BIOMED_TEXT_4 = (
    "Gut microbiome composition significantly influences response to immune checkpoint inhibitors in "
    "metastatic melanoma. Patients with higher baseline abundance of Faecalibacterium prausnitzii "
    "exhibited improved progression-free survival (median 31.2 vs 11.9 months; HR 0.45, 95% CI 0.23-0.88) "
    "when treated with anti-PD-1 therapy (Science 2021;371:595-603)."
)

BIOMED_TEXT_5 = (
    "Single-cell RNA sequencing reveals distinct tumor-infiltrating lymphocyte subsets associated with "
    "response to neoadjuvant chemotherapy in triple-negative breast cancer. CD8+ T cells expressing "
    "TCF7 and IL-7R were enriched in pathological complete responders (pCR), with 89% pCR rate when "
    "these subsets comprised >15% of TILs (Nature Med 2022;28:50-61)."
)


class TestUnstructuredAPIDocumentLoader:
    """Test suite for UnstructuredAPIDocumentLoader with biomedical context."""

    def test_get_file_paths_non_existent_dir(self):
        """Test directory validation for non-existent path."""
        loader = UnstructuredAPIDocumentLoader()
        with pytest.raises(ValueError, match="Directory does not exist"):
            loader._get_all_file_paths_from_directory("/non/existent/path")

    def test_get_file_paths_not_a_directory(self, tmp_path):
        """Test directory validation for file path."""
        file_path = tmp_path / "test.pdf"
        file_path.touch()

        loader = UnstructuredAPIDocumentLoader()
        with pytest.raises(ValueError, match="Path is not a directory"):
            loader._get_all_file_paths_from_directory(str(file_path))

    def test_get_file_paths_valid_directory(self, tmp_path):
        """Test successful file path retrieval from directory."""
        # Create test directory structure
        (tmp_path / "oncology").mkdir()
        (tmp_path / "cardiology").mkdir()

        (tmp_path / "RAAS_mechanisms.pdf").touch()
        (tmp_path / "oncology" / "immunotherapy_microbiome.pdf").touch()
        (tmp_path / "cardiology" / "SGLT2_cardiorenal.pdf").touch()
        (
            tmp_path / "cardiology" / "clinical_trial_data.xlsx"
        ).touch()  # Should be included as well

        loader = UnstructuredAPIDocumentLoader()
        paths = loader._get_all_file_paths_from_directory(str(tmp_path))

        # Verify all files are returned (not just PDFs)
        assert len(paths) == 4
        expected_files = [
            str(tmp_path / "RAAS_mechanisms.pdf"),
            str(tmp_path / "oncology" / "immunotherapy_microbiome.pdf"),
            str(tmp_path / "cardiology" / "SGLT2_cardiorenal.pdf"),
            str(tmp_path / "cardiology" / "clinical_trial_data.xlsx"),
        ]
        assert sorted(paths) == sorted(expected_files)

    def test_get_file_paths_empty_directory(self, tmp_path):
        """Test empty directory handling."""
        loader = UnstructuredAPIDocumentLoader()
        paths = loader._get_all_file_paths_from_directory(str(tmp_path))
        assert paths == []

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_basic(self, mock_loader, tmp_path):
        """Test document transformation with basic API."""
        # Setup test files
        (tmp_path / "RAAS_review.pdf").touch()
        (tmp_path / "SGLT2_trials.pdf").touch()

        # Configure mock loader responses
        mock_loader.return_value.load.side_effect = [
            [
                Document(page_content=BIOMED_TEXT_1),
                Document(page_content=BIOMED_TEXT_2),
            ],
            [Document(page_content=BIOMED_TEXT_3)],
        ]

        # Execute transformation
        loader = UnstructuredAPIDocumentLoader(
            partition_via_api=True,
            api_key="test-key-123",
            url="https://api.unstructured.io/v0.0.16",
        )
        documents = loader.transform_documents(str(tmp_path))

        # Verify loader initialization
        assert mock_loader.call_count == 2

        # Check that both files were processed (order may vary due to rglob)
        expected_calls = [
            call(
                file_path=str(tmp_path / "RAAS_review.pdf"),
                partition_via_api=True,
                post_processors=None,
                api_key="test-key-123",
                client=None,
                url="https://api.unstructured.io/v0.0.16",
                web_url=None,
            ),
            call(
                file_path=str(tmp_path / "SGLT2_trials.pdf"),
                partition_via_api=True,
                post_processors=None,
                api_key="test-key-123",
                client=None,
                url="https://api.unstructured.io/v0.0.16",
                web_url=None,
            ),
        ]
        # Use any_order=True to allow for different file order from rglob
        mock_loader.assert_has_calls(expected_calls, any_order=True)

        # Verify document content
        assert len(documents) == 3
        # Check that the expected content is present in the returned documents
        all_content = [doc.page_content for doc in documents]
        assert any(BIOMED_TEXT_1 in content for content in all_content)
        assert any(BIOMED_TEXT_2 in content for content in all_content)
        assert any(BIOMED_TEXT_3 in content for content in all_content)

        # Verify that specific terms are present
        content_str = " ".join(all_content)
        assert "ACEIs" in content_str  # Key term
        assert "HR 0.79" in content_str  # Clinical trial statistic
        assert "CRISPR-Cas9" in content_str  # Gene editing terminology
        assert "NEJM 2021" in content_str  # Journal reference

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_with_post_processors(self, mock_loader, tmp_path):
        """Test transformation with post-processors on biomedical text."""
        (tmp_path / "cancer_therapy.pdf").touch()

        # Define biomedical-specific post-processors
        processors = [
            lambda x: x.replace("PD-1", "programmed cell death protein 1"),
            lambda x: x.replace("TILs", "tumor-infiltrating lymphocytes"),
        ]

        # Configure mock with oncology text that contains terms to be replaced
        original_content = BIOMED_TEXT_4  # This contains "PD-1" but not "TILs"
        mock_loader.return_value.load.return_value = [
            Document(page_content=original_content)
        ]

        loader = UnstructuredAPIDocumentLoader(post_processors=processors)
        documents = loader.transform_documents(str(tmp_path))

        # Verify that the loader was called with the correct post-processors
        mock_loader.assert_called_once()
        assert mock_loader.call_args[1]["post_processors"] == processors

        # Since we're mocking, the content will be the original content
        content = documents[0].page_content
        assert "Faecalibacterium" in content  # Original term preserved
        assert "HR 0.45" in content  # Statistics preserved

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_api_client(self, mock_loader, tmp_path):
        """Test transformation with pre-configured client for biomedical processing."""
        (tmp_path / "gene_editing.pdf").touch()

        # Create mock client
        mock_client = Mock()
        # The client is passed to UnstructuredLoader which handles the API calls

        loader = UnstructuredAPIDocumentLoader(
            client=mock_client, partition_via_api=True, api_key="biomed-api-key-789"
        )
        # Configure mock loader to return expected content
        mock_loader.return_value.load.return_value = [
            Document(page_content=BIOMED_TEXT_3)
        ]
        documents = loader.transform_documents(str(tmp_path))

        # Verify client propagation and biomedical content
        mock_loader.assert_called_once()
        assert mock_loader.call_args[1]["client"] is mock_client
        assert "CRISPR-Cas9" in documents[0].page_content
        assert "BCL11A" in documents[0].page_content
        assert "NEJM 2021" in documents[0].page_content

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_no_files(self, mock_loader, tmp_path):
        """Test transformation with empty directory."""
        loader = UnstructuredAPIDocumentLoader()
        documents = loader.transform_documents(str(tmp_path))

        assert documents == []
        mock_loader.assert_not_called()

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_mixed_file_types(self, mock_loader, tmp_path):
        """Test transformation processing all file types in biomedical context."""
        # Create realistic biomedical directory structure
        (tmp_path / "clinical_trial_data.pdf").touch()
        (tmp_path / "patient_data.csv").touch()
        (tmp_path / "histology_image.png").touch()
        (tmp_path / "supplemental_materials.docx").touch()

        # Since all files will be processed, and each call to load() returns the same
        # content, we need to set up the mock to return different content for each call
        # or just verify that the expected number of calls were made
        mock_loader.return_value.load.side_effect = [
            [Document(page_content=BIOMED_TEXT_5)],  # For clinical_trial_data.pdf
            [Document(page_content=BIOMED_TEXT_5)],  # For patient_data.csv
            [Document(page_content=BIOMED_TEXT_5)],  # For histology_image.png
            [Document(page_content=BIOMED_TEXT_5)],  # For supplemental_materials.docx
        ]

        loader = UnstructuredAPIDocumentLoader()
        documents = loader.transform_documents(str(tmp_path))

        # All files should be processed (not just PDFs)
        assert len(documents) == 4
        # All documents will have the same content since we're mocking with same content
        for doc in documents:
            assert "Single-cell RNA sequencing" in doc.page_content
            assert "triple-negative breast cancer" in doc.page_content
        assert mock_loader.call_count == 4
        # Check that all files were processed by examining the call_args_list
        called_paths = []
        for call_item in mock_loader.call_args_list:
            # call_item is a MockCall object; get the file_path from kwargs
            if call_item.kwargs and "file_path" in call_item.kwargs:
                called_paths.append(call_item.kwargs["file_path"])
            # Fallback to positional args if needed
            elif call_item.args:
                called_paths.append(call_item.args[0])
            # Fallback to old format if needed
            elif call_item[0]:
                called_paths.append(call_item[0][0])

        assert any("clinical_trial_data.pdf" in path for path in called_paths)
        assert any("patient_data.csv" in path for path in called_paths)
        assert any("histology_image.png" in path for path in called_paths)
        assert any("supplemental_materials.docx" in path for path in called_paths)

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_recursive_search(self, mock_loader, tmp_path):
        """Test recursive directory search in biomedical research context."""
        # Create nested biomedical research structure
        (tmp_path / "2023_publications").mkdir()
        (tmp_path / "2023_publications" / "oncology").mkdir()
        (tmp_path / "2023_publications" / "oncology" / "immunotherapy").mkdir()
        (tmp_path / "2022_publications").mkdir()

        # Create realistic publication files
        (tmp_path / "2023_publications" / "RAAS_review.pdf").touch()
        (tmp_path / "2023_publications" / "oncology" / "microbiome_impact.pdf").touch()
        (
            tmp_path
            / "2023_publications"
            / "oncology"
            / "immunotherapy"
            / "TIL_subsets.pdf"
        ).touch()
        (tmp_path / "2022_publications" / "SGLT2_mechanisms.pdf").touch()

        # Configure mock responses with different biomedical content
        mock_loader.return_value.load.side_effect = [
            [Document(page_content=BIOMED_TEXT_1)],  # RAAS_review
            [Document(page_content=BIOMED_TEXT_4)],  # microbiome_impact
            [Document(page_content=BIOMED_TEXT_5)],  # TIL_subsets
            [Document(page_content=BIOMED_TEXT_2)],  # SGLT2_mechanisms
        ]

        loader = UnstructuredAPIDocumentLoader()
        documents = loader.transform_documents(str(tmp_path))

        # Verify all PDFs in nested structure are processed
        assert len(documents) == 4
        assert mock_loader.call_count == 4

        # Verify specific biomedical content in the documents (order may vary)
        all_content = [doc.page_content for doc in documents]
        assert any("ACEIs" in content for content in all_content)  # RAAS review
        assert any(
            "Faecalibacterium" in content for content in all_content
        )  # microbiome paper
        # The original text doesn't contain "tumor-infiltrating lymphocytes" but "TILs"
        # So we need to check for the original term
        assert any("TIL" in content for content in all_content)  # TIL subsets
        assert any("empagliflozin" in content for content in all_content)  # SGLT2 paper

        # Verify recursive path handling
        # Check that the call_args_list is not empty and has the expected structure
        assert len(mock_loader.call_args_list) == 4
        called_paths = []
        for call_item in mock_loader.call_args_list:
            # call_item is a MockCall object; get the file_path from kwargs
            if call_item.kwargs and "file_path" in call_item.kwargs:
                called_paths.append(call_item.kwargs["file_path"])
            # Fallback to positional args if needed
            elif call_item.args:
                called_paths.append(call_item.args[0])
            # Fallback to old format if needed
            elif call_item[0]:
                called_paths.append(call_item[0][0])

        expected_paths = [
            str(tmp_path / "2023_publications" / "RAAS_review.pdf"),
            str(tmp_path / "2023_publications" / "oncology" / "microbiome_impact.pdf"),
            str(
                tmp_path
                / "2023_publications"
                / "oncology"
                / "immunotherapy"
                / "TIL_subsets.pdf"
            ),
            str(tmp_path / "2022_publications" / "SGLT2_mechanisms.pdf"),
        ]

        # Check that all expected paths are in the called paths (order may vary)
        for expected_path in expected_paths:
            found = any(expected_path in called_path for called_path in called_paths)
            assert found, f"Expected path {expected_path} not found in {called_paths}"

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_with_empty_response(self, mock_loader, tmp_path):
        """Test handling of empty API responses for biomedical documents."""
        (tmp_path / "empty_response.pdf").touch()

        # Simulate API returning no content (possible with corrupted PDFs)
        mock_loader.return_value.load.return_value = []

        loader = UnstructuredAPIDocumentLoader()
        documents = loader.transform_documents(str(tmp_path))

        assert len(documents) == 0
        assert mock_loader.call_count == 1
        # Verify the correct file was passed to the loader by checking call_args_list
        called_args = mock_loader.call_args_list
        assert len(called_args) == 1
        # Get the first call's file_path argument
        call_item = called_args[0]
        # call_item is a MockCall object; get the file_path from kwargs
        if call_item.kwargs and "file_path" in call_item.kwargs:
            assert "empty_response.pdf" in call_item.kwargs["file_path"]
        # Fallback to positional args if needed
        elif call_item.args:
            assert "empty_response.pdf" in call_item.args[0]
        # Fallback to old format if needed
        elif call_item[0]:
            assert "empty_response.pdf" in call_item[0][0]
        else:
            # If positional args are not in the expected format
            raise AssertionError(f"Expected call with file_path, got {call_item}")

    @patch(
        "rag_pipelines.utils.unstructured.unstructured_api_loader.UnstructuredLoader"
    )
    def test_transform_documents_with_malformed_biomedical_text(
        self, mock_loader, tmp_path
    ):
        """Test handling of malformed biomedical text with special characters."""
        (tmp_path / "genetics_study.pdf").touch()

        # Simulate biomedical text with special characters and encoding issues
        malformed_text = (
            "CYP2C19*2 (rs4244285) loss-of-function alleles impair clopidogrel activation. "
            "Patients with ≥1 *2 allele had higher risk of stent thrombosis (HR 2.31; 95% CI 1.67-3.19). "
            "Next-generation sequencing revealed novel variants: c.681G>A (p.Trp227Ter) and c.431G>T (p.Gly144Val)."
        )

        mock_loader.return_value.load.return_value = [
            Document(page_content=malformed_text)
        ]

        loader = UnstructuredAPIDocumentLoader()
        documents = loader.transform_documents(str(tmp_path))

        # Verify special biomedical characters are preserved
        content = documents[0].page_content
        assert "CYP2C19*2" in content
        assert "rs4244285" in content
        assert "p.Trp227Ter" in content
        assert "HR 2.31" in content
