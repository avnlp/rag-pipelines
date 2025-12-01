"""Test suite for MetadataExtractor class."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag_pipelines.utils.metadata_extractor.metadata_extractor import MetadataExtractor


class TestMetadataExtractor:
    """Test suite for MetadataExtractor class."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM instance."""
        return Mock(spec=BaseChatModel)

    @pytest.fixture
    def extractor(self, mock_llm):
        """MetadataExtractor instance with mocked LLM."""
        return MetadataExtractor(llm=mock_llm)

    @pytest.fixture
    def biomedical_texts(self):
        """Sample biomedical texts from PubMed."""
        return {
            "cancer_study": (
                "A randomized controlled trial of 500 patients with metastatic non-small cell lung cancer "
                "showed that pembrolizumab combined with chemotherapy significantly improved overall survival "
                "(HR 0.65, 95% CI 0.55-0.78, p<0.001). The median progression-free survival was 8.3 months "
                "in the experimental group versus 5.6 months in the control group. Adverse events included "
                "fatigue (25%), nausea (18%), and rash (12%)."
            ),
            "diabetes_research": (
                "This multicenter study evaluated the efficacy of semaglutide in 1200 patients with type 2 diabetes. "
                "HbA1c reduction from baseline was -1.8% in the semaglutide group compared to -0.5% in the placebo group. "
                "Weight loss of ≥5% was achieved by 68% of patients. The study duration was 52 weeks."
            ),
            "alzheimers_trial": (
                "Phase 3 clinical trial of aducanumab in early Alzheimer's disease demonstrated dose-dependent "
                "reduction in amyloid plaque burden as measured by PET imaging. The high-dose group showed "
                "a 59% reduction in amyloid plaques compared to placebo. Cognitive decline was slowed by 22% "
                "on the CDR-SB scale in the treatment group."
            ),
        }

    @pytest.fixture
    def valid_schema(self):
        """Valid metadata schema for biomedical research."""
        return {
            "properties": {
                "disease_condition": {
                    "type": "string",
                    "description": "Primary disease or condition studied",
                },
                "intervention": {
                    "type": "string",
                    "description": "Intervention or treatment used",
                },
                "sample_size": {
                    "type": "number",
                    "description": "Number of participants in the study",
                },
                "primary_endpoint": {
                    "type": "string",
                    "description": "Primary outcome measure",
                },
                "is_randomized": {
                    "type": "boolean",
                    "description": "Whether the study was randomized",
                },
                "study_phase": {
                    "type": "string",
                    "enum": ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Preclinical"],
                    "description": "Clinical trial phase",
                },
            }
        }

    def test_init_with_valid_llm(self, mock_llm):
        """Test initialization with valid LLM."""
        extractor = MetadataExtractor(llm=mock_llm)
        assert extractor.llm == mock_llm

    def test_init_with_invalid_llm(self):
        """Test initialization with invalid LLM does not raise error."""
        extractor = MetadataExtractor(llm="not_an_llm")
        assert extractor.llm == "not_an_llm"

    def test_invoke_with_valid_schema_and_text(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test successful metadata extraction with valid schema and text."""
        mock_result = SimpleNamespace(
            disease_condition="non-small cell lung cancer",
            intervention="pembrolizumab",
            sample_size=500,
            primary_endpoint="overall survival",
            is_randomized=True,
            study_phase="Phase 3",
        )

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            text = biomedical_texts["cancer_study"]
            result = extractor.invoke(text, valid_schema)

            expected_result = {
                "disease_condition": "non-small cell lung cancer",
                "intervention": "pembrolizumab",
                "sample_size": 500,
                "primary_endpoint": "overall survival",
                "is_randomized": True,
                "study_phase": "Phase 3",
            }

            assert result == expected_result
            mock_llm.with_structured_output.assert_called_once()

    def test_invoke_with_missing_properties_key(self, extractor, biomedical_texts):
        """Test invoke raises ValueError when schema missing properties key."""
        invalid_schema = {"wrong_key": {"field1": {"type": "string"}}}

        with pytest.raises(ValueError, match="Schema must contain a 'properties' key."):
            extractor.invoke(biomedical_texts["cancer_study"], invalid_schema)

    def test_invoke_with_unsupported_field_type(self, extractor, biomedical_texts):
        """Test invoke raises ValueError for unsupported field types."""
        invalid_schema = {
            "properties": {"invalid_field": {"type": "array"}}
        }  # arrays not supported

        with pytest.raises(
            ValueError, match="Unsupported type 'array' for field 'invalid_field'"
        ):
            extractor.invoke(biomedical_texts["cancer_study"], invalid_schema)

    def test_invoke_with_enum_on_non_string_field(self, extractor, biomedical_texts):
        """Test invoke raises ValueError when enum used on non-string field."""
        invalid_schema = {
            "properties": {
                "sample_size": {"type": "number", "enum": [100, 200, 300]}
            }  # enum not allowed for numbers
        }

        with pytest.raises(
            ValueError, match="Enum is only supported for string fields"
        ):
            extractor.invoke(biomedical_texts["cancer_study"], invalid_schema)

    def test_invoke_with_partial_extraction(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test extraction where only some fields are present in text."""
        # Create a mock result object that behaves like a Pydantic model
        mock_result = SimpleNamespace(
            disease_condition="type 2 diabetes",
            intervention="semaglutide",
            sample_size=1200,
            primary_endpoint="HbA1c reduction",
            is_randomized=None,  # Not mentioned in text
            study_phase=None,  # Not mentioned in text
        )

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            text = biomedical_texts["diabetes_research"]
            result = extractor.invoke(text, valid_schema)

            # Only non-None fields should be included
            expected_result = {
                "disease_condition": "type 2 diabetes",
                "intervention": "semaglutide",
                "sample_size": 1200,
                "primary_endpoint": "HbA1c reduction",
            }

            assert result == expected_result
            assert "is_randomized" not in result
            assert "study_phase" not in result

    def test_invoke_with_llm_exception(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test invoke returns empty dict when LLM raises exception."""
        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.side_effect = Exception("LLM API error")

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            result = extractor.invoke(biomedical_texts["cancer_study"], valid_schema)

            assert result == {}

    def test_invoke_with_validation_error(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test invoke handles Pydantic validation errors."""
        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            # Create a simple exception to simulate validation error
            mock_chain.invoke.side_effect = Exception("Validation error")

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            result = extractor.invoke(biomedical_texts["cancer_study"], valid_schema)

            assert result == {}

    def test_transform_documents_success(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test transform_documents with successful metadata extraction."""
        # Create mock result objects that behave like Pydantic models
        mock_result1 = SimpleNamespace(
            disease_condition="lung cancer",
            intervention="pembrolizumab",
            sample_size=500,
            primary_endpoint="overall survival",
            is_randomized=True,
            study_phase="Phase 3",
        )

        mock_result2 = SimpleNamespace(
            disease_condition="type 2 diabetes",
            intervention="semaglutide",
            sample_size=1200,
            primary_endpoint="HbA1c reduction",
            is_randomized=True,
            study_phase="Phase 4",
        )

        mock_result3 = SimpleNamespace(
            disease_condition="Alzheimer's disease",
            intervention="aducanumab",
            sample_size=800,
            primary_endpoint="cognitive decline",
            is_randomized=True,
            study_phase="Phase 3",
        )

        mock_results = [mock_result1, mock_result2, mock_result3]

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.side_effect = mock_results

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            # Create input documents
            documents = [
                Document(
                    page_content=biomedical_texts["cancer_study"],
                    metadata={"source": "pubmed_1"},
                ),
                Document(
                    page_content=biomedical_texts["diabetes_research"],
                    metadata={"source": "pubmed_2"},
                ),
                Document(
                    page_content=biomedical_texts["alzheimers_trial"],
                    metadata={"source": "pubmed_3"},
                ),
            ]

            transformed = extractor.transform_documents(documents, valid_schema)

            # Verify results
            assert len(transformed) == 3

            # Check first document
            assert transformed[0].page_content == biomedical_texts["cancer_study"]
            assert transformed[0].metadata["source"] == "pubmed_1"
            assert transformed[0].metadata["disease_condition"] == "lung cancer"
            assert transformed[0].metadata["intervention"] == "pembrolizumab"
            assert transformed[0].metadata["sample_size"] == 500

            # Check second document
            assert transformed[1].page_content == biomedical_texts["diabetes_research"]
            assert transformed[1].metadata["source"] == "pubmed_2"
            assert transformed[1].metadata["disease_condition"] == "type 2 diabetes"
            assert transformed[1].metadata["intervention"] == "semaglutide"
            assert transformed[1].metadata["sample_size"] == 1200

            # Check third document
            assert transformed[2].page_content == biomedical_texts["alzheimers_trial"]
            assert transformed[2].metadata["source"] == "pubmed_3"
            assert transformed[2].metadata["disease_condition"] == "Alzheimer's disease"
            assert transformed[2].metadata["intervention"] == "aducanumab"
            assert transformed[2].metadata["sample_size"] == 800

    def test_transform_documents_with_extraction_failure(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test transform when metadata extraction fails for some documents."""
        # Create mock result objects that behave like Pydantic models
        mock_result1 = SimpleNamespace(
            disease_condition="lung cancer",
            intervention="pembrolizumab",
            sample_size=500,
            primary_endpoint="overall survival",
            is_randomized=True,
            study_phase="Phase 3",
        )

        mock_result3 = SimpleNamespace(
            disease_condition="Alzheimer's disease",
            intervention="aducanumab",
            sample_size=800,
            primary_endpoint="cognitive decline",
            is_randomized=True,
            study_phase="Phase 3",
        )

        mock_results = [mock_result1, Exception("Extraction failed"), mock_result3]

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.side_effect = mock_results

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            documents = [
                Document(
                    page_content=biomedical_texts["cancer_study"],
                    metadata={"source": "pubmed_1"},
                ),
                Document(
                    page_content=biomedical_texts["diabetes_research"],
                    metadata={"source": "pubmed_2"},
                ),
                Document(
                    page_content=biomedical_texts["alzheimers_trial"],
                    metadata={"source": "pubmed_3"},
                ),
            ]

            transformed = extractor.transform_documents(documents, valid_schema)

            # All documents should be processed,
            # failed extraction returns empty metadata
            assert len(transformed) == 3

            # First document - successful extraction
            assert "disease_condition" in transformed[0].metadata
            assert "intervention" in transformed[0].metadata

            # Second document - failed extraction, only original metadata
            assert transformed[1].metadata == {"source": "pubmed_2"}

            # Third document - successful extraction
            assert "disease_condition" in transformed[2].metadata
            assert "intervention" in transformed[2].metadata

    def test_boolean_field_extraction(self, extractor, mock_llm, biomedical_texts):
        """Test extraction of boolean fields."""
        boolean_schema = {
            "properties": {
                "is_randomized": {
                    "type": "boolean",
                    "description": "Is the study randomized?",
                },
                "is_multicenter": {
                    "type": "boolean",
                    "description": "Is the study multicenter?",
                },
            }
        }

        mock_result = SimpleNamespace(
            is_randomized=True,
            is_multicenter=False,
        )

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            result = extractor.invoke(
                biomedical_texts["diabetes_research"], boolean_schema
            )

            assert result == {"is_randomized": True, "is_multicenter": False}

    def test_enum_field_extraction(self, extractor, mock_llm, biomedical_texts):
        """Test extraction with enum constraints."""
        enum_schema = {
            "properties": {
                "study_phase": {
                    "type": "string",
                    "enum": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
                    "description": "Clinical trial phase",
                }
            }
        }

        mock_result = SimpleNamespace(
            study_phase="Phase 3",
        )

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            result = extractor.invoke(biomedical_texts["alzheimers_trial"], enum_schema)

            assert result == {"study_phase": "Phase 3"}

    @patch("rag_pipelines.utils.metadata_extractor.metadata_extractor.tqdm")
    def test_transform_documents_progress_bar(
        self, mock_tqdm, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test that transform_documents uses tqdm progress bar."""
        # Mock tqdm to return the input list unchanged
        mock_tqdm.return_value = [
            Document(
                page_content=biomedical_texts["cancer_study"],
                metadata={"source": "pubmed_1"},
            )
        ]

        # Mock successful extraction
        class MockResult:
            def __init__(self):
                self.disease_condition = "test"

        mock_chain = Mock()
        mock_chain.invoke.return_value = MockResult()

        mock_structured_llm = Mock()
        mock_structured_llm.with_structured_output.return_value = mock_chain
        mock_llm.with_structured_output.return_value = mock_structured_llm

        documents = [
            Document(
                page_content=biomedical_texts["cancer_study"],
                metadata={"source": "pubmed_1"},
            )
        ]

        extractor.transform_documents(documents, valid_schema)

        # Verify tqdm was called
        mock_tqdm.assert_called_once_with(documents)

    def test_empty_documents_list(self, extractor, valid_schema):
        """Test transform_documents with empty documents list."""
        result = extractor.transform_documents([], valid_schema)
        assert result == []

    def test_none_values_are_excluded(
        self, extractor, mock_llm, biomedical_texts, valid_schema
    ):
        """Test that None values are excluded from final result."""
        mock_result = SimpleNamespace(
            disease_condition="test disease",
            intervention=None,
            sample_size=None,
            primary_endpoint=None,
            is_randomized=None,
            study_phase=None,
        )

        with patch(
            "rag_pipelines.utils.metadata_extractor.metadata_extractor.ChatPromptTemplate"
        ) as mock_prompt_class:
            # Mock the chain
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_result

            # Mock the prompt
            mock_prompt = Mock()
            mock_prompt.__or__ = Mock(return_value=mock_chain)

            # Make from_messages return our mock prompt
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Mock structured LLM
            mock_llm.with_structured_output.return_value = Mock()

            result = extractor.invoke(biomedical_texts["cancer_study"], valid_schema)

            assert result == {"disease_condition": "test disease"}
            assert len(result) == 1
