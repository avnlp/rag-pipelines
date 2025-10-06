"""Test the ContextualReranker class."""

from unittest.mock import Mock, patch

import pytest
import torch
from langchain_core.documents import Document

from rag_pipelines.utils.contextual_ranker.contextual_ranker import ContextualReranker


@pytest.fixture
def model_path():
    return "test_model_path"


@pytest.fixture
def query():
    return "What are the effects of ACE inhibitors on cardiovascular health in elderly patients?"


@pytest.fixture
def documents():
    return [
        Document(
            page_content="The renin-angiotensin system plays a crucial role in cardiovascular regulation. ACE inhibitors effectively reduce blood pressure by blocking the conversion of angiotensin I to angiotensin II, leading to vasodilation and reduced aldosterone secretion. Clinical trials have demonstrated significant benefits in reducing cardiovascular events.",
            metadata={"source": "pubmed_12345", "pmid": "12345"},
        ),
        Document(
            page_content="Diabetes mellitus significantly increases the risk of cardiovascular complications. Patients with both diabetes and hypertension benefit from ACE inhibitor therapy, which provides renal protection in addition to cardiovascular benefits. Long-term studies show reduced incidence of microalbuminuria and progression to overt nephropathy.",
            metadata={"source": "pubmed_67890", "pmid": "67890"},
        ),
        Document(
            page_content="Aging is associated with increased arterial stiffness and altered cardiovascular responses. Elderly patients often present with isolated systolic hypertension, which responds well to ACE inhibitor therapy. However, careful monitoring is required due to increased risk of hyperkalemia and acute kidney injury in this population.",
            metadata={"source": "pubmed_54321", "pmid": "54321"},
        ),
    ]


class TestContextualReranker:
    """Test the ContextualReranker class."""

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_init_cuda_available(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path
    ):
        """Test initialization when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        reranker = ContextualReranker(model_path)

        assert reranker.device == "cuda"
        assert reranker.dtype == torch.bfloat16
        mock_tokenizer.assert_called_once_with(model_path, use_fast=True)
        mock_model.assert_called_once_with(model_path, torch_dtype=torch.bfloat16)
        mock_model_instance.to.assert_called_once_with("cuda")
        mock_model_instance.eval.assert_called_once()
        assert mock_tokenizer_instance.pad_token == "<eos>"
        assert mock_tokenizer_instance.padding_side == "left"

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_init_cuda_not_available(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path
    ):
        """Test initialization when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "pad"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        reranker = ContextualReranker(model_path)

        assert reranker.device == "cpu"
        assert reranker.dtype == torch.float32
        mock_tokenizer.assert_called_once_with(model_path, use_fast=True)
        mock_model.assert_called_once_with(model_path, torch_dtype=torch.float32)
        mock_model_instance.to.assert_called_once_with("cpu")
        mock_model_instance.eval.assert_called_once()
        assert mock_tokenizer_instance.pad_token == "pad"
        assert mock_tokenizer_instance.padding_side == "left"

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_init_with_instruction(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path
    ):
        """Test initialization with custom instruction."""
        mock_cuda_available.return_value = True
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        instruction = "Please analyze the clinical relevance of the following document in the context of cardiovascular health."
        reranker = ContextualReranker(model_path, instruction=instruction)

        assert reranker.instruction == instruction

    def test_format_prompts_without_instruction(self, model_path):
        """Test prompt formatting without instruction."""
        with (
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
            ),
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
            ),
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            reranker = ContextualReranker(model_path)

            documents = [
                "ACE inhibitors block the conversion of angiotensin I to angiotensin II.",
                "Diabetes increases cardiovascular risk significantly.",
            ]
            expected_prompts = [
                "Check whether a given document contains information helpful to answer the query.\n<Document> ACE inhibitors block the conversion of angiotensin I to angiotensin II.\n<Query> test query ??",
                "Check whether a given document contains information helpful to answer the query.\n<Document> Diabetes increases cardiovascular risk significantly.\n<Query> test query ??",
            ]

            result = reranker._format_prompts("test query", documents)
            assert result == expected_prompts

    def test_format_prompts_with_instruction(self, model_path):
        """Test prompt formatting with instruction."""
        with (
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
            ),
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
            ),
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            instruction = "Analyze clinical relevance"
            reranker = ContextualReranker(model_path, instruction=instruction)

            documents = [
                "ACE inhibitors block the conversion of angiotensin I to angiotensin II.",
                "Diabetes increases cardiovascular risk significantly.",
            ]
            expected_prompts = [
                f"Check whether a given document contains information helpful to answer the query.\n<Document> ACE inhibitors block the conversion of angiotensin I to angiotensin II.\n<Query> test query {instruction} ??",
                f"Check whether a given document contains information helpful to answer the query.\n<Document> Diabetes increases cardiovascular risk significantly.\n<Query> test query {instruction} ??",
            ]

            result = reranker._format_prompts("test query", documents)
            assert result == expected_prompts

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_rerank_empty_documents(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path, query
    ):
        """Test rerank with empty document list."""
        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "pad"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        reranker = ContextualReranker(model_path)

        result = reranker.rerank(query, [])
        assert result == []

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_rerank_single_document(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path, query
    ):
        """Test rerank with single document."""
        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "pad"
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Mock the tokenizer call to return proper tensors
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance

        # Mock the model output with proper tensor structure
        mock_outputs = Mock()
        # Shape: [batch_size, seq_len, vocab_size]
        logits = torch.randn(1, 4, 100)  # batch_size=1, seq_len=4, vocab_size=100
        logits[:, -1, 0] = torch.tensor([0.8])  # Set score for token 0 at last position
        mock_outputs.logits = logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.to.return_value = mock_model_instance

        reranker = ContextualReranker(model_path)

        single_doc = [
            Document(
                page_content="ACE inhibitors effectively reduce blood pressure in elderly patients with hypertension."
            )
        ]
        result = reranker.rerank(query, single_doc)

        assert len(result) == 1
        assert (
            result[0].page_content
            == "ACE inhibitors effectively reduce blood pressure in elderly patients with hypertension."
        )

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_rerank_multiple_documents(
        self,
        mock_cuda_available,
        mock_model,
        mock_tokenizer,
        model_path,
        query,
        documents,
    ):
        """Test rerank with multiple documents."""
        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "pad"
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Mock tokenizer encoding - fix the return value structure
        def mock_tokenizer_call(prompts, return_tensors, padding, truncation):
            batch_size = len(prompts)
            max_len = 5
            input_ids = torch.randint(1, 100, (batch_size, max_len))
            attention_mask = torch.ones((batch_size, max_len))
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        mock_tokenizer_instance.side_effect = mock_tokenizer_call

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance

        # Mock model outputs with different logits to test sorting
        # Shape: [batch_size, seq_len, vocab_size]
        mock_outputs = Mock()
        batch_size = 3
        seq_len = 5
        vocab_size = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Set specific values for the first token at the last position for each sequence
        logits[:, -1, 0] = torch.tensor([0.5, 1.2, -0.3])  # These will be the scores
        mock_outputs.logits = logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.to.return_value = mock_model_instance

        reranker = ContextualReranker(model_path)

        # Test documents with different content
        docs = [
            Document(
                page_content="ACE inhibitors block the conversion of angiotensin I to angiotensin II, reducing blood pressure.",
                metadata={"source": "pubmed_cardio_1", "pmid": "11111"},
            ),
            Document(
                page_content="Statins are effective in reducing cholesterol levels and cardiovascular events.",
                metadata={"source": "pubmed_cardio_2", "pmid": "22222"},
            ),
            Document(
                page_content="Exercise and lifestyle modifications significantly improve cardiovascular health outcomes.",
                metadata={"source": "pubmed_cardio_3", "pmid": "33333"},
            ),
        ]

        result = reranker.rerank(query, docs)

        # Check that all documents are returned
        assert len(result) == 3

        # Verify that metadata is preserved
        for i, doc in enumerate(docs):
            found = any(
                res.page_content == doc.page_content and res.metadata == doc.metadata
                for res in result
            )
            assert found, f"Document {i} with metadata was not preserved"

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_rerank_document_ordering(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path, query
    ):
        """Test that documents are properly sorted by score."""
        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "pad"
        mock_tokenizer.return_value = mock_tokenizer_instance

        def mock_tokenizer_call(prompts, return_tensors, padding, truncation):
            batch_size = len(prompts)
            max_len = 5
            input_ids = torch.randint(1, 100, (batch_size, max_len))
            attention_mask = torch.ones((batch_size, max_len))
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        mock_tokenizer_instance.side_effect = mock_tokenizer_call

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance

        # Mock outputs with known logits to ensure predictable sorting
        mock_outputs = Mock()
        batch_size = 3
        seq_len = 5
        vocab_size = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Set scores in descending order: [2.0, 1.0, 0.0]
        # -> should maintain order after sorting: [2.0, 1.0, 0.0]
        logits[:, -1, 0] = torch.tensor([1.0, 2.0, 0.0])
        mock_outputs.logits = logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.to.return_value = mock_model_instance

        reranker = ContextualReranker(model_path)

        # Create documents with known content to verify ordering
        docs = [
            Document(
                page_content="ACE inhibitors have significant benefits for cardiovascular health in elderly patients.",
                metadata={"score": 1.0},
            ),
            Document(
                page_content="Clinical trials show ACE inhibitors reduce cardiovascular events by 25% in elderly populations.",
                metadata={"score": 2.0},
            ),  # Should come first
            Document(
                page_content="General health guidelines for elderly patients.",
                metadata={"score": 0.0},
            ),  # Should come last
        ]

        result = reranker.rerank(query, docs)

        # Check the order:
        # doc2 (score 2.0) should come first,
        # then doc1 (score 1.0), then doc3 (score 0.0)

        assert (
            result[0].page_content
            == "Clinical trials show ACE inhibitors reduce cardiovascular events by 25% in elderly populations."
        )
        assert (
            result[1].page_content
            == "ACE inhibitors have significant benefits for cardiovascular health in elderly patients."
        )
        assert (
            result[2].page_content == "General health guidelines for elderly patients."
        )

    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
    )
    @patch(
        "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available"
    )
    def test_rerank_with_instruction(
        self, mock_cuda_available, mock_model, mock_tokenizer, model_path, query
    ):
        """Test rerank with custom instruction."""
        mock_cuda_available.return_value = False
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "pad"
        mock_tokenizer.return_value = mock_tokenizer_instance

        def mock_tokenizer_call(prompts, return_tensors, padding, truncation):
            # Verify that the prompts contain the instruction
            for prompt in prompts:
                assert "Analyze clinical relevance" in prompt

            batch_size = len(prompts)
            max_len = 5
            input_ids = torch.randint(1, 100, (batch_size, max_len))
            attention_mask = torch.ones((batch_size, max_len))
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        mock_tokenizer_instance.side_effect = mock_tokenizer_call

        mock_model_instance = Mock()
        mock_model_instance.eval = Mock(return_value=None)
        mock_model.return_value = mock_model_instance

        mock_outputs = Mock()
        batch_size = 2
        seq_len = 5
        vocab_size = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, -1, 0] = torch.tensor([0.5, 1.0])
        mock_outputs.logits = logits
        mock_model_instance.return_value = mock_outputs
        mock_model_instance.to.return_value = mock_model_instance

        instruction = "Analyze clinical relevance"
        reranker = ContextualReranker(model_path, instruction=instruction)

        docs = [
            Document(
                page_content="ACE inhibitors effectively reduce blood pressure in elderly patients."
            ),
            Document(
                page_content="Regular exercise improves cardiovascular health outcomes."
            ),
        ]

        result = reranker.rerank(query, docs)
        assert len(result) == 2

    def test_rerank_model_calls(self, model_path, query):
        """Test that model and tokenizer are called with correct parameters."""
        with (
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoTokenizer.from_pretrained"
            ) as mock_tokenizer,
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.AutoModelForCausalLM.from_pretrained"
            ) as mock_model,
            patch(
                "rag_pipelines.utils.contextual_ranker.contextual_ranker.torch.cuda.is_available",
                return_value=False,
            ) as _,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = "pad"
            mock_tokenizer_instance.padding_side = "left"

            def verify_tokenizer_params(prompts, return_tensors, padding, truncation):
                # Verify parameters
                assert return_tensors == "pt"
                assert padding is True
                assert truncation is True

                return {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                }

            mock_tokenizer_instance.side_effect = verify_tokenizer_params
            mock_tokenizer.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.eval = Mock(return_value=None)
            mock_model.return_value = mock_model_instance

            mock_outputs = Mock()
            # Create proper tensor shape: [batch_size=2, seq_len=3, vocab_size=2]
            mock_outputs.logits = torch.tensor(
                [
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # First sequence
                    [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],  # Second sequence
                ]
            )
            mock_model_instance.return_value = mock_outputs
            mock_model_instance.to.return_value = mock_model_instance

            reranker = ContextualReranker(model_path)

            docs = [
                Document(page_content="ACE inhibitors reduce cardiovascular risk."),
                Document(page_content="Statins lower cholesterol levels."),
            ]
            reranker.rerank(query, docs)

            # Verify tokenizer was called
            assert mock_tokenizer_instance.called
            # Verify model was called
            assert mock_model_instance.called
