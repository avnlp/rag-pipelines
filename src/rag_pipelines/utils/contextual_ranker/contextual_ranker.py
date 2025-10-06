"""Ranker that uses the Contextual Rerank models for reordering documents.

The contextual rerank models can be used in two ways:
1. Contextual AI API
2. Load the Contextual rerank models from HuggingFace.

This component loads the Contextual reranker models from HuggingFace and uses them to
score documents based on their relevance to a given query.
"""

from typing import List

import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer


class ContextualReranker:
    """Ranker that uses the Contextual Rerank models for reordering documents.

    The reranker loads the Contextual Rerank models from Hugging Face and uses them to
    score documents based on their relevance to a given query.

    It uses a prompt-based scoring mechanism where each document is scored by
    the model's logit for token ID 0 at the last position of a structured prompt.
    Documents are then sorted in descending order of their scores.
    """

    def __init__(self, model_path: str, instruction: str = "") -> None:
        """Initializes the ContextualReranker with the contextual rerank model.

        Args:
            model_path: Path or identifier of a HuggingFace model.
            instruction: Optional natural language instruction to refine the query
                context during reranking.
                If provided, it is appended to the query in the prompt template.
        """
        self.model_path = model_path
        self.instruction = instruction

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)  # type: ignore[no-untyped-call]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _format_prompts(self, query: str, documents: List[str]) -> List[str]:
        """Constructs structured prompts for scoring each document against the query.

        Each prompt follows the template:
        "Check whether a given document contains information helpful to
        answer the query.
        <Document> {doc}
        <Query> {query}{instruction} ??"

        Args:
            query: The user query to evaluate document relevance against.
            documents: A list of document texts to be scored.

        Returns:
            A list of formatted prompts, one per document.
        """
        instruction_suffix = f" {self.instruction}" if self.instruction else ""
        return [
            (
                f"Check whether a given document contains information helpful to answer the query.\n"
                f"<Document> {doc}\n"
                f"<Query> {query}{instruction_suffix} ??"
            )
            for doc in documents
        ]

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Reranks a list of documents by their relevance to the query.

        The method preserves all original document metadata and returns a new list
        sorted from most to least relevant. Scoring is performed by extracting the
        model's logit for token ID 0 at the final position of each formatted prompt.

        Args:
            query: The input query string used to assess document relevance.
            documents: A list of LangChain Document objects to rerank.

        Returns:
            A list of Document objects sorted in descending order of relevance scores.
            If the input list is empty, returns an empty list.
        """
        if not documents:
            return []

        doc_texts = [doc.page_content for doc in documents]
        prompts = self._format_prompts(query, doc_texts)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract logits for token ID 0 at the last token position
        last_token_logits = outputs.logits[:, -1, :]
        scores = last_token_logits[:, 0].to(torch.bfloat16).float().tolist()

        # Sort documents by score in descending order
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]
