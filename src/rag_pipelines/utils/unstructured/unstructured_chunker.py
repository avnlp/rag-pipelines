"""Unstructured Chunker: Chunking documents using strategies provided by Unstructured.

This class provides functionality for chunking unstructured documents
using different strategies from the `unstructured` library. It supports:

- "basic" chunking: Splits documents into chunks based on character limits with
    optional overlap and combining of smaller text elements.
- "by_title" chunking: Splits documents into sections based on titles, considering
    options like maximum characters and allowing sections to span multiple pages.

The chunker can be configured with various parameters for each strategy, allowing
fine-grained control over the resulting chunks.

Example Usage:
```python
from rag_pipelines.utils.unstructured.unstructured_chunker import UnstructuredChunker

# Initialize the chunker
chunker = UnstructuredChunker()

# Load your documents
documents = [...]

# Chunk the documents using the "basic" strategy
chunked_documents = chunker.transform_documents(documents)
```
"""

import logging
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, NarrativeText


logger = logging.getLogger(__name__)


class UnstructuredChunker:
    """A class for chunking documents using the Unstructured library.

    Supports both "basic" and "by_title" chunking strategies.

    Attributes:
        chunking_strategy (Literal["basic", "by_title"]): The strategy
            to use for chunking elements.
        max_characters (int): Maximum number of characters in each chunk
            for the "basic" strategy.
        new_after_n_chars (int): Number of characters after which a new
            chunk is forced for the "basic" strategy.
        overlap (int): Number of characters to overlap between chunks
            for the "basic" strategy.
        overlap_all (bool): Whether to overlap all chunks for the
            "basic" strategy.
        combine_text_under_n_chars (Optional[int]): Maximum characters
            to combine smaller text elements for the "basic" strategy.
        include_orig_elements (Optional[bool]): Whether to include
            original elements in the output for the "basic" strategy.
        multipage_sections (Optional[bool]): Whether to allow sections
            to span multiple pages for the "by_title" strategy.
    """

    def __init__(
        self,
        chunking_strategy: Literal["basic", "by_title"] = "basic",
        max_characters: int = 500,
        new_after_n_chars: int = 500,
        overlap: int = 0,
        overlap_all: bool = False,
        combine_text_under_n_chars: Optional[int] = None,
        include_orig_elements: Optional[bool] = None,
        multipage_sections: Optional[bool] = None,
    ):
        """Initialize the chunker with the specified chunking strategy and parameters.

        Args:
            chunking_strategy (Literal["basic", "by_title"], optional):
                Chunking strategy to use. Defaults to "basic".
            max_characters (int, optional): Maximum characters in a chunk
                for the "basic" strategy. Defaults to 500.
            new_after_n_chars (int, optional): Characters after which a new
                chunk is forced for the "basic" strategy. Defaults to 500.
            overlap (int, optional): Characters to overlap between chunks
                for the "basic" strategy. Defaults to 0.
            overlap_all (bool, optional): Overlap all chunks for the
                "basic" strategy. Defaults to False.
            combine_text_under_n_chars (Optional[int], optional): Combine
                text elements under this limit for the "basic" strategy.
                Defaults to None.
            include_orig_elements (Optional[bool], optional): Include
                original elements in output for the "basic" strategy.
                Defaults to None.
            multipage_sections (Optional[bool], optional): Allow sections
                to span multiple pages for the "by_title" strategy.
                Defaults to None.
        """
        self.chunking_strategy = chunking_strategy
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.overlap = overlap
        self.overlap_all = overlap_all
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.include_orig_elements = include_orig_elements
        self.multipage_sections = multipage_sections

    def _convert_documents_to_elements(
        self, documents: list[Document]
    ) -> tuple[list[NarrativeText], list[dict[str, Any]]]:
        """Convert a list of LangChain documents to unstructured NarrativeText elements.

        This method takes in a list of LangChain Document objects and converts each
        document into a NarrativeText element. It also extracts and stores the metadata
        of each document separately in a list.

        Args:
            documents (List[Document]): A list of LangChain Document
                objects to be converted to NarrativeText elements.

        Returns:
            tuple[list[NarrativeText], list[dict[str, Any]]]:
                - A list of unstructured NarrativeText elements, where
                  each element corresponds to a document's text.
                - A list of metadata dictionaries, where each dictionary
                  corresponds to the metadata of a document in the
                  input list.

        """
        elements = []
        element_metadatas = []

        for document in documents:
            # Convert each document into a NarrativeText element
            element = NarrativeText(text=document.page_content)
            elements.append(element)

            # Store the metadata separately
            element_metadatas.append(document.metadata)

        logger.debug(f"Converted {len(documents)} documents to elements.")

        return elements, element_metadatas

    def _convert_chunked_elements_to_documents(
        self, elements: list[Element]
    ) -> list[Document]:
        """Convert a list of chunked unstructured elements back to LangChain documents.

        Args:
            elements (List[Element]): List of chunked unstructured elements.

        Returns:
            List[Document]: List of LangChain documents converted from elements.
        """
        documents = []
        for element in elements:
            document = Document(
                page_content=element.text, metadata=element.metadata.to_dict()
            )
            documents.append(document)
        logger.debug(f"Converted {len(elements)} chunked elements to documents.")
        return documents

    def transform_documents(self, documents: list[Document]) -> list[Document]:
        """Chunks the provided documents based on the configured strategy.

        Args:
            documents (List[Document]): List of documents to be chunked.

        Returns:
            List[Document]: List of chunked documents.

        Raises:
            ValueError: If no documents are provided or if an unsupported
                chunking strategy is specified.
        """
        if not documents:
            msg = "No documents provided for transformation."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            f"Transforming {len(documents)} documents using strategy: {self.chunking_strategy}"
        )

        all_chunked_documents = []

        # Convert each document to unstructured elements and separate metadata
        elements, element_metadatas = self._convert_documents_to_elements(documents)

        # Apply the selected chunking strategy
        for i, element in enumerate(elements):
            metadata = element_metadatas[i]  # Get the metadata for the current document

            if self.chunking_strategy == "basic":
                chunked_elements = chunk_elements(
                    [element],  # Process one element at a time
                    max_characters=self.max_characters,
                    new_after_n_chars=self.new_after_n_chars,
                    overlap=self.overlap,
                    overlap_all=self.overlap_all,
                    include_orig_elements=self.include_orig_elements,
                )
            elif self.chunking_strategy == "by_title":
                chunked_elements = chunk_by_title(
                    [element],  # Process one element at a time
                    max_characters=self.max_characters,
                    new_after_n_chars=self.new_after_n_chars,
                    overlap=self.overlap,
                    overlap_all=self.overlap_all,
                    include_orig_elements=self.include_orig_elements,
                    multipage_sections=self.multipage_sections,
                    combine_text_under_n_chars=self.combine_text_under_n_chars,
                )
            else:
                msg = f"Unsupported chunking strategy: {self.chunking_strategy}"
                logger.error(msg)
                raise ValueError(msg)

            logger.info(f"Chunked element into {len(chunked_elements)} sub-elements.")

            # Add metadata to each chunk and convert back to Document
            for chunk in chunked_elements:
                chunked_document = Document(page_content=chunk.text, metadata=metadata)
                all_chunked_documents.append(chunked_document)

        logger.info(
            f"Combined all chunked documents into {len(all_chunked_documents)} documents."
        )
        return all_chunked_documents
