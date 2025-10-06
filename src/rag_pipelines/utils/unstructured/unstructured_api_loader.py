"""Unstructured API Document Loader: Loading documents using the Unstructured API.

This class provides functionality for loading and transforming unstructured documents
using the Unstructured API.
It supports extracting text, tables, and images from various document formats.
"""

from pathlib import Path
from typing import Callable, Optional

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from unstructured_client import UnstructuredClient


class UnstructuredAPIDocumentLoader:
    """A class for loading and transforming documents using the Unstructured API.

    This class provides functionality for extracting text, tables, and images
    from documents using the Unstructured API.
    """

    def __init__(
        self,
        partition_via_api: bool = False,
        post_processors: Optional[list[Callable[[str], str]]] = None,
        api_key: Optional[str] = None,
        client: Optional[UnstructuredClient] = None,
        url: Optional[str] = None,
        web_url: Optional[str] = None,
    ):
        """Initialize the document loader with configuration parameters.

        Args:
            partition_via_api (bool): Whether to partition the document via the API.
            post_processors (Optional[List[Callable[[str], str]]]): List of
                post-processors to apply to the document.
            api_key (Optional[str]): API key for the Unstructured API.
            client (Optional[UnstructuredClient]): Unstructured client for the API.
            url (Optional[str]): URL for the Unstructured API.
            web_url (Optional[str]): Web URL for the Unstructured API.
        """
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        self.api_key = api_key
        self.client = client
        self.url = url
        self.web_url = web_url

    def _get_all_file_paths_from_directory(self, directory_path: str) -> list[str]:
        """Retrieve all file paths from a given directory (recursively).

        Args:
            directory_path (str): Path to the directory.

        Returns:
            List[str]: A list of file paths.

        Raises:
            ValueError: If the directory does not exist or is not a directory.
        """
        path = Path(directory_path).resolve()  # Convert to absolute path

        if not path.exists():
            msg = f"Directory does not exist: {directory_path}"
            raise ValueError(msg)
        if not path.is_dir():
            msg = f"Path is not a directory: {directory_path}"
            raise ValueError(msg)

        return [
            str(file) for file in path.rglob("*") if file.is_file()
        ]  # Get only files

    def transform_documents(self, directory_path: str) -> list[Document]:
        """Transform all documents in the given directory into structured format.

        This method loads PDFs from the specified directory and processes them
        using the UnstructuredPDFLoader.

        Args:
            directory_path (str): Path to the directory containing PDF files.

        Returns:
            List[Document]: A list of structured documents.
        """
        file_paths = self._get_all_file_paths_from_directory(directory_path)

        documents: list[Document] = []

        for file in file_paths:
            loader = UnstructuredLoader(
                file_path=file,
                partition_via_api=self.partition_via_api,
                post_processors=self.post_processors,
                api_key=self.api_key,
                client=self.client,
                url=self.url,
                web_url=self.web_url,
            )
            parsed_documents = loader.load()
            documents.extend(parsed_documents)

        return documents
