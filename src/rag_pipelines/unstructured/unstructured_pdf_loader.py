from pathlib import Path
from typing import Optional

import weave
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document


class UnstructuredDocumentLoader(weave.Model):
    """A class for loading and transforming unstructured PDF documents.

    This class provides functionality for extracting text, tables, and images
    from PDFs using different processing strategies.
    """

    strategy: str
    mode: str
    include_page_breaks: bool
    infer_table_structure: bool
    ocr_languages: Optional[str]
    languages: Optional[list[str]]
    hi_res_model_name: Optional[str]
    extract_images_in_pdf: bool
    extract_image_block_types: Optional[list[str]]
    extract_image_block_output_dir: Optional[str]
    extract_image_block_to_payload: bool
    starting_page_number: int
    extract_forms: bool
    form_extraction_skip_tables: bool

    def __init__(
        self,
        strategy: str = "hi_res",
        mode: str = "elements",
        include_page_breaks: bool = False,
        infer_table_structure: bool = False,
        ocr_languages: Optional[str] = None,
        languages: Optional[list[str]] = None,
        hi_res_model_name: Optional[str] = None,
        extract_images_in_pdf: bool = False,
        extract_image_block_types: Optional[list[str]] = None,
        extract_image_block_output_dir: Optional[str] = None,
        extract_image_block_to_payload: bool = False,
        starting_page_number: int = 1,
        extract_forms: bool = False,
        form_extraction_skip_tables: bool = True,
    ):
        """Initialize the document loader with configuration parameters.

        Args:
            strategy (str): The strategy for document processing (e.g., "hi_res").
            mode (str): The mode of extraction (e.g., "elements").
            include_page_breaks (bool): Whether to include page breaks.
            infer_table_structure (bool): Whether to infer table structures.
            ocr_languages (Optional[str]): Languages for OCR processing.
            languages (Optional[List[str]]): List of languages for document processing.
            hi_res_model_name (Optional[str]): Model name for high-resolution processing.
            extract_images_in_pdf (bool): Whether to extract images from PDFs.
            extract_image_block_types (Optional[List[str]]): Types of image blocks to extract.
            extract_image_block_output_dir (Optional[str]): Directory to save extracted images.
            extract_image_block_to_payload (bool): Whether to add extracted images to payload.
            starting_page_number (int): Page number from which extraction should start.
            extract_forms (bool): Whether to extract form data.
            form_extraction_skip_tables (bool): Whether to skip tables during form extraction.
        """
        super().__init__(
            strategy=strategy,
            mode=mode,
            include_page_breaks=include_page_breaks,
            infer_table_structure=infer_table_structure,
            ocr_languages=ocr_languages,
            languages=languages,
            hi_res_model_name=hi_res_model_name,
            extract_images_in_pdf=extract_images_in_pdf,
            extract_image_block_types=extract_image_block_types,
            extract_image_block_output_dir=extract_image_block_output_dir,
            extract_image_block_to_payload=extract_image_block_to_payload,
            starting_page_number=starting_page_number,
            extract_forms=extract_forms,
            form_extraction_skip_tables=form_extraction_skip_tables,
        )

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

        return [str(file) for file in path.rglob("*") if file.is_file()]  # Get only files

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
            loader = UnstructuredPDFLoader(
                file_path=file,
                mode=self.mode,
                strategy=self.strategy,
                include_page_breaks=self.include_page_breaks,
                infer_table_structure=self.infer_table_structure,
                ocr_languages=self.ocr_languages,
                languages=self.languages,
                hi_res_model_name=self.hi_res_model_name,
                extract_images_in_pdf=self.extract_images_in_pdf,
                extract_image_block_types=self.extract_image_block_types,
                extract_image_block_output_dir=self.extract_image_block_output_dir,
                extract_image_block_to_payload=self.extract_image_block_to_payload,
                starting_page_number=self.starting_page_number,
                extract_forms=self.extract_forms,
                form_extraction_skip_tables=self.form_extraction_skip_tables,
            )
            parsed_documents = loader.load()
            documents.extend(parsed_documents)

        return documents
