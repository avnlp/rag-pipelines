"""Process and Index the FinanceBench dataset."""

import asyncio
import os
from pathlib import Path

import fsspec
import yaml
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

from rag_pipelines.utils import (
    EnrichmentConfig,
    EnrichmentMode,
    MetadataEnricher,
    UnstructuredChunker,
    UnstructuredDocumentLoader,
)


async def main() -> None:
    """Process and Index the FinanceBench dataset."""
    # Load environment variables
    load_dotenv()

    # Load config
    with open("financebench_indexing_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Get the JSON schema definition from config
    metadata_schema = config.get("metadata_schema", {})

    # Initialize embeddings
    embedding_cfg = config["embedding"]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["model_name"])

    # Fetch 10K, 10Q, 8K, and Earnings Call Transcripts from FinanceBench repository
    dataset_cfg = config["dataset"]
    # Directory to store the PDFs
    documents_dir = Path("pdfs")
    documents_dir.mkdir(exist_ok=True, parents=True)

    # Initialize the GitHub filesystem
    fs = fsspec.filesystem("github", org=dataset_cfg["org"], repo=dataset_cfg["repo"])

    # List files in the "pdfs/" directory in the repository
    repo_pdf_path = "pdfs/"
    pdf_files = fs.ls(repo_pdf_path)

    # Iterate over files and download them
    for filename in pdf_files:
        # Full path in the repository
        file_path_in_repo = f"{repo_pdf_path}{filename}"
        # Full local path
        local_file_path = documents_dir / filename
        try:
            # Download the file to the local destination
            fs.get(file_path_in_repo, local_file_path.as_posix())
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    # Load text from PDFs
    unstructured_loader_cfg = config["unstructured_loader"]
    pdf_loader = UnstructuredDocumentLoader(
        strategy=unstructured_loader_cfg["strategy"],
        mode=unstructured_loader_cfg["mode"],
        include_page_breaks=unstructured_loader_cfg["include_page_breaks"],
        infer_table_structure=unstructured_loader_cfg["infer_table_structure"],
        ocr_languages=unstructured_loader_cfg["ocr_languages"],
        languages=unstructured_loader_cfg["languages"],
        extract_images_in_pdf=unstructured_loader_cfg["extract_images_in_pdf"],
        extract_forms=unstructured_loader_cfg["extract_forms"],
        form_extraction_skip_tables=unstructured_loader_cfg[
            "form_extraction_skip_tables"
        ],
    )

    documents = pdf_loader.transform_documents(str(documents_dir))

    # Chunk documents
    chunking_cfg = config["chunking"]
    chunker = UnstructuredChunker(
        chunking_strategy=chunking_cfg["chunking_strategy"],
        max_characters=chunking_cfg["max_characters"],
        new_after_n_chars=chunking_cfg["new_after_n_chars"],
        overlap=chunking_cfg["overlap"],
        overlap_all=chunking_cfg["overlap_all"],
        combine_text_under_n_chars=chunking_cfg["combine_text_under_n_chars"],
        include_orig_elements=chunking_cfg["include_orig_elements"],
        multipage_sections=chunking_cfg["multipage_sections"],
    )
    chunked_documents = chunker.transform_documents(documents)

    # Enrich metadata using MetadataEnricher
    # With three layers: structural, user-defined schema, and RAG-optimized fields
    enricher_cfg = config.get("metadata_enricher", {})
    enrichment_config = EnrichmentConfig(
        mode=EnrichmentMode(enricher_cfg.get("mode", "full")),
        batch_size=enricher_cfg.get("batch_size", 10),
    )
    metadata_enricher = MetadataEnricher(enrichment_config)
    transformed_documents = await metadata_enricher.atransform_documents(
        chunked_documents,
        metadata_schema,
    )

    # Apply default values for missing metadata fields
    metadata_defaults = config.get("metadata_defaults", {})
    default_value = metadata_defaults.get("default_value", "")
    if default_value:
        metadata_schema_properties = metadata_schema.get("properties", {})
        for doc in transformed_documents:
            for field_name in metadata_schema_properties:
                if field_name not in doc.metadata:
                    doc.metadata[field_name] = default_value

    # Index into Milvus
    vectorstore_config = config["vectorstore"]

    Milvus.from_documents(
        documents=transformed_documents,
        embedding=embeddings,
        builtin_function=BM25BuiltInFunction(),
        vector_field=vectorstore_config["vector_fields"],
        collection_name=vectorstore_config["collection_name"],
        connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
        },
        consistency_level=vectorstore_config["consistency_level"],
        drop_old=vectorstore_config["drop_old"],
    )


if __name__ == "__main__":
    asyncio.run(main())
