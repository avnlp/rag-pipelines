"""Process and Index the HealthBench dataset."""

import asyncio
import os

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

from rag_pipelines.utils import (
    EnrichmentConfig,
    EnrichmentMode,
    MetadataEnricher,
    UnstructuredChunker,
)


async def main() -> None:
    """Process and Index the HealthBench dataset."""
    # Load environment variables
    load_dotenv()

    # Load config
    with open("healthbench_indexing_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Get the JSON schema definition from config
    metadata_schema = config.get("metadata_schema", {})

    # Initialize embeddings
    embedding_cfg = config["embedding"]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_cfg["model_name"])

    # Load dataset
    dataset_cfg = config["dataset"]
    dataset = load_dataset(
        dataset_cfg["path"],
        name=dataset_cfg["split_name"],
        split=dataset_cfg["split"],
    )

    # Build documents from multi-turn conversations
    documents = []
    for example in dataset:
        # Combine multi-turn conversation into single text
        conversation_text = ""
        for message in example["prompt"]:
            # Each message has 'content' and 'role' keys
            role = message.get("role", "unknown")
            content = message.get("content", "")
            conversation_text += f"{role}: {content}\n"

        # Create document with conversation as main content
        doc = Document(page_content=conversation_text)

        # Extract and add metadata from HealthBench structure
        doc.metadata["prompt_id"] = example.get("prompt_id", "")
        doc.metadata["example_tags"] = example.get("example_tags", [])

        # Add information about rubrics
        if "rubrics" in example:
            doc.metadata["rubric_count"] = len(example["rubrics"])
            doc.metadata["rubric_axes"] = list(
                {
                    tag
                    for rubric in example["rubrics"]
                    for tag in rubric.get("tags", [])
                    if "axis:" in tag
                }
            )

        # Add ideal completion if available
        if "ideal_completions_data" in example and example["ideal_completions_data"]:
            ideal_data = example["ideal_completions_data"]
            doc.metadata["has_ideal_completion"] = True
            if "ideal_completions_group" in ideal_data:
                doc.metadata["completion_group"] = ideal_data["ideal_completions_group"]

        documents.append(doc)

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
