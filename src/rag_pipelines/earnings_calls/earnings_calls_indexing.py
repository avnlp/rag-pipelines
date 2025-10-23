"""Process and Index the Earnings Calls dataset."""

import os

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

from rag_pipelines.utils import MetadataExtractor, UnstructuredChunker


def main() -> None:
    """Process and Index the Earnings Calls dataset."""
    # Load environment variables
    load_dotenv()

    # Load config
    with open("earnings_calls_indexing_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize LLM
    llm_cfg = config["llm"]
    extractor_llm = ChatGroq(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        max_retries=llm_cfg["max_retries"],
    )

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

    # Build document list from Earnings Call transcripts
    documents = []
    for row in dataset:
        doc = Document(
            page_content=row["transcript"],
            metadata={
                "year": row["q"].split("-")[0],
                "quarter": row["q"].split("-")[1],
                "company": row["ticker"],
            },
        )
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

    # Extract metadata
    metadata_schema = config["metadata_schema"]
    metadata_extractor = MetadataExtractor(llm=extractor_llm)
    transformed_documents = metadata_extractor.transform_documents(
        chunked_documents, metadata_schema
    )

    # Fill missing metadata fields
    expected_fields = set(metadata_schema["properties"].keys())
    default_value = config["metadata_defaults"]["default_value"]

    for doc in transformed_documents:
        for field in expected_fields:
            if field not in doc.metadata:
                doc.metadata[field] = default_value

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
    main()
