import argparse
import logging

# Import or define your QdrantVectorDB
# Here we provide a sample implementation that supports both dense and optional sparse embeddings.
from typing import Any, List, Optional

import weave
from dataloaders.langchain import FinanceBenchDataloader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from rag_pipelines.embeddings.sparse_fastembed_milvus import SparseEmbeddings
from rag_pipelines.unstructured.unstructured_chunker import UnstructuredChunker
from rag_pipelines.unstructured.unstructured_pdf_loader import UnstructuredDocumentLoader
from rag_pipelines.utils.logging import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class QdrantVectorDB(weave.Model):
    """Interface for interacting with a Qdrant vector database.

    Provides functionalities for:
      - Connecting to a Qdrant instance.
      - Creating (or recreating) a collection.
      - Upserting documents with dense (and optionally sparse) embeddings.
    """

    host: str
    port: int
    collection_name: str
    text_field: str
    vector_dimension: int
    dense_embedding_model: Optional[HuggingFaceEmbeddings] = None

    client: Optional[QdrantClient] = None

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_dimension: int,
        text_field: str = "text",
        dense_embedding_model: Optional[HuggingFaceEmbeddings] = None,
    ) -> None:
        """Initialize the QdrantVectorDB client and collection.

        Args:
            host (str): Qdrant server hostname.
            port (int): Qdrant server port.
            collection_name (str): Name of the Qdrant collection.
            vector_dimension (int): Dimension of the dense vectors.
            text_field (str): Field in the payload containing document text.
            dense_embedding_model (Optional[HuggingFaceEmbeddings]): Model to generate dense embeddings.
        """
        super().__init__(
            host=host,
            port=port,
            collection_name=collection_name,
            vector_dimension=vector_dimension,
            text_field=text_field,
            dense_embedding_model=dense_embedding_model,
        )
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.text_field = text_field
        self.dense_embedding_model = dense_embedding_model

        self.client = self._initialize_client()
        self._initialize_collection()

    def _initialize_client(self) -> QdrantClient:
        """Initialize and return a Qdrant client instance.

        Returns:
            QdrantClient: The initialized Qdrant client.
        """
        try:
            client = QdrantClient(host=self.host, port=self.port)
            logger.info("Successfully initialized Qdrant client.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _initialize_collection(self) -> None:
        """Initialize (or recreate) the Qdrant collection.

        Creates the collection with the specified vector parameters using cosine distance.
        """
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' is ready in Qdrant.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise

    @weave.op()
    def add_documents(
        self,
        documents: list[Document],
        dense_embedding_model: HuggingFaceEmbeddings,
        sparse_embedding_model: Optional[Any] = None,
    ) -> None:
        """Add documents to the Qdrant collection with dense (and optionally sparse) embeddings.

        Each document is processed to generate a dense vector. If a sparse embedding model is provided,
        its output is also added to the document payload.

        Args:
            documents (List[Document]): List of documents to be indexed.
            dense_embedding_model (HuggingFaceEmbeddings): Model for generating dense embeddings.
            sparse_embedding_model (Optional[Any]): Optional model for generating sparse embeddings.

        Raises:
            ValueError: If the documents list is empty.
            RuntimeError: If dense_embedding_model is not provided.
        """
        if not documents:
            msg = "The documents list is empty. Provide valid documents to add."
            raise ValueError(msg)

        if dense_embedding_model is None:
            msg = "Dense embedding model is required for document vectorization."
            raise RuntimeError(msg)

        points: list[PointStruct] = []
        for i, doc in enumerate(documents):
            # Generate dense embedding for the document text.
            dense_vector = dense_embedding_model.embed([doc.text])[0]
            payload = {
                self.text_field: doc.text,
                "metadata": getattr(doc, "metadata", {}),
            }

            # If a sparse embedding model is provided, generate the sparse embedding and include it in the payload.
            if sparse_embedding_model is not None:
                sparse_vector = sparse_embedding_model.embed([doc.text])[0]
                payload["sparse_vector"] = sparse_vector

            point = PointStruct(
                id=i,
                vector=dense_vector,
                payload=payload,
            )
            points.append(point)

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully added {len(points)} documents to Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant: {e}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the FinanceBench pipeline to load, process, chunk, embed, and index documents in Qdrant."
    )

    # FinanceBench dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PatronusAI/financebench",
        help="Name of the FinanceBench dataset to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:1]",
        help="Dataset split to use (e.g., 'train[:1]').",
    )

    # PDF directory for unstructured document loader
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="pdfs/",
        help="Directory path containing PDF files.",
    )

    # UnstructuredDocumentLoader parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="fast",
        help="Processing strategy for the unstructured document loader.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="elements",
        help="Extraction mode for the unstructured document loader.",
    )

    # Qdrant connection parameters
    parser.add_argument(
        "--qdrant_host",
        type=str,
        default="localhost",
        help="Qdrant server host.",
    )
    parser.add_argument(
        "--qdrant_port",
        type=int,
        default=6333,
        help="Qdrant server port.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="financebench",
        help="Name of the Qdrant collection to create/use.",
    )
    parser.add_argument(
        "--vector_dimension",
        type=int,
        default=768,
        help="Dimension of the dense embeddings.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the FinanceBench document processing pipeline using Qdrant.

    The pipeline performs the following steps:
      1. Initializes Weave tracing.
      2. Loads a subset of the FinanceBench dataset.
      3. Retrieves PDF documents from the specified directory.
      4. Processes PDFs using the UnstructuredDocumentLoader.
      5. Chunks documents using the UnstructuredChunker.
      6. Generates dense and sparse embeddings.
      7. Sets up a Qdrant vector database and indexes the documents.
    """
    args = parse_arguments()

    # Initialize Weave tracing
    weave.init("financebench_test")

    # Initialize FinanceBench dataloader and load the corpus PDFs
    dataloader = FinanceBenchDataloader(
        dataset_name=args.dataset_name,
        split=args.split,
    )
    dataloader.get_corpus_pdfs()

    # Load and transform PDF documents from the provided directory
    unstructured_document_loader = UnstructuredDocumentLoader(
        strategy=args.strategy,
        mode=args.mode,
    )
    documents = unstructured_document_loader.transform_documents(args.pdf_dir)
    print("Loaded Documents:")
    print(documents)

    # Chunk the documents using the UnstructuredChunker
    chunker = UnstructuredChunker()
    chunked_documents = chunker.transform_documents(documents)
    print("Chunked Documents:")
    print(chunked_documents)

    # Initialize dense and sparse embedding models
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    sparse_embeddings = SparseEmbeddings(model_name="prithvida/Splade_PP_en_v1")

    # Initialize the Qdrant vector database client
    qdrant_vector_db = QdrantVectorDB(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection_name=args.collection_name,
        vector_dimension=args.vector_dimension,
        text_field="text",
        dense_embedding_model=embeddings,
    )

    # Add documents to Qdrant using both dense and sparse embeddings
    qdrant_vector_db.add_documents(
        documents=chunked_documents,
        dense_embedding_model=embeddings,
        sparse_embedding_model=sparse_embeddings,
    )
    print("Documents have been indexed successfully in Qdrant.")


if __name__ == "__main__":
    main()
