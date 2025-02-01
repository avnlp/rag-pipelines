import logging
from typing import Optional

import weave
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from rag_pipelines.utils.logging import LoggerFactory

# Set up logging via a custom logger factory.
logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class QdrantVectorDB(weave.Model):
    """Interface for interacting with a Qdrant vector database.

    This class provides functionalities for:
    - Connecting to a Qdrant instance.
    - Creating or using an existing collection.
    - Upserting vectorized documents into the collection.
    """

    host: str
    port: int
    api_key: Optional[str] = None
    collection_name: str
    text_field: str
    dense_embedding_model: Optional[HuggingFaceEmbeddings] = None
    vector_dimension: int

    client: Optional[QdrantClient] = None

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_dimension: int,
        text_field: str = "text",
        dense_embedding_model: Optional[HuggingFaceEmbeddings] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the QdrantVectorDB client and collection.

        Args:
            host (str): The Qdrant server hostname.
            port (int): The Qdrant server port.
            collection_name (str): The name of the Qdrant collection.
            vector_dimension (int): Dimension of the dense vectors.
            text_field (str): Field name in the payload containing document text.
            dense_embedding_model (Optional[HuggingFaceEmbeddings]): Embedding model for vectorization.
            api_key (Optional[str]): API key for Qdrant authentication (if required).
        """
        super().__init__(
            host=host,
            port=port,
            collection_name=collection_name,
            text_field=text_field,
            dense_embedding_model=dense_embedding_model,
            vector_dimension=vector_dimension,
            api_key=api_key,
        )
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.text_field = text_field
        self.dense_embedding_model = dense_embedding_model
        self.vector_dimension = vector_dimension

        # Initialize the Qdrant client and collection.
        self.client = self._initialize_client()
        self._initialize_collection()

    def _initialize_client(self) -> QdrantClient:
        """Initialize and return a Qdrant client instance.

        Returns:
            QdrantClient: An instance of the Qdrant client.

        Raises:
            Exception: If the client initialization fails.
        """
        try:
            client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
            logger.info("Successfully initialized Qdrant client.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _initialize_collection(self) -> None:
        """Initialize the Qdrant collection.

        Creates the collection if it does not exist. Uses cosine distance as the metric.

        Raises:
            Exception: If collection creation fails.
        """
        try:
            # Create collection with specified vector parameters.
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE,  # You can change the distance metric if needed.
                ),
            )
            logger.info(f"Collection '{self.collection_name}' is ready in Qdrant.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise

    @weave.op()
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the Qdrant collection with dense embeddings.

        This method processes each document, generates its dense embedding, and upserts it into Qdrant.

        Args:
            documents (List[Document]): List of documents to be added.

        Raises:
            ValueError: If the provided document list is empty.
            RuntimeError: If a dense embedding model is not provided.
        """
        if not documents:
            msg = "The documents list is empty. Provide valid documents to add."
            raise ValueError(msg)

        if self.dense_embedding_model is None:
            msg = "Dense embedding model is not provided for document vectorization."
            raise RuntimeError(msg)

        points: list[PointStruct] = []
        # Process each document, generate an embedding, and prepare a point for upsertion.
        for i, doc in enumerate(documents):
            # Generate embedding for the document text.
            vector = self.dense_embedding_model.embed([doc.text])[0]
            # Prepare a point with an autogenerated id.
            point = PointStruct(
                id=i,
                vector=vector,
                payload={
                    self.text_field: doc.text,
                    "metadata": getattr(doc, "metadata", {}),
                },
            )
            points.append(point)

        try:
            # Upsert the points into the collection.
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully added {len(points)} documents to Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant: {e}")
            raise
