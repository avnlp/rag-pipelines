import logging
from typing import Any, Optional

import weave
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from rag_pipelines.utils.logging import LoggerFactory

# Set up logging via a custom logger factory.
logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class QdrantRetriever(weave.Model):
    """Hybrid document retriever utilizing Qdrant for dense (semantic) search.

    This retriever leverages a dense embedding model to encode queries and then searches
    the Qdrant collection for relevant documents.
    """

    host: str
    port: int
    api_key: Optional[str] = None
    collection_name: str
    text_field: str
    dense_embedding_model: Optional[HuggingFaceEmbeddings] = None
    top_k: int

    client: Optional[QdrantClient] = None

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        top_k: int = 4,
        text_field: str = "text",
        dense_embedding_model: Optional[HuggingFaceEmbeddings] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the QdrantRetriever with specified parameters.

        Args:
            host (str): Qdrant server hostname.
            port (int): Qdrant server port.
            collection_name (str): Name of the Qdrant collection to search.
            top_k (int): Number of top search results to return.
            text_field (str): Field name in the payload containing document text.
            dense_embedding_model (Optional[HuggingFaceEmbeddings]): Model for dense embeddings.
            api_key (Optional[str]): API key for Qdrant authentication (if required).
        """
        super().__init__(
            host=host,
            port=port,
            collection_name=collection_name,
            top_k=top_k,
            text_field=text_field,
            dense_embedding_model=dense_embedding_model,
            api_key=api_key,
        )
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.top_k = top_k
        self.text_field = text_field
        self.dense_embedding_model = dense_embedding_model

        self.client = self._initialize_client()

    def _initialize_client(self) -> QdrantClient:
        """Initialize and return a Qdrant client instance.

        Returns:
            QdrantClient: An authenticated Qdrant client instance.

        Raises:
            Exception: If client initialization fails.
        """
        try:
            client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
            logger.info("Qdrant client initialized successfully for retriever.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    @weave.op()
    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant documents based on the input query.

        Args:
            state (dict[str, Any]): Dictionary containing:
                - "question" (str): The search query.

        Returns:
            dict[str, Any]: Updated state with:
                - "documents" (List[dict]): List of retrieved documents.
                - "question" (str): The original query.

        Raises:
            ValueError: If the 'question' key is missing or empty.
            RuntimeError: If a dense embedding model is not provided.
        """
        question = state.get("question")
        if not question:
            msg = "Input state must contain a 'question' key with a non-empty string."
            raise ValueError(msg)

        if self.dense_embedding_model is None:
            msg = "Dense embedding model is not provided for query encoding."
            raise RuntimeError(msg)

        # Generate the dense embedding for the query.
        query_vector = self.dense_embedding_model.embed([question])[0]
        # Perform the search in Qdrant.
        try:
            search_results: list[PointStruct] = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
            )
        except Exception as e:
            logger.error(f"Search in Qdrant failed: {e}")
            raise

        # Transform search results into a list of document-like dictionaries.
        documents = []
        for result in search_results:
            doc = {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get(self.text_field, ""),
                "metadata": result.payload.get("metadata", {}),
            }
            documents.append(doc)

        return {"documents": documents, "question": question}
