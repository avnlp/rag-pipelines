import logging
from typing import Optional

import weave
import weaviate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weave import Model
from weaviate import WeaviateClient
from weaviate.classes.init import Auth

from rag_pipelines.utils.logging import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class WeaviateVectorDB(Model):
    """Interface for interacting with Weaviate vector databases.

    Provides functionalities for:
    - Connecting to a Weaviate instance.
    - Managing a collection.
    - Upserting and querying vectorized documents.
    """

    cluster_url: str
    api_key: str
    headers: Optional[dict[str, str]] = None
    collection_name: Optional[str] = None
    text_field: str
    dense_embedding_model: Optional[HuggingFaceEmbeddings] = None
    client: Optional[WeaviateClient] = None
    vector_store: Optional[WeaviateVectorStore] = None

    def __init__(
        self,
        cluster_url: str,
        api_key: str,
        headers: Optional[dict[str, str]] = None,
        collection_name: Optional[str] = None,
        text_field: str = "text",
        dense_embedding_model: Optional[HuggingFaceEmbeddings] = None,
    ) -> None:
        """Initialize the WeaviateVectorDB client.

        Args:
            cluster_url (str): URL of the Weaviate cluster.
            api_key (str): API key for Weaviate authentication.
            headers (Optional[Dict[str, str]]): Additional HTTP headers for requests.
            collection_name (Optional[str]): The name of the collection in Weaviate.
            text_field (str): The key representing the text content in Weaviate. Defaults to "text".
            dense_embedding_model (Optional[HuggingFaceEmbeddings]): Embedding model for vectorization.
        """
        super().__init__(
            cluster_url=cluster_url,
            api_key=api_key,
            headers=headers,
            collection_name=collection_name,
            text_field=text_field,
            dense_embedding_model=dense_embedding_model,
        )

        self.cluster_url = cluster_url
        self.api_key = api_key
        self.headers = headers or {}
        self.collection_name = collection_name
        self.text_field = text_field
        self.dense_embedding_model = dense_embedding_model

        self.client = self._initialize_client()
        self.vector_store = self._initialize_vector_store() if self.client else None

    def _initialize_client(self) -> Optional[WeaviateClient]:
        """Initialize and return a Weaviate client instance.

        Returns:
            Optional[WeaviateClient]: The initialized Weaviate client, or None if initialization fails.
        """
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=Auth.api_key(api_key=self.api_key),
                headers=self.headers,
            )
            logger.info("Successfully initialized Weaviate client.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            return None

    def _initialize_vector_store(self) -> Optional[WeaviateVectorStore]:
        """Initialize and return the Weaviate vector store.

        Returns:
            Optional[WeaviateVectorStore]: The initialized Weaviate vector store, or None if initialization fails.
        """
        if not self.client:
            logger.error("Cannot initialize vector store without a valid Weaviate client.")
            return None

        if not self.collection_name:
            logger.error("Collection name must be specified to initialize the vector store.")
            return None

        try:
            vector_store = WeaviateVectorStore(
                index_name=self.collection_name,
                client=self.client,
                text_key=self.text_field,
                embedding=self.dense_embedding_model,
            )
            logger.info("Successfully initialized Weaviate vector store.")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate vector store: {e}")
            return None

    @weave.op()
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the Weaviate collection with hybrid embeddings.

        Args:
            documents (List[Document]): List of documents to be added to Weaviate.

        Raises:
            ValueError: If the provided document list is empty.
            RuntimeError: If the vector store is not initialized.
        """
        if not documents:
            msg = "The documents list is empty. Provide valid documents to add."
            raise ValueError(msg)

        if not self.vector_store:
            msg = "Vector store is not initialized. Ensure Weaviate is connected and configured correctly."
            raise RuntimeError(msg)

        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to Weaviate.")
        except Exception as e:
            logger.error(f"Failed to add documents to Weaviate: {e}")
            raise
