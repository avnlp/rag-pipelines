import logging
from typing import Any, Optional

import weave
import weaviate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate import WeaviateClient
from weaviate.classes.init import Auth

from rag_pipelines.utils.logging import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class WeaviateRetriever(weave.Model):
    """Hybrid document retriever utilizing Weaviate for dense and sparse search methods.

    This retriever integrates dense embeddings (e.g., transformer-based models)
    with sparse retrieval techniques (e.g., BM25 or SPLADE) to enhance search results.
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
        top_k: int = 4,
    ) -> None:
        """Initialize the Weaviate retriever with the specified parameters.

        Args:
            cluster_url (str): URL of the Weaviate cluster.
            api_key (str): API key for authentication.
            headers (Optional[Dict[str, str]]): Additional headers for Weaviate client.
            collection_name (Optional[str]): The name of the collection to search.
            text_field (str): Field name containing document text. Defaults to "text".
            dense_embedding_model (Optional[HuggingFaceEmbeddings]): Model for dense embeddings.
            top_k (int): Number of top results to retrieve. Defaults to 4.
        """
        super().__init__(
            cluster_url=cluster_url,
            api_key=api_key,
            headers=headers,
            collection_name=collection_name,
            text_field=text_field,
            dense_embedding_model=dense_embedding_model,
            top_k=top_k,
        )

        self.cluster_url = cluster_url
        self.api_key = api_key
        self.headers = headers or {}
        self.collection_name = collection_name
        self.text_field = text_field
        self.dense_embedding_model = dense_embedding_model
        self.top_k = top_k

        self.client = self._initialize_client()
        self.vector_store = self._initialize_vector_store()

    def _initialize_client(self) -> WeaviateClient:
        """Initialize and authenticate the Weaviate client.

        Returns:
            WeaviateClient: An authenticated Weaviate client instance.

        Raises:
            Exception: If connection fails.
        """
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=Auth.api_key(api_key=self.api_key),
                headers=self.headers,
            )
            logger.info("Weaviate client initialized successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise

    def _initialize_vector_store(self) -> WeaviateVectorStore:
        """Initialize the Weaviate vector store for semantic search.

        Returns:
            WeaviateVectorStore: A vector store instance for document retrieval.

        Raises:
            Exception: If initialization fails.
        """
        try:
            vector_store = WeaviateVectorStore(
                index_name=self.collection_name,
                client=self.client,
                text_key=self.text_field,
                embedding=self.dense_embedding_model,
            )
            logger.info("Weaviate vector store initialized successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate vector store: {e}")
            raise

    @weave.op()
    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant documents based on the input query.

        Args:
            state (Dict[str, Any]): Dictionary containing:
                - "question" (str): The search query.

        Returns:
            Dict[str, Any]: Updated state with:
                - "documents" (List[Any]): List of retrieved documents.
                - "question" (str): The original query.

        Example:
            ```python
            retriever = WeaviateRetriever(cluster_url, api_key, collection_name="my_collection")
            state = {"question": "What is quantum computing?"}
            updated_state = retriever(state)
            print(updated_state["documents"])  # Output: Retrieved documents
            ```
        """
        question = state.get("question")
        if not question:
            msg = "Input state must contain a 'question' key with a non-empty string."
            raise ValueError(msg)

        documents = self.vector_store.semantic_search(question, self.top_k)
        return {"documents": documents, "question": question}
