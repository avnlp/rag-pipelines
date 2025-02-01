import os
from typing import Any, Optional

from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec


class PineconeVectorDB:
    """Manage interactions with a Pinecone vector index for hybrid search.

    This class facilitates initializing a Pinecone index, adding documents with dense and sparse embeddings,
    and configuring a hybrid retriever for effective search across both embeddings.

    Attributes:
        pinecone_client (Pinecone): Client instance for interacting with Pinecone services.
        index (Pinecone.Index): The active Pinecone index for storing and retrieving vector data.
        hybrid_retriever (Optional[PineconeHybridSearchRetriever]): A retriever supporting hybrid search combining dense and sparse embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "default",
        dimension: int = 512,
        metric: str = "dotproduct",
        region: str = "us-east-1",
        cloud: str = "aws",
        **kwargs: Any,
    ) -> None:
        """Initialize the PineconeHybridVectorDB.

        This sets up the Pinecone client, creates an index if it does not already exist,
        and prepares it for hybrid search using dense and sparse embeddings.

        Args:
            api_key (Optional[str]): Pinecone API key; if not provided, attempts to retrieve it from environment variables.
            index_name (str): Name of the Pinecone index to use or create. Defaults to "default".
            dimension (int): Dimensionality of vectors in the index. Defaults to 512.
            metric (str): Similarity metric for vector search. Defaults to "dotproduct".
            region (str): Region where the index will be created. Defaults to "us-east-1".
            cloud (str): Cloud provider for the index. Defaults to "aws".
            **kwargs: Additional parameters for configuring the index.

        Raises:
            OSError: If the API key is not provided and cannot be retrieved from environment variables.
        """
        # Retrieve API key from environment or provided input
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            msg = "Pinecone API key is missing; provide it directly or via environment variables."
            raise OSError(msg)

        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=self.api_key)

        # Set up index parameters
        self.index_name = index_name
        self.index_params = {
            "dimension": dimension,
            "metric": metric,
            "region": region,
            "cloud": cloud,
            **kwargs,
        }

        # Initialize or create the specified index
        self.initialize_index()

        # Access the specified index
        self.index = self.pinecone_client.Index(self.index_name)

        # Hybrid retriever, initially unset
        self.hybrid_retriever: Optional[PineconeHybridSearchRetriever] = None

    def initialize_index(self) -> None:
        """Create the Pinecone index if it does not exist.

        This method checks for the existence of the index specified during initialization.
        If the index does not exist, it creates one with the specified parameters.

        Raises:
            Exception: If there is an issue creating or accessing the index.
        """
        # Check if the index exists; if not, create it
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.index_params["dimension"],
                metric=self.index_params["metric"],
                spec=ServerlessSpec(cloud=self.index_params["cloud"], region=self.index_params["region"]),
            )

    def add_documents(
        self,
        documents: list[Document],
        dense_embedding_model,
        sparse_embedding_model,
        namespace: str,
    ) -> None:
        """Add documents to the Pinecone index with hybrid embeddings.

        This method processes a list of documents, generates dense and sparse embeddings,
        and stores them in the Pinecone index under the specified namespace.

        Args:
            documents (List[Document]): List of documents to be added, each containing `page_content` and metadata.
            dense_embedding_model: Model used to generate dense embeddings for the documents.
            sparse_embedding_model: Model used to generate sparse embeddings for the documents.
            namespace (str): Namespace in the Pinecone index to isolate these documents.

        Raises:
            ValueError: If documents list is empty or lacks required attributes (`page_content` or `metadata`).
        """
        # Validate input documents
        if not documents:
            msg = "The documents list is empty, provide valid documents to add."
            raise ValueError(msg)

        # Extract text, metadata, and unique IDs from the documents
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        ids = [doc.id for doc in documents]

        # Initialize the hybrid retriever if it hasn't been initialized yet
        if not self.hybrid_retriever:
            self.hybrid_retriever = PineconeHybridSearchRetriever(
                embeddings=dense_embedding_model,
                sparse_encoder=sparse_embedding_model,
                index=self.index,
            )

        # Add texts to the Pinecone index using the retriever
        self.hybrid_retriever.add_texts(texts=texts, ids=ids, metadatas=metadata_list, namespace=namespace)
