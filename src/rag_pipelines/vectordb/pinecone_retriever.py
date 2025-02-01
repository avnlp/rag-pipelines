from typing import Any, Optional

import weave
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone.data.index import Index
from pinecone_text.sparse import SpladeEncoder


class PineconeRetriever(weave.Model):
    """Combine dense and sparse retrieval methods using Pinecone.

    This class sets up a hybrid retriever that leverages dense embeddings (e.g., transformer-based)
    and sparse embeddings (e.g., BM25 or SPLADE) to enhance retrieval accuracy. By balancing
    between dense contextual similarity and sparse keyword-based matching, it improves search results.

    Attributes:
        index (Optional[Index]): The Pinecone index instance used for document retrieval.
        dense_embedding_model (Optional[HuggingFaceEmbeddings]): Model for generating dense embeddings.
        sparse_embedding_model (Optional[SpladeEncoder]): Model for generating sparse embeddings.
        namespace (str): Namespace for isolating the retrieval space within the Pinecone index.
        alpha (float): Weighting factor for dense vs. sparse retrieval. Closer to 1 favors dense, closer to 0 favors sparse.
        top_k (int): Maximum number of top documents to retrieve.
        hybrid_retriever (Optional[PineconeHybridSearchRetriever]): Configured retriever instance.
    """

    index: Optional[Index] = None
    dense_embedding_model: Optional[HuggingFaceEmbeddings] = None
    sparse_embedding_model: Optional[SpladeEncoder] = None
    namespace: str
    alpha: float
    top_k: int
    hybrid_retriever: Optional[PineconeHybridSearchRetriever] = None

    def __init__(
        self,
        index: Index,
        dense_embedding_model: HuggingFaceEmbeddings,
        sparse_embedding_model: SpladeEncoder,
        namespace: str,
        alpha: float = 0.5,
        top_k: int = 4,
    ) -> None:
        """Initialize the hybrid retriever with specified parameters.

        This constructor configures the retriever to balance between dense and sparse retrieval
        methods and sets up a namespace for isolated searches in the Pinecone index.

        Args:
            index (Index): The Pinecone index instance for document retrieval.
            dense_embedding_model (HuggingFaceEmbeddings): Model for generating dense embeddings.
            sparse_embedding_model (SpladeEncoder): Model for generating sparse embeddings.
            namespace (str): String to isolate the retrieval context within the Pinecone index.
            alpha (float): Weighting factor for dense vs. sparse retrieval, in the range [0, 1]. Defaults to 0.5.
            top_k (int): Maximum number of documents to retrieve. Defaults to 4.
        """
        super().__init__(
            index=index,
            dense_embedding_model=dense_embedding_model,
            sparse_embedding_model=sparse_embedding_model,
            namespace=namespace,
            alpha=alpha,
            top_k=top_k,
        )

        self.index = index
        self.dense_embedding_model = dense_embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.namespace = namespace
        self.alpha = alpha
        self.top_k = top_k

        # Initialize the hybrid retriever
        if not self.hybrid_retriever:
            self.hybrid_retriever = PineconeHybridSearchRetriever(
                embeddings=self.dense_embedding_model,
                sparse_encoder=self.sparse_embedding_model,
                index=self.index,
                namespace=self.namespace,
                alpha=self.alpha,
                top_k=self.top_k,
            )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve documents and update the state with the results.

        This method accepts a state dictionary containing a query string under the key "question."
        It uses the hybrid retriever to fetch relevant documents and updates the state with the
        retrieved documents.

        Args:
            state (Dict[str, Any]): Input state containing:
                - "question" (str): The query string to retrieve documents for.

        Returns:
            Dict[str, Any]: Updated state containing:
                - "documents" (List[Any]): Retrieved documents matching the query.
                - "question" (str): The original query string.

        Example:
            ```python
            retriever = PineconeHybridRetriever(index, dense_model, sparse_model, "my_namespace")
            state = {"question": "What is quantum computing?"}
            updated_state = retriever(state)
            print(updated_state["documents"])  # Output: Retrieved documents
            ```
        """
        question = state["question"]
        documents = self.hybrid_retriever.invoke(question)

        return {"documents": documents, "question": question}
