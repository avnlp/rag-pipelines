from typing import Any, Optional

import weave
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus import (
    Collection,
    WeightedRanker,
)

from rag_pipelines.embeddings.sparse_fastembed_milvus import SparseEmbeddings


TEXT_FIELD = "text"
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"


class MilvusRetriever(weave.Model):
    """Combine dense and sparse retrieval methods using Milvus.

    This class sets up a hybrid retriever that leverages dense embeddings (e.g., transformer-based)
    and sparse embeddings (e.g., BM25 or SPLADE) to enhance retrieval accuracy. By balancing
    between dense contextual similarity and sparse keyword-based matching, it improves search results.
    """

    collection: Optional[Collection] = None
    dense_embedding_model: Optional[HuggingFaceEmbeddings] = None
    sparse_embedding_model: Optional[SparseEmbeddings] = None
    anns_fields: Optional[list[str]] = None
    field_search_params: Optional[list[dict[str, Any]]] = None
    text_field: str = TEXT_FIELD
    top_k: int = 4
    rerank: Optional[WeightedRanker] = WeightedRanker(0.5, 0.5)
    hybrid_retriever: Optional[MilvusCollectionHybridSearchRetriever] = None

    def __init__(
        self,
        collection: Collection,
        dense_embedding_model: HuggingFaceEmbeddings,
        sparse_embedding_model: SparseEmbeddings,
        anns_fields: Optional[list[str]] = None,
        field_search_params: Optional[list[dict[str, Any]]] = None,
        text_field: str = TEXT_FIELD,
        rerank: Optional[WeightedRanker] = WeightedRanker(0.5, 0.5),
        top_k: int = 4,
    ) -> None:
        """Initialize the hybrid retriever with specified parameters.

        This constructor configures the retriever to balance between dense and sparse retrieval
        methods.

        Args:
            collection (Collection): The Milvus collection instance for document retrieval.
            dense_embedding_model (HuggingFaceEmbeddings): Model for generating dense embeddings.
            sparse_embedding_model (SparseEmbeddings): Model for generating sparse embeddings.
            anns_fields (Optional[list[str]]): List of fields to search in the collection.
            field_search_params (Optional[dict[str, Any]]): Parameters for field-specific search.
            text_field (str): Field name for text content. Defaults to "text".
            rerank (Optional[WeightedRanker]): Weighted ranker for reranking results.
            top_k (int): Maximum number of top documents to retrieve.
        """
        super().__init__(
            collection=collection,
            dense_embedding_model=dense_embedding_model,
            sparse_embedding_model=sparse_embedding_model,
            anns_fields=anns_fields,
            field_search_params=field_search_params,
            text_field=text_field,
            top_k=top_k,
            rerank=rerank,
        )

        if field_search_params is None:
            field_search_params = [
                {"metric_type": "IP"},
                {"metric_type": "IP", "params": {}},
            ]
        if anns_fields is None:
            anns_fields = [DENSE_FIELD, SPARSE_FIELD]

        self.collection = collection
        self.dense_embedding_model = dense_embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.anns_fields = anns_fields
        self.field_search_params = field_search_params
        self.text_field = text_field
        self.top_k = top_k
        self.rerank = rerank

        self._initialize_retriever()

    def _initialize_retriever(self):
        # Check if the retriever has already been initialized
        self.hybrid_retriever = MilvusCollectionHybridSearchRetriever(
            collection=self.collection,
            rerank=self.rerank,
            anns_fields=self.anns_fields,
            field_embeddings=[self.dense_embedding_model, self.sparse_embedding_model],
            field_search_params=self.field_search_params,
            text_field=self.text_field,
            top_k=self.top_k,
        )

    @weave.op()
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
            retriever = MilvusRetriever(collection, dense_model, sparse_model)
            state = {"question": "What is quantum computing?"}
            updated_state = retriever(state)
            print(updated_state["documents"])  # Output: Retrieved documents
            ```
        """
        question = state["question"]
        documents = self.hybrid_retriever.invoke(question)

        return {"documents": documents, "question": question}
