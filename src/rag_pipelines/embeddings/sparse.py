from typing import Any, Optional

import weave
from pinecone_text.sparse import SpladeEncoder


class SparseEmbeddings(weave.Model):
    """Generate sparse embeddings for documents and queries using the FastEmbedSparse model.

    Attributes:
        model_kwargs (Optional[dict[str, Any]]): Additional configuration parameters for the model.
        sparse_embedding_model (SpladeEncoder): The FastEmbedSparse model initialized with the specified parameters.
    """

    model_kwargs: Optional[dict[str, Any]]
    sparse_embedding_model: Optional[SpladeEncoder] = None

    def __init__(
        self,
        model_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize the SparseEmbeddings class with the specified model and configurations.

        Args:
            model_kwargs (Optional[dict[str, Any]]): Additional model configuration parameters for initialization.
        """
        super().__init__(model_kwargs=model_kwargs)

        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # Initialize the sparse embedding model with specified parameters
        self.sparse_embedding_model = SpladeEncoder(**self.model_kwargs)

    @weave.op()
    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        """Embed a list of texts and return their sparse embeddings.

        Args:
            texts (list[str]): A list of document texts to embed.

        Returns:
            list[dict[str, float]]: A list of sparse embedding dictionaries for each document text.
                                    Each dictionary maps terms to their corresponding weights.
        """
        return self.sparse_embedding_model.encode_documents(texts)

    @weave.op()
    def embed_query(self, text: str) -> dict[str, float]:
        """Embed a single query text and return its sparse embedding.

        Args:
            text (str): The query text to embed.

        Returns:
            dict[str, float]: A sparse embedding dictionary for the query text, where keys are terms
                                and values are term weights.
        """
        return self.sparse_embedding_model.encode_queries([text])
