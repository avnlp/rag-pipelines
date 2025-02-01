from typing import Any, Optional

import weave
from langchain_qdrant.fastembed_sparse import FastEmbedSparse


class SparseEmbeddings(weave.Model):
    """Generate sparse embeddings for documents and queries using the FastEmbedSparse model.

    Attributes:
        model_name (str): The name of the sparse embedding model to use.
        model_kwargs (Optional[dict[str, Any]]): Additional configuration parameters for the model.
        sparse_embedding_model (FastEmbedSparse): The initialized FastEmbedSparse model with the specified parameters.
    """

    def __init__(
        self,
        model_name: str = "prithvida/Splade_PP_en_v1",
        model_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize the SparseEmbeddings class with the specified model and configurations.

        Args:
            model_name (str): The name of the sparse embedding model. Defaults to "prithvida/Splade_PP_en_v1".
            model_kwargs (Optional[dict[str, Any]]): Additional model configuration parameters for initialization. Defaults to None.
        """
        self.model_name = model_name
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # Initialize the sparse embedding model with specified parameters
        self.sparse_embedding_model = FastEmbedSparse(model_name=self.model_name, **self.model_kwargs)

    @weave.op()
    def embed_texts(self, texts: list[str]) -> list[dict[str, float]]:
        """Embed a list of texts and return their sparse embeddings.

        Args:
            texts (list[str]): A list of document texts to embed.

        Returns:
            list[dict[str, float]]: A list of sparse embedding dictionaries for each document text.
                                    Each dictionary maps terms to their corresponding weights.
        """
        return self.sparse_embedding_model.embed_documents(texts)

    @weave.op()
    def embed_query(self, text: str) -> dict[str, float]:
        """Embed a single query text and return its sparse embedding.

        Args:
            text (str): The query text to embed.

        Returns:
            dict[str, float]: A sparse embedding dictionary for the query text, where keys are terms
                                and values are term weights.
        """
        return self.sparse_embedding_model.embed_query(text)
