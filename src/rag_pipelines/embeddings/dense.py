from typing import Any, Optional

import weave
from langchain_huggingface import HuggingFaceEmbeddings


class DenseEmbeddings(weave.Model):
    """Generate dense embeddings for documents and queries using a specified SentenceTransformer model.

    This class leverages HuggingFace's `HuggingFaceEmbeddings` to compute dense embeddings for input text.

    Attributes:
        model_name (str): The name of the pre-trained embedding model to use.
        model_kwargs (Optional[Dict[str, Any]]): Additional configuration parameters for the embedding model.
        encode_kwargs (Optional[Dict[str, Any]]): Parameters for fine-tuning the behavior of the encoding process.
        embedding_model (HuggingFaceEmbeddings): The initialized HuggingFace embeddings model with the specified settings.
    """

    model_name: str
    model_kwargs: Optional[dict[str, Any]]
    encode_kwargs: Optional[dict[str, Any]]
    show_progress: bool
    embedding_model: Optional[HuggingFaceEmbeddings] = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: Optional[dict[str, Any]] = None,
        encode_kwargs: Optional[dict[str, Any]] = None,
        show_progress: bool = True,
    ):
        """Initialize the DenseEmbeddings class with the specified model and configurations.

        Args:
            model_name (str): The name of the pre-trained embedding model. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            model_kwargs (Optional[dict[str, Any]]): Additional model configuration parameters for initialization. Defaults to None.
            encode_kwargs (Optional[dict[str, Any]]): Parameters for encoding settings. Defaults to None.
            show_progress (bool): Whether to display progress during model operations. Defaults to True.
        """
        if encode_kwargs is None:
            encode_kwargs = {"normalize_embeddings": True}
        if model_kwargs is None:
            model_kwargs = {"device": "cpu"}
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=show_progress,
        )

        self.model_name = model_name
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.encode_kwargs = encode_kwargs if encode_kwargs is not None else {}

        # Initialize the embedding model with the specified parameters
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
            show_progress=show_progress,
        )

    @weave.op()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their embeddings.

        Args:
            texts (list[str]): A list of texts to embed.

        Returns:
            list[list[float]]: A list of embedding vectors corresponding to each input text.
        """
        return self.embedding_model.embed_documents(texts)

    @weave.op()
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text and returns its embedding.

        Args:
            text (str): The query text to be embedded.

        Returns:
            List[float]: The embedding vector for the query text.
        """
        return self.embedding_model.embed_query(text)
