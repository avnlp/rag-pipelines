from typing import Optional

import weave
from langchain_core.documents import Document
from pymilvus import (
    Collection,
    CollectionSchema,
    connections,
)


TEXT_FIELD = "text"
METADATA_FIELD = "metadata"
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"


class MilvusVectorDB(weave.Model):
    """Manage interactions with a Milvus vector index for hybrid search.

    This class facilitates initializing a Milvus Collection, adding documents with dense and sparse embeddings,
    and configuring a hybrid retriever for effective search across both embeddings.
    """

    uri: str
    token: str
    collection_name: str
    collection_schema: Optional[CollectionSchema] = None
    dense_field: str
    sparse_field: str
    text_field: str
    metadata_field: str
    dense_index_params: Optional[dict] = None
    sparse_index_params: Optional[dict] = None
    create_new_collection: bool
    collection: Optional[Collection] = None

    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: str = "default",
        collection_schema: Optional[CollectionSchema] = None,
        dense_field: str = DENSE_FIELD,
        sparse_field: str = SPARSE_FIELD,
        text_field: str = TEXT_FIELD,
        metadata_field: str = METADATA_FIELD,
        dense_index_params: Optional[dict] = None,
        sparse_index_params: Optional[dict] = None,
        create_new_collection: bool = False,
    ) -> None:
        """Initialize the MilvusVectorDB.

        This sets up the Milvus client, creates a collection if it does not already exist,
        and prepares it for hybrid search using dense and sparse embeddings.

        Args:
            uri (str): Milvus server URI.
            token (str): Milvus server token.
            collection_name (str): Name of the Milvus collection to use or create. Defaults to "default".
            collection_schema (Optional[CollectionSchema]): Schema for the Milvus collection. Defaults to None.
            dense_field (str): Field name for dense embeddings. Defaults to "dense_vector".
            sparse_field (str): Field name for sparse embeddings. Defaults to "sparse_vector".
            text_field (str): Field name for text content. Defaults to "text".
            metadata_field (str): Field name for metadata. Defaults to "metadata".
            dense_index_params (Optional[dict]): Index parameters for dense embeddings. Defaults to None.
            sparse_index_params (Optional[dict]): Index parameters for sparse embeddings. Defaults to None.
            create_new_collection (bool): Flag indicating whether to create a new collection. Defaults to False.
        """
        super().__init__(
            uri=uri,
            token=token,
            collection_name=collection_name,
            collection_schema=collection_schema,
            dense_field=dense_field,
            sparse_field=sparse_field,
            text_field=text_field,
            metadata_field=metadata_field,
            dense_index_params=dense_index_params,
            sparse_index_params=sparse_index_params,
            create_new_collection=create_new_collection,
        )

        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.collection_schema = collection_schema
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.dense_field = dense_field
        self.sparse_field = sparse_field
        self.dense_index_params = dense_index_params or {}
        self.sparse_index_params = sparse_index_params or {}

        connections.connect(uri=self.uri, token=self.token)

        if create_new_collection:
            self.create_collection()

        else:
            self.collection = Collection(name=self.collection_name)

    def create_collection(self) -> None:
        """Create the Milvus collection if it does not already exist.

        This method connects to the specified Milvus server and creates a new collection
        with the provided schema, dense field, and index parameters.
        """
        self.collection = Collection(
            name=self.collection_name, schema=self.collection_schema
        )

        self.collection.create_index(self.dense_field, self.dense_index_params)
        self.collection.create_index(self.sparse_field, self.sparse_index_params)
        self.collection.flush()

    @weave.op()
    def add_documents(
        self,
        documents: list[Document],
        dense_embedding_model,
        sparse_embedding_model,
    ) -> None:
        """Add documents to the Milvus collection with hybrid embeddings.

        This method processes a list of documents, generates dense and sparse embeddings, and adds them to the Milvus collection.

        Args:
            documents (list[Document]): List of documents to add to the collection.
            dense_embedding_model: Model for generating dense embeddings.
            sparse_embedding_model: Model for generating sparse embeddings.
            text_field (str): Field name for text content. Defaults to "text".
        """
        # Validate input documents
        if not documents:
            msg = "The documents list is empty, provide valid documents to add."
            raise ValueError(msg)

        # Extract text and metadata from the documents
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]

        # Generate dense and sparse embeddings
        entities = []
        for text, metadata in zip(texts, metadata_list):
            dense_embedding = dense_embedding_model.embed_documents([text])[0]
            sparse_embedding = sparse_embedding_model.embed_documents([text])[0]
            entity = {
                self.text_field: text,
                self.dense_field: dense_embedding,
                self.sparse_field: sparse_embedding,
                self.metadata_field: metadata,
            }
            entities.append(entity)
        self.collection.insert(entities)
        self.collection.load()
