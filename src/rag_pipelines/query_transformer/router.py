from typing import Any


class SubQueryRouter:
    """Route sub-queries to specific retrieval mechanisms and combines responses.

    Attributes:
        llm (Any): The LLM responsible for generating sub-queries and synthesizing answers.
        vectordb (Any): The vector database (e.g., Milvus) for retrieving relevant documents.
        embedding_model (Any): The model for generating vector embeddings from sub-queries.

    Methods:
        __call__(query: str) -> str:
            Processes a complex query, breaks it into sub-queries, retrieves relevant documents,
            and synthesizes a final response.
    """

    def __init__(self, llm: Any, vectordb: Any, embedding_model: Any):
        """Initialize the router with an LLM, a vector database, and an embedding model.

        Args:
            llm (Any): The LLM used for sub-query generation and response synthesis.
            vectordb (Any): The vector database for retrieving relevant documents.
            embedding_model (Any): Model for generating embeddings from queries.
        """
        self.llm = llm
        self.vectordb = vectordb
        self.embedding_model = embedding_model

    def generate_sub_queries(self, query: str) -> List[str]:
        """Use the LLM to generate sub-queries from a complex query.

        Args:
            query (str): The user-provided complex query.

        Returns:
            List[str]: A list of simpler sub-queries.
        """
        prompt = f"Break down the following query into simpler sub-queries:\n\n{query}\n\nSub-queries:"
        response = self.llm.generate(prompt)
        return response.split(
            "\n"
        )  # Assuming the LLM returns sub-queries as newline-separated text.

    def retrieve_relevant_documents(self, sub_queries: List[str]) -> List[Document]:
        """Retrieve relevant documents for each sub-query from the vector database.

        Args:
            sub_queries (List[str]): A list of sub-queries.

        Returns:
            List[Document]: A list of relevant documents retrieved from the vector database.
        """
        retrieved_docs = []
        for sub_query in sub_queries:
            query_embedding = self.embedding_model.embed(sub_query)
            results = self.vectordb.search(
                query_embedding, top_k=5
            )  # Retrieve top-K relevant chunks
            retrieved_docs.extend(results)
        return retrieved_docs

    def synthesize_response(self, query: str, documents: List[Document]) -> str:
        """Use the LLM to synthesize a final response based on retrieved documents.

        Args:
            query (str): The original complex query.
            documents (List[Document]): Relevant documents retrieved from the vector database.

        Returns:
            str: The synthesized response.
        """
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"Based on the following retrieved information, answer the query:\n\nQuery: {query}\n\nContext:\n{context}\n\nResponse:"
        return self.llm.generate(prompt)

    def __call__(self, query: str) -> str:
        """Process a query by breaking it into sub-queries, retrieving relevant documents, and synthesizing a response.

        Args:
            query (str): The user query.

        Returns:
            str: The final synthesized response.
        """
        sub_queries = self.generate_sub_queries(query)
        documents = self.retrieve_relevant_documents(sub_queries)
        return self.synthesize_response(query, documents)
