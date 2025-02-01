from dspy import ChainOfThought, Module
from dspy.primitives.prediction import Prediction

from rag_pipelines.prompts.dspy_answer_generation import GenerateAnswer
from rag_pipelines.vectordb.weaviate import WeaviateVectorDB


class DSPyRAG(Module):
    """DSPyRAG Module for Retrieval-Augmented Generation (RAG) using DSPy.

    This module integrates retrieval from a Weaviate vector database and answer
    generation using a chain-of-thought prompt. It is designed to first retrieve
    relevant passages (context) based on a user's question and then generate an answer
    leveraging the retrieved context.
    """

    def __init__(self, weaviate_db: WeaviateVectorDB):
        """Initialize the DSPyRAG module.

        Args:
            weaviate_db (WeaviateVectorDB): An instance of the Weaviate vector database
                used for retrieving relevant documents/passages.
        """
        # Initialize the parent Module class
        super().__init__()
        # Save the provided Weaviate vector database for later retrieval operations.
        self.weaviate_db = weaviate_db
        # Initialize the chain-of-thought answer generator with the provided prompt generator.
        self.generate_answer = ChainOfThought(GenerateAnswer)

    def retrieve(self, question: str) -> Prediction:
        """Retrieve relevant context from Weaviate based on the input question.

        This method queries the Weaviate vector database to find and return relevant
        passages that serve as context for generating an answer.

        Args:
            question (str): The input question for which context is to be retrieved.

        Returns:
            Prediction: A Prediction object containing the retrieved passages.
                        The 'passages' attribute holds the context retrieved from Weaviate.
        """
        # Query the Weaviate database to retrieve relevant context
        context = self.weaviate_db.retrieve(question)
        # Wrap the retrieved context in a Prediction object
        return Prediction(passages=context)

    def forward(self, question: str) -> Prediction:
        """Generate an answer based on the retrieved context for the input question.

        This method performs two main steps:
          1. It retrieves context from the Weaviate database by calling the retrieve() method.
          2. It generates an answer using the chain-of-thought answer generator with the
             retrieved context and the input question.

        Args:
            question (str): The input question to be answered.

        Returns:
            Prediction: A Prediction object containing:
                - context: The retrieved passages used for generating the answer.
                - answer: The generated answer to the question.
        """
        # Retrieve the context (passages) from Weaviate using the retrieve method.
        retrieved_context = self.retrieve(question).passages

        # Generate an answer using the chain-of-thought generator.
        # The generator takes both the retrieved context and the question as inputs.
        prediction = self.generate_answer(context=retrieved_context, question=question)

        # Return a new Prediction containing both the context and the generated answer.
        return Prediction(context=retrieved_context, answer=prediction.answer)
