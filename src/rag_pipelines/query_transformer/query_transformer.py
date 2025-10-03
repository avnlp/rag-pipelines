from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QueryTransformer:
    """Transform user questions into optimized versions for document retrieval.

    This class leverages a Large Language Model (LLM) to rewrite user questions by analyzing their semantic
    intent, making them more effective for retrieval tasks in pipelines involving LLMs.

    Attributes:
        llm (Any): The language model used for query transformation.

    Methods:
        transform_query(question: str) -> dict[str, str]:
            Transform an input question into an optimized version for retrieval.
        __call__(state: dict[str, Any]) -> dict[str, Any]:
            Process a state dictionary containing a question and documents, returning the transformed question.
    """

    def __init__(self, llm: Any):
        """Initialize the QueryTransformer with a language model.

        Args:
            llm (Any): The large language model that processes and transforms the query.
                The LLM must implement an `invoke` method to process the input and generate the transformed query.
        """
        self.llm = llm

    def transform_query(self, question: str) -> dict[str, str]:
        """Transform a given user question into an optimized form for document retrieval.

        This method uses the LLM to analyze the input question, infer its semantic intent, and
        return a more effective version for retrieval purposes.

        Args:
            question (str): The user question to transform.

        Returns:
            dict[str, str]: A dictionary containing the transformed question under the key 'question'.

        Example:
            ```python
            original_question = "What are the benefits of cloud computing?"
            transformer = QueryTransformer(llm)
            transformed_query = transformer.transform_query(original_question)
            print(
                transformed_query["question"]
            )  # Output: 'What advantages does cloud computing provide?'
            ```
        """
        self.prompt = PromptTemplate.from_template(
            (
                "You are generating questions that are well optimized for retrieval.\n"
                "Look at the input and try to reason about the underlying semantic intent.\n"
                "Here is the initial question:\n"
                "-------\n"
                "{question}\n"
                "-------\n"
                "Provide an improved question without any preamble, only respond with the updated question:"
            ),
            input_variables=["question"],
        )

        # Define the chain that processes the question using the prompt, LLM, and output parser
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Invoke the chain to process the input question and return the optimized version
        rewritten_question = self.chain.invoke({"question": question})

        return {"question": rewritten_question}

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process a state dictionary to transform a question for optimized retrieval.

        This method takes the current state, transforms the question using the LLM,
        and returns the updated state containing the optimized question alongside
        the original documents.

        Args:
            state (dict[str, Any]): A dictionary with the following keys:
                - 'question': The original user question.
                - 'documents': A list of documents (passed through unchanged).

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'documents': The original documents.
                - 'question': The transformed question.

        Example:
            ```python
            state = {
                "question": "What are the benefits of cloud computing?",
                "documents": [{"content": "Sample document"}],
            }
            transformer = QueryTransformer(llm)
            new_state = transformer(state)
            print(new_state["question"])  # Optimized question
            ```
        """
        question = state["question"]
        documents = state["documents"]

        rewritten_question = self.transform_query(question)

        return {"documents": documents, "question": rewritten_question}
