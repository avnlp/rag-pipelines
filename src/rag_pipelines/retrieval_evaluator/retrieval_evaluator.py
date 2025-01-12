from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class RetrievalEvaluator:
    """Evaluate the relevance of documents in relation to a user question.

    This class leverages a Large Language Model (LLM) to assign a binary relevance score ("yes" or "no") to each document
    based on its ability to answer the given question. The relevance assessment ensures that only relevant documents are
    used in subsequent stages of a pipeline, such as answer generation.

    Attributes:
        llm (Any): The language model used for relevance evaluation.
        prompt (PromptTemplate): A template that formats the question and document for the LLM.
        chain (LangChain Chain): A chain linking the prompt, LLM, and output parser to process relevance scoring.
    """

    def __init__(self, llm: Any):
        """Initialize the RetrievalEvaluator with a language model and set up the evaluation chain.

        The initializer constructs a prompt template and connects it to the LLM and output parser,
        creating a chain that facilitates binary relevance scoring.

        Args:
            llm (Any): The language model used for evaluating document relevance.
        """
        self.llm = llm
        self.prompt = PromptTemplate(
            template=(
                "You are a grader assessing the relevance of a retrieved document to a user question.\n"
                "Here is the retrieved document:\n\n{context}\n\n"
                "Here is the user question: {question}\n"
                "If the document is related to the user question in meaning and contains facts that answer the question, grade it as relevant.\n"
                "The goal is to filter out erroneous retrievals.\n"
                "Give a binary score ('yes' or 'no') to indicate whether the document is relevant to the question.\n"
                "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."
            ),
            input_variables=["question", "context"],
        )
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def score_documents(self, question: str, documents: list[Any]) -> list[dict[str, Any]]:
        """Score the relevance of each document with respect to the given user question.

        This method evaluates the relevance of each document by passing the user question and the document's content
        to the LLM. The LLM provides a binary score ('yes' or 'no') based on whether the document is relevant.

        Args:
            question (str): The user's question to evaluate the documents against.
            documents (list[Any]): A list of documents to be evaluated.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing:
                - `document`: The original document.
                - `relevance_score`: The binary relevance score ('yes' or 'no').

        Example:
            ```python
            evaluator = RetrievalEvaluator(llm)
            question = "What are the benefits of cloud computing?"
            documents = [
                Document(page_content="Cloud computing provides scalability."),
                Document(page_content="It is raining outside.")
            ]
            results = evaluator.score_documents(question, documents)
            print(results)
            # Output: [{'document': <Document>, 'relevance_score': 'yes'}, {'document': <Document>, 'relevance_score': 'no'}]
            ```
        """
        scored_documents = []

        # Evaluate each document's relevance
        for doc in documents:
            score = self.chain.invoke(
                {
                    "question": question,
                    "context": doc.page_content,
                }
            )
            grade = score["score"]
            scored_documents.append({"document": doc, "relevance_score": grade})

        return scored_documents
