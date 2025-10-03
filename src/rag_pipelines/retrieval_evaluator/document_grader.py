from typing import Any

from rag_pipelines.retrieval_evaluator import RetrievalEvaluator


class DocumentGrader:
    """Grade documents and determine if a web search is required.

    This class evaluates document relevance using the `RetrievalEvaluator` and applies a predefined threshold
    to decide whether the retrieved documents are sufficient for answering a query. If the proportion of relevant
    documents falls below the threshold, it recommends conducting a web search.

    Attributes:
        threshold (float): The minimum relevance ratio required to avoid a web search.
        retrieval_evaluator (RetrievalEvaluator): An instance used for scoring document relevance.

    Methods:
        grade_documents(question: str, documents: list[Any]) -> dict[str, Any]:
            Evaluate and filter relevant documents while determining if a web search is needed.

        __call__(state: dict[str, Any]) -> dict[str, Any]:
            Process the current state, filter documents, and decide if a web search is necessary.
    """

    def __init__(self, retrieval_evaluator: RetrievalEvaluator, threshold: float = 0.4):
        """Initialize the DocumentGrader with a threshold and evaluator.

        Args:
            retrieval_evaluator (RetrievalEvaluator): An instance for evaluating document relevance.
            threshold (float): The minimum ratio of relevant documents required to avoid a web search. Defaults to 0.4.
        """
        self.threshold = threshold
        self.retrieval_evaluator = retrieval_evaluator

    def grade_documents(self, question: str, documents: list[Any]) -> dict[str, Any]:
        """Grade documents for relevance and filter out irrelevant ones.

        This method evaluates the relevance of documents using the `RetrievalEvaluator`. Relevant documents
        are retained, and the ratio of relevant documents to total documents is compared against a threshold
        to determine if a web search should be performed.

        Args:
            question (str): The user's query to evaluate document relevance.
            documents (list[Any]): A list of retrieved documents. Each document must have a `page_content` attribute.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'documents' (list[Any]): A list of relevant documents passing the threshold.
                - 'run_web_search' (str): A recommendation for a web search ("Yes" or "No").

        Example:
            ```python
            grader = DocumentGrader(threshold=0.4, retrieval_evaluator=evaluator)
            results = grader.grade_documents("What is AI?", documents)
            print(results["run_web_search"])  # Output: 'Yes' or 'No'
            ```
        """
        scored_documents = self.retrieval_evaluator.score_documents(
            question=question, documents=documents
        )

        relevant_docs = []
        relevant_count = 0

        for scored_doc in scored_documents:
            if scored_doc["relevance_score"] == "yes":
                relevant_docs.append(scored_doc["document"])
                relevant_count += 1

        # Calculate relevance ratio
        relevance_ratio = (
            relevant_count / len(scored_documents) if scored_documents else 0
        )

        # Determine if a web search is needed
        run_web_search = "Yes" if relevance_ratio <= self.threshold else "No"

        return {
            "documents": relevant_docs,
            "run_web_search": run_web_search,
        }

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the state to filter documents and decide if a web search is necessary.

        Args:
            state (dict[str, Any]): The current state containing:
                - 'question' (str): The user's query.
                - 'documents' (list[Any]): The list of retrieved documents.

        Returns:
            dict[str, Any]: An updated state with:
                - 'documents' (list[Any]): The filtered relevant documents.
                - 'question' (str): The original query.
                - 'web_search' (str): A web search recommendation ("Yes" or "No").

        Example:
            ```python
            state = {"question": "What is the capital of France?", "documents": [...]}
            updated_state = grader(state)
            print(updated_state["web_search"])  # Output: 'Yes' or 'No'
            ```
        """
        question = state["question"]
        documents = state["documents"]

        graded_results = self.grade_documents(question=question, documents=documents)

        return {
            "documents": graded_results["documents"],
            "question": question,
            "web_search": graded_results["run_web_search"],
        }
