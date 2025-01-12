import logging
from typing import Any

from rag_pipelines.utils import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class QueryDecisionMaker:
    """Make decisions on the next step in the retrieval-augmented generation pipeline.

    This class evaluates the relevance of retrieved documents and determines the appropriate action:
    1. Transform the user's query for a web search if the retrieved documents are irrelevant.
    2. Generate an answer using the relevant documents if available.

    Designed for use in graph-based state management systems, this class processes the current state
    to guide the next step in the pipeline.

    Methods:
        __call__(state: dict[str, Any]) -> str:
            Determine the next action ("transform_query" or "generate") based on the state.
    """

    def __call__(self, state: dict[str, Any]) -> str:
        """Determine the next step in the pipeline based on document relevance.

        This method evaluates the relevance of retrieved documents, as indicated in the `web_search` field.
        If a web search is required due to irrelevant documents, it suggests transforming the query.
        Otherwise, it suggests generating an answer based on the relevant documents.

        Args:
            state (dict[str, Any]): The current state of the pipeline, containing:
                - `question` (str): The user's query.
                - `web_search` (str): A binary decision ("Yes" or "No") indicating if a web search is required.
                - `documents` (list): A list of retrieved and graded documents.

        Returns:
            str: The next step in the pipeline:
                - `"transform_query"`: If documents are irrelevant and a web search is needed.
                - `"generate"`: If relevant documents are available for answering the query.

        Example:
            ```python
            state = {
                "question": "What is the capital of France?",
                "web_search": "Yes",
                "documents": []
            }
            decision_maker = QueryDecisionMaker()
            next_step = decision_maker(state)
            print(next_step)  # Output: "transform_query"
            ```
        """
        logger.info("ASSESSING GRADED DOCUMENTS")
        web_search = state["web_search"]

        if web_search == "Yes":
            # All documents have been filtered as irrelevant
            logger.info("DECISION: DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY")
            return "transform_query"
        else:
            # Relevant documents are available, proceed to generate the answer
            logger.info("DECISION: GENERATE")
            return "generate"
