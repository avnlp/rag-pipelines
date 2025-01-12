from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document


class WebSearch:
    """Perform web searches and append results to filtered documents.

    This class uses the DuckDuckGo search tool to perform web searches based on user queries. The search results
    are converted into `Document` objects and appended to an existing list of documents for further use.

    Attributes:
        search_tool (DuckDuckGoSearchRun): The DuckDuckGo search tool used to perform web searches.
    """

    def __init__(self):
        """Initialize the WebSearch class with the DuckDuckGo search tool.

        Example:
            ```python
            web_searcher = WebSearch()
            ```
        """
        # Initialize the DuckDuckGo search tool
        self.search_tool = DuckDuckGoSearchRun()

    def search(self, question: str, filtered_docs: list[Document]) -> list[Document]:
        """Perform a web search using DuckDuckGo and append results to filtered documents.

        This method conducts a web search based on the provided question and appends the resulting web search
        results to the given list of filtered documents.

        Args:
            question (str): The user query used for the web search.
            filtered_docs (list[Document]): A list of existing documents to which web search results will be added.

        Returns:
            list[Document]: An updated list of documents containing the original filtered documents along with
            the newly retrieved web search results.

        Example:
            ```python
            question = "What is the latest AI research?"
            filtered_docs = [Document(page_content="Existing content")]

            updated_docs = web_searcher.search(question, filtered_docs)
            print(updated_docs)
            ```
        """
        # Conduct a web search using the DuckDuckGo search tool
        search_results = self.search_tool.invoke({"query": question})

        # Create a Document object from the search results
        web_results = Document(page_content=search_results)

        # Append the web search results to the filtered documents list
        filtered_docs.append(web_results)

        # Return the updated list of documents
        return filtered_docs

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the input state and append web search results to its documents.

        This callable interface extracts a question and a list of documents from the input state, performs a web
        search, and appends the results to the documents in the state.

        Args:
            state (dict[str, Any]): A dictionary containing:
                - 'question' (str): The user query for the web search.
                - 'documents' (list[Document]): Existing documents to which web search results will be added.

        Returns:
            dict[str, Any]: An updated state dictionary containing:
                - 'question' (str): The original query.
                - 'documents' (list[Document]): The updated list of documents with appended web search results.

        Example:
            ```python
            state = {
                "question": "What is the latest AI research?",
                "documents": [Document(page_content="Existing content")]
            }
            web_searcher = WebSearch()
            updated_state = web_searcher(state)
            print(updated_state["documents"])
            ```
        """
        # Extract the question and documents from the state
        question = state["question"]
        documents = state["documents"]

        # Perform a web search and append the results to the existing documents
        updated_docs = self.search(question=question, filtered_docs=documents)

        # Return the updated state with the new documents
        return {"question": question, "documents": updated_docs}
