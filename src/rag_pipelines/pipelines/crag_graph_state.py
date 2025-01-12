from typing_extensions import TypedDict


class CRAGGraphState(TypedDict):
    """Represents the state of the graph for the Corrective Retrieval-Augmentation-Generation (CRAG) pipeline.

    Attributes:
        question (str): The input question for the pipeline.
        generation (str): The generated response from the LLM.
        web_search (str): Indicates whether a web search is required (e.g., "yes" or "no").
        documents (List[str]): A list of relevant documents retrieved or processed.
    """

    question: str
    generation: str
    web_search: str
    documents: list[str]
