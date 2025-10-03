import os
from typing import Any, Optional

import weave
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from weave import Model
from weave.integrations.langchain import WeaveTracer

from rag_pipelines.llms.groq import ChatGroqGenerator
from rag_pipelines.pipelines.crag_graph_state import CRAGGraphState
from rag_pipelines.query_transformer import QueryTransformer
from rag_pipelines.retrieval_evaluator import DocumentGrader, QueryDecisionMaker
from rag_pipelines.websearch import WebSearch


# Disable global tracing explicitly
os.environ["WEAVE_TRACE_LANGCHAIN"] = "false"


class CorrectiveRAGPipeline(Model):
    """A corrective retrieval-augmented generation (RAG) pipeline using Weave for tracing and LangChain components.

    This pipeline integrates document retrieval, relevance evaluation, grading, query transformation, web search,
    and LLM-based response generation to implement a corrective RAG system. It utilizes Weave for tracing execution
    details and LangChain components for processing.

    Attributes:
    -----------
    retriever : Optional[PineconeHybridSearchRetriever]
        The retrieval model used to fetch relevant documents based on a query.
    prompt : Optional[ChatPromptTemplate]
        The prompt template to generate questions for the LLM.
    generator : Optional[ChatGroqGenerator]
        The language model used to generate responses.
    grader : Optional[DocumentGrader]
        Grades documents based on evaluation results.
    query_transformer : Optional[QueryTransformer]
        Transforms user queries to optimize retrieval.
    web_search : Optional[WebSearch]
        Performs web search for additional context.
    tracing_project_name : str
        The name of the Weave project for tracing.
    tracer : Optional[WeaveTracer]
        The tracer used to record execution details with Weave.
    weave_params : Dict[str, Any]
        Parameters for initializing Weave.

    Methods:
    --------
    _initialize_weave(**weave_params):
        Initializes Weave with the specified tracing project name and parameters.
    _build_crag_graph() -> CompiledStateGraph:
        Builds and compiles the corrective RAG workflow graph.
    predict(question: str) -> str:
        Executes the corrective RAG pipeline for the given question and returns the generated response.
    """

    retriever: Optional[PineconeHybridSearchRetriever] = None
    prompt: Optional[ChatPromptTemplate] = None
    generator: Optional[ChatGroqGenerator] = None
    grader: Optional[DocumentGrader] = None
    query_transformer: Optional[QueryTransformer] = None
    web_search: Optional[WebSearch] = None
    tracing_project_name: str
    weave_params: dict[str, Any]
    tracer: Optional[WeaveTracer] = None

    def __init__(
        self,
        retriever: PineconeHybridSearchRetriever,
        prompt: ChatPromptTemplate,
        generator: ChatGroqGenerator,
        grader: DocumentGrader,
        query_transformer: QueryTransformer,
        web_search: WebSearch,
        tracing_project_name: str = "corrective_rag",
        weave_params: Optional[dict[str, Any]] = None,
    ):
        """Initializes the HybridCorrectiveRAGPipeline.

        Args:
        -----
        retriever : PineconeHybridSearchRetriever
            The retrieval model used to fetch documents for the RAG pipeline.
        prompt : ChatPromptTemplate
            The prompt template used to create questions for the LLM.
        generator : ChatGroqGenerator
            The language model used for response generation.
        grader : DocumentGrader
            Component to grade the relevance of evaluated documents.
        query_transformer : QueryTransformer
            Component to transform the user query.
        web_search : WebSearch
            Component to perform web search for additional context.
        tracing_project_name : str
            The name of the Weave project for tracing. Defaults to "corrective_rag".
        weave_params : Dict[str, Any]
            Additional parameters for initializing Weave.
        """
        if weave_params is None:
            weave_params = {}
        super().__init__(
            retriever=retriever,
            prompt=prompt,
            generator=generator,
            grader=grader,
            query_transformer=query_transformer,
            web_search=web_search,
            tracing_project_name=tracing_project_name,
            weave_params=weave_params,
        )

        self.retriever = retriever
        self.prompt = prompt
        self.generator = generator
        self.grader = grader
        self.query_transformer = query_transformer
        self.web_search = web_search
        self.tracing_project_name = tracing_project_name

        if weave_params:
            self._initialize_weave(**weave_params)
        else:
            self._initialize_weave()

    def _initialize_weave(self, **weave_params) -> None:
        """Initializes Weave with the specified tracing project name.

        Sets up the Weave environment and creates a tracer for monitoring pipeline execution.

        Args:
        -----
        **weave_params: Dict[str, Any]
            Additional parameters for configuring Weave.
        """
        weave.init(self.tracing_project_name, **weave_params)
        self.tracer = WeaveTracer()

    def _build_crag_graph(self) -> CompiledStateGraph:
        """Builds and compiles the corrective RAG workflow graph.

        The graph defines the flow between components like retrieval, grading, query transformation,
        web search, and generation.

        Returns:
        --------
        CompiledStateGraph:
            The compiled state graph representing the corrective RAG pipeline workflow.
        """
        crag_workflow = StateGraph(CRAGGraphState)

        # Define the nodes
        crag_workflow.add_node("retrieve", self.retriever)
        crag_workflow.add_node("grade_documents", self.grader)
        crag_workflow.add_node("generate", self.generator)
        crag_workflow.add_node("transform_query", self.query_transformer)
        crag_workflow.add_node("web_search_node", self.web_search)

        # Define edges between nodes
        crag_workflow.add_edge(START, "retrieve")
        crag_workflow.add_edge("retrieve", "grade_documents")
        crag_workflow.add_conditional_edges(
            "grade_documents",
            QueryDecisionMaker(),
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        crag_workflow.add_edge("transform_query", "web_search_node")
        crag_workflow.add_edge("web_search_node", "generate")
        crag_workflow.add_edge("generate", END)

        # Compile the graph
        crag_pipeline = crag_workflow.compile()
        return crag_pipeline

    @weave.op()
    def predict(self, question: str) -> str:
        """Executes the corrective RAG pipeline with the given question.

        The pipeline retrieves documents, evaluates and grades their relevance, and generates a final response
        using the LLM.

        Args:
        -----
        question : str
            The input question to be answered.

        Returns:
        --------
        str:
            The final answer generated by the LLM.

        Example:
        --------
        ```python
        pipeline = HybridCorrectiveRAGPipeline(...)
        answer = pipeline.predict("What are the latest AI trends?")
        print(answer)
        ```
        """
        config = {"callbacks": [self.tracer]}

        crag_graph = self._build_crag_graph()
        response = crag_graph.invoke(question, config=config)

        return response
