import os
from typing import Any, Optional, Union

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag_pipelines.prompts import RAG_PROMPT


class ChatGroqGenerator(weave.Model):
    """Interact with the ChatGroq model to generate responses based on user queries and documents.

    This class provides an interface for generating responses using the ChatGroq model.
    It handles prompt formatting, LLM invocation, document integration, and result generation.

    Attributes:
        model (str): The name of the ChatGroq model to use.
        api_key (str): The API key for authenticating with the ChatGroq service.
        llm (ChatGroq): An instance of the ChatGroq language model.
        llm_params (dict): Additional configuration parameters for the ChatGroq model.
        prompt_template (ChatPromptTemplate, optional): A template for structuring prompts.

    Methods:
        format_user_prompts(prompts): Format user prompts for ChatGroq.
        predict(user_prompts, system_prompt, invoke_inputs): Generate a response using ChatGroq.
        format_docs(docs): Prepare document content for integration into model inputs.
    """

    model: str
    api_key: str
    llm_params: dict[str, Union[str, int, float, bool, str]]
    llm: Optional[ChatGroq] = None
    prompt_template: Optional[ChatPromptTemplate] = None

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        llm_params: Optional[dict[str, Union[str, int, float, bool, str]]] = None,
        prompt_template: Optional[ChatPromptTemplate] = None,
    ):
        """Initialize the ChatGroqGenerator with configuration parameters.

        Args:
            model (str): The name of the ChatGroq model to use.
            api_key (Optional[str]): API key for the ChatGroq service. If not provided,
                the "GROQ_API_KEY" environment variable will be used.
            llm_params (Optional[dict]): Additional parameters for configuring the ChatGroq model.
            prompt_template (Optional[ChatPromptTemplate]): A template for structuring messages.

        Raises:
            ValueError: If the API key is not provided and the "GROQ_API_KEY" environment variable is not set.
        """
        if llm_params is None:
            llm_params = {}
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key is None:
            msg = "GROQ_API_KEY is not set. Please provide an API key or set it as an environment variable."
            raise ValueError(msg)

        super().__init__(model=model, api_key=api_key, llm_params=llm_params)
        self.model = model
        self.api_key = api_key
        self.llm_params = llm_params
        self.prompt_template = prompt_template
        self.llm = ChatGroq(model=self.model, api_key=self.api_key, **llm_params)

    @weave.op()
    def format_user_prompts(self, prompts: list[str]) -> list[dict]:
        """Format a list of user prompts for compatibility with the ChatGroq API.

        This method processes text and image prompts, ensuring they are properly
        structured for model input.

        Args:
            prompts (list[str]): A list of user prompts, where each prompt is a string.

        Returns:
            list[dict]: A list of formatted prompts, each represented as a dictionary.
        """
        content = []
        for prompt in prompts:
            if prompt.startswith("data:image/png;base64,") or prompt.startswith("data:image/jpeg;base64,"):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": prompt, "detail": "high"},
                    }
                )
            else:
                content.append({"type": "text", "text": prompt})
        return content

    @weave.op()
    def predict(
        self,
        user_prompts: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
        invoke_inputs: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a response from the ChatGroq model using provided inputs.

        Args:
            user_prompts (Optional[list[str]]): A list of user prompts to process.
            system_prompt (Optional[str]): A system-level instruction for the model.
            invoke_inputs (Optional[dict[str, Any]]): Additional keyword arguments for LLM invocation.

        Returns:
            str: The response text generated by the ChatGroq model.
        """
        messages: list[Union[SystemMessage, HumanMessage]] = []

        if system_prompt:
            messages.append(SystemMessage(content=[{"type": "text", "text": system_prompt}]))

        user_prompt_contents = self.format_user_prompts(user_prompts)
        messages.append(HumanMessage(content=user_prompt_contents))

        if not self.prompt_template:
            self.prompt_template = ChatPromptTemplate.from_messages(messages)

        llm_chain = self.prompt_template | self.llm | StrOutputParser()
        return llm_chain.invoke(**(invoke_inputs or {}))

    def format_docs(self, docs: list) -> str:
        """Format retrieved documents into a string for model input.

        Args:
            docs (list): A list of document objects retrieved during the pipeline's execution.

        Returns:
            str: A formatted string of document contents, ready for input to the ChatGroq model.
        """
        formatted_context = []
        for doc in docs:
            date = doc.metadata.get("filing_date", "")
            accession_no = doc.metadata.get("accession_no", "")
            summary = doc.metadata.get("summary", "")
            image_descriptions = doc.metadata.get("image_descriptions", "")

            doc_text = (
                f"# Report {accession_no} filed on {date}:\n\n"
                f"## An excerpt from the report:\n\n{doc.page_content}\n\n"
                f"## Image descriptions:\n\n{image_descriptions}\n\n"
                f"## Summary:\n\n{summary}\n\n"
            )
            formatted_context.append(doc_text)

        return "\n".join(formatted_context)

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate a response using the current state of user prompts and graded documents.

        Args:
            state (dict[str, Any]): The current state, containing:
                - 'question': The user question.
                - 'documents': A list of graded documents.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'documents': The original documents.
                - 'question': The user question.
                - 'generation': The generated response.
        """
        question = state["question"]
        documents = state["documents"]

        context = self.format_docs(documents)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", RAG_PROMPT),
            ]
        )

        response = self.predict(invoke_inputs={"question": question, "context": context})

        return {"documents": documents, "question": question, "generation": response}
