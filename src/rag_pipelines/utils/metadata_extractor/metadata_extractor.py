"""Metadata Extractor: Extracts structured metadata from text based on a JSON schema."""

from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, create_model
from tqdm import tqdm


class MetadataExtractor:
    """Extracts structured metadata from text using a language model and a JSON schema.

    This class leverages a function-calling or structured-output-capable LLM to extract
    metadata fields defined by a user-provided JSON schema. The schema is dynamically
    converted into a Pydantic model, which is used to enforce type safety and
    validation. Only fields that are successfully extracted (i.e., not null) are
    included in the result.

    This metadata can be used for filtering search results, or in the case of queries,
    it can be used to filter out documents that are not relevant to the query.

    Attributes:
        llm (BaseChatModel): The language model used for metadata extraction. Must
        support structured output (e.g., via `.with_structured_output()`).
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initializes the MetadataExtractor with a language model.

        Args:
            llm (BaseChatModel): A LangChain-compatible chat model that supports
                structured output (e.g., ChatGroq, ChatOpenAI). The model should be
                capable of returning Pydantic-model-compliant responses.
        """
        self.llm = llm

    def invoke(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts metadata from the given text according to the provided JSON schema.

        The method constructs a dynamic Pydantic model from the schema, instructs the
        LLM to extract only explicitly stated information, and returns a dictionary
        containing only the successfully extracted (non-null) fields.

        Args:
            text (str): The input text from which metadata should be extracted.
            schema (Dict[str, Any]): A JSON schema defining the expected metadata
                structure. Must contain a top-level "properties" key.
                Each property may specify:
                - "type": one of "string", "number", or "boolean"
                - "enum": (optional) a list of allowed values (for string fields)
                - "description": (optional) a field description used in the prompt

        Example:
                    {
                        "properties": {
                            "movie_title": {"type": "string"},
                            "rating": {"type": "number"},
                            "is_positive": {"type": "boolean"},
                            "tone": {"type": "string", "enum": ["positive", "negative"]}
                        }
                    }

        Returns:
            Dict[str, Any]: A dictionary of extracted metadata. Only fields that were
                present and non-null in the LLM's response are included. Fields that
                could not be extracted are omitted entirely.

        Raises:
            ValueError: If the provided schema does not contain a "properties" key.
            ValueError: If any property type is not string, number, or boolean.

        Note:
            The LLM is explicitly instructed not to hallucinate or use placeholder
            values (e.g., "Unknown"). Missing fields are returned as null by the model
            and then excluded from the final result.
        """
        if "properties" not in schema:
            raise ValueError("Schema must contain a 'properties' key.")

        properties = schema.get("properties", {})

        # Validate and filter properties to only allow string, number, and boolean types
        allowed_types = {"string", "number", "boolean"}
        validated_properties = {}

        for name, spec in properties.items():
            json_type = spec.get("type", "string")
            if json_type not in allowed_types:
                raise ValueError(
                    f"Unsupported type '{json_type}' for field '{name}'. "
                    f"Only {allowed_types} are allowed."
                )
            validated_properties[name] = spec

        # Build dynamic Pydantic model — all fields optional to allow None
        field_definitions: Dict[str, Any] = {}
        for name, spec in validated_properties.items():
            json_type = spec.get("type", "string")
            description = spec.get("description", "")
            enum_vals = spec.get("enum")

            type_map = {
                "string": str,
                "number": float,
                "boolean": bool,
            }
            py_type = type_map.get(json_type, str)

            if enum_vals is not None:
                # Only allow enum for string types
                if json_type == "string":
                    py_type = Literal[tuple(enum_vals)]  # type: ignore
                else:
                    raise ValueError(
                        f"Enum is only supported for string fields. "
                        f"Field '{name}' has type '{json_type}'."
                    )

            # Make every field optional so model can return null
            py_type = Optional[py_type]  # type: ignore[assignment]
            field_definitions[name] = (
                py_type,
                Field(default=None, description=description),
            )

        dynamic_pydantic_model = create_model("ExtractedMetadata", **field_definitions)

        # Clear, strict prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise metadata extraction system. "
                    "Extract only the fields specified in the schema from the input text. "
                    "For any field not explicitly mentioned in the text, return null for that field. "
                    "Do NOT use placeholders like 'Unknown', 'N/A', 'Not specified', or make up values. "
                    "Only use facts directly stated in the input.",
                ),
                ("human", "{input}"),
            ]
        )

        structured_llm = self.llm.with_structured_output(dynamic_pydantic_model)
        chain = prompt | structured_llm

        try:
            result_obj = chain.invoke({"input": text})
            result: Dict[str, Any] = {}
            for field in validated_properties:
                value = getattr(result_obj, field, None)
                # Only include field if it's not None
                if value is not None:
                    result[field] = value
            return result
        except Exception:
            # On any error (e.g., parsing, validation, LLM failure), return empty dict
            return {}

    def transform_documents(
        self, documents: List[Document], schema: Dict[str, Any]
    ) -> List[Document]:
        """Applies metadata extraction to a list of LangChain Documents.

        For each document, metadata is extracted from its `page_content` using the
        provided schema and merged into the document's existing metadata. The
        original document content is preserved.

        Args:
            documents (List[Document]): A list of LangChain Document objects to process.
            schema (Dict[str, Any]): A JSON schema defining the metadata structure to
            extract. See `invoke()` for schema format details.

        Returns:
            List[Document]: A new list of Document objects with enriched metadata.
                Each document's `metadata` field is updated with the extracted fields
                (excluding any that were null or missing).
        """
        transformed_documents: List[Document] = []

        for doc in tqdm(documents):
            extracted = self.invoke(doc.page_content, schema)
            new_metadata = {**doc.metadata, **extracted}
            transformed_documents.append(
                Document(page_content=doc.page_content, metadata=new_metadata)
            )

        return transformed_documents
