"""Layer 3: Fixed Schema extraction (RAG-optimized automatic fields)."""

from typing import Any, Dict, TypeVar

from cachetools import TTLCache
from cachetools_async import cached

from rag_pipelines.baml_client.type_builder import TypeBuilder


T = TypeVar("T")


_fixed_cache: TTLCache[Any, Any] = TTLCache(maxsize=2000, ttl=3600)


class FixedSchemaEnricher:
    """Layer 3: Fixed Schema extraction (RAG-optimized automatic fields).

    Extracts a predefined set of fields designed to optimize RAG performance:
    questions_answered, chunk_summary, keywords, content_type, and header_text.
    These fields enable semantic search, result ranking, and query-document
    matching. Only runs in FULL enrichment mode due to higher LLM cost.
    """

    def __init__(self, baml_client: Any) -> None:
        """Initialize the fixed schema enricher.

        Args:
            baml_client: The BAML client for LLM calls. Must have
                ExtractMetadata function available.
        """
        self.baml_client = baml_client

    @cached(cache=_fixed_cache)  # type: ignore[misc]
    async def extract(self, text: str) -> Dict[str, Any]:
        """Extract RAG-optimized fixed fields from text.

        Applies a curated schema designed to improve RAG retrieval quality
        and ranking. All fields are optional - the LLM can decline extraction
        if fields don't apply to the chunk.

        Results are cached by text content hash to avoid redundant LLM calls
        for identical chunks. The global fixed cache is shared across all
        enricher instances to maximize hit rates during batch processing.

        Args:
            text: The text to extract metadata from.

        Returns:
            Dictionary of fixed schema fields with non-empty values.
            Returns empty dict if extraction fails.
        """
        # Define the fixed schema for RAG optimization. These fields are
        # specifically chosen to improve retrieval relevance and result ranking
        # in RAG systems.
        fixed_schema = {
            "properties": {
                "questions_answered": {
                    "type": "string",
                    "description": "2-3 natural questions that this chunk answers",
                },
                "chunk_summary": {
                    "type": "string",
                    "description": "1-2 sentence summary of the chunk",
                },
                "keywords": {
                    "type": "string",
                    "description": "3-7 important terms from the chunk",
                },
                "content_type": {
                    "type": "string",
                    "enum": [
                        "definition",
                        "example",
                        "procedure",
                        "conceptual",
                        "numerical",
                        "analytical",
                        "comparative",
                    ],
                    "description": "Classification of the chunk content",
                },
                "header_text": {
                    "type": "string",
                    "description": "Descriptive title for the chunk (2-8 words)",
                },
            }
        }

        tb = TypeBuilder()
        baml_definitions = []
        class_properties = []

        # Build BAML definitions for fixed schema. Process each field in the
        # predefined schema, handling type conversion and enum generation.
        for name, spec in fixed_schema.get("properties", {}).items():
            spec_dict = spec if isinstance(spec, dict) else {}
            json_type = spec_dict.get("type", "string")
            description = spec_dict.get("description", "")
            enum_vals = spec_dict.get("enum")

            # Map JSON types to BAML types. This schema only uses string types.
            baml_type_map = {
                "string": "string",
                "number": "float",
                "integer": "int",
                "boolean": "bool",
            }

            if json_type not in baml_type_map:
                continue

            baml_type = baml_type_map[json_type]

            # Handle enum fields by generating BAML enum definitions. content_type
            # field uses enums to enforce consistent content classification.
            if enum_vals and json_type == "string":
                # Convert snake_case field name to PascalCase enum name.
                # Example: content_type -> ContentTypeEnum
                parts = name.split("_")
                enum_name = "".join(p.capitalize() for p in parts) + "Enum"
                enum_def = f"enum {enum_name} {{\n"
                for val in enum_vals:
                    formatted_val = val[0].upper() + val[1:] if val else val
                    enum_def += f" {formatted_val}\n"
                enum_def += "}"
                baml_definitions.append(enum_def)
                baml_type = enum_name

            # Build field definition with optional type (?) to allow LLM to
            # decline extraction if field is not relevant to the content.
            field_def = f"  {name} {baml_type}?"
            if description:
                field_def += f' @description("{description}")'
            class_properties.append(field_def)

        # Build complete BAML class matching the fixed schema. Class name must be
        # DynamicMetadata to match ExtractMetadata BAML function signature.
        baml_class_def = (
            "dynamic class DynamicMetadata {\n" + "\n".join(class_properties) + "\n}"
        )
        baml_definitions.append(baml_class_def)

        full_baml_string = "\n".join(baml_definitions)
        tb.add_baml(full_baml_string)

        try:
            # Call ExtractMetadata BAML function with fixed schema. TypeBuilder
            # passes dynamically generated schema via baml_options.
            result_obj = await self.baml_client.ExtractMetadata(
                text, baml_options={"tb": tb}
            )

            # Extract only fields defined in fixed schema, filtering None values
            # and empty strings to avoid polluting metadata.
            result_dict = {}
            for field_name in fixed_schema.get("properties", {}):
                field_value = getattr(result_obj, field_name, None)
                if field_value is not None:
                    result_dict[field_name] = field_value

            return {k: v for k, v in result_dict.items() if v is not None and v != ""}

        except Exception:
            # Gracefully handle LLM failures, type mismatches, or malformed
            # responses. Returns empty dict to allow pipeline to continue.
            return {}
