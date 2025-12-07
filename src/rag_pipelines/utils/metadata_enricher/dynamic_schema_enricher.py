"""Layer 2: Dynamic Schema extraction (user-defined fields)."""

import hashlib
import json
from typing import Any, Dict

from cachetools import TTLCache
from cachetools_async import cached

from rag_pipelines.baml_client.type_builder import TypeBuilder


_dynamic_cache: TTLCache[Any, Any] = TTLCache(maxsize=2000, ttl=3600)


def _make_schema_key(
    *args: Any, text: str | None = None, schema: Dict[str, Any] | None = None
) -> tuple[str, str]:
    """Generate cache key from text and schema."""
    if text is None and args:
        text = str(args[0])

    text_hash = hashlib.sha256((text or "").encode()).hexdigest()

    if schema is None:
        schema_hash = "none"
    else:
        schema_str = json.dumps(schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()

    return (text_hash, schema_hash)


class DynamicSchemaEnricher:
    """Layer 2: Dynamic Schema extraction (user-defined fields).

    Extracts LLM-powered metadata fields specified in user-provided JSON schema.
    Uses BAML's TypeBuilder to dynamically generate typed classes matching the
    schema, enabling structured LLM output extraction. Supports custom fields
    with descriptions and enums. Gracefully handles schema mismatches and LLM
    failures by returning empty dict.
    """

    def __init__(self, baml_client: Any) -> None:
        """Initialize the dynamic enricher.

        Args:
            baml_client: The BAML client for LLM calls. Must have
                ExtractMetadata function available.
        """
        self.baml_client = baml_client

    @cached(cache=_dynamic_cache, key=_make_schema_key)  # type: ignore[misc]
    async def extract(self, text: str, user_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user-defined metadata fields from text.

        Converts JSON schema to BAML class definitions, uses ExtractMetadata
        LLM function to extract values, and returns only non-empty results.

        Results are cached by both text content and schema structure to avoid
        redundant LLM calls. The global dynamic cache is shared across enricher
        instances to maximize hit rates when processing documents with the same
        schema and text content.

        Args:
            text: The text to extract metadata from.
            user_schema: JSON schema defining user fields. Expected format:
                {"properties": {"field_name": {"type": "string", ...}, ...}}

        Returns:
            Dictionary of extracted user-defined fields with non-empty values.
            Returns empty dict if schema is invalid or extraction fails.
        """
        if not user_schema or "properties" not in user_schema:
            return {}

        # TypeBuilder enables dynamic BAML class generation at runtime,
        # allowing us to support arbitrary user-defined schemas without
        # pre-defining all possible field combinations.
        tb = TypeBuilder()
        baml_definitions = []
        class_properties = []

        for name, spec in user_schema.get("properties", {}).items():
            json_type = spec.get("type", "string")
            description = spec.get("description", "")
            enum_vals = spec.get("enum")

            # Map JSON Schema types to BAML types. Unsupported types are skipped
            # to allow gradual schema evolution.
            baml_type_map = {
                "string": "string",
                "number": "float",
                "integer": "int",
                "boolean": "bool",
            }

            if json_type not in baml_type_map:
                continue

            baml_type = baml_type_map[json_type]

            # Handle enum fields by generating BAML enum definitions. Enum values
            # are capitalized for valid BAML identifier formatting (e.g., "high"
            # becomes "High" in the enum).
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
            # decline extraction if field is not applicable to the text.
            field_def = f"  {name} {baml_type}?"
            if description:
                field_def += f' @description("{description}")'
            class_properties.append(field_def)

        if not class_properties:
            return {}

        # Build complete BAML class matching user schema structure.
        # Uses "dynamic" keyword to indicate this is generated at runtime.
        baml_class_def = (
            "dynamic class DynamicMetadata {\n" + "\n".join(class_properties) + "\n}"
        )
        baml_definitions.append(baml_class_def)

        full_baml_string = "\n".join(baml_definitions)
        tb.add_baml(full_baml_string)

        try:
            # Call ExtractMetadata BAML function with dynamically generated
            # schema. TypeBuilder passes schema via baml_options.
            result_obj = await self.baml_client.ExtractMetadata(
                text, baml_options={"tb": tb}
            )

            # Extract only fields present in user schema, filtering None values
            # and empty strings to avoid polluting metadata with missing data.
            result_dict = {}
            for field_name in user_schema.get("properties", {}):
                field_value = getattr(result_obj, field_name, None)
                if field_value is not None:
                    result_dict[field_name] = field_value

            return {k: v for k, v in result_dict.items() if v is not None and v != ""}

        except Exception:
            # Gracefully handle LLM failures, type mismatches, or malformed
            # responses. Returns empty dict to allow pipeline to continue.
            return {}
