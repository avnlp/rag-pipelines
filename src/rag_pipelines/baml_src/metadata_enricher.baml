// Metadata Enricher

// Define a base dynamic class that can be extended at runtime
class DynamicMetadata {
  // This class is marked as dynamic and will be modified by TypeBuilder at runtime
  @@dynamic
}

// Define the function that will perform the extraction
// It takes the input text and returns an instance of the DynamicMetadata class
function ExtractMetadata(text: string) -> DynamicMetadata {
  client GroqFallbackClient
  prompt #"
    <instruction>
    You are a precise metadata extraction system.
    Extract the following fields from the provided text if they are explicitly mentioned.
    The fields to extract are defined by the schema provided.
    </instruction>

    <input_text>
    {{ text }}
    </input_text>

    {{ ctx.output_format }}
    Ensure the output is valid JSON matching the specified schema.
    Do not include fields that are not found in the text. Omit them entirely.
 "#
}

// An empty schema will be passed here since the dynamic fields have not been defined yet
test TestMetadataExtraction {
    functions [ExtractMetadata]
    args {
        text "Apple Inc. reported strong results for Q3 2024. The company's revenue reached $81.8 billion, marking a significant increase. The CEO mentioned that iPhone sales drove most of the growth."
    }
}

// Test case for dynamic metadata extraction with a specific schema
test TestDynamicMetadataExtraction {
    functions [ExtractMetadata]
    type_builder {
        // Define the dynamic class based on the schema
        dynamic class DynamicMetadata {
          company_name string?
          quarter string?
          revenue float?
          is_positive bool?
        }
    }
    args {
      text "Apple Inc. reported strong results for Q3 2024. The company's revenue reached $81.8 billion, marking a significant increase. The CEO mentioned that iPhone sales drove most of the growth."
    }
}