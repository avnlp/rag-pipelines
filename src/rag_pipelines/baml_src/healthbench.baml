// HealthBench

// The `HealthBenchAnswer` class defines the structured output we expect from the LLM.
class HealthBenchAnswer {
  // The `chain_of_thought` field captures the LLM's reasoning process.
  chain_of_thought string @description("Step-by-step reasoning for how the medical response was derived from the context.")
  // The `summary` field contains the final, summarized answer.
  summary string @description("The final, concise medical answer to the user's question.")
}

// The HealthBenchAnswer is generated based on the context and question
function GenerateHealthBenchAnswer(context: string, question: string) -> HealthBenchAnswer {
    // The `GroqFallbackClient` tries a sequence of providers (Groq, then Cerebras, then SambaNova)
    client GroqFallbackClient

    // The `{{ ctx.output_format }}` uses the schema from the `Answer` class
    prompt #"
    {{ _.role("system") }}
    You are a medical AI assistant providing safe, evidence-based health information.

    CRITICAL SAFETY GUIDELINES:
    - For emergency symptoms (chest pain, difficulty breathing, severe bleeding, etc.), immediately recommend emergency services
    - Always emphasize the importance of professional medical evaluation
    - Do not provide definitive diagnoses without proper medical examination
    - Recommend appropriate level of care based on symptom severity
    - Provide clear, actionable advice while acknowledging limitations
    - Prioritize patient safety above all else.

    Use the retrieved clinical context to inform your response while adhering to established medical guidelines.

    Your response MUST be in the following format:
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Clinical Conversation Context:
    ---
    {{ context }}
    ---

    Current Patient Question: {{ question }}
"#
}

test TestHealthBenchQuestion {
    functions [GenerateHealthBenchAnswer]
    args {
        context "A 45-year-old male presents with chest pain radiating to the left arm, shortness of breath, and diaphoresis. ECG shows ST elevation in leads II, III, and aVF. Cardiac enzymes are elevated."
        question "What is the most likely diagnosis for this patient?"
    }
}