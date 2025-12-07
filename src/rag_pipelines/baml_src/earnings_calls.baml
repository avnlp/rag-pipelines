// Earnings Calls

// The `Answer` class defines the structured output we expect from the LLM.
class Answer {
  // The `chain_of_thought` field captures the LLM's reasoning process.
  chain_of_thought string @description("Step-by-step reasoning for how the answer was derived from the context.")
  // The `summary` field contains the final, summarized answer.
  summary string @description("The final, concise answer to the user's question.")
}

// The Answer is generated based on the context and question
function GenerateAnswer(context: string, question: string) -> Answer {
  // The `GroqFallbackClient` tries a sequence of providers (Groq, then Cerebras, then SambaNova)
  client GroqFallbackClient

  // The `{{ ctx.output_format }}` uses the schema from the `Answer` class
  prompt #"
    {{ _.role("system") }}
    You are an expert earnings call financial analyst. Use the retrieved context
    to answer the question precisely. Provide a detailed and accurate
    answer based on the provided context.

    Your response MUST be in the following format:
    {{ ctx.output_format }}

    {{ _.role("user") }}
    Context:
    ---
    {{ context }}
    ---

    Question: {{ question }}
  "#
}

test TestEarningsQuestion {
  functions [GenerateAnswer]
  args {
    context "In Q4 2023, Apple's revenue was $119.6 billion, driven by strong iPhone sales. The services division also saw significant growth, reaching an all-time revenue record."
    question "What was Apple's revenue in Q4 2023 and what were the main drivers?"
  }
}