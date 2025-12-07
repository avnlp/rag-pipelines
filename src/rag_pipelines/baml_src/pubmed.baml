// PubMedQA

// The `PubMedAnswer` class defines the structured output we expect from the LLM.
class PubMedAnswer {
  // The `chain_of_thought` field captures the LLM's reasoning process.
  chain_of_thought string @description("Step-by-step reasoning for how the medical answer was derived from the context.")
  // The `answer` field contains the final, medical answer.
  answer string @description("The final, concise medical answer to the question based on the provided context.")
}

// The PubMedAnswer is generated based on the context and question
function GeneratePubMedAnswer(context: string, question: string) -> PubMedAnswer {
    // The `GroqFallbackClient` tries a sequence of providers (Groq, then Cerebras, then SambaNova)
    client GroqFallbackClient

    // The `{{ ctx.output_format }}` uses the schema from the `PubMedAnswer` class
    prompt #"
    {{ _.role("system") }}
    You are an expert biomedical researcher. Use the retrieved context
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

test TestPubMedQuestion {
    functions [GeneratePubMedAnswer]
    args {
    context "Endothelial dysfunction is a key mechanism in the development of pulmonary arterial hypertension (PAH). MicroRNA-124 has been implicated in vascular biology and inflammation. This study investigated the role of microRNA-124 in pulmonary arterial endothelial cells and its potential therapeutic implications in PAH. METHODS: We analyzed microRNA-124 expression in lung tissues from PAH patients and controls. Human pulmonary arterial endothelial cells (HPAECs) were transfected with microRNA-124 mimics or inhibitors, and cell proliferation, apoptosis, and migration were assessed. A rat model of PAH was induced by monocrotaline injection, and the therapeutic effect of microRNA-124 overexpression was evaluated. RESULTS: MicroRNA-124 expression was significantly downregulated in lung tissues from PAH patients compared to controls. In HPAECs, microRNA-124 overexpression inhibited cell proliferation and migration while promoting apoptosis. In the rat PAH model, microRNA-124 overexpression ameliorated pulmonary vascular remodeling and improved right ventricular function."
    question "Is microRNA-124 downregulated in pulmonary arterial hypertension and does its overexpression ameliorate the condition in experimental models?"
    }
}