// MetaMedQA

// The `MetaMedQAAnswer` class defines the structured output we expect from the LLM.
class MetaMedQAAnswer {
  // The `chain_of_thought` field captures the LLM's reasoning process.
  chain_of_thought string @description("Step-by-step reasoning for how the medical answer was derived from the context.")
  // The `answer` field contains the final, medical answer.
  answer string @description("The final, concise medical answer to the question based on the provided context.")
}

// The MetaMedQAAnswer is generated based on the context and question
function GenerateMetaMedQAAnswer(context: string, question: string) -> MetaMedQAAnswer {
    // The `GroqFallbackClient` tries a sequence of providers (Groq, then Cerebras, then SambaNova)
    client GroqFallbackClient

    // The `{{ ctx.output_format }}` uses the schema from the `MetaMedQAAnswer` class
    prompt #"
    {{ _.role("system") }}
    You are an expert medical educator and diagnostician. Use the retrieved context
    to analyze the clinical case and identify the most likely answer from the options provided.

    CRITICAL INSTRUCTIONS:
    1. Base your reasoning SOLELY on the provided medical context
    2. For medical questions, prioritize the most specific and pathophysiologically relevant information
    3. Explain the clinical reasoning that connects the patient's presentation to the correct answer
    4. If multiple options seem plausible, identify which one best explains ALL the clinical findings
    5. Address why key distractors are less likely based on the clinical context

    Provide your answer in this exact format:
    **Answer: [LETTER]**
    **Explanation:** [Your clinical reasoning]

    Your response MUST be in the following format:
    {{ ctx.output_format }}

    {{ _.role("user") }}
    CLINICAL CONTEXT:
    {{ context }}

    QUESTION:
    {{ question }}

    Based on the clinical context above, analyze this case and provide your answer.
"#
}

test TestMetaMedQAQuestion {
  functions [GenerateMetaMedQAAnswer]
  args {
      context  #"A 3-month-old infant presenting with poor feeding, respiratory distress, and a holosystolic murmur at the lower left sternal border is highly suggestive of a congenital heart defect. The clinical presentation, including failure to thrive, feeding difficulties, and a characteristic murmur, is classic for a ventricular septal defect (VSD), which is the most common congenital heart defect associated with 22q11.2 deletion syndrome (DiGeorge syndrome).

      22q11.2 deletion syndrome is a genetic disorder caused by a microdeletion on chromosome 22. Key clinical features include:
      - Congenital heart defects (present in 75-80% of cases)
      - Characteristic facial features
      - Thymic hypoplasia
      - Hypocalcemia due to parathyroid hypoplasia
      - Cleft palate or velopharyngeal insufficiency
      - Developmental delay

      The holosystolic murmur at the lower left sternal border is most consistent with a VSD, particularly a perimembranous VSD, which is commonly seen in 22q11.2 deletion syndrome. The murmur is typically high-pitched and harsh, radiating throughout the precordium. The absence of cyanosis (no history of turning blue) suggests the defect is not causing right-to-left shunting (acyanotic heart disease).

      The other options are less likely:
      - Deletion of genes on chromosome 7 is associated with Williams syndrome, which typically presents with supravalvular aortic stenosis and elfin facies.
      - Lithium exposure in utero is associated with Ebstein's anomaly of the tricuspid valve, not VSD.
      - Maternal alcohol consumption is associated with fetal alcohol syndrome, which has different clinical features including growth retardation, facial dysmorphisms, and neurodevelopmental issues.

      The diagnosis of 22q11.2 deletion syndrome can be confirmed with chromosomal microarray analysis or FISH testing for the 22q11.2 deletion. Early diagnosis is important for managing potential complications such as hypocalcemia, immune deficiency, and planning for surgical correction of cardiac defects.

      Management typically involves a multidisciplinary team including pediatric cardiology, genetics, immunology, and developmental specialists. Surgical repair of the VSD is usually recommended if there is evidence of heart failure, failure to thrive, or significant left-to-right shunting."#

      question "A 3-month-old infant is brought to her pediatrician because she coughs and seems to have difficulty breathing while feeding. In addition, she seems to have less energy compared to other babies and appears listless throughout the day. She was born by cesarean section to a G1P1 woman with no prior medical history and had a normal APGAR score at birth. Her parents say that she has never been observed to turn blue. Physical exam reveals a high-pitched holosystolic murmur that is best heard at the lower left sternal border. The most likely cause of this patient's symptoms is associated with which of the following abnormalities?\n\nA: 22q11 deletion\nB: Deletion of genes on chromosome 7\nC: Lithium exposure in utero\nD: Maternal alcohol consumption\nE: None of the above\nF: I don't know or cannot answer"
  }
}