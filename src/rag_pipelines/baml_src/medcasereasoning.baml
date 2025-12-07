// MedCaseReasoning

// The `MedCaseReasoningAnswer` class defines the structured output we expect from the LLM.
class MedCaseReasoningAnswer {
  // The `chain_of_thought` field captures the LLM's reasoning process.
  chain_of_thought string @description("Step-by-step reasoning for how the medical diagnosis was derived from the context.")
  // The `summary` field contains the final, summarized answer.
  diagnosis string @description("The final, concise medical diagnosis answer to the case prompt.")
}

// The MedCaseReasoningAnswer is generated based on the context and question
function GenerateMedCaseReasoningAnswer(context: string, question: string) -> MedCaseReasoningAnswer {
    // The `GroqFallbackClient` tries a sequence of providers (Groq, then Cerebras, then SambaNova)
    client GroqFallbackClient

    // The `{{ ctx.output_format }}` uses the schema from the `Answer` class
    prompt #"
    {{ _.role("system") }}
    You are an expert medical diagnostician. Use the retrieved context
    to answer the medical case question precisely. Provide a detailed and accurate
    diagnosis based on the provided context.

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

test TestMedCaseReasoningQuestion {
 functions [GenerateMedCaseReasoningAnswer]
 args {
    context " Background A traumatic neuroma is a hyperplastic lesion caused by trauma or surgery that involves the peripheral nerves and is not considered to be a true neoplasm . It may occur in any part of the body, including the head, neck, gallbladder and thigh . The clinical features of a traumatic neuroma include the formation of a solitary nodule less than 2 cm in diameter, neuralgic pain, tenderness, paresthesias and increased pain on palpation over the lesion [ 2 , 3 ]. The recommended treatment of a traumatic neuroma is simple excision rather than nerve resection or alcohol blocks . In the oral region, a traumatic neuroma is a rare disorder that occurs most commonly at the mental foramen, lower lip, tongue and intra-osseous areas [ 4 , 5 ]. It is extremely rare in the palate. There have been no previously published reports of a diffuse traumatic neuroma. Here we describe the treatment outcomes of an unusual diffuse traumatic neuroma occurring in the palate. Case presentation A 30-year-old Japanese man was referred to our clinic complaining of pain, tenderness, and swelling on the left side of his palate for the past several months. Our initial clinical examination found that his left palatine mucosa was significantly swollen compared with his right side. The swelling was diffuse, and its borders were unclear (Fig. 1a ). The swelling was especially pronounced in the left molar region of his palate, and his pain was exacerbated with the application of direct pressure to the lesion. The swelling exhibited increased signal intensity on T2-weighted magnetic resonance images (MRI, Fig. 1b ). All of his left maxillary teeth were healthy, and no specific abnormalities in his left maxilla and maxillary sinus were observed on panoramic X-ray and computed tomography (CT) images. His medical and familial histories were unremarkable, and he was not aware of any history of trauma or inflammation of his head or neck. We administered antibiotics orally (cefcapene pivoxil hydrochloride hydrate 100 mg tablet every 8 hours) and non-steroidal anti-inflammatory drugs (loxoprofen sodium hydrate 60 mg tablet every 8 hours) for 7 days because we suspected that his symptoms were due to inflammation secondary to an infection. However, his symptoms did not improve. Fig. 1 a Photograph showing an obvious swelling of the left palatine mucosa ( arrows ). b A T2-weighted magnetic resonance image showing the lesion as an area of high-signal intensity ( arrow ) Based on his clinical and radiographic examinations, our initial diagnosis was a soft tissue tumor. An incisional biopsy was performed and histopathologic examination of the lesion revealed haphazard nodes and inflammatory cells in a fibrous stroma (Fig. 2a, b ). Immunohistochemical analysis revealed significant staining for the neural marker S-100 in the bundles within the node (Fig. 2c ). Factor VIII staining was positive in the fibrous stroma, but not in the bundles (Fig. 2d ). These findings led to the diagnosis of a traumatic neuroma. The patient underwent resection of the tumor with a 5-mm margin using an electric scalpel under general anesthesia. Although the border of the mass was unclear and diffuse, the extent of the tumor could be determined based on the MRI images. As the tumor was conglutinated with a part of his palatine bone, we saucerized the bone surface including the overlying mucosa and the periosteum. His left greater palatine nerve was resected as the possible origin of the tumor. The open surgical wound was covered with a collagen-based artificial dermis (Terudermis, Olympus Terumo Biomaterials Corp, Tokyo, Japan) and a surgical splint. The size of the specimen was approximately 6×3 cm (Fig. 3 ). The pathologic findings of the surgical specimen were the same as those of the incisional biopsy specimen, and the patient’s greater palatine nerve exhibited no pathologic changes. The tumor was not entirely encapsulated and some tumor cells were observed within the surgical margins. We decided to continue strict clinical follow-up without additional surgery because the traumatic neuroma is not a true neoplasm, and his pain subsided immediately after the surgery. No clinical evidence of a recurrence has been observed in the 3 years since the surgery. Fig. 2 a Many haphazard nodes can be seen within the fibrous stroma (hematoxylin and eosin stain ×100) and b hematoxylin and eosin stain × 200. c A strong positive signal for S-100 of the bundles within the nodes is evident (S-100 protein ×100). d Factor VIII was positive in the microvessels within the fibrous stroma, but did not stain the nerve bundles (Factor VIII ×200) Fig. 3 The lesion excised from the left side of the palatine mucosa was approximately 6×3 cm Discussion Traumatic neuromas, also called amputation neuromas, are not true neoplasms but reactive proliferations of neural tissue that occur after transection or damage to a nerve bundle. Jones and Franklin  reported that the frequency of traumatic neuromas was 0.34 % in the oral region. The most common sites for a traumatic neuroma in the head and neck are the inferior alveolar nerve."
    question "A 30-year-old man presented with several months of pain, tenderness, and swelling on the left side of his palate. On examination, the left palatine mucosa was diffusely swollen—most pronounced in the molar region—with poorly defined borders and increased pain on palpation. T2-weighted MRI demonstrated a high-signal-intensity lesion in that area. Panoramic radiography and CT scans of the maxilla and sinuses were unremarkable, and all left maxillary teeth appeared healthy. His medical and family histories were unremarkable, and he denied any history of trauma or inflammation in the head or neck. Empiric treatment with oral antibiotics and nonsteroidal anti-inflammatory drugs for 7 days produced no improvement. Based on the clinical and radiographic findings, a soft-tissue tumor was suspected, and an incisional biopsy was performed."
  }
}