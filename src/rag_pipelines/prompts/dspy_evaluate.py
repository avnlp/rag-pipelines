from dspy import InputField, OutputField, Signature


class Assess(Signature):
    """Assessment Signature for Evaluating Answer Quality.

    This signature defines the input and output fields required for an evaluation module
    that assesses the quality of an answer. The module considers the context in which the
    answer is provided, the evaluation criteria (question), and the answer itself, and outputs
    a rating on a scale from 1 to 5.

    Attributes:
        context (InputField): The background or supporting information used to answer the question.
        assessed_question (InputField): The evaluation criterion or the specific question that guides the assessment.
        assessed_answer (InputField): The answer provided that will be evaluated.
        assessment_answer (OutputField): The resulting quality rating, typically a number between 1 and 5.
    """

    # Input field containing the context that supports the answer.
    context = InputField(desc="The context for answering the question.")
    # Input field specifying the evaluation criterion or the question to be assessed.
    assessed_question = InputField(desc="The evaluation criterion.")
    # Input field holding the answer that is being evaluated.
    assessed_answer = InputField(desc="The answer to the question.")
    # Output field that will hold the final assessment score, which is a rating between 1 and 5.
    assessment_answer = OutputField(desc="A rating between 1 and 5.")
