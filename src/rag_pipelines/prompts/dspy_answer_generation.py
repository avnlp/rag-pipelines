from dspy import InputField, OutputField, Signature


class GenerateAnswer(Signature):
    """DSPy Signature for Answer Generation.

    This signature specifies the input and output fields for a module responsible for generating
    answers to questions. It takes in a context, which may include relevant facts or background information,
    and a question. Based on these inputs, the module produces a short and precise answer.

    Attributes:
        context (InputField): An input field containing background information or relevant facts that aid in answering the question.
        question (InputField): An input field representing the question for which an answer is to be generated.
        answer (OutputField): An output field that will contain the generated answer, designed to be short and precise.
    """

    # Input field for context information that may include relevant facts.
    context = InputField(desc="may contain relevant facts")
    # Input field for the question to be answered.
    question = InputField()
    # Output field for the generated short and precise answer.
    answer = OutputField(desc="short and precise answer")
