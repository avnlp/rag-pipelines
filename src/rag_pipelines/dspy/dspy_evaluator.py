from dspy import ChainOfThought

from rag_pipelines.prompts.dspy_evaluate import Assess


class DSPyEvaluator:
    """Evaluator for DSPy RAG using an LLM-based metric.

    This class provides a static method to evaluate the quality of a retrieved answer using
    LLM-generated assessments. The evaluation is based on several criteria such as detail,
    faithfulness, and overall quality. Each criterion is evaluated via a chain-of-thought prompt,
    and the resulting scores are combined to produce a final metric.
    """

    @staticmethod
    def llm_metric(gold, pred, trace=None):
        """Compute an LLM-based evaluation metric for a given prediction.

        This method evaluates the predicted answer based on three criteria:
        1. Detail: Whether the answer is sufficiently detailed.
        2. Faithfulness: Whether the answer is grounded in the provided context.
        3. Overall: A general assessment of how well the answer addresses the question.

        Each criterion is assessed using a ChainOfThought process with the Assess signature,
        and the resulting scores are combined into a final metric.

        Args:
            gold: An object containing the gold standard or reference question (expects a 'question' attribute).
            pred: An object containing the predicted answer and its context (expects 'answer' and 'context' attributes).
            trace: Optional trace parameter for debugging or logging purposes (not used in this implementation).

        Returns:
            float: The final evaluation metric as a normalized score (typically between 0 and 1).

        Process:
            - Extract the predicted answer, context, and gold question.
            - Print the test question and predicted answer for debugging.
            - Define evaluation questions for each criterion (detail, faithfulness, overall).
            - Use ChainOfThought with the Assess signature to evaluate each criterion.
            - Print the intermediate scores for faithfulness, detail, and overall assessment.
            - Calculate the total score by combining the individual scores with weighting.
            - Normalize the score and return it.
        """
        # Extract predicted answer and context from the prediction object
        predicted_answer = pred.answer
        context = pred.context
        question = gold.question

        # Debugging: Print the test question and the predicted answer.
        print(f"Test Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")

        # Define evaluation questions for each assessment criterion.
        detail_q = "Is the assessed answer detailed?"
        faithful_q = "Is the assessed text grounded in the context?"
        overall_q = f"Rate how well this answer answers the question `{question}`"

        # Evaluate 'detail' criterion: Uses a placeholder context ("N/A") since detail may not depend on context.
        detail = ChainOfThought(Assess)(context="N/A", assessed_question=detail_q, assessed_answer=predicted_answer)

        # Evaluate 'faithfulness' criterion: Uses the provided context to check grounding of the answer.
        faithful = ChainOfThought(Assess)(
            context=context, assessed_question=faithful_q, assessed_answer=predicted_answer
        )

        # Evaluate 'overall' quality: Considers how well the answer addresses the question, using the context.
        overall = ChainOfThought(Assess)(context=context, assessed_question=overall_q, assessed_answer=predicted_answer)

        # Debugging: Print the individual assessment scores.
        print(f"Faithfulness: {faithful.assessment_answer}")
        print(f"Detail: {detail.assessment_answer}")
        print(f"Overall: {overall.assessment_answer}")

        # Combine the assessment scores with a weighting scheme:
        # Faithfulness is weighted double compared to the others.
        total = (
            float(detail.assessment_answer) + float(faithful.assessment_answer) * 2 + float(overall.assessment_answer)
        )

        # Normalize the total score by dividing by the maximum possible weighted score (5.0).
        return total / 5.0
