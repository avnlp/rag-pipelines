from typing import Optional, Union

import weave
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from weave import Scorer


class FaithfulnessScorer(Scorer):
    """Evaluate the faithfulness of LLM generated outputs.

    This scorer uses DeepEval's `Faithfulness` Metric.

    The faithfulness metric measures the quality of your LLM generation by evaluating whether the `actual_output` factually aligns with the contents of your `retrieval_context`.

    Attributes:
        threshold (float): The minimum score required to pass the faithfulness check, defaults to 0.5.
        model (str): The LLM model used for evaluation, defaults to "gpt-4".
        include_reason (bool): Whether to include an explanation for the assigned score, defaults to True.
        strict_mode (bool): When True, enforces binary scoring (1 for perfect alignment, 0 otherwise).
            Overrides the threshold to 1. Defaults to False.
        async_mode (bool): Whether to perform scoring asynchronously, defaults to True.
        verbose (bool): Whether to display intermediate steps during metric computation, defaults to False.
        truths_extraction_limit (Optional[int]): Limits the number of key facts to extract from the retrieval
            context for evaluation, ordered by importance. Defaults to None.
        metric (FaithfulnessMetric): An instance of DeepEval's `FaithfulnessMetric` for scoring.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str = "gpt-4",
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose: bool = False,
        truths_extraction_limit: Optional[int] = None,
    ):
        """Initialize the Faithfulness Scorer with DeepEval's Faithfulness Metric.

        Args:
            threshold (float): The minimum score required to pass the faithfulness check, defaults to 0.5.
            model (str): The LLM model used for evaluation, defaults to "gpt-4".
            include_reason (bool): Whether to include an explanation for the assigned score, defaults to True.
            strict_mode (bool): Enforces binary scoring (1 for perfect alignment, 0 otherwise).
                Overrides the threshold to 1. Defaults to False.
            async_mode (bool): Whether to perform scoring asynchronously, defaults to True.
            verbose (bool): Whether to display intermediate steps during metric computation, defaults to False.
            truths_extraction_limit (Optional[int]): Limits the number of key facts to extract from the retrieval
                context for evaluation, ordered by importance. Defaults to None.
        """
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose = verbose
        self.truths_extraction_limit = truths_extraction_limit

        self.metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
            verbose=self.verbose,
        )

    @weave.op
    def score(
        self,
        input: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        retrieval_context: Optional[list[str]] = None,
        context: Optional[list[str]] = None,
    ) -> dict[str, Union[str, float]]:
        """Evaluate the faithfulness of an LLM generated response.

        Faithfulness is calculated as:

        Faithfulness = (Number of Truthful Claims) / (Total Number of Claims).

        The Faithfulness Metric evaluates all claims in the `actual_output` and checks
        whether they are truthful based on the facts in the `retrieval_context`. Claims
        are marked truthful if they align with or do not contradict any facts in the context.

        Args:
            input (str): The input query or prompt that triggered the output.
            actual_output (str): The LLM generated response to evaluate.
            expected_output (Optional[str]): The expected or reference output, defaults to None.
            retrieval_context (Optional[list[str]]): The context containing factual information to compare against.
            context (Optional[list[str]]): Additional context for the evaluation, defaults to None.

        Returns:
            dict[str, Union[str, float]]: A dictionary containing:
                - "score" (float): The computed faithfulness score.
                - "reason" (str): A detailed explanation for the assigned score.
        """
        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
            context=context,
        )

        result: dict[str, Union[str, float]] = {}

        self.metric.measure(test_case)
        result = {"score": self.metric.score, "reason": self.metric.reason}

        return result
