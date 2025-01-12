from typing import Optional, Union

import weave
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from weave import Scorer


class HallucinationScorer(Scorer):
    """Evaluate the factual alignment of the generated output with the provided context.

    This scorer uses DeepEval's `Hallucination` Metric to assess how well the generated output
    aligns with the reference context.

    The Hallucination metric determines whether your LLM generates factually correct information by comparing the `actual_output` to the provided `context`.

    Attributes:
        threshold (float): A float representing the minimum passing threshold, defaults to 0.5.
        model (str): The LLM model to use for scoring, defaults to "gpt-4".
        include_reason (bool): Whether to include a reason for the evaluation score, defaults to True.
        strict_mode (bool): A boolean which when set to True, enforces a binary metric score: 1 for perfection,
        0 otherwise. It also overrides the current threshold and sets it to 1. Defaults to False.
        async_mode (bool): Whether to use asynchronous scoring, defaults to True.
        verbose (bool): Whether to print the intermediate steps used to calculate said metric to the console, defaults
        to False.
        metric (HallucinationMetric): The DeepEval HallucinationMetric.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str = "gpt-4",
        include_reason: bool = True,
        strict_mode: bool = True,
        async_mode: bool = True,
        verbose: bool = False,
    ):
        """Initialize the Hallucination scorer using DeepEval's Hallucination Metric.

        Args:
            threshold (float): A float representing the minimum passing threshold, defaults to 0.5.
            model (str): The LLM model to use for scoring, defaults to "gpt-4".
            include_reason (bool): Whether to include a reason for the evaluation score, defaults to True.
            strict_mode (bool): A boolean which when set to True, enforces a binary metric score: 1 for perfection,
            0 otherwise. It also overrides the current threshold and sets it to 1. Defaults to False.
            async_mode (bool): Whether to use asynchronous scoring, defaults to True.
            verbose (bool): Whether to print the intermediate steps used to calculate said metric to the console, defaults
            to False.
        """
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose = verbose

        self.metric = HallucinationMetric(
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
        """Evaluate the factual alignment of the generated output with the provided context.

        The Hallucination Score is calculated according to the following equation:

        Hallucination = Number of Contradicted Contexts / Total Number of Contexts

        The Hallucination Score uses an LLM to determine, for each context in `contexts`, whether there are any contradictions to the `actual_output`.

        Although extremely similar to the Faithfulness Scorer, the Hallucination Score is calculated differently since it uses `contexts` as the source of truth instead. Since `contexts` is the ideal segment of your knowledge base relevant to a specific input, the degree of hallucination can be measured by the degree of which the `contexts` is disagreed upon.

        Args:
            input (str): The input query or prompt that triggered the output.
            actual_output (str): The LLM generated response to evaluate.
            expected_output (Optional[str]): The expected or reference output, defaults to None.
            retrieval_context (Optional[list[str]]): The context containing factual information to compare against.
            context (Optional[list[str]]): Additional context for the evaluation, defaults to None.

        Returns:
            dict[str, Union[str, float]]: A dictionary containing:
                - "score" (float): The computed hallucination score.
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
