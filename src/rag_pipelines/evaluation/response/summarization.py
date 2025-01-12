from typing import Optional, Union

import weave
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from weave import Scorer


class SummarizationScorer(Scorer):
    """Summarization Scorer.

    This scorer uses DeepEval's `Summarization` Metric to assess how well the generated output
    aligns with the reference context.

    The summarization metric uses LLMs to determine whether the LLM application is generating factually correct
    summaries while including the neccessary details from the original text.

    Attributes:
        threshold (float): Minimum passing threshold, defaults to 0.5.
        model (str): LLM model for scoring, defaults to "gpt-4".
        assessment_questions: a list of close-ended questions that can be answered with either a 'yes' or a 'no'.
        These are questions you want your summary to be able to ideally answer,
        and is especially helpful if you already know what a good summary for your use case looks like. If
        include_reason (bool): Include reason for the evaluation score, defaults to True.
        strict_mode (bool): Enforces binary metric scoring (1 or 0), defaults to False.
        async_mode (bool): Use asynchronous scoring, defaults to True.
        verbose (bool): Print intermediate steps used for scoring, defaults to False.
        truths_extraction_limit (Optional[int]): Maximum number of factual truths to extract
            from the retrieval_context. Defaults to None.
        metric (SummarizationMetric): An instance of DeepEval's `SummarizationMetric` for scoring.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str = "gpt-4",
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose: bool = False,
        assessment_questions: Optional[list[str]] = None,
        n: Optional[int] = 5,
        truths_extraction_limit: Optional[int] = None,
    ):
        """Initialize the Summarization Scorer with DeepEval's Summarization Metric.

        Args:
            threshold (float): Minimum passing threshold, defaults to 0.5.
            model (str): LLM model for scoring, defaults to "gpt-4".
            include_reason (bool): Include reason for the evaluation score, defaults to True.
            strict_mode (bool): Enforces binary metric scoring (1 or 0), defaults to False.
            async_mode (bool): Use asynchronous scoring, defaults to True.
            verbose (bool): Print intermediate steps used for scoring, defaults to False.
            assessment_questions (Optional[list[str]]): a list of close-ended questions that can be answered with either
                a 'yes' or a 'no'. These are questions you want your summary to be able to ideally answer, and is
                especially helpful if you already know what a good summary for your use case looks like. If
                `assessment_questions` is not provided, the metric will generate a set of `assessment_questions` at
                evaluation time.
            n (Optional[int]): The number of assessment questions to generate when `assessment_questions` is not
                provided. Defaults to 5.
            truths_extraction_limit (Optional[int]): Maximum number of factual truths to extract
                from the retrieval_context. Defaults to None.
        """
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose = verbose
        self.assessment_questions = assessment_questions
        self.n = n
        self.truths_extraction_limit = truths_extraction_limit

        self.metric = SummarizationMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
            verbose=self.verbose,
            assessment_questions=self.assessment_questions,
            n=self.n,
            truths_extraction_limit=self.truths_extraction_limit,
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
        """Evaluate the quality of summarization of an LLM generated response.

        The Summarization score is calculated according to the following equation:

        Summarization = min(Alignment Score, Coverage Score)

        where,
        - Alignment Score: determines whether the summary contains hallucinated or contradictory information to the original text.
        - Coverage Score: determines whether the summary contains the neccessary information from the original text.


        While the Alignment Score is similar to that of the Hallucination Score, the Coverage Score is first calculated
        by generating n closed-ended questions that can only be answered with either a 'yes or a 'no', before
        calculating the ratio of which the original text and summary yields the same answer.

        Args:
            input (str): The input query or prompt that triggered the output.
            actual_output (str): The LLM generated response to evaluate.
            expected_output (Optional[str]): The expected or reference output, defaults to None.
            retrieval_context (Optional[list[str]]): The context containing factual information to compare against.
            context (Optional[list[str]]): Additional context for the evaluation, defaults to None.

        Returns:
            dict[str, Union[str, float]]: A dictionary containing:
                - "score" (float): The computed summarization score.
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
