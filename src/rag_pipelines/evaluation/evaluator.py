from typing import Any

import weave
from weave import Dataset, Evaluation, Model, Scorer


class Evaluator(Model):
    """Evaluate a model on a dataset using a list of scorers.

    Attributes:
        evaluation_name (str): The name of the evaluation run.
        evaluation_dataset (Dataset): The dataset used for evaluation.
        evaluation_scorers (list[Scorer]): A list of scorer objects used to evaluate the pipeline.
        pipeline (Model): The pipeline (model) to be evaluated.
    """

    def __init__(
        self,
        evaluation_name: str,
        evaluation_dataset: Dataset,
        evaluation_scorers: list[Scorer],
        pipeline: Model,
    ):
        """Initialize the Evaluator instance with the specified evaluation parameters.

        Args:
            evaluation_name (str): A unique identifier for the evaluation run.
            evaluation_dataset (Dataset): A `Dataset` object representing the data for evaluation.
            evaluation_scorers (list[Scorer]): A list of `Scorer` objects that calculate various metrics.
            pipeline (Model): The model or pipeline to evaluate.
        """
        self.evaluation_name = evaluation_name
        self.evaluation_dataset = evaluation_dataset
        self.evaluation_scorers = evaluation_scorers
        self.pipeline = pipeline

    @weave.op()
    def evaluate(self) -> dict[str, Any]:
        """Perform evaluation of the pipeline using the specified dataset and scorers.

        This method creates an `Evaluation` object, executes the evaluation process, and
        returns the results as a dictionary.

        Returns:
            dict[str, Any]: A dictionary where keys are scorer names and values are evaluation results.
        """
        evaluation = Evaluation(
            name=self.evaluation_name,
            dataset=self.evaluation_dataset,
            scorers=self.evaluation_scorers,
        )

        try:
            evaluation_results = evaluation.evaluate(self.pipeline)
        except Exception as exception:
            msg = f"Evaluation run failed: {exception}"
            raise RuntimeError(msg) from exception

        return evaluation_results
