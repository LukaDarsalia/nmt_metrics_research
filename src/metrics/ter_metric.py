"""
TER (Translation Edit Rate) metric implementation for NMT processors.

This module provides TER score calculation using HuggingFace evaluate library.
TER measures the number of edits required to transform hypothesis into reference.
Can be run standalone from command line.
"""

from typing import List
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
import os

# Add parent directory to path to import base_metric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.base_metric import BaseMetric


class TERMetric(BaseMetric):
    """TER metric implementation using HuggingFace evaluate."""

    def __init__(self):
        super().__init__("TER")
        self._ter_metric = None

    def get_dependencies(self) -> List[str]:
        """Return required packages for TER metric."""
        return ["evaluate", "datasets"]

    def _initialize_ter(self):
        """Initialize TER metric lazily."""
        if self._ter_metric is None:
            try:
                import evaluate
                self._ter_metric = evaluate.load("ter")
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import evaluate. Install with: pip install evaluate datasets\n"
                    f"Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load TER metric: {e}")

    def calculate_scores(
            self,
            hypotheses: List[str],
            references: List[str],
            sources: List[str] = None
    ) -> List[float]:
        """
        Calculate TER scores for hypothesis-reference pairs.

        TER measures the number of edits required to transform the hypothesis
        into the reference. Lower TER scores indicate better translations.
        This method returns 1 - (TER/100) so higher scores are better.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (ignored for TER)

        Returns:
            List of inverted TER scores (higher is better)
        """
        self.validate_inputs(hypotheses, references, sources)
        self._initialize_ter()

        scores = []
        for hyp, ref in zip(hypotheses, references):
            try:
                ter_score = self._ter_metric.compute(
                    predictions=[hyp],
                    references=[[ref]]
                )["score"]

                # Invert TER so higher scores are better (like other metrics)
                # TER ranges from 0 to 100+, we convert to 0-1 scale where 1 is best
                inverted_score = max(0.0, 1 - (ter_score / 100))
                scores.append(inverted_score)

            except Exception as e:
                print(f"Warning: TER calculation failed for pair, using 0.0: {e}")
                scores.append(0.0)

        return scores


def main():
    """Main function for CLI usage."""
    metric = TERMetric()
    return metric.run_from_cli()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
