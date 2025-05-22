"""
ROUGE metric implementation for NMT evaluation_metrics.

This module provides ROUGE-L F1 score calculation using rouge_score library.
ROUGE-L measures the longest common subsequence between hypothesis and reference.
Can be run standalone from command line.
"""

from typing import List
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
import os

# Add parent directory to path to import base_metric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_metric import BaseMetric


class ROUGEMetric(BaseMetric):
    """ROUGE-L metric implementation using rouge_score library."""

    def __init__(self):
        super().__init__("ROUGE")
        self._rouge_scorer = None

    def get_dependencies(self) -> List[str]:
        """Return required packages for ROUGE metric."""
        return ["rouge_score"]

    def _initialize_rouge(self):
        """Initialize ROUGE scorer lazily."""
        if self._rouge_scorer is None:
            try:
                from rouge_score.rouge_scorer import RougeScorer
                self._rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import rouge_score. Install with: pip install rouge_score\n"
                    f"Error: {e}"
                )

    def calculate_scores(
            self,
            hypotheses: List[str],
            references: List[str],
            sources: List[str] = None
    ) -> List[float]:
        """
        Calculate ROUGE-L F1 scores for hypothesis-reference pairs.

        ROUGE-L measures the longest common subsequence between hypothesis
        and reference, providing a recall-oriented evaluation_metrics metric.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (ignored for ROUGE)

        Returns:
            List of ROUGE-L F1 scores (0-1 scale)
        """
        self.validate_inputs(hypotheses, references, sources)
        self._initialize_rouge()

        scores = []
        for hyp, ref in zip(hypotheses, references):
            try:
                score = self._rouge_scorer.score(ref, hyp)['rougeL'].fmeasure
                scores.append(score)
            except Exception as e:
                print(f"Warning: ROUGE calculation failed for pair, using 0.0: {e}")
                scores.append(0.0)

        return scores


def main():
    """Main function for CLI usage."""
    metric = ROUGEMetric()
    return metric.run_from_cli()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
