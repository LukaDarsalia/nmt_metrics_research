"""
BLEU metric implementation for NMT processors.

This module provides BLEU score calculation using sacrebleu library.
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


class BLEUMetric(BaseMetric):
    """BLEU metric implementation using sacrebleu."""

    def __init__(self):
        super().__init__("BLEU")
        self._bleu = None

    def get_dependencies(self) -> List[str]:
        """Return required packages for BLEU metric."""
        return ["sacrebleu"]

    def _initialize_bleu(self):
        """Initialize BLEU scorer lazily."""
        if self._bleu is None:
            try:
                from sacrebleu.metrics import BLEU
                self._bleu = BLEU(effective_order=True)
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import sacrebleu. Install with: pip install sacrebleu\n"
                    f"Error: {e}"
                )

    def calculate_scores(
            self,
            hypotheses: List[str],
            references: List[str],
            sources: List[str] = None
    ) -> List[float]:
        """
        Calculate BLEU scores for hypothesis-reference pairs.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (ignored for BLEU)

        Returns:
            List of BLEU scores (0-100 scale)
        """
        self.validate_inputs(hypotheses, references, sources)
        self._initialize_bleu()

        scores = []
        for hyp, ref in zip(hypotheses, references):
            try:
                score = self._bleu.sentence_score(hyp, [ref]).score
                scores.append(score)
            except Exception as e:
                print(f"Warning: BLEU calculation failed for pair, using 0.0: {e}")
                scores.append(0.0)

        return scores


def main():
    """Main function for CLI usage."""
    metric = BLEUMetric()
    return metric.run_from_cli()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
