"""
CHRF++ metric implementation for NMT processors.

This module provides CHRF++ score calculation using sacrebleu library.
CHRF++ is particularly suitable for morphologically rich languages like Georgian.
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


class CHRFMetric(BaseMetric):
    """CHRF++ metric implementation using sacrebleu."""

    def __init__(self):
        super().__init__("CHRF++")
        self._chrf = None

    def get_dependencies(self) -> List[str]:
        """Return required packages for CHRF++ metric."""
        return ["sacrebleu"]

    def _initialize_chrf(self):
        """Initialize CHRF++ scorer lazily."""
        if self._chrf is None:
            try:
                from sacrebleu.metrics import CHRF
                # CHRF++ uses word_order=2
                self._chrf = CHRF(word_order=2)
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
        Calculate CHRF++ scores for hypothesis-reference pairs.

        CHRF++ considers character-level n-grams and is particularly
        suitable for morphologically rich languages like Georgian.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (ignored for CHRF++)

        Returns:
            List of CHRF++ scores (0-100 scale)
        """
        self.validate_inputs(hypotheses, references, sources)
        self._initialize_chrf()

        scores = []
        for hyp, ref in zip(hypotheses, references):
            try:
                score = self._chrf.sentence_score(hyp, [ref]).score
                scores.append(score)
            except Exception as e:
                print(f"Warning: CHRF++ calculation failed for pair, using 0.0: {e}")
                scores.append(0.0)

        return scores


def main():
    """Main function for CLI usage."""
    metric = CHRFMetric()
    return metric.run_from_cli()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
