"""
COMET metric implementation for NMT processors.

This module provides COMET score calculation using the comet library.
COMET uses neural-based processors and requires source, hypothesis, and reference texts.
Can be run standalone from command line.
"""

from typing import List, Optional
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
import os
import argparse

# Add parent directory to path to import base_metric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.base_metric import BaseMetric


class COMETMetric(BaseMetric):
    """COMET metric implementation using comet library."""

    def __init__(self, batch_size: int = 8, sample_size: Optional[int] = None):
        super().__init__("COMET")
        self._comet_model = None
        self.batch_size = batch_size
        self.sample_size = sample_size

    def get_dependencies(self) -> List[str]:
        """Return required packages for COMET metric."""
        return ["unbabel-comet", "torch", "transformers"]

    def _initialize_comet(self):
        """Initialize COMET model lazily with robust error handling."""
        if self._comet_model is None:
            try:
                # Import comet components
                from comet import download_model, load_from_checkpoint

                print("Loading COMET model... This may take a while on first run.")

                # Download and load the model
                model_path = download_model("Unbabel/wmt22-comet-da")
                self._comet_model = load_from_checkpoint(model_path)

                print("✅ COMET model loaded successfully")

            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import comet. Install with: pip install unbabel-comet\n"
                    f"You may also need: pip install torch transformers\n"
                    f"Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load COMET model: {e}")

    def calculate_scores(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: List[str] = None
    ) -> List[float]:
        """
        Calculate COMET scores using neural-based processors.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (required for COMET)

        Returns:
            List of COMET scores
        """
        if sources is None:
            raise ValueError("COMET requires source texts. Please provide sources parameter.")

        self.validate_inputs(hypotheses, references, sources)
        self._initialize_comet()

        # Prepare data for COMET
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]

        # Limit sample size if specified
        if self.sample_size is not None:
            if self.sample_size < len(data):
                print(f"Evaluating COMET on {self.sample_size} samples (out of {len(data)})")
                data = data[:self.sample_size]

        try:
            # Run COMET prediction
            model_output = self._comet_model.predict(data, batch_size=self.batch_size, progress_bar=True)

            # Extract scores (handle different output formats)
            if hasattr(model_output, 'scores'):
                scores = model_output.scores
            elif isinstance(model_output, list):
                scores = model_output
            else:
                scores = [float(model_output)] * len(data)

            # Convert to list if needed
            if not isinstance(scores, list):
                scores = scores.tolist() if hasattr(scores, 'tolist') else [float(scores)]

            # Extend to full length if sample_size was used
            if self.sample_size is not None and len(scores) < len(hypotheses):
                avg_score = sum(scores) / len(scores) if scores else 0.0
                scores.extend([avg_score] * (len(hypotheses) - len(scores)))
                print(f"Extended {len(data)} COMET scores to {len(hypotheses)} using average: {avg_score:.4f}")

            return scores

        except Exception as e:
            print(f"Warning: COMET calculation failed: {e}")
            # Return zeros if COMET fails
            return [0.0] * len(hypotheses)

    def run_from_cli(self) -> None:
        """
        Run the metric from command line interface.
        Overridden to add COMET-specific parameters.
        """
        parser = argparse.ArgumentParser(
            description=f'Calculate {self.metric_name} scores for NMT processors'
        )
        parser.add_argument(
            '--input', '-i',
            required=True,
            help='Path to input CSV file containing processors data'
        )
        parser.add_argument(
            '--output', '-o',
            required=True,
            help='Path to output JSON file for metric scores'
        )
        parser.add_argument(
            '--batch-size', '-b',
            type=int,
            default=8,
            help='Batch size for COMET model inference (default: 8)'
        )
        parser.add_argument(
            '--sample-size', '-s',
            type=int,
            default=None,
            help='Limit number of samples to evaluate (default: all)'
        )

        args = parser.parse_args()

        # Update batch size and sample size
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size

        try:
            print(f"🚀 Running {self.metric_name} metric processors...")
            print(f"  Batch size: {self.batch_size}")
            if self.sample_size:
                print(f"  Sample size: {self.sample_size}")

            # Load data
            print(f"Loading data from {args.input}...")
            data = self.load_data(args.input)
            print(f"Loaded {len(data)} processors pairs")

            # Extract text components
            hypotheses = data['targetText'].tolist()
            references = data['referenceText'].tolist()
            sources = data['sourceText'].tolist()

            # Calculate scores
            print(f"Calculating {self.metric_name} scores...")
            scores = self.calculate_scores(hypotheses, references, sources)

            # Save results
            self.save_scores(scores, args.output)

            # Print summary statistics
            import numpy as np
            print(f"\n{self.metric_name} Score Statistics:")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")

        except Exception as e:
            print(f"❌ Error running {self.metric_name}: {e}")
            return 1

        return 0


def main():
    """Main function for CLI usage."""
    metric = COMETMetric()
    return metric.run_from_cli()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
