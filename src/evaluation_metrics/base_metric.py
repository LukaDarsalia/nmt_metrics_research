"""
Abstract base class for NMT evaluation_metrics metrics.

This module defines the interface that all metric implementations must follow.
"""

import argparse
import json
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

class BaseMetric(ABC):
    """Abstract base class for all NMT evaluation_metrics metrics."""

    def __init__(self, metric_name: str):
        """
        Initialize the metric.

        Args:
            metric_name: Name of the metric (e.g., 'BLEU', 'CHRF++')
        """
        self.metric_name = metric_name

    @abstractmethod
    def calculate_scores(
            self,
            hypotheses: List[str],
            references: List[str],
            sources: List[str] = None
    ) -> List[float]:
        """
        Calculate metric scores for hypothesis-reference pairs.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (optional, needed for some metrics like COMET)

        Returns:
            List of metric scores
        """
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Return list of required packages for this metric.

        Returns:
            List of package names required for this metric
        """
        pass

    def validate_inputs(
            self,
            hypotheses: List[str],
            references: List[str],
            sources: List[str] = None
    ) -> None:
        """
        Validate input data.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations
            sources: List of source texts (optional)

        Raises:
            ValueError: If inputs are invalid
        """
        if len(hypotheses) != len(references):
            raise ValueError("Hypotheses and references must have the same length")

        if sources is not None and len(sources) != len(hypotheses):
            raise ValueError("Sources must have the same length as hypotheses")

        if len(hypotheses) == 0:
            raise ValueError("Input lists cannot be empty")

    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        Load evaluation_metrics data from CSV file.

        Args:
            input_path: Path to input CSV file

        Returns:
            DataFrame with evaluation_metrics data
        """
        required_columns = ['targetText', 'referenceText', 'sourceText']

        try:
            data = pd.read_csv(input_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {input_path}: {e}")

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove rows with missing values in required columns
        data = data.dropna(subset=required_columns)

        if len(data) == 0:
            raise ValueError("No valid data rows after removing NaN values")

        return data

    def save_scores(self, scores: List[float], output_path: str) -> None:
        """
        Save metric scores to file.

        Args:
            scores: List of metric scores
            output_path: Path to output file
        """
        try:
            # Save as JSON for easy loading
            result = {
                'metric_name': self.metric_name,
                'scores': scores,
                'num_samples': len(scores)
            }

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved {len(scores)} {self.metric_name} scores to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save scores to {output_path}: {e}")

    def run_from_cli(self) -> None:
        """
        Run the metric from command line interface.
        """
        parser = argparse.ArgumentParser(
            description=f'Calculate {self.metric_name} scores for NMT evaluation_metrics'
        )
        parser.add_argument(
            '--input', '-i',
            required=True,
            help='Path to input CSV file containing evaluation_metrics data'
        )
        parser.add_argument(
            '--output', '-o',
            required=True,
            help='Path to output JSON file for metric scores'
        )

        args = parser.parse_args()

        try:
            print(f"🚀 Running {self.metric_name} metric evaluation_metrics...")

            # Load data
            print(f"Loading data from {args.input}...")
            data = self.load_data(args.input)
            print(f"Loaded {len(data)} evaluation_metrics pairs")

            # Extract text components
            hypotheses = data['targetText'].tolist()
            references = data['referenceText'].tolist()
            sources = data['sourceText'].tolist() if 'sourceText' in data.columns else None

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


if __name__ == "__main__":
    # This shouldn't be run directly, but we can show usage
    print("This is the base metric class. Use specific metric implementations instead.")
    print("Available metrics: BLEU, CHRF++, TER, ROUGE, COMET")
