"""
CSV processor for NMT metrics evaluation using the metrics registry.

This module provides a simplified interface for processing CSV files containing
machine translation data and adding metric scores as additional columns.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Import our utilities and registry
from src.utils.utils import load_metric_scores
from src.config.metrics_registry import get_registry, initialize_registry


class CSVMetricsProcessor:
    """
    Processes CSV files containing translation data and adds metric scores.

    This class provides a simplified interface for evaluating machine translations
    using multiple metrics and saving the results with scores as additional columns.
    Uses the central metrics registry for discovering and managing metrics.
    """

    def __init__(self, auto_discover_metrics: bool = True):
        """
        Initialize the CSV metrics processor.

        Args:
            auto_discover_metrics (bool): Whether to auto-discover metrics from the filesystem
        """
        self.registry = get_registry()
        if not self.registry._initialized:
            initialize_registry(auto_discover=auto_discover_metrics)
        self.temp_dir = None

    def setup_temp_directory(self) -> str:
        """
        Create temporary directory for intermediate results.

        Returns:
            str: Path to the created temporary directory
        """
        self.temp_dir = tempfile.mkdtemp(prefix='csv_metrics_')
        print(f"📁 Created temporary directory: {self.temp_dir}")
        return self.temp_dir

    def validate_input_csv(self, input_path: str) -> pd.DataFrame:
        """
        Validate and load the input CSV file.

        Args:
            input_path (str): Path to the input CSV file

        Returns:
            pd.DataFrame: Loaded and validated DataFrame

        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If required columns are missing or data is invalid
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Check required columns
        required_columns = ['sourceText', 'targetText', 'referenceText']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove rows with missing values in required columns
        original_length = len(df)
        df = df.dropna(subset=required_columns)

        if len(df) == 0:
            raise ValueError("No valid data rows after removing rows with missing values")

        if len(df) < original_length:
            print(f"⚠️  Removed {original_length - len(df)} rows with missing values")

        print(f"✅ Validated input CSV: {len(df)} valid rows")
        return df

    def get_available_metrics(self, check_dependencies: bool = False) -> List[str]:
        """
        Get list of available metrics.

        Args:
            check_dependencies (bool): Whether to check if dependencies are installed

        Returns:
            List[str]: List of available metric names
        """
        return self.registry.get_available_metrics(check_dependencies=check_dependencies)

    def print_available_metrics(self) -> None:
        """Print information about all available metrics."""
        self.registry.print_registry_info()

    def run_metric(
        self,
        metric_name: str,
        input_path: str,
        additional_args: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Run a single metric evaluation using the registry.

        Args:
            metric_name (str): Name of the metric (e.g., 'BLEU', 'CHRF++')
            input_path (str): Path to input CSV file
            additional_args (Optional[List[str]]): Additional command line arguments

        Returns:
            Optional[str]: Path to output file if successful, None otherwise
        """
        metric_info = self.registry.get_metric_info(metric_name)
        if not metric_info:
            print(f"⚠️  Unknown metric: {metric_name}")
            return None

        module_path = metric_info.module_path

        # Check dependencies
        deps_status = self.registry.check_dependencies(metric_name)
        missing_deps = [dep for dep, available in deps_status.items() if not available]
        if missing_deps:
            print(f"⚠️  Missing dependencies for {metric_name}: {missing_deps}")
            print(f"   Install with: pip install {' '.join(missing_deps)}")
            return None

        output_path = os.path.join(
            self.temp_dir,
            f"{metric_name.lower().replace('+', 'plus').replace('-', '_')}_scores.json"
        )

        # Build command - use module execution
        cmd = [sys.executable, '-m', module_path, '--input', input_path, '--output', output_path]
        if additional_args:
            # Filter out None values and convert properly
            filtered_args = []
            for arg in additional_args:
                if arg is not None and str(arg) != 'None':
                    filtered_args.append(str(arg))
            cmd.extend(filtered_args)

        print(f"\n🚀 Running {metric_name}...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                print(f"✅ {metric_name} completed successfully")
                if result.stdout:
                    print(result.stdout)
                return output_path
            else:
                print(f"❌ {metric_name} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                return None

        except subprocess.TimeoutExpired:
            print(f"❌ {metric_name} timed out after 1 hour")
            return None
        except Exception as e:
            print(f"❌ {metric_name} failed with exception: {e}")
            return None

    def run_all_metrics(
        self,
        input_path: str,
        metrics_to_run: Optional[List[str]] = None,
        comet_args: Optional[Dict[str, Union[str, int]]] = None,
        llm_args: Optional[Dict[str, Union[str, int]]] = None
    ) -> Dict[str, str]:
        """
        Run all specified metrics on the input data.

        Args:
            input_path (str): Path to input CSV file
            metrics_to_run (Optional[List[str]]): List of metrics to run (default: basic metrics)
            comet_args (Optional[Dict[str, Union[str, int]]]): Arguments for COMET metric
                - batch_size (int): Batch size for COMET evaluation
                - sample_size (int): Limit COMET evaluation to N samples
            llm_args (Optional[Dict[str, Union[str, int]]]): Arguments for LLM metrics
                - model (str): LLM model name
                - api_key (str): API key for LLM service
                - batch_size (int): Number of samples to evaluate
                - max_retries (int): Maximum retries for failed requests
                - env_path (str): Path to .env file
                - progress_file (str): Progress file path

        Returns:
            Dict[str, str]: Dictionary mapping metric names to their output file paths
        """
        if metrics_to_run is None:
            # Default to basic, fast metrics that don't require heavy dependencies
            basic_metrics = ['BLEU', 'CHRF++', 'TER', 'ROUGE']
            available_metrics = self.get_available_metrics(check_dependencies=True)
            metrics_to_run = [m for m in basic_metrics if m in available_metrics]
            print(f"🎯 Using default metrics: {metrics_to_run}")

        # Validate requested metrics
        available_metrics = self.registry.get_metric_names()
        invalid_metrics = [m for m in metrics_to_run if m not in available_metrics]
        if invalid_metrics:
            print(f"⚠️  Unknown metrics: {invalid_metrics}")
            print(f"📋 Available metrics: {available_metrics}")
            metrics_to_run = [m for m in metrics_to_run if m in available_metrics]

        if not metrics_to_run:
            raise ValueError("No valid metrics specified")

        results_paths = {}

        for metric_name in metrics_to_run:
            # Prepare additional arguments based on metric type
            additional_args = None

            # Handle COMET-specific arguments
            if metric_name == 'COMET' and comet_args:
                additional_args = []
                if 'batch_size' in comet_args and comet_args['batch_size'] is not None:
                    additional_args.extend(['--batch-size', str(comet_args['batch_size'])])
                if 'sample_size' in comet_args and comet_args['sample_size'] is not None:
                    additional_args.extend(['--sample-size', str(comet_args['sample_size'])])
                # If no arguments were added, set to None
                if not additional_args:
                    additional_args = None

            # Handle LLM-specific arguments
            elif metric_name.startswith('LLM-') and llm_args:
                additional_args = []

                # Set evaluation mode based on metric name
                if metric_name == 'LLM-Reference-Based':
                    additional_args.extend(['--mode', 'reference_based'])
                elif metric_name == 'LLM-Reference-Free':
                    additional_args.extend(['--mode', 'reference_free'])

                # Add other LLM arguments if provided
                if 'model' in llm_args and llm_args['model'] is not None:
                    additional_args.extend(['--model', str(llm_args['model'])])
                if 'api_key' in llm_args and llm_args['api_key'] is not None:
                    additional_args.extend(['--api-key', str(llm_args['api_key'])])
                if 'batch_size' in llm_args and llm_args['batch_size'] is not None:
                    additional_args.extend(['--batch-size', str(llm_args['batch_size'])])
                if 'max_retries' in llm_args and llm_args['max_retries'] is not None:
                    additional_args.extend(['--max-retries', str(llm_args['max_retries'])])
                if 'env_path' in llm_args and llm_args['env_path'] is not None:
                    additional_args.extend(['--env-path', str(llm_args['env_path'])])
                if 'progress_file' in llm_args and llm_args['progress_file'] is not None:
                    additional_args.extend(['--progress-file', str(llm_args['progress_file'])])

            output_path = self.run_metric(metric_name, input_path, additional_args)
            if output_path:
                results_paths[metric_name] = output_path

        return results_paths

    def add_scores_to_dataframe(
        self,
        df: pd.DataFrame,
        metric_scores: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Add metric scores as new columns to the DataFrame.

        Args:
            df (pd.DataFrame): Original DataFrame with translation data
            metric_scores (Dict[str, List[float]]): Dictionary mapping metric names to score lists

        Returns:
            pd.DataFrame: DataFrame with added score columns

        Raises:
            ValueError: If score lists don't match DataFrame length
        """
        result_df = df.copy()

        for metric_name, scores in metric_scores.items():
            if scores is None:
                print(f"⚠️  Skipping {metric_name} - no scores available")
                continue

            # Create clean column name
            column_name = f"{metric_name.lower().replace('+', 'plus').replace('-', '_')}_score"

            if len(scores) != len(df):
                print(f"⚠️  Score length mismatch for {metric_name}: {len(scores)} vs {len(df)}")
                # Pad or truncate to match DataFrame length
                if len(scores) < len(df):
                    padded_scores = scores + [np.nan] * (len(df) - len(scores))
                    result_df[column_name] = padded_scores
                else:
                    result_df[column_name] = scores[:len(df)]
            else:
                result_df[column_name] = scores

        return result_df

    def process_csv(
        self,
        input_path: str,
        output_path: str,
        metrics_to_run: Optional[List[str]] = None,
        comet_args: Optional[Dict[str, Union[str, int]]] = None,
        llm_args: Optional[Dict[str, Union[str, int]]] = None
    ) -> pd.DataFrame:
        """
        Process a CSV file and add metric scores as additional columns.

        Args:
            input_path (str): Path to input CSV file with required columns
            output_path (str): Path to save the output CSV file with scores
            metrics_to_run (Optional[List[str]]): List of metrics to evaluate (default: basic metrics)
            comet_args (Optional[Dict[str, Union[str, int]]]): COMET-specific arguments
            llm_args (Optional[Dict[str, Union[str, int]]]): LLM-specific arguments

        Returns:
            pd.DataFrame: DataFrame with original data and added metric score columns

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input data is invalid
            RuntimeError: If metric evaluation fails
        """
        print(f"🚀 Starting CSV metrics processing...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Metrics: {metrics_to_run or 'default (basic metrics)'}")

        # Setup temporary directory
        self.setup_temp_directory()

        try:
            # Validate and load input CSV
            df = self.validate_input_csv(input_path)

            # Run all metrics
            print(f"\n📊 Running metrics evaluation...")
            metric_result_paths = self.run_all_metrics(
                input_path,
                metrics_to_run,
                comet_args,
                llm_args
            )

            if not metric_result_paths:
                raise RuntimeError("No metrics completed successfully")

            # Load metric scores
            print(f"\n📥 Loading metric scores...")
            metric_scores = load_metric_scores(metric_result_paths)

            # Add scores to DataFrame
            print(f"\n📝 Adding scores to DataFrame...")
            result_df = self.add_scores_to_dataframe(df, metric_scores)

            # Save result
            result_df.to_csv(output_path, index=False)
            print(f"✅ Saved results to: {output_path}")

            # Print summary
            score_columns = [col for col in result_df.columns if col.endswith('_score')]
            print(f"\n📈 Summary:")
            print(f"  - Original columns: {len(df.columns)}")
            print(f"  - Added score columns: {len(score_columns)}")
            print(f"  - Total columns: {len(result_df.columns)}")
            print(f"  - Score columns: {score_columns}")

            # Print score statistics
            if score_columns:
                print(f"\n📊 Score Statistics:")
                for col in score_columns:
                    scores = result_df[col].dropna()
                    if len(scores) > 0:
                        print(f"  {col}:")
                        print(f"    Mean: {scores.mean():.4f}")
                        print(f"    Std:  {scores.std():.4f}")
                        print(f"    Min:  {scores.min():.4f}")
                        print(f"    Max:  {scores.max():.4f}")

            return result_df

        finally:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Process CSV file and add NMT metric scores as columns'
    )
    parser.add_argument(
        '--input', '-i',
        help='Path to input CSV file (must contain: sourceText, targetText, referenceText)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to output CSV file with added score columns'
    )
    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        default=None,  # Will use default basic metrics
        help='Metrics to evaluate (default: basic metrics like BLEU, CHRF++, TER, ROUGE)'
    )
    parser.add_argument(
        '--list-metrics',
        action='store_true',
        help='List all available metrics and exit'
    )
    parser.add_argument(
        '--check-dependencies',
        action='store_true',
        help='Check dependencies for all metrics and exit'
    )
    parser.add_argument(
        '--debug-dependencies',
        metavar='METRIC_NAME',
        help='Debug dependency imports for a specific metric and exit'
    )

    # COMET arguments
    parser.add_argument(
        '--comet-batch-size',
        type=int,
        default=8,
        help='Batch size for COMET evaluation (default: 8)'
    )
    parser.add_argument(
        '--comet-sample-size',
        type=int,
        default=None,
        help='Limit COMET evaluation to N samples (default: all)'
    )

    # LLM arguments
    parser.add_argument(
        '--llm-model',
        default='claude-sonnet-4-20250514',
        help='LLM model to use for evaluation (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--llm-api-key',
        help='API key for LLM evaluation (default: from ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--llm-batch-size',
        type=int,
        default=None,
        help='Number of samples to evaluate with LLM (default: all data)'
    )
    parser.add_argument(
        '--llm-max-retries',
        type=int,
        default=3,
        help='Maximum retries for failed LLM requests (default: 3)'
    )
    parser.add_argument(
        '--llm-env-path',
        help='Path to .env file for LLM configuration'
    )
    parser.add_argument(
        '--llm-progress-file',
        help='Path to save/load LLM batch progress'
    )

    args = parser.parse_args()

    # Create processor
    processor = CSVMetricsProcessor()

    # Handle special commands
    if args.list_metrics:
        processor.print_available_metrics()
        return 0

    if args.check_dependencies:
        print("🔍 Checking dependencies for all metrics:")
        registry = get_registry()
        for metric_name in registry.get_metric_names():
            deps_status = registry.check_dependencies(metric_name)
            status_symbol = "✅" if all(deps_status.values()) else "❌"
            print(f"{status_symbol} {metric_name}")
            for dep, available in deps_status.items():
                dep_symbol = "✅" if available else "❌"
                print(f"    {dep_symbol} {dep}")
        return 0

    if args.debug_dependencies:
        registry = get_registry()
        registry.debug_dependency_imports(args.debug_dependencies)
        return 0

    # Validate required arguments for processing
    if not args.input:
        parser.error("--input is required when processing CSV files")
    if not args.output:
        parser.error("--output is required when processing CSV files")

    # Prepare arguments for metrics
    comet_args = {
        'batch_size': args.comet_batch_size,
        'sample_size': args.comet_sample_size
    }

    llm_args = {
        'model': args.llm_model,
        'api_key': args.llm_api_key,
        'batch_size': args.llm_batch_size,
        'max_retries': args.llm_max_retries,
        'env_path': args.llm_env_path,
        'progress_file': args.llm_progress_file
    }

    try:
        result_df = processor.process_csv(
            args.input,
            args.output,
            args.metrics,
            comet_args,
            llm_args
        )

        print(f"\n✅ Processing completed successfully!")
        print(f"Results saved to: {args.output}")
        return 0

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
