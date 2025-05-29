"""
Main runner for NMT metrics evaluation using the metrics registry.

This script orchestrates the evaluation of multiple NMT metrics against human scores,
calculates correlations, and generates comprehensive reports with visualizations.
Now uses the centralized metrics registry for better organization and maintainability.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Import our utilities and registry
from src.utils.utils import (
    calculate_correlations,
    preprocess_human_scores,
    load_metric_scores,
    print_evaluation_results,
    analyze_significance,
    get_best_metrics
)
from src.config.metrics_registry import get_registry, initialize_registry


class NMTMetricsRunner:
    """
    Main runner for comprehensive NMT metrics evaluation against human scores.

    This class orchestrates metric evaluation, correlation analysis, and report generation
    using the centralized metrics registry for better organization and maintainability.
    """

    def __init__(self, auto_discover_metrics: bool = True):
        """
        Initialize the NMT metrics runner.

        Args:
            auto_discover_metrics (bool): Whether to auto-discover metrics from filesystem
        """
        self.registry = get_registry()
        if not self.registry._initialized:
            initialize_registry(auto_discover=auto_discover_metrics)
        self.temp_dir = None
        self.results = {}

    def setup_temp_directory(self) -> str:
        """
        Create temporary directory for intermediate results.

        Returns:
            str: Path to the created temporary directory
        """
        self.temp_dir = tempfile.mkdtemp(prefix='nmt_eval_')
        print(f"📁 Created temporary directory: {self.temp_dir}")
        return self.temp_dir

    def get_available_metrics(self, check_dependencies: bool = False) -> List[str]:
        """
        Get list of available metrics using the registry.

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
            metric_name (str): Name of the metric
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

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
            comet_args: Optional[Dict[str, str]] = None,
            llm_args: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Run all specified metrics using the registry.

        Args:
            input_path (str): Path to input CSV file
            metrics_to_run (Optional[List[str]]): List of metrics to run (default: basic metrics)
            comet_args (Optional[Dict[str, str]]): Additional arguments for COMET metric
            llm_args (Optional[Dict[str, str]]): Additional arguments for LLM metrics

        Returns:
            Dict[str, str]: Dictionary mapping metric names to their output file paths
        """
        if metrics_to_run is None:
            # Default to commonly used metrics, checking availability
            default_metrics = ['BLEU', 'CHRF++', 'TER', 'ROUGE', 'COMET', 'LLM-Reference-Based', 'LLM-Reference-Free']
            available_metrics = self.get_available_metrics(check_dependencies=False)
            metrics_to_run = [m for m in default_metrics if m in available_metrics]
            print(f"🎯 Using available metrics: {metrics_to_run}")

        # Validate requested metrics against registry
        available_metrics = self.registry.get_metric_names()
        invalid_metrics = [m for m in metrics_to_run if m not in available_metrics]
        if invalid_metrics:
            print(f"⚠️  Unknown metrics (not in registry): {invalid_metrics}")
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
                for arg_name, cmd_flag in [
                    ('model', '--model'),
                    ('api_key', '--api-key'),
                    ('batch_size', '--batch-size'),
                    ('max_retries', '--max-retries'),
                    ('env_path', '--env-path'),
                    ('progress_file', '--progress-file')
                ]:
                    if arg_name in llm_args and llm_args[arg_name] is not None:
                        additional_args.extend([cmd_flag, str(llm_args[arg_name])])

            output_path = self.run_metric(metric_name, input_path, additional_args)
            if output_path:
                results_paths[metric_name] = output_path

        return results_paths

    def calculate_all_correlations(
        self,
        metric_scores: Dict[str, List[float]],
        human_scores: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations for all metrics.

        Args:
            metric_scores (Dict[str, List[float]]): Dictionary mapping metric names to their scores
            human_scores (List[float]): List of human evaluation scores

        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping metric names to their correlation results
        """
        correlation_results = {}

        for metric_name, scores in metric_scores.items():
            if scores is None:
                print(f"⚠️  Skipping {metric_name} - no scores available")
                continue

            if len(scores) != len(human_scores):
                print(f"⚠️  Skipping {metric_name} - score length mismatch ({len(scores)} vs {len(human_scores)})")
                continue

            print(f"Calculating correlations for {metric_name}...")
            correlation_results[metric_name] = calculate_correlations(scores, human_scores)

        return correlation_results

    def create_visualizations(
        self,
        results: Dict[str, Dict[str, float]],
        metric_scores: Dict[str, List[float]],
        human_scores: List[float],
        output_dir: str
    ) -> None:
        """
        Create comprehensive visualizations of results.

        Args:
            results (Dict[str, Dict[str, float]]): Correlation results for each metric
            metric_scores (Dict[str, List[float]]): Dictionary mapping metric names to their scores
            human_scores (List[float]): List of human evaluation scores
            output_dir (str): Directory to save visualizations
        """
        print("📈 Creating visualizations...")

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Correlation comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NMT Metrics Evaluation Results', fontsize=16, fontweight='bold')

        # Extract correlation data
        metrics = list(results.keys())
        correlation_types = ['pearson', 'spearman', 'kendall']

        corr_data = []
        for metric in metrics:
            row = [results[metric][corr_type] for corr_type in correlation_types]
            corr_data.append(row)

        corr_df = pd.DataFrame(corr_data, columns=correlation_types, index=metrics)

        # Heatmap of correlations
        sns.heatmap(corr_df, annot=True, cmap='RdYlBu_r', center=0,
                   fmt='.3f', ax=axes[0,0], cbar_kws={'label': 'Correlation'})
        axes[0,0].set_title('Correlation Coefficients with Human Scores')

        # Bar plot of Pearson correlations
        pearson_scores = [results[metric]['pearson'] for metric in metrics]
        bars = axes[0,1].bar(metrics, pearson_scores, alpha=0.8, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Pearson Correlation with Human Scores')
        axes[0,1].set_ylabel('Pearson Correlation')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, pearson_scores):
            if not np.isnan(score):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Scatter plots showing metric vs human scores (first 2 metrics)
        available_metrics = [m for m in metrics if m in metric_scores and metric_scores[m] is not None]

        for i, metric_name in enumerate(available_metrics[:2]):
            ax = axes[1, i]
            scores = metric_scores[metric_name]

            ax.scatter(scores, human_scores, alpha=0.6, s=30)
            ax.set_xlabel(f'{metric_name} Score')
            ax.set_ylabel('Human Score')
            ax.set_title(f'{metric_name} vs Human Scores')
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient to plot
            corr = results[metric_name]['pearson']
            if not np.isnan(corr):
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # If less than 2 metrics, fill remaining subplot
        if len(available_metrics) < 2:
            axes[1,1].text(0.5, 0.5, 'Insufficient metric data\nfor visualization',
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Metric vs Human Scores')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Radar chart of average correlations
        if len(metrics) > 2:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            avg_scores = []
            for metric in metrics:
                correlations = []
                for corr_type in correlation_types:
                    if not np.isnan(results[metric][corr_type]):
                        correlations.append(results[metric][corr_type])
                avg_scores.append(np.mean(correlations) if correlations else 0)

            avg_scores += avg_scores[:1]  # Complete the circle

            ax.plot(angles, avg_scores, 'o-', linewidth=2, label='Average Correlation')
            ax.fill(angles, avg_scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, max(avg_scores) * 1.1 if max(avg_scores) > 0 else 1)
            ax.set_title('Average Correlation Scores\n(All Three Measures)', pad=20)
            ax.grid(True)

            plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
            plt.show()

        print(f"✅ Visualizations saved to {output_dir}")

    def save_comprehensive_results(
        self,
        results: Dict[str, Dict[str, float]],
        metric_scores: Dict[str, List[float]],
        human_scores: List[float],
        output_path: str
    ) -> None:
        """
        Save comprehensive results including correlations and metric scores.

        Args:
            results (Dict[str, Dict[str, float]]): Correlation results for each metric
            metric_scores (Dict[str, List[float]]): Dictionary mapping metric names to their scores
            human_scores (List[float]): List of human evaluation scores
            output_path (str): Path to save results CSV
        """
        print(f"💾 Saving comprehensive results to {output_path}...")

        # Create a comprehensive DataFrame
        num_samples = len(human_scores)

        # Start with human scores
        base_data = {'human_score': human_scores}

        # Add metric scores
        for metric_name, scores in metric_scores.items():
            if scores is not None:
                column_name = f'{metric_name.lower().replace("+", "plus").replace("-", "_")}_score'
                if len(scores) == num_samples:
                    base_data[column_name] = scores
                else:
                    print(f"⚠️  {metric_name} has different length ({len(scores)}), padding/truncating")
                    # Pad or truncate to match
                    if len(scores) < num_samples:
                        padded = scores + [np.nan] * (num_samples - len(scores))
                        base_data[column_name] = padded
                    else:
                        base_data[column_name] = scores[:num_samples]

        # Create DataFrame
        df = pd.DataFrame(base_data)

        # Add correlation results as metadata
        correlations_summary = {
            'correlation_results': results,
            'summary_statistics': {},
            'registry_info': {
                'available_metrics': self.registry.get_metric_names(),
                'used_metrics': list(metric_scores.keys())
            }
        }

        # Calculate summary statistics
        for metric_name, scores in metric_scores.items():
            if scores is not None:
                correlations_summary['summary_statistics'][metric_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'count': len(scores)
                }

        # Save main data
        df.to_csv(output_path, index=False)

        # Save correlation summary
        summary_path = output_path.replace('.csv', '_correlations.json')
        with open(summary_path, 'w') as f:
            json.dump(correlations_summary, f, indent=2)

        print(f"✅ Saved {len(df)} samples to {output_path}")
        print(f"✅ Saved correlation summary to {summary_path}")

    def generate_report(
        self,
        results: Dict[str, Dict[str, float]],
        input_path: str,
        num_samples: int
    ) -> None:
        """
        Generate a comprehensive text report.

        Args:
            results (Dict[str, Dict[str, float]]): Correlation results for each metric
            input_path (str): Path to input data file
            num_samples (int): Number of evaluation samples
        """
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)

        # Basic info
        print(f"\nEVALUATION SUMMARY:")
        print(f"   Dataset: {input_path}")
        print(f"   Total evaluation pairs: {num_samples}")
        print(f"   Metrics evaluated: {len(results)}")

        # Registry info
        print(f"   Available metrics in registry: {len(self.registry.get_metric_names())}")
        print(f"   Registry metrics: {', '.join(self.registry.get_metric_names())}")

        # Print detailed results table
        print_evaluation_results(results)

        # Find best metrics
        best_pearson, best_spearman, best_kendall, best_overall = get_best_metrics(results)

        print(f"\n🏆 BEST PERFORMING METRICS:")
        if best_pearson:
            print(f"  Pearson correlation:  {best_pearson} ({results[best_pearson]['pearson']:.4f})")
        if best_spearman:
            print(f"  Spearman correlation: {best_spearman} ({results[best_spearman]['spearman']:.4f})")
        if best_kendall:
            print(f"  Kendall correlation:  {best_kendall} ({results[best_kendall]['kendall']:.4f})")
        if best_overall:
            avg_corr = np.mean([results[best_overall][t] for t in ['pearson', 'spearman', 'kendall']
                               if not np.isnan(results[best_overall][t])])
            print(f"  Best overall metric:  {best_overall} (avg: {avg_corr:.4f})")

        # Statistical significance analysis
        significance = analyze_significance(results)

        print(f"\n📈 STATISTICAL SIGNIFICANCE ANALYSIS:")
        for level, metrics_list in significance.items():
            if metrics_list:
                level_name = level.replace('_', ' ').title()
                thresholds = {
                    'highly_significant': 'p < 0.001',
                    'significant': 'p < 0.01',
                    'marginally_significant': 'p < 0.05',
                    'not_significant': 'p ≥ 0.05'
                }
                print(f"  {level_name} ({thresholds[level]}):")
                for metric, p_val in metrics_list:
                    print(f"    {metric}: p = {p_val:.6f}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_overall:
            print(f"  1. Use {best_overall} as the primary automatic evaluation metric")
            print(f"  2. Consider ensemble approaches combining top-performing metrics")

        # Show top 3 metrics
        avg_correlations = {}
        for metric in results.keys():
            correlations = [results[metric][t] for t in ['pearson', 'spearman', 'kendall']
                           if not np.isnan(results[metric][t])]
            if correlations:
                avg_correlations[metric] = np.mean(correlations)

        if avg_correlations:
            sorted_metrics = sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)
            print(f"  3. Top 3 metrics for this dataset:")
            for i, (metric, score) in enumerate(sorted_metrics[:3], 1):
                print(f"      {i}. {metric} (avg: {score:.4f})")

        print(f"  4. Validate results on additional test sets")
        print(f"  5. Consider task-specific optimization of metric parameters")

        print("=" * 80)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive NMT metrics evaluation using registry'
    )
    parser.add_argument(
        '--input', '-i',
        help='Path to input CSV file containing evaluation data'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to output CSV file for comprehensive results'
    )
    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        default=None,  # Will use available metrics from registry
        help='Metrics to evaluate (default: all available from registry)'
    )
    parser.add_argument(
        '--list-metrics',
        action='store_true',
        help='List all available metrics from registry and exit'
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
    parser.add_argument(
        '--no-standardize',
        action='store_true',
        help='Skip z-standardization of human scores by evaluator'
    )
    parser.add_argument(
        '--min-evaluations',
        type=int,
        default=5,
        help='Minimum evaluations per evaluator for standardization (default: 5)'
    )
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
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory for output files and visualizations (default: current)'
    )

    # LLM arguments
    parser.add_argument(
        '--llm-model',
        default='claude-sonnet-4-20250514',
        help='LLM model to use for evaluation (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--llm-api-key',
        help='Anthropic API key for LLM evaluation (default: from ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--llm-batch-size',
        type=int,
        default=None,
        help='Number of samples to evaluate with LLM (default: all data, max: 100000)'
    )
    parser.add_argument(
        '--llm-max-retries',
        type=int,
        default=3,
        help='Maximum retries for failed LLM requests (default: 3)'
    )
    parser.add_argument(
        '--llm-env-path',
        help='Path to .env file for LLM configuration (default: auto-search)'
    )
    parser.add_argument(
        '--llm-progress-file',
        help='Path to save/load LLM batch progress (default: auto-generated)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    runner = NMTMetricsRunner()

    # Handle special commands
    if args.list_metrics:
        runner.print_available_metrics()
        return 0

    if args.check_dependencies:
        print("🔍 Checking dependencies for all metrics:")
        for metric_name in runner.registry.get_metric_names():
            deps_status = runner.registry.check_dependencies(metric_name)
            status_symbol = "✅" if all(deps_status.values()) else "❌"
            print(f"{status_symbol} {metric_name}")
            for dep, available in deps_status.items():
                dep_symbol = "✅" if available else "❌"
                print(f"    {dep_symbol} {dep}")
        return 0

    if args.debug_dependencies:
        runner.registry.debug_dependency_imports(args.debug_dependencies)
        return 0

    # Validate required arguments for evaluation
    if not args.input:
        parser.error("--input is required when running evaluation")
    if not args.output:
        parser.error("--output is required when running evaluation")

    try:
        # Setup
        runner.setup_temp_directory()

        print("🚀 Starting comprehensive NMT metrics evaluation using registry...")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Metrics: {args.metrics or 'available from registry'}")

        # Load and preprocess data FIRST
        print("\n📁 Loading evaluation data...")
        data = pd.read_csv(args.input)
        print(f"Loaded {len(data)} evaluation records")

        # Preprocess human scores (this may reduce the dataset size)
        if not args.no_standardize:
            data, score_column = preprocess_human_scores(data, args.min_evaluations)
        else:
            score_column = 'score'
            print("Skipping score standardization")

        # Save preprocessed data to temporary file
        preprocessed_data_path = os.path.join(runner.temp_dir, 'preprocessed_data.csv')
        data.to_csv(preprocessed_data_path, index=False)
        print(f"📁 Saved preprocessed data ({len(data)} samples) to temporary file")

        human_scores = data[score_column].tolist()

        # Run all metrics on the PREPROCESSED data
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

        metric_result_paths = runner.run_all_metrics(
            preprocessed_data_path,  # Use preprocessed data, not original
            args.metrics,
            comet_args,
            llm_args
        )

        if not metric_result_paths:
            print("❌ No metrics completed successfully")
            return 1

        # Load metric scores
        print("\nLoading metric scores...")
        metric_scores = load_metric_scores(metric_result_paths)

        # Calculate correlations (now shapes should match)
        print("\n🔗 Calculating correlations...")
        correlation_results = runner.calculate_all_correlations(metric_scores, human_scores)

        if not correlation_results:
            print("❌ No correlation results available")
            return 1

        # Generate report
        runner.generate_report(correlation_results, args.input, len(data))

        # Create visualizations
        runner.create_visualizations(
            correlation_results,
            metric_scores,
            human_scores,
            args.output_dir
        )

        # Save comprehensive results
        runner.save_comprehensive_results(
            correlation_results,
            metric_scores,
            human_scores,
            args.output
        )

        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"Visualizations saved to: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        runner.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
