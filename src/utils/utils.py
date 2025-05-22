"""
Utility functions for NMT metrics evaluation_metrics.

This module provides correlation calculation and data preprocessing functions.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')


def calculate_correlations(
        metric_scores: List[float],
        human_scores: List[float]
) -> Dict[str, float]:
    """
    Calculate correlation coefficients between metric and human scores.

    Computes Pearson, Spearman, and Kendall's tau correlation coefficients
    along with their p-values to assess how well the metric correlates
    with human judgments. Handles constant arrays gracefully.

    Args:
        metric_scores: List of automatic metric scores
        human_scores: List of corresponding human evaluation_metrics scores

    Returns:
        Dictionary containing correlation coefficients and p-values:
        - 'pearson': Pearson correlation coefficient
        - 'spearman': Spearman rank correlation coefficient
        - 'kendall': Kendall's tau correlation coefficient
        - 'p_value_pearson': P-value for Pearson correlation
        - 'p_value_spearman': P-value for Spearman correlation
        - 'p_value_kendall': P-value for Kendall correlation

    Raises:
        ValueError: If input lists have different lengths or are empty
    """
    if len(metric_scores) != len(human_scores):
        raise ValueError("Metric scores and human scores must have the same length")

    if len(metric_scores) == 0:
        raise ValueError("Input lists cannot be empty")

    # Convert to numpy arrays for easier checking
    metric_array = np.array(metric_scores)
    human_array = np.array(human_scores)

    # Check for constant arrays or invalid values
    metric_std = np.std(metric_array)
    human_std = np.std(human_array)

    # Initialize results with NaN
    results = {
        'pearson': np.nan,
        'spearman': np.nan,
        'kendall': np.nan,
        'p_value_pearson': np.nan,
        'p_value_spearman': np.nan,
        'p_value_kendall': np.nan
    }

    # Check if arrays are constant or contain only NaN/inf
    if (metric_std == 0 or human_std == 0 or
            np.all(np.isnan(metric_array)) or np.all(np.isnan(human_array)) or
            np.all(np.isinf(metric_array)) or np.all(np.isinf(human_array))):
        print("Warning: Constant or invalid array detected, returning NaN correlations")
        return results

    # Remove any NaN or inf pairs
    valid_mask = (np.isfinite(metric_array) & np.isfinite(human_array))
    if np.sum(valid_mask) < 2:
        print("Warning: Less than 2 valid data points, returning NaN correlations")
        return results

    clean_metric = metric_array[valid_mask]
    clean_human = human_array[valid_mask]

    # Calculate correlations with error handling
    try:
        pearson, p_value_pearson = pearsonr(clean_metric, clean_human)
        results['pearson'] = pearson
        results['p_value_pearson'] = p_value_pearson
    except Exception as e:
        print(f"Warning: Pearson correlation failed: {e}")

    try:
        spearman, p_value_spearman = spearmanr(clean_metric, clean_human)
        results['spearman'] = spearman
        results['p_value_spearman'] = p_value_spearman
    except Exception as e:
        print(f"Warning: Spearman correlation failed: {e}")

    try:
        kendall, p_value_kendall = kendalltau(clean_metric, clean_human)
        results['kendall'] = kendall
        results['p_value_kendall'] = p_value_kendall
    except Exception as e:
        print(f"Warning: Kendall correlation failed: {e}")

    return results


def preprocess_human_scores(
        data: pd.DataFrame,
        min_evaluations: int = 5
) -> Tuple[pd.DataFrame, str]:
    """
    Preprocess human scores by z-standardizing within each evaluator.

    This method:
    1. Removes evaluators with fewer than min_evaluations
    2. Z-standardizes scores within each evaluator (mean=0, std=1)
    3. Returns cleaned data with standardized scores

    Args:
        data: DataFrame containing evaluation_metrics data with 'score' and 'createdBy_id' columns
        min_evaluations: Minimum number of evaluations required per evaluator

    Returns:
        Tuple of (DataFrame with z-standardized scores, score_column_name)

    Raises:
        ValueError: If required columns are missing or no data remains after filtering
    """
    if 'createdBy_id' not in data.columns:
        print("Warning: 'createdBy_id' column not found. Skipping evaluator standardization.")
        return data.copy(), 'score'

    print(f"Preprocessing human scores...")
    print(f"Original data: {len(data)} evaluations from {data['createdBy_id'].nunique()} evaluators")

    # Calculate evaluations per evaluator
    evaluator_counts = data['createdBy_id'].value_counts()

    # Filter evaluators with sufficient data
    valid_evaluators = evaluator_counts[evaluator_counts >= min_evaluations].index
    filtered_data = data[data['createdBy_id'].isin(valid_evaluators)].copy()

    removed_evaluators = len(evaluator_counts) - len(valid_evaluators)
    removed_evaluations = len(data) - len(filtered_data)

    print(f"Removed {removed_evaluators} evaluators with < {min_evaluations} evaluations")
    print(f"Removed {removed_evaluations} evaluations ({removed_evaluations / len(data) * 100:.1f}%)")

    if len(filtered_data) == 0:
        raise ValueError(f"No evaluators have >= {min_evaluations} evaluations")

    # Z-standardize scores within each evaluator
    def z_standardize_group(group):
        if len(group) < 2:  # Can't standardize with < 2 points
            return group
        mean_score = group['score'].mean()
        std_score = group['score'].std()
        if std_score == 0:  # All scores are the same
            group['standardized_score'] = 0.0
        else:
            group['standardized_score'] = (group['score'] - mean_score) / std_score
        return group

    # Apply z-standardization
    standardized_data = filtered_data.groupby('createdBy_id').apply(z_standardize_group)
    standardized_data = standardized_data.reset_index(drop=True)

    print(
        f"Final data: {len(standardized_data)} evaluations from {standardized_data['createdBy_id'].nunique()} evaluators")

    # Show standardization effects
    original_stats = filtered_data['score'].describe()
    standardized_stats = standardized_data['standardized_score'].describe()

    print(
        f"Score statistics before standardization: mean={original_stats['mean']:.2f}, std={original_stats['std']:.2f}")
    print(
        f"Score statistics after standardization:  mean={standardized_stats['mean']:.2f}, std={standardized_stats['std']:.2f}")

    return standardized_data, 'standardized_score'


def load_metric_scores(metric_results_paths: Dict[str, str]) -> Dict[str, List[float]]:
    """
    Load metric scores from JSON files saved by individual metrics.

    Args:
        metric_results_paths: Dictionary mapping metric names to their result file paths

    Returns:
        Dictionary mapping metric names to their score lists
    """
    import json

    metric_scores = {}

    for metric_name, path in metric_results_paths.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                metric_scores[metric_name] = data['scores']
                print(f"✅ Loaded {len(data['scores'])} {metric_name} scores from {path}")
        except Exception as e:
            print(f"❌ Failed to load {metric_name} scores from {path}: {e}")
            metric_scores[metric_name] = None

    return metric_scores


def print_evaluation_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print formatted evaluation_metrics results table.

    Args:
        results: Dictionary mapping metric names to their correlation results
    """
    print("\n" + "=" * 80)
    print("METRICS EVALUATION RESULTS")
    print("=" * 80)

    # Header
    header = f"{'Metric':<12} | {'Pearson':<8} | {'p-val':<8} | {'Spearman':<8} | {'p-val':<8} | {'Kendall':<8} | {'p-val':<8}"
    print(header)
    print("-" * len(header))

    # Results for each metric
    for metric_name, corr_results in results.items():
        row = (
            f"{metric_name:<12} | "
            f"{corr_results['pearson']:<8.4f} | "
            f"{corr_results['p_value_pearson']:<8.4f} | "
            f"{corr_results['spearman']:<8.4f} | "
            f"{corr_results['p_value_spearman']:<8.4f} | "
            f"{corr_results['kendall']:<8.4f} | "
            f"{corr_results['p_value_kendall']:<8.4f}"
        )
        print(row)

    print("=" * 80)


def analyze_significance(results: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze statistical significance of correlations.

    Args:
        results: Dictionary mapping metric names to their correlation results

    Returns:
        Dictionary categorizing metrics by significance level
    """
    significance_levels = {
        'highly_significant': [],  # p < 0.001
        'significant': [],  # p < 0.01
        'marginally_significant': [],  # p < 0.05
        'not_significant': []  # p >= 0.05
    }

    for metric in results.keys():
        # Get minimum p-value across all correlation types
        p_values = [
            results[metric].get('p_value_pearson', 1.0),
            results[metric].get('p_value_spearman', 1.0),
            results[metric].get('p_value_kendall', 1.0)
        ]
        min_p = min(p for p in p_values if not np.isnan(p))

        if min_p < 0.001:
            significance_levels['highly_significant'].append((metric, min_p))
        elif min_p < 0.01:
            significance_levels['significant'].append((metric, min_p))
        elif min_p < 0.05:
            significance_levels['marginally_significant'].append((metric, min_p))
        else:
            significance_levels['not_significant'].append((metric, min_p))

    return significance_levels


def get_best_metrics(results: Dict[str, Dict[str, float]]) -> Tuple[str, str, str, str]:
    """
    Find the best performing metrics for each correlation type.

    Args:
        results: Dictionary mapping metric names to their correlation results

    Returns:
        Tuple of (best_pearson, best_spearman, best_kendall, best_overall)
    """
    metrics = list(results.keys())

    # Filter out metrics with NaN correlations
    def get_valid_metrics(correlation_type):
        return [m for m in metrics if not np.isnan(results[m][correlation_type])]

    valid_pearson_metrics = get_valid_metrics('pearson')
    valid_spearman_metrics = get_valid_metrics('spearman')
    valid_kendall_metrics = get_valid_metrics('kendall')

    best_pearson = max(valid_pearson_metrics, key=lambda m: results[m]['pearson']) if valid_pearson_metrics else None
    best_spearman = max(valid_spearman_metrics,
                        key=lambda m: results[m]['spearman']) if valid_spearman_metrics else None
    best_kendall = max(valid_kendall_metrics, key=lambda m: results[m]['kendall']) if valid_kendall_metrics else None

    # Calculate average correlation for overall ranking (excluding NaN)
    avg_correlations = {}
    for metric in metrics:
        correlations = []
        for corr_type in ['pearson', 'spearman', 'kendall']:
            if not np.isnan(results[metric][corr_type]):
                correlations.append(results[metric][corr_type])

        if correlations:  # Only calculate average if we have valid correlations
            avg_correlations[metric] = np.mean(correlations)
        else:
            avg_correlations[metric] = np.nan

    # Find best overall metric (excluding NaN)
    valid_avg_correlations = {k: v for k, v in avg_correlations.items() if not np.isnan(v)}
    best_overall = max(valid_avg_correlations, key=valid_avg_correlations.get) if valid_avg_correlations else None

    return best_pearson, best_spearman, best_kendall, best_overall
