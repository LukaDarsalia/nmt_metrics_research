"""Script for evaluating NMT metrics on Georgian dataset.

This script evaluates the performance of various machine translation metrics
(BLEU, CHRF++, TER, ROUGE, COMET) on a Georgian dataset and compares them
with human evaluation scores using correlation measures (Pearson, Spearman, Kendall's tau).

Usage:
    python -m src.evaluate_metrics [--data_path DATA_PATH]

Args:
    --data_path: Path to the evaluation data CSV file (default: data/results.csv)
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Check for required packages
missing_packages = []

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
except ImportError:
    missing_packages.append("scipy")

# Import NMT metric libraries
try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    missing_packages.append("sacrebleu")

try:
    import nltk
    from nltk.translate.ter_score import corpus_ter
except ImportError:
    missing_packages.append("nltk")

try:
    from rouge import Rouge
except ImportError:
    missing_packages.append("rouge-score")

try:
    import comet_scores
except ImportError:
    missing_packages.append("unbabel-comet")

# If any packages are missing, print instructions and exit
if missing_packages:
    print("The following required packages are missing:")
    for package in missing_packages:
        print(f"  - {package}")
    print("\nPlease install the required packages:")
    print("pip install " + " ".join(missing_packages))
    print("\nIf you're installing nltk, also download the required data:")
    print("python -c \"import nltk; nltk.download('punkt')\"")
    exit(1)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load evaluation data from CSV file.

    Args:
        data_path: Path to the evaluation data CSV file

    Returns:
        DataFrame containing evaluation data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    return pd.read_csv(data_path)


def calculate_bleu(hypotheses: List[str], references: List[str]) -> List[float]:
    """
    Calculate BLEU scores for each hypothesis-reference pair.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations

    Returns:
        List of BLEU scores
    """
    bleu = BLEU()
    scores = []

    for hyp, ref in zip(hypotheses, references):
        score = bleu.sentence_score(hyp, [ref]).score
        scores.append(score)

    return scores


def calculate_chrf(hypotheses: List[str], references: List[str]) -> List[float]:
    """
    Calculate CHRF++ scores for each hypothesis-reference pair.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations

    Returns:
        List of CHRF++ scores
    """
    chrf = CHRF(word_order=2)  # CHRF++ uses word_order=2
    scores = []

    for hyp, ref in zip(hypotheses, references):
        score = chrf.sentence_score(hyp, [ref]).score
        scores.append(score)

    return scores


def calculate_ter(hypotheses: List[str], references: List[str]) -> List[float]:
    """
    Calculate TER scores for each hypothesis-reference pair.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations

    Returns:
        List of TER scores
    """
    scores = []

    for hyp, ref in zip(hypotheses, references):
        # Lower score is better for TER, so we invert it (1 - TER) to make it comparable with other metrics
        score = 1 - corpus_ter([hyp.split()], [[ref.split()]])
        scores.append(score)

    return scores


def calculate_rouge(hypotheses: List[str], references: List[str]) -> List[float]:
    """
    Calculate ROUGE scores for each hypothesis-reference pair.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations

    Returns:
        List of ROUGE-L F1 scores
    """
    rouge = Rouge()
    scores = []

    for hyp, ref in zip(hypotheses, references):
        try:
            # Use ROUGE-L F1 score
            score = rouge.get_scores(hyp, ref)[0]['rouge-l']['f']
            scores.append(score)
        except ValueError:
            # Handle empty hypothesis or reference
            scores.append(0.0)

    return scores


def calculate_comet(hypotheses: List[str], references: List[str], sources: List[str]) -> List[float]:
    """
    Calculate COMET scores for each hypothesis-reference-source triplet.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations
        sources: List of source texts

    Returns:
        List of COMET scores
    """
    try:
        # Use the default COMET model
        model = comet_scores.load_from_checkpoint("wmt20-comet-da")

        data = [{"src": src, "mt": hyp, "ref": ref} 
                for src, hyp, ref in zip(sources, hypotheses, references)]

        scores = model.predict(data, batch_size=8, gpus=0)
        return scores
    except Exception as e:
        print(f"Error calculating COMET scores: {e}")
        print("Returning zeros for COMET scores")
        return [0.0] * len(hypotheses)


def calculate_correlations(metric_scores: List[float], human_scores: List[float]) -> Dict[str, float]:
    """
    Calculate correlation coefficients between metric scores and human scores.

    Args:
        metric_scores: List of metric scores
        human_scores: List of human scores

    Returns:
        Dictionary containing Pearson, Spearman, and Kendall's tau correlation coefficients
    """
    pearson, p_value_pearson = pearsonr(metric_scores, human_scores)
    spearman, p_value_spearman = spearmanr(metric_scores, human_scores)
    kendall, p_value_kendall = kendalltau(metric_scores, human_scores)

    return {
        'pearson': pearson,
        'spearman': spearman,
        'kendall': kendall,
        'p_value_pearson': p_value_pearson,
        'p_value_spearman': p_value_spearman,
        'p_value_kendall': p_value_kendall
    }


def evaluate_metrics(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all metrics and calculate correlations with human scores.

    Args:
        data: DataFrame containing evaluation data

    Returns:
        Dictionary mapping metric names to correlation results
    """
    # Extract data
    human_scores = data['score'].tolist()
    hypotheses = data['targetText'].tolist()
    references = data['referenceText'].tolist()
    sources = data['sourceText'].tolist()

    # Calculate metric scores
    results = {}

    print("Calculating BLEU scores...")
    bleu_scores = calculate_bleu(hypotheses, references)
    results['BLEU'] = calculate_correlations(bleu_scores, human_scores)

    print("Calculating CHRF++ scores...")
    chrf_scores = calculate_chrf(hypotheses, references)
    results['CHRF++'] = calculate_correlations(chrf_scores, human_scores)

    print("Calculating TER scores...")
    ter_scores = calculate_ter(hypotheses, references)
    results['TER'] = calculate_correlations(ter_scores, human_scores)

    print("Calculating ROUGE scores...")
    rouge_scores = calculate_rouge(hypotheses, references)
    results['ROUGE'] = calculate_correlations(rouge_scores, human_scores)

    print("Calculating COMET scores...")
    try:
        comet_scores = calculate_comet(hypotheses, references, sources)
        results['COMET'] = calculate_correlations(comet_scores, human_scores)
    except Exception as e:
        print(f"Error calculating COMET scores: {e}")
        print("Skipping COMET evaluation")

    return results


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print evaluation results in a formatted table.

    Args:
        results: Dictionary mapping metric names to correlation results
    """
    print("\nCorrelation Results:")
    print("-" * 80)
    print(f"{'Metric':<10} | {'Pearson':<10} | {'Spearman':<10} | {'Kendall':<10}")
    print("-" * 80)

    for metric, correlations in results.items():
        pearson = correlations['pearson']
        spearman = correlations['spearman']
        kendall = correlations['kendall']

        print(f"{metric:<10} | {pearson:.4f}    | {spearman:.4f}    | {kendall:.4f}")

    print("-" * 80)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate NMT metrics on Georgian dataset")
    parser.add_argument("--data_path", default="data/results.csv", help="Path to evaluation data CSV file")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} evaluation samples")

    print("Evaluating metrics...")
    results = evaluate_metrics(data)

    print_results(results)


if __name__ == "__main__":
    main()
