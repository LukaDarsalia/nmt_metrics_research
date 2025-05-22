# NMT Metrics Research

A comprehensive, modular framework for evaluating Neural Machine Translation (NMT) metrics against human evaluation scores, with a focus on Georgian language translation quality assessment.

## 🎯 Overview

This project provides a robust evaluation system for machine translation metrics by comparing automatic metric scores with human evaluation judgments. The framework features:

- **Modular Architecture**: Each metric runs independently, preventing library conflicts
- **Comprehensive Analysis**: Multiple correlation measures and statistical significance testing
- **Visual Analytics**: Detailed plots and visualizations for better understanding
- **Georgian Language Focus**: Optimized for morphologically rich languages
- **PyTorch Integration**: Support for neural-based custom metrics

## ✨ Key Features

- ✅ **Isolated Metrics**: No more import failures - each metric runs separately
- ✅ **CLI Interface**: Run individual metrics or complete evaluations from command line
- ✅ **Robust Error Handling**: Continue evaluation even if one metric fails
- ✅ **Statistical Analysis**: Pearson, Spearman, and Kendall correlations with significance testing
- ✅ **Rich Visualizations**: Correlation plots, heatmaps, and radar charts
- ✅ **Data Preprocessing**: Automatic z-standardization by evaluator
- ✅ **Comprehensive Output**: Detailed results in CSV and JSON formats

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LukaDarsalia/nmt_metrics_research.git
   cd nmt_metrics_research
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### Run Complete Evaluation
```bash
python -m src.evaluation_metrics.runner --input data/results.csv --output metrics_results.csv
```

#### Run Without COMET (if installation issues)
```bash
python -m src.evaluation_metrics.runner --input data/results.csv --output metrics_results.csv \
    --metrics BLEU CHRF++ TER ROUGE
```

#### Run Individual Metrics
```bash
# Test BLEU metric
python -m src.evaluation_metrics.bleu_metric.py --input data/results.csv --output bleu_results.csv 

```

## 📊 Metrics Evaluated

### Traditional Metrics

1. **BLEU** (Bilingual Evaluation Understudy)
   - Precision-based n-gram overlap metric
   - Range: 0-100 (higher = better)
   - Good for: Fluency assessment

2. **CHRF++** (Character n-gram F-score)
   - Character-level metric with word order
   - Range: 0-100 (higher = better)
   - Good for: Morphologically rich languages like Georgian

3. **TER** (Translation Edit Rate)
   - Edit distance-based metric (inverted for consistency)
   - Range: 0-1 (higher = better, inverted from original TER)
   - Good for: Error analysis

4. **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
   - Recall-oriented longest common subsequence
   - Range: 0-1 (higher = better)
   - Good for: Content preservation

### Neural Metrics

5. **COMET** (Crosslingual Optimized Metric for Evaluation of Translation)
   - Neural metric using contextual embeddings
   - Range: ~-1 to 1 (higher = better)
   - Good for: Overall quality with source context

## 📈 Correlation Analysis

The framework evaluates metrics using three correlation measures:

### Correlation Types

1. **Pearson Correlation (r)**
   - Measures linear relationships
   - Sensitive to outliers
   - Most commonly reported

2. **Spearman Correlation (ρ)**
   - Measures monotonic relationships
   - Uses rank ordering
   - Robust to outliers and non-linear patterns

3. **Kendall's Tau (τ)**
   - Conservative rank correlation
   - Less sensitive to ties
   - More reliable with small samples

### Interpretation Guidelines

| Correlation | Strength | Interpretation |
|-------------|----------|----------------|
| 0.8 - 1.0   | Excellent | Metric is highly reliable |
| 0.6 - 0.8   | Good     | Metric is quite reliable |
| 0.4 - 0.6   | Moderate | Metric has some utility |
| 0.2 - 0.4   | Weak     | Metric is questionable |
| 0.0 - 0.2   | Very weak| Metric is unreliable |

### Statistical Significance

- **p < 0.001**: Highly significant (***)
- **p < 0.01**: Significant (**)
- **p < 0.05**: Marginally significant (*)
- **p ≥ 0.05**: Not significant

## 🔧 Advanced Usage

### Custom Configuration

```bash
# Skip score standardization
python -m src.evaluation_metrics.runner --input data/results.csv --output metrics_results.csv --no-standardize

# Adjust evaluator filtering threshold
python -m src.evaluation_metrics.runner --input data/results.csv --output metrics_results.csv --min-evaluations 3

# COMET with custom parameters
python -m src.evaluation_metrics.runner --input data/results.csv --output metrics_results.csv \
    --comet-batch-size 4 --comet-sample-size 100
```

### Creating Custom Metrics

```python
from base_metric import BaseMetric
from typing import List

class CustomMetric(BaseMetric):
    def __init__(self):
        super().__init__("CustomMetric")
    
    def get_dependencies(self) -> List[str]:
        return ["your_required_packages"]
    
    def calculate_scores(self, hypotheses: List[str], 
                        references: List[str], 
                        sources: List[str] = None) -> List[float]:
        # Your metric implementation here
        scores = []
        for hyp, ref in zip(hypotheses, references):
            score = your_calculation_function(hyp, ref)
            scores.append(score)
        return scores
```

## 📁 Project Structure

```
nmt_metrics_research/
├── data/                                  # Evaluation datasets
│   ├── results.csv                       # Main human evaluation data
│   ├── flores_devtest_google_sheet.csv   # Reference translations
│   └── flores_devtest_*.csv              # Additional test sets
├── src/
│   ├── evaluation_metrics/               # 🆕 New modular evaluation system
│   │   ├── base_metric.py               # Abstract base class for metrics
│   │   ├── bleu_metric.py               # BLEU implementation
│   │   ├── chrf_metric.py               # CHRF++ implementation
│   │   ├── ter_metric.py                # TER implementation
│   │   ├── rouge_metric.py              # ROUGE implementation
│   │   ├── comet_metric.py              # COMET implementation
│   │   ├── runner.py                    # Main evaluation orchestrator
│   │   ├── utils.py                     # Correlation and preprocessing utilities
│   │   ├── debug_utility.py             # Debug and analysis tools
│   │   ├── setup_script.py              # Installation verification
│   │   └── pytorch_integration.py       # PyTorch-based custom metrics
│   ├── data/                            # Data handling modules
│   │   ├── data_fetcher.py             # Database data fetching
│   │   └── db_connector.py             # Database connection management
│   └── utils/                           # General utilities
│       ├── config.py                   # Configuration management
│       └── utils.py                    # General utility functions
├── notebooks/                           # Jupyter notebooks
│   └── analyse_labeled_data.ipynb      # Data analysis notebook
├── tests/                              # Test suite
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 📊 Output Files

After running the evaluation, you'll get:

1. **`evaluation_results.csv`** - Complete dataset with all metric scores
2. **`evaluation_results_correlations.json`** - Correlation coefficients and p-values
3. **`evaluation_results.png`** - Correlation visualization plots
4. **`radar_chart.png`** - Performance comparison radar chart

### Example Output

```
📊 METRICS EVALUATION RESULTS
================================================================================
Metric       | Pearson  | p-val    | Spearman | p-val    | Kendall  | p-val    
--------------------------------------------------------------------------------
CHRF++       | 0.7234   | 0.0001   | 0.7156   | 0.0001   | 0.5234   | 0.0002   
BLEU         | 0.6789   | 0.0003   | 0.6654   | 0.0004   | 0.4789   | 0.0010   
ROUGE        | 0.5432   | 0.0045   | 0.5321   | 0.0052   | 0.3876   | 0.0120   
TER          | 0.4321   | 0.0234   | 0.4123   | 0.0287   | 0.2987   | 0.0456   
COMET        | 0.8123   | 0.0000   | 0.7987   | 0.0000   | 0.6234   | 0.0000   

🏆 Best Overall Metric: COMET (avg correlation: 0.7448)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add your metric to `src/evaluation_metrics/`
4. Follow the `BaseMetric` interface
5. Add tests and documentation
6. Submit a pull request

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{nmt_metrics_research,
  title={Modular Framework for NMT Metrics Evaluation on Georgian Language},
  author={Luka Darsalia; Ketevan Bakhturidze; Saba Sturua},
  year={2025},
  url={https://github.com/LukaDarsalia/nmt_metrics_research}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Built with ❤️ for Georgian NMT research**