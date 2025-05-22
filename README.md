# NMT Metrics Research

This repository contains tools for evaluating Neural Machine Translation (NMT) metrics on Georgian language datasets.

## Overview

The project aims to evaluate the performance of various machine translation metrics by comparing them with human evaluation scores. The metrics are evaluated using correlation measures to determine which automatic metric best reflects human judgment on Georgian translation data.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nmt_metrics_research.git
   cd nmt_metrics_research
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data (required for TER metric):
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

### Evaluating NMT Metrics

You can evaluate the performance of various machine translation metrics on a Georgian dataset using the provided scripts.

#### Option 1: Using the convenience script

The easiest way to run the evaluation is to use the `run_evaluation.py` script:

```
python run_evaluation.py
```

This will run the evaluation with default parameters.

#### Option 2: Using the module directly

Alternatively, you can run the evaluation module directly:

```
python -m src.evaluate_metrics [--data_path DATA_PATH]
```

Arguments:
- `--data_path`: Path to the evaluation data CSV file (default: data/results.csv)

Example:
```
python -m src.evaluate_metrics --data_path data/results.csv
```

### Metrics Evaluated

The script evaluates the following metrics:

1. **BLEU** (Bilingual Evaluation Understudy): A precision-based metric that measures the overlap of n-grams between the hypothesis and reference translations.

2. **CHRF++**: A character-level metric that considers character n-grams and word n-grams, making it suitable for morphologically rich languages like Georgian.

3. **TER** (Translation Edit Rate): Measures the number of edits required to transform the hypothesis into the reference.

4. **ROUGE**: A recall-oriented metric that measures the overlap of n-grams between the hypothesis and reference.

5. **COMET**: A neural-based metric that uses contextual embeddings to evaluate translation quality.

### Correlation Measures

The script compares the metric scores with human evaluation scores using three correlation measures:

1. **Pearson correlation coefficient**: Measures the linear correlation between two variables.

2. **Spearman rank correlation**: Measures the monotonic relationship between two variables.

3. **Kendall's tau**: Measures the ordinal association between two variables.

## Project Structure

- `data/`: Contains the evaluation data
  - `results.csv`: Human evaluation scores for translations
  - `flores_devtest_google_sheet.csv`: Reference translations

- `src/`: Source code
  - `data/`: Data handling modules
    - `data_fetcher.py`: Fetches evaluation data from database
    - `db_connector.py`: Manages database connections
  - `utils/`: Utility modules
    - `config.py`: Configuration loading and management
  - `evaluate_metrics.py`: Script for evaluating NMT metrics

- `notebooks/`: Jupyter notebooks for analysis
  - `analyse_labeled_data.ipynb`: Notebook for analyzing labeled data
