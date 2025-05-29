# NMT Metrics Research

A comprehensive, modular framework for evaluating Neural Machine Translation (NMT) metrics against human evaluation scores, with a focus on Georgian language translation quality assessment. Now includes training capabilities for custom COMET models.

## 🎯 Overview

This project provides a robust evaluation system for machine translation metrics by comparing automatic metric scores with human evaluation judgments. The framework features:

- **Modular Architecture**: Each metric runs independently, preventing library conflicts
- **Comprehensive Analysis**: Multiple correlation measures and statistical significance testing
- **Visual Analytics**: Detailed plots and visualizations for better understanding
- **Georgian Language Focus**: Optimized for morphologically rich languages
- **PyTorch Integration**: Support for neural-based custom metrics
- **COMET Training**: Train custom COMET models on your own data

## ✨ Key Features

- ✅ **Isolated Metrics**: No more import failures - each metric runs separately
- ✅ **CLI Interface**: Run individual metrics or complete evaluations from command line
- ✅ **Robust Error Handling**: Continue evaluation even if one metric fails
- ✅ **Statistical Analysis**: Pearson, Spearman, and Kendall correlations with significance testing
- ✅ **Rich Visualizations**: Correlation plots, heatmaps, and radar charts
- ✅ **Data Preprocessing**: Automatic z-standardization by evaluator
- ✅ **Comprehensive Output**: Detailed results in CSV and JSON formats
- ✅ **Custom Model Training**: Train your own COMET models with PyTorch Lightning

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
python -m src.processors.evaluation_runner --input data/results.csv --output metrics_results.csv
```

#### Run Without COMET (if installation issues)
```bash
python -m src.processors.evaluation_runner --input data/results.csv --output metrics_results.csv \
    --metrics BLEU CHRF++ TER ROUGE
```

#### Run Individual Metrics
```bash
# Test BLEU metric
python -m src.metrics.bleu_metric --input data/results.csv --output bleu_results.csv 

# Test CHRF++ metric  
python -m src.metrics.chrf_metric --input data/results.csv --output chrf_results.csv

# Test LLM-based evaluation
python -m src.metrics.llm_metric --mode reference_based --input data/results.csv --output llm_results.csv
```

## 🎓 Training Custom COMET Models

### Prerequisites

Download a pre-trained COMET checkpoint from HuggingFace:
```bash
# Example: Download XCOMET-XL
wget https://huggingface.co/Unbabel/wmt22-comet-da/resolve/main/checkpoints/model.ckpt?download=true
```

### Data Preparation

Prepare your training and validation data in the correct format:

```bash
python -m src.training.prepare_data \
    --input_train raw_train_data.csv \
    --input_valid raw_valid_data.csv \
    --output_train train.csv \
    --output_valid valid.csv
```

The input CSV files should contain the following columns:
- `sourceText`: Source language text
- `targetText`: Machine translation output
- `referenceText`: Reference translation
- `score` or `llm_reference_based_score`: Quality scores
- `createdBy_id`: Evaluator ID (for validation data)

### Training

Train a custom COMET model using the prepared data:

```bash
comet-train \
    --cfg src/training/configs/models/regression_model.yaml \
    --load_from_checkpoint path/to/checkpoint.cpkg
```

### Configuration

Customize training parameters by modifying the YAML files in `src/training/configs/`:

- **`models/regression_model.yaml`**: Main model configuration
  - Learning rates, batch size, dropout
  - Training and validation data paths
  - Loss function and optimizer settings

- **`trainer.yaml`**: PyTorch Lightning trainer settings
  - GPU/CPU selection
  - Number of epochs
  - Gradient accumulation
  - Validation frequency

- **`early_stopping.yaml`**: Early stopping configuration
  - Monitored metric (default: val_kendall)
  - Patience and mode

- **`model_checkpoint.yaml`**: Model checkpointing settings
  - Save criteria and frequency
  - Checkpoint naming pattern

#### Example: Modifying Training Parameters

To change batch size and learning rate, edit `src/training/configs/models/regression_model.yaml`:

```yaml
regression_metric:
  init_args:
    batch_size: 16  # Increase batch size
    learning_rate: 2e-05  # Adjust learning rate
    encoder_learning_rate: 2e-05
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
   - Can be customized through training

### LLM-based Metrics

6. **LLM-Reference-Based**
   - Uses LLM to evaluate translation quality with reference
   - Range: 0-100 (higher = better)
   - Good for: Nuanced quality assessment

7. **LLM-Reference-Free**
   - Uses LLM to evaluate translation quality without reference
   - Range: 0-100 (higher = better)
   - Good for: Cases where references are unavailable

## 🔧 Advanced Usage

### Custom Configuration

```bash
# Skip score standardization
python -m src.processors.evaluation_runner --input data/results.csv --output metrics_results.csv --no-standardize

# Adjust evaluator filtering threshold
python -m src.processors.evaluation_runner --input data/results.csv --output metrics_results.csv --min-evaluations 3

# COMET with custom parameters
python -m src.processors.evaluation_runner --input data/results.csv --output metrics_results.csv \
    --comet-batch-size 4 --comet-sample-size 100

# LLM evaluation with custom settings
python -m src.processors.evaluation_runner --input data/results.csv --output metrics_results.csv \
    --llm-model claude-sonnet-4-20250514 --llm-batch-size 50
```

### Creating Custom Metrics

```python
from src.metrics.base_metric import BaseMetric
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
│   ├── config/                          # Configuration modules
│   │   └── metrics_registry.py          # Central metrics registry
│   ├── data/                            # Data handling modules
│   │   ├── data_fetcher.py             # Database data fetching
│   │   └── db_connector.py             # Database connection management
│   ├── metrics/                         # Metric implementations
│   │   ├── base_metric.py               # Abstract base class for metrics
│   │   ├── bleu_metric.py               # BLEU implementation
│   │   ├── chrf_metric.py               # CHRF++ implementation
│   │   ├── comet_metric.py              # COMET implementation
│   │   ├── llm_metric.py                # LLM-based evaluation
│   │   ├── rouge_metric.py              # ROUGE implementation
│   │   └── ter_metric.py                # TER implementation
│   ├── processors/                      # Processing and evaluation modules
│   │   ├── csv_processor.py             # CSV processing with metrics
│   │   └── evaluation_runner.py         # Main evaluation orchestrator
│   ├── training/                        # COMET training modules
│   │   ├── prepare_data.py             # Data preparation script
│   │   └── configs/                    # Training configuration files
│   │       ├── early_stopping.yaml     # Early stopping settings
│   │       ├── model_checkpoint.yaml   # Checkpointing settings
│   │       ├── trainer.yaml            # PyTorch Lightning trainer config
│   │       └── models/
│   │           └── regression_model.yaml # COMET model configuration
│   └── utils/                           # Utility modules
│       ├── config.py                   # Configuration management
│       ├── llm_prompts.py              # LLM evaluation prompts
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

For training, outputs include:
- Model checkpoints in the specified directory
- Training logs and metrics
- Best model checkpoint based on validation performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add your metric to `src/metrics/`
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