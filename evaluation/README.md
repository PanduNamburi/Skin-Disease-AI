# Model Evaluation

This directory contains scripts and results for evaluating the trained skin disease detection model.

## Quick Start

Run the evaluation script from the project root:

```bash
python evaluation/evaluate_model.py
```

Or from within this directory:

```bash
cd evaluation
python evaluate_model.py
```

## Output Files

All results are saved to `evaluation/results/` directory:

### Visualizations
- **`confusion_matrix.png`** - Visual confusion matrix showing both raw counts and normalized percentages

### Data Files
- **`confusion_matrix.csv`** - Confusion matrix as CSV (rows = true labels, columns = predicted labels)
- **`metrics_per_class.csv`** - Per-class metrics (Precision, Recall, F1 Score, Support) in CSV format
- **`metrics_summary.json`** - Complete metrics in JSON format for programmatic access
- **`evaluation_metrics.txt`** - Human-readable detailed report

## Metrics Included

### Overall Metrics
- **Accuracy** - Overall percentage of correct predictions

### Average Metrics
- **Weighted Average** - Accounts for class imbalance
- **Macro Average** - Simple average across all classes
- **Micro Average** - Global calculation

### Per-Class Metrics
- **Precision** - Percentage of positive predictions that were correct
- **Recall** - Percentage of actual positives correctly identified
- **F1 Score** - Harmonic mean of precision and recall
- **Support** - Number of samples for each class

## Using the Data Files

### CSV Files
The CSV files can be opened in Excel, Google Sheets, or any data analysis tool:

```python
import pandas as pd

# Load per-class metrics
df = pd.read_csv('evaluation/results/metrics_per_class.csv')
print(df)

# Load confusion matrix
cm = pd.read_csv('evaluation/results/confusion_matrix.csv', index_col=0)
print(cm)
```

### JSON File
The JSON file provides structured access to all metrics:

```python
import json

with open('evaluation/results/metrics_summary.json', 'r') as f:
    metrics = json.load(f)

print(f"Overall Accuracy: {metrics['overall']['accuracy_percent']:.2f}%")
print(f"Weighted F1 Score: {metrics['averages']['weighted']['f1_score_percent']:.2f}%")

# Access per-class metrics
for class_metric in metrics['per_class']:
    print(f"{class_metric['class_name']}: F1={class_metric['f1_score_percent']:.2f}%")
```

## Requirements

All required packages are in the main `requirements.txt`:
- torch, torchvision
- scikit-learn
- matplotlib, seaborn
- numpy, pandas
