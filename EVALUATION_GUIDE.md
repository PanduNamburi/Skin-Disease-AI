# Model Evaluation Guide

This guide explains how to evaluate your trained skin disease model and generate comprehensive metrics.

## Quick Start

Run the evaluation script from the project root:

```bash
python evaluation/evaluate_model.py
```

Or navigate to the evaluation directory:

```bash
cd evaluation
python evaluate_model.py
```

## What the Script Does

The evaluation script (`evaluation/evaluate_model.py`) will:

1. **Load your trained model** from `best_skindisease_model.pt`
2. **Evaluate on the test set** from `SkinDisease/SkinDisease/test/`
3. **Generate metrics**:
   - Overall Accuracy
   - Precision (per-class, weighted, macro, micro)
   - Recall (per-class, weighted, macro, micro)
   - F1 Score (per-class, weighted, macro, micro)
4. **Create visualizations**:
   - Confusion Matrix (both counts and normalized percentages)
5. **Export data in multiple formats**:
   - CSV files for easy analysis
   - JSON file for programmatic access
   - Text report for human reading

## Output Files

After running the script, you'll find results in the `evaluation/results/` directory:

### Visualizations
- `confusion_matrix.png` - Visual confusion matrix with counts and percentages

### Data Files
- `confusion_matrix.csv` - Confusion matrix as CSV
- `metrics_per_class.csv` - Per-class metrics in CSV format
- `metrics_summary.json` - Complete metrics in JSON format
- `evaluation_metrics.txt` - Detailed text report with all metrics

## Metrics Explained

### Accuracy
The overall percentage of correct predictions.

### Precision
The percentage of positive predictions that were actually correct. High precision means fewer false positives.

### Recall (Sensitivity)
The percentage of actual positives that were correctly identified. High recall means fewer false negatives.

### F1 Score
The harmonic mean of precision and recall. Provides a balanced measure of model performance.

### Weighted vs Macro vs Micro Averages

- **Weighted Average**: Accounts for class imbalance by weighting each class by its support (number of samples)
- **Macro Average**: Simple average across all classes (treats all classes equally)
- **Micro Average**: Calculates metrics globally by counting total true positives, false negatives, and false positives

## Requirements

All required packages are already in `requirements.txt`:
- torch, torchvision
- scikit-learn
- matplotlib, seaborn
- numpy

## Troubleshooting

### Model Not Found
If you get an error about the model file not being found:
- Make sure you've trained the model first using `train_skindisease.py`
- The script looks for `best_skindisease_model.pt` in the project root

### Dataset Not Found
If you get an error about the dataset:
- Make sure your dataset is located at `SkinDisease/SkinDisease/test/`
- The test folder should contain subdirectories for each disease class

### Architecture Mismatch
The script automatically tries different ResNet architectures (101, 50, 18) if the initial load fails. If all fail, check that your model architecture matches what was used during training.
