"""
Evaluate the trained skin disease model and generate comprehensive metrics:
- Confusion Matrix
- Accuracy
- Precision (per-class and weighted/macro averages)
- Recall (per-class and weighted/macro averages)
- F1 Score (per-class and weighted/macro averages)

All results are saved to the evaluation/results/ directory in multiple formats:
- confusion_matrix.png (visualization)
- evaluation_metrics.txt (human-readable report)
- metrics_per_class.csv (structured CSV data)
- metrics_summary.json (structured JSON data)
- confusion_matrix.csv (confusion matrix as CSV)
"""

import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


def get_device() -> torch.device:
    """Return CUDA device if available, else MPS (Apple GPU) or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_test_directory(project_root: Path) -> Path:
    """
    Find the test directory by checking common dataset locations.
    Returns the test directory path if found, raises FileNotFoundError otherwise.
    """
    possible_locations = [
        project_root / "SkinDisease" / "SkinDisease" / "test",  # Nested structure
        project_root / "SkinDisease" / "test",  # Direct structure
        project_root / "test",  # Root level
    ]
    
    for test_dir in possible_locations:
        if test_dir.is_dir():
            return test_dir
    
    # If not found, provide helpful error message
    raise FileNotFoundError(
        f"Could not find test dataset directory. Checked:\n"
        + "\n".join(f"  - {loc}" for loc in possible_locations)
        + f"\n\nPlease ensure your dataset test folder exists in one of these locations."
    )


def create_test_dataloader(test_dir: Path, batch_size: int = 32, num_workers: int = 4):
    """
    Create test dataloader from the test directory.
    """
    
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    input_size = 224
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader, test_dataset


def create_model(num_classes: int) -> nn.Module:
    """
    Create a ResNet-101 model matching the architecture used in training.
    """
    model = models.resnet101(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate the model and return predictions and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating model on test set...")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """
    Create and save a confusion matrix visualization.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)
    
    # Plot 2: Normalized percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Percentage (%)'}
    )
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_title("Confusion Matrix (Normalized %)", fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix saved to: {save_path}")


def save_confusion_matrix_csv(cm, class_names, save_path: Path):
    """
    Save confusion matrix as CSV file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(save_path)
    print(f"Confusion matrix CSV saved to: {save_path}")


def save_metrics_csv(y_true, y_pred, class_names, precision, recall, f1, support, save_path: Path):
    """
    Save per-class metrics as CSV file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'Class': class_names,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1_Score': f1 * 100,
        'Support': support
    }
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Per-class metrics CSV saved to: {save_path}")


def save_metrics_json(y_true, y_pred, class_names, save_path: Path):
    """
    Save all metrics as JSON file for easy programmatic access.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Build JSON structure
    metrics_data = {
        'overall': {
            'accuracy': float(accuracy),
            'accuracy_percent': float(accuracy * 100)
        },
        'averages': {
            'weighted': {
                'precision': float(precision_weighted),
                'precision_percent': float(precision_weighted * 100),
                'recall': float(recall_weighted),
                'recall_percent': float(recall_weighted * 100),
                'f1_score': float(f1_weighted),
                'f1_score_percent': float(f1_weighted * 100)
            },
            'macro': {
                'precision': float(precision_macro),
                'precision_percent': float(precision_macro * 100),
                'recall': float(recall_macro),
                'recall_percent': float(recall_macro * 100),
                'f1_score': float(f1_macro),
                'f1_score_percent': float(f1_macro * 100)
            },
            'micro': {
                'precision': float(precision_micro),
                'precision_percent': float(precision_micro * 100),
                'recall': float(recall_micro),
                'recall_percent': float(recall_micro * 100),
                'f1_score': float(f1_micro),
                'f1_score_percent': float(f1_micro * 100)
            }
        },
        'per_class': [
            {
                'class_name': class_name,
                'precision': float(precision[i]),
                'precision_percent': float(precision[i] * 100),
                'recall': float(recall[i]),
                'recall_percent': float(recall[i] * 100),
                'f1_score': float(f1[i]),
                'f1_score_percent': float(f1[i] * 100),
                'support': int(support[i])
            }
            for i, class_name in enumerate(class_names)
        ],
        'confusion_matrix': {
            'classes': class_names,
            'matrix': cm.tolist()
        }
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics JSON saved to: {save_path}")


def save_metrics_report(y_true, y_pred, class_names, save_path: Path):
    """
    Calculate and save comprehensive metrics report as text file.
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class and average metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Weighted averages (accounts for class imbalance)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Macro averages (simple average across classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Micro average (same as accuracy for multiclass)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    # Save to file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("SKIN DISEASE MODEL EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        
        f.write("AVERAGE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Weighted Average (accounts for class imbalance):\n")
        f.write(f"  Precision: {precision_weighted * 100:.2f}%\n")
        f.write(f"  Recall: {recall_weighted * 100:.2f}%\n")
        f.write(f"  F1 Score: {f1_weighted * 100:.2f}%\n\n")
        
        f.write(f"Macro Average (simple average across classes):\n")
        f.write(f"  Precision: {precision_macro * 100:.2f}%\n")
        f.write(f"  Recall: {recall_macro * 100:.2f}%\n")
        f.write(f"  F1 Score: {f1_macro * 100:.2f}%\n\n")
        
        f.write(f"Micro Average:\n")
        f.write(f"  Precision: {precision_micro * 100:.2f}%\n")
        f.write(f"  Recall: {recall_micro * 100:.2f}%\n")
        f.write(f"  F1 Score: {f1_micro * 100:.2f}%\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(
                f"{class_name:<25} {precision[i]*100:>10.2f}% {recall[i]*100:>10.2f}% "
                f"{f1[i]*100:>10.2f}% {support[i]:>10}\n"
            )
        f.write("\n")
        
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")
        f.write(report)
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Metrics report saved to: {save_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"\nWeighted Averages:")
    print(f"  Precision: {precision_weighted * 100:.2f}%")
    print(f"  Recall: {recall_weighted * 100:.2f}%")
    print(f"  F1 Score: {f1_weighted * 100:.2f}%")
    print(f"\nMacro Averages:")
    print(f"  Precision: {precision_macro * 100:.2f}%")
    print(f"  Recall: {recall_macro * 100:.2f}%")
    print(f"  F1 Score: {f1_macro * 100:.2f}%")
    print("=" * 80 + "\n")
    
    return precision, recall, f1, support


def main():
    import sys
    
    # Setup paths - adjust for script location in evaluation/ directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Allow custom dataset path via command line argument
    custom_dataset_path = None
    if len(sys.argv) > 1:
        custom_dataset_path = Path(sys.argv[1])
        if not custom_dataset_path.exists():
            print(f"Error: Custom dataset path does not exist: {custom_dataset_path}")
            return
        print(f"Using custom dataset path: {custom_dataset_path}")
    
    model_path = project_root / "best_skindisease_model.pt"
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train_skindisease.py")
        return
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Find test directory
    if custom_dataset_path:
        # Use custom path - check if it's a test directory or contains test subdirectory
        if (custom_dataset_path / "test").is_dir():
            test_dir = custom_dataset_path / "test"
        elif custom_dataset_path.name == "test" and custom_dataset_path.is_dir():
            test_dir = custom_dataset_path
        else:
            print(f"Error: Custom path must be a test directory or contain a 'test' subdirectory")
            print(f"Path provided: {custom_dataset_path}")
            return
        print(f"Using custom test dataset at: {test_dir}")
    else:
        try:
            test_dir = find_test_directory(project_root)
            print(f"Found test dataset at: {test_dir}")
        except FileNotFoundError as e:
            print(f"\n{'='*80}")
            print("DATASET NOT FOUND")
            print(f"{'='*80}")
            print(str(e))
            print(f"\nTo use a custom dataset path, run:")
            print(f"  python evaluation/evaluate_model.py /path/to/your/dataset")
            print(f"  (where /path/to/your/dataset contains a 'test' folder)")
            print(f"{'='*80}\n")
            return
    
    # Create test dataloader
    test_loader, test_dataset = create_test_dataloader(
        test_dir=test_dir,
        batch_size=32,
        num_workers=4,
    )
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"\nFound {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Create and load model
    print(f"\nLoading model from {model_path}...")
    model = create_model(num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative model architectures...")
        
        # Try ResNet50
        try:
            model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_classes),
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded as ResNet50!")
        except Exception:
            # Try ResNet18
            try:
                model = models.resnet18(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, num_classes),
                ).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Loaded as ResNet18!")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                return
    
    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # Create output directory in evaluation/results/
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate and save confusion matrix visualization
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Save confusion matrix as CSV
    cm_csv_path = output_dir / "confusion_matrix.csv"
    save_confusion_matrix_csv(cm, class_names, cm_csv_path)
    
    # Generate and save metrics report (text)
    metrics_path = output_dir / "evaluation_metrics.txt"
    precision, recall, f1, support = save_metrics_report(y_true, y_pred, class_names, metrics_path)
    
    # Save per-class metrics as CSV
    metrics_csv_path = output_dir / "metrics_per_class.csv"
    save_metrics_csv(y_true, y_pred, class_names, precision, recall, f1, support, metrics_csv_path)
    
    # Save all metrics as JSON
    metrics_json_path = output_dir / "metrics_summary.json"
    save_metrics_json(y_true, y_pred, class_names, metrics_json_path)
    
    print(f"\n{'='*80}")
    print(f"All evaluation results saved to: {output_dir}")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print(f"  - confusion_matrix.png (visualization)")
    print(f"  - confusion_matrix.csv (confusion matrix data)")
    print(f"  - evaluation_metrics.txt (detailed text report)")
    print(f"  - metrics_per_class.csv (per-class metrics)")
    print(f"  - metrics_summary.json (all metrics in JSON format)")


if __name__ == "__main__":
    main()
