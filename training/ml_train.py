"""
Train classical ML models (SVM, RandomForest, LogisticRegression) on handcrafted
features extracted from the SkinDisease image dataset.

This effectively creates "Dataset A" (tabular features) from your image dataset
and trains multiple models, saving them under the models/ folder.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from training.utils import (
    get_project_root,
    get_skin_image_root,
    extract_color_histogram,
    load_image_for_features,
)


def build_dataset_a() -> pd.DataFrame:
    """
    Walk the SkinDisease/train directory, extract handcrafted features from
    each image, and return a DataFrame with columns:
        feature_0 ... feature_N, label

    This is your "Dataset A" for classical ML.
    """
    image_root = get_skin_image_root() / "train"
    rows = []

    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in class_dir.glob("*.jpeg"):
            try:
                img = load_image_for_features(img_path)
                feats = extract_color_histogram(img)
                row = dict(
                    **{f"f_{i}": float(v) for i, v in enumerate(feats)},
                    label=label,
                )
                rows.append(row)
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    df = pd.DataFrame(rows)
    return df


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def train_and_evaluate_ml():
    project_root = get_project_root()
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    csv_path = dataset_dir / "dataset_a_features.csv"

    if csv_path.exists():
        print(f"Loading existing Dataset A from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Building Dataset A from images (this may take a while)...")
        df = build_dataset_a()
        df.to_csv(csv_path, index=False)
        print(f"Saved Dataset A to {csv_path}")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "svm": SVC(kernel="rbf", probability=True, C=2.0, gamma="scale"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced_subsample"
        ),
        "log_reg": LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs"
        ),
    }

    metrics_dir = project_root / "training" / "ml_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    unique_labels = sorted(np.unique(y_train))

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc * 100:.2f}%")
        report = classification_report(y_test, y_pred, target_names=unique_labels)
        print(report)

        # Save model
        model_path = models_dir / f"{name}_skin_ml.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")

        # Save metrics text report
        report_path = metrics_dir / f"{name}_classification_report.txt"
        with report_path.open("w") as f:
            f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
            f.write(report)

        # Save confusion matrix plot
        cm_path = metrics_dir / f"{name}_confusion_matrix.png"
        save_confusion_matrix(
            y_test, y_pred, labels=unique_labels, out_path=cm_path, title=f"{name} Confusion Matrix"
        )
        print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    train_and_evaluate_ml()


