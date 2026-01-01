"""
Train deep learning models (custom CNN and ResNet18 transfer learning)
on the SkinDisease image dataset and save them under models/.
Also computes accuracy, precision, recall, F1, and confusion matrices.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from training.utils import get_skin_image_root, get_project_root


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
):
    root = get_skin_image_root()
    train_dir = root / "train"
    test_dir = root / "test"

    input_size = 224
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
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

    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    num_train = len(full_train_dataset)
    num_val = int(val_split * num_train)
    num_train = num_train - num_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_train_dataset.classes


class SimpleCNN(nn.Module):
    """Small custom CNN for educational purposes."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_resnet_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    total = len(y_true)
    loss = running_loss / total
    acc = accuracy_score(y_true, y_pred)
    return loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)

        preds = outputs.argmax(dim=1)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    total = len(y_true)
    loss = running_loss / total
    acc = accuracy_score(y_true, y_pred)
    return loss, acc, np.array(y_true), np.array(y_pred)


def save_dl_metrics(
    y_true,
    y_pred,
    class_names,
    model_name: str,
    metrics_dir: Path,
):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Text metrics
    txt_path = metrics_dir / f"{model_name}_metrics.txt"
    with txt_path.open("w") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Precision (weighted): {prec * 100:.2f}%\n")
        f.write(f"Recall (weighted): {rec * 100:.2f}%\n")
        f.write(f"F1 (weighted): {f1 * 100:.2f}%\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig_path = metrics_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(fig_path)
    plt.close(fig)


def train_deep_models():
    device = get_device()
    print(f"Using device: {device}")

    batch_size = 32
    num_epochs = 10

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=batch_size
    )
    num_classes = len(class_names)

    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    metrics_dir = project_root / "training" / "dl_metrics"

    # --- Train Simple CNN ---
    print("\n=== Training Simple CNN ===")
    cnn = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\n[CNN] Epoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            cnn, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(cnn, val_loader, criterion, device)
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(cnn.state_dict(), models_dir / "simple_cnn_skindisease.pt")
            print(
                f"Saved best Simple CNN with val acc: {best_val_acc * 100:.2f}%"
            )

    # Evaluate on test set
    if (models_dir / "simple_cnn_skindisease.pt").exists():
        cnn.load_state_dict(
            torch.load(models_dir / "simple_cnn_skindisease.pt", map_location=device)
        )
    test_loss, test_acc, y_true, y_pred = evaluate(
        cnn, test_loader, criterion, device
    )
    print(f"[CNN] Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    save_dl_metrics(
        y_true, y_pred, class_names, "simple_cnn", metrics_dir=metrics_dir
    )

    # --- Train ResNet18 Transfer Learning ---
    print("\n=== Training ResNet18 (Transfer Learning) ===")
    resnet = create_resnet_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(resnet.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_acc = 0.0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, num_epochs + 1):
        print(f"\n[ResNet18] Epoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            resnet, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(resnet, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(resnet.state_dict(), models_dir / "resnet18_skindisease.pt")
            print(
                f"Saved best ResNet18 with val acc: {best_val_acc * 100:.2f}%"
            )

    if (models_dir / "resnet18_skindisease.pt").exists():
        resnet.load_state_dict(
            torch.load(models_dir / "resnet18_skindisease.pt", map_location=device)
        )
    test_loss, test_acc, y_true, y_pred = evaluate(
        resnet, test_loader, criterion, device
    )
    print(f"[ResNet18] Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    save_dl_metrics(
        y_true, y_pred, class_names, "resnet18", metrics_dir=metrics_dir
    )


if __name__ == "__main__":
    train_deep_models()


