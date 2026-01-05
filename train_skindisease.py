import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, models, transforms


def get_device() -> torch.device:
    """Return CUDA device if available, else MPS (Apple GPU) or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
):
    """
    Create train/val/test dataloaders from the SkinDisease folder structure.

    Expects:
        data_root / "SkinDisease" / "train"
        data_root / "SkinDisease" / "test"
    """
    train_dir = data_root / "SkinDisease" / "train"
    test_dir = data_root / "SkinDisease" / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train/test folders under {data_root / 'SkinDisease'}"
        )

    # Enhanced augmentation for better generalization - more aggressive
    input_size = 224
    train_transform = transforms.Compose(
        [
            transforms.Resize((280, 280)),  # Resize larger for better cropping
            transforms.RandomCrop(input_size),  # Random crop for better augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),  # Increased rotation
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.15, 0.15), 
                scale=(0.85, 1.15),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.4, 
                contrast=0.4, 
                saturation=0.4, 
                hue=0.15
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.RandomGrayscale(p=0.1),  # Sometimes convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.4), ratio=(0.3, 3.3)),
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

    # For validation, use eval transforms (no augmentation)
    # random_split keeps the transform from the original dataset,
    # so we override it here.
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

    return train_loader, val_loader, test_loader, full_train_dataset


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()
        
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal loss with label smoothing
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
        
        pt = (smooth_targets * probs).sum(dim=1)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def create_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Create a ResNet-101 model (more capacity than ResNet-50) with ImageNet-pretrained 
    backbone and a new classification head for the given number of classes.
    
    Args:
        freeze_backbone: If True, freeze backbone layers initially (for transfer learning)
    """
    # Use ResNet101 for better capacity and accuracy
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    
    # Freeze backbone layers initially
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.fc.in_features
    # Enhanced classification head with more capacity
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
    
    # Always unfreeze the new head
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def unfreeze_backbone_layers(model, epoch, total_epochs):
    """
    Progressively unfreeze backbone layers during training.
    This allows fine-tuning the entire network after initial head training.
    """
    # Check if backbone is still frozen
    backbone_frozen = not any(p.requires_grad for n, p in model.named_parameters() if 'fc' not in n)
    
    if epoch == total_epochs // 3 and backbone_frozen:  # Unfreeze at 1/3 of training
        print("Unfreezing backbone layers for fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_per_class(model, dataloader, class_names, device, csv_path: Path):
    """
    Compute per-class accuracy on the given dataloader and save as a CSV file.

    CSV columns:
        class_name, correct, total, accuracy
    """
    model.eval()
    num_classes = len(class_names)
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _, preds = outputs.max(1)

        for t, p in zip(targets, preds):
            total_per_class[int(t.item())] += 1
            if int(t.item()) == int(p.item()):
                correct_per_class[int(t.item())] += 1

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "correct", "total", "accuracy_percent"])
        for idx, name in enumerate(class_names):
            total = total_per_class[idx]
            correct = correct_per_class[idx]
            acc = (correct / total * 100.0) if total > 0 else 0.0
            writer.writerow([name, correct, total, f"{acc:.2f}"])

    print(f"Per-class accuracy CSV written to: {csv_path}")


def main():
    # Adjust this if you move the script
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "SkinDisease"

    device = get_device()
    print(f"Using device: {device}")

    batch_size = 32  # Optimal batch size for ResNet50
    num_epochs = 50  # Increased from 30 for better convergence
    learning_rate = 2e-4  # Slightly higher initial LR
    early_stopping_patience = 10  # Stop if no improvement for 10 epochs

    # get full_train_dataset so we can compute class weights
    train_loader, val_loader, test_loader, full_train_dataset = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        val_split=0.15,
        num_workers=4,
    )

    class_names = full_train_dataset.classes
    print(f"Found {len(class_names)} classes:")
    print(class_names)

    model = create_model(num_classes=len(class_names), freeze_backbone=True).to(device)

    # --- Enhanced loss: Focal Loss with class weights ---
    targets_tensor = torch.tensor(full_train_dataset.targets)
    class_counts = torch.bincount(targets_tensor)
    class_weights = (class_counts.sum() / class_counts).to(torch.float32)
    class_weights = class_weights.to(device)
    
    # Use Focal Loss with label smoothing for better handling of hard examples
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    
    # Separate learning rates for backbone and head
    optimizer = torch.optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 
             'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': [p for n, p in model.named_parameters() if 'fc' in n], 
             'lr': learning_rate}  # Higher LR for head
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
    
    # Use CosineAnnealingWarmRestarts for better learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=8,  # Initial restart period (adjusted for 30 epochs)
        T_mult=2,  # Period multiplier
        eta_min=1e-6,  # Minimum learning rate
    )
    
    # Also use ReduceLROnPlateau as backup
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_acc = 0.0
    best_model_path = project_root / "best_skindisease_model.pt"
    patience_counter = 0
    target_accuracy = 0.75  # Target: 75% accuracy

    print(f"\n🎯 Training Goal: Achieve >{target_accuracy*100:.0f}% accuracy")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Progressive unfreezing
        unfreeze_backbone_layers(model, epoch, num_epochs)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm=1.0
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update both schedulers
        scheduler.step()
        plateau_scheduler.step(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"✅ New best model saved! Val accuracy: {best_val_acc * 100:.2f}%"
            )
            
            # Check if we reached target
            if best_val_acc >= target_accuracy:
                print(f"\n🎉 TARGET ACHIEVED! Accuracy: {best_val_acc * 100:.2f}% >= {target_accuracy*100:.0f}%")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience and best_val_acc >= target_accuracy:
            print(f"\n✅ Early stopping: Target accuracy achieved and no improvement for {early_stopping_patience} epochs")
            break

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    if best_val_acc >= target_accuracy:
        print(f"✅ SUCCESS: Achieved target accuracy of {target_accuracy*100:.0f}%!")
    else:
        print(f"⚠️  Target accuracy ({target_accuracy*100:.0f}%) not reached. Current: {best_val_acc * 100:.2f}%")
    print("=" * 60)

    # Load best model for testing
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    # Compute and save per-class accuracy on the test set
    per_class_csv = project_root / "per_class_accuracy.csv"
    evaluate_per_class(model, test_loader, class_names, device, per_class_csv)


if __name__ == "__main__":
    main()


