# Model Training Guide

This guide explains how to train your skin disease detection model.

## Prerequisites

1. **Dataset**: You need a skin disease image dataset organized in a specific folder structure
2. **Python Environment**: Virtual environment with required packages installed
3. **Hardware**: GPU recommended (CUDA/MPS) but CPU will work (slower)

## Dataset Structure

Your dataset must be organized as follows:

```
SkinDisease/
└── SkinDisease/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   └── ...
    │   └── ...
    └── test/
        ├── class1/
        │   └── ...
        ├── class2/
        │   └── ...
        └── ...
```

**Important**: 
- Each disease class should be in its own folder
- Images can be `.jpg`, `.jpeg`, or `.png` format
- The `train` folder is used for training (will be split into train/validation)
- The `test` folder is used for final evaluation

## Step 1: Set Up Dataset

### Option A: If you have the dataset elsewhere

```bash
# Create the directory structure
mkdir -p SkinDisease/SkinDisease

# Move or link your dataset
# If your dataset is at /path/to/your/dataset:
mv /path/to/your/dataset/* SkinDisease/SkinDisease/

# OR create a symbolic link:
ln -s /path/to/your/dataset SkinDisease/SkinDisease
```

### Option B: Download a dataset

If you need to download a skin disease dataset, common sources include:
- Kaggle datasets (search for "skin disease classification")
- Medical image repositories
- Research datasets

Make sure the dataset follows the folder structure above.

## Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

## Step 3: Verify Dataset Location

Check that your dataset is in the correct location:

```bash
# Should show train and test directories
ls -la SkinDisease/SkinDisease/

# Should show class folders
ls SkinDisease/SkinDisease/train/
ls SkinDisease/SkinDisease/test/
```

## Step 4: Run Training

### Main Training Script (ResNet101)

```bash
python train_skindisease.py
```

This script will:
- Load images from `SkinDisease/SkinDisease/train/` and `test/`
- Split training data into train/validation (85%/15%)
- Train a ResNet101 model with transfer learning
- Use Focal Loss for handling class imbalance
- Save the best model to `best_skindisease_model.pt`
- Display training progress and metrics

### Training Parameters

The script uses these default settings (you can modify in `train_skindisease.py`):

- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate**: 2e-4
- **Early Stopping**: 10 epochs patience
- **Target Accuracy**: 75%

### Alternative Training Scripts

You can also train other models:

```bash
# Train Simple CNN and ResNet18 models
python training/dl_train.py

# Train classical ML models (SVM, RandomForest, LogisticRegression)
python training/ml_train.py
```

## Step 5: Monitor Training

During training, you'll see output like:

```
Epoch 1/50
------------------------------------------------------------
Train Loss: 1.2345 | Train Acc: 45.67% | Val Loss: 1.1234 | Val Acc: 48.90%
Current Learning Rate: 0.000200
✅ New best model saved! Val accuracy: 48.90%
```

## Step 6: Training Outputs

After training completes:

- **Best Model**: `best_skindisease_model.pt` (saved automatically)
- **Per-Class Accuracy**: `per_class_accuracy.csv` (if generated)
- **Training Logs**: Console output with metrics

## Troubleshooting

### Error: "Expected train/test folders under..."

**Solution**: Make sure your dataset structure matches exactly:
```
SkinDisease/SkinDisease/train/
SkinDisease/SkinDisease/test/
```

### Error: "No module named 'torch'"

**Solution**: Activate your virtual environment:
```bash
source venv/bin/activate
```

### Training is too slow

**Solutions**:
- Use GPU if available (CUDA for NVIDIA, MPS for Apple Silicon)
- Reduce batch size (edit `batch_size` in script)
- Reduce number of epochs
- Use a smaller model (ResNet18 instead of ResNet101)

### Out of Memory Error

**Solutions**:
- Reduce batch size (e.g., from 32 to 16 or 8)
- Use a smaller model
- Reduce image size in transforms

## Training Tips

1. **GPU Usage**: Training is much faster on GPU. The script will automatically detect and use:
   - CUDA (NVIDIA GPUs)
   - MPS (Apple Silicon Macs)
   - CPU (fallback)

2. **Early Stopping**: The model stops training if validation accuracy doesn't improve for 10 epochs

3. **Model Checkpoints**: The best model (highest validation accuracy) is automatically saved

4. **Class Imbalance**: The script uses Focal Loss with class weights to handle imbalanced datasets

5. **Data Augmentation**: Extensive augmentation is applied during training to improve generalization

## Next Steps After Training

Once training is complete:

1. **Evaluate the model**:
   ```bash
   python evaluation/evaluate_model.py
   ```

2. **Use the model** in your Django application - it will automatically load `best_skindisease_model.pt`

3. **Check metrics** in `evaluation/results/` directory

## Training Time Estimates

- **CPU**: ~2-4 hours per epoch (50 epochs = 100-200 hours)
- **GPU (CUDA)**: ~5-10 minutes per epoch (50 epochs = 4-8 hours)
- **MPS (Apple Silicon)**: ~10-20 minutes per epoch (50 epochs = 8-16 hours)

*Times vary based on dataset size and hardware*
