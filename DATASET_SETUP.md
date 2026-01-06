# Dataset Setup Guide

**You cannot train the model without the dataset.** This guide will help you obtain and set up your dataset.

## Quick Answer

You have **3 options**:

1. **Find your existing dataset** (if you trained before)
2. **Download a new dataset** (from Kaggle or other sources)
3. **Use the existing trained model** (skip training, just evaluate)

---

## Option 1: Find Your Existing Dataset

Since you have `best_skindisease_model.pt`, you likely trained before. Let's find where your dataset might be:

### Check Common Locations

```bash
# Check Desktop
find ~/Desktop -type d -name "*Skin*" -o -name "*skin*" 2>/dev/null

# Check Downloads
find ~/Downloads -type d -name "*Skin*" -o -name "*skin*" 2>/dev/null

# Check Documents
find ~/Documents -type d -name "*Skin*" -o -name "*skin*" 2>/dev/null

# Check entire home directory (may take a while)
find ~ -maxdepth 4 -type d -name "*Skin*" 2>/dev/null | head -20
```

### If You Find It

Once you locate your dataset, set it up:

```bash
# Create the expected directory structure
mkdir -p SkinDisease/SkinDisease

# Move or link your dataset
# If found at /path/to/your/dataset:
mv /path/to/your/dataset/* SkinDisease/SkinDisease/

# OR create a symbolic link (saves disk space):
ln -s /path/to/your/dataset SkinDisease/SkinDisease
```

---

## Option 2: Download a Dataset

### Popular Skin Disease Datasets

#### A. Kaggle Datasets

1. **Skin Disease Classification Dataset**
   - Search Kaggle for: "skin disease classification"
   - Popular datasets:
     - "Skin Disease Classification Dataset"
     - "Dermatology Dataset"
     - "Skin Lesion Classification"

2. **How to Download from Kaggle:**
   ```bash
   # Install Kaggle API (if not installed)
   pip install kaggle
   
   # Set up Kaggle API credentials (get from kaggle.com/account)
   # Place kaggle.json in ~/.kaggle/
   
   # Download dataset (example)
   kaggle datasets download -d dataset-name
   unzip dataset-name.zip
   ```

#### B. ISIC Archive (International Skin Imaging Collaboration)
- Website: https://www.isic-archive.com/
- Large collection of skin lesion images
- Requires registration

#### C. HAM10000 Dataset
- Popular skin cancer classification dataset
- Available on Kaggle and other repositories
- Contains 10,000+ images

### After Downloading

Organize your dataset to match the expected structure:

```
SkinDisease/
└── SkinDisease/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── ...
```

**Important**: Each disease class must be in its own folder!

---

## Option 3: Use Existing Model (Skip Training)

If you just want to **evaluate** your existing model and don't need to retrain:

1. **You only need the TEST dataset** (not the full training set)
2. Place just the test folder:
   ```bash
   mkdir -p SkinDisease/SkinDisease
   # Copy only test folder
   cp -r /path/to/test/folder SkinDisease/SkinDisease/test
   ```
3. Run evaluation:
   ```bash
   python evaluation/evaluate_model.py
   ```

---

## Verify Dataset Structure

After setting up your dataset, verify it's correct:

```bash
# Check structure
ls -la SkinDisease/SkinDisease/
# Should show: train/ and test/

# Check train folder has class directories
ls SkinDisease/SkinDisease/train/
# Should show: class1/, class2/, etc.

# Check test folder has class directories
ls SkinDisease/SkinDisease/test/
# Should show: class1/, class2/, etc.

# Check images exist
ls SkinDisease/SkinDisease/train/class1/ | head -5
# Should show image files (.jpg, .jpeg, .png)
```

---

## Quick Setup Script

I've created a helper script (`setup_dataset.py`) that can help you:
- Check if dataset exists
- Guide you through setup
- Verify dataset structure

Run it:
```bash
python setup_dataset.py
```

---

## Common Issues

### "Dataset not found" error
- Make sure path is: `SkinDisease/SkinDisease/train/` and `test/`
- Check folder names are exactly `train` and `test` (lowercase)

### "No images found" error
- Verify images are in class subfolders, not directly in train/test
- Check image formats (.jpg, .jpeg, .png are supported)

### Dataset too large
- Use symbolic links instead of copying
- Consider using only a subset for testing

---

## Next Steps

Once your dataset is set up:

1. **Verify structure** (see above)
2. **Train the model**:
   ```bash
   source venv/bin/activate
   python train_skindisease.py
   ```
3. **Evaluate the model**:
   ```bash
   python evaluation/evaluate_model.py
   ```

---

## Need Help?

- Check `TRAINING_GUIDE.md` for detailed training instructions
- Check `EVALUATION_GUIDE.md` for evaluation instructions
- Run `python setup_dataset.py` for interactive help
