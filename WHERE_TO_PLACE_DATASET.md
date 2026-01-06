# Where to Place Your Dataset

## Exact Location

Place your dataset in this **exact location**:

```
/Users/pandunamburi/Desktop/Skin-Disease-AI-main/
└── SkinDisease/
    └── SkinDisease/
        ├── train/
        │   ├── class1/
        │   │   ├── image1.jpg
        │   │   └── image2.jpg
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

## Step-by-Step Instructions

### Option 1: If you have a dataset folder ready

```bash
# Navigate to your project directory
cd /Users/pandunamburi/Desktop/Skin-Disease-AI-main

# Create the directory structure
mkdir -p SkinDisease/SkinDisease

# If your dataset is already organized with train/test folders:
# Move it to the expected location
mv /path/to/your/dataset/* SkinDisease/SkinDisease/

# OR if your dataset IS the train/test structure:
# Move the entire folder
mv /path/to/your/dataset SkinDisease/SkinDisease
```

### Option 2: Create symbolic link (if dataset is large/elsewhere)

```bash
# Create directory structure
mkdir -p SkinDisease

# Create symbolic link (saves disk space, doesn't copy files)
ln -s /path/to/your/dataset SkinDisease/SkinDisease
```

### Option 3: Manual folder creation

1. **In your project root** (`/Users/pandunamburi/Desktop/Skin-Disease-AI-main/`):
   - Create folder: `SkinDisease`
   - Inside `SkinDisease`, create folder: `SkinDisease`
   - Inside that, create folders: `train` and `test`
   - Place your class folders inside `train/` and `test/`

## Visual Guide

Your project structure should look like this:

```
Skin-Disease-AI-main/                    ← Project root
├── SkinDisease/                        ← Create this
│   └── SkinDisease/                    ← Create this
│       ├── train/                      ← Your training images go here
│       │   ├── Acne/
│       │   ├── Eczema/
│       │   ├── Psoriasis/
│       │   └── ... (other disease classes)
│       └── test/                       ← Your test images go here
│           ├── Acne/
│           ├── Eczema/
│           ├── Psoriasis/
│           └── ... (same classes as train)
├── train_skindisease.py
├── evaluation/
├── models/
└── ... (other project files)
```

## Quick Setup Commands

Run these commands in your project directory:

```bash
# Navigate to project
cd /Users/pandunamburi/Desktop/Skin-Disease-AI-main

# Create structure
mkdir -p SkinDisease/SkinDisease/train
mkdir -p SkinDisease/SkinDisease/test

# Now copy or move your dataset folders into:
# - SkinDisease/SkinDisease/train/  (for training images)
# - SkinDisease/SkinDisease/test/    (for test images)
```

## Verify It's Correct

After placing your dataset, verify with:

```bash
# Check structure exists
ls -la SkinDisease/SkinDisease/

# Should show:
# train/  test/

# Check train folder has classes
ls SkinDisease/SkinDisease/train/

# Should show class folders like:
# Acne/  Eczema/  Psoriasis/  ...

# Check test folder has classes
ls SkinDisease/SkinDisease/test/

# Should show same class folders

# Check images exist
ls SkinDisease/SkinDisease/train/Acne/ | head -3

# Should show image files (.jpg, .jpeg, .png)
```

## Important Notes

1. **Exact Path**: The path must be exactly `SkinDisease/SkinDisease/` (nested)
2. **Folder Names**: Must be `train` and `test` (lowercase)
3. **Class Folders**: Each disease class must be in its own folder
4. **Images**: Images can be `.jpg`, `.jpeg`, or `.png`

## Example

If your dataset is currently at `/Users/pandunamburi/Downloads/my_skin_dataset/`:

```bash
cd /Users/pandunamburi/Desktop/Skin-Disease-AI-main
mkdir -p SkinDisease/SkinDisease
mv /Users/pandunamburi/Downloads/my_skin_dataset/* SkinDisease/SkinDisease/
```

Or if your dataset already has `train/` and `test/` folders:

```bash
cd /Users/pandunamburi/Desktop/Skin-Disease-AI-main
mkdir -p SkinDisease
mv /Users/pandunamburi/Downloads/my_skin_dataset SkinDisease/SkinDisease
```

## Need Help?

Run the setup helper:
```bash
python setup_dataset.py
```

This will guide you through the process interactively.
