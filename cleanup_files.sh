#!/bin/bash
# Script to clean up unnecessary files

echo "=== File Cleanup Script ==="
echo ""
echo "This will remove:"
echo "  1. .DS_Store files (macOS system files)"
echo "  2. training_output.log (training log - can be regenerated)"
echo "  3. Python cache files (__pycache__, *.pyc)"
echo "  4. Helper scripts (check_training.sh, monitor_training.sh, setup_dataset.py)"
echo "  5. Duplicate per_class_accuracy.csv (if evaluation/results has it)"
echo "  6. Build artifacts (skin_disease_app/build/)"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo "Cleaning up..."

# Remove .DS_Store files
find . -name ".DS_Store" -type f -delete
echo "✓ Removed .DS_Store files"

# Remove training log
if [ -f "training_output.log" ]; then
    rm training_output.log
    echo "✓ Removed training_output.log"
fi

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "✓ Removed Python cache files"

# Remove helper scripts (optional - ask user)
if [ -f "check_training.sh" ]; then
    rm check_training.sh
    echo "✓ Removed check_training.sh"
fi

if [ -f "monitor_training.sh" ]; then
    rm monitor_training.sh
    echo "✓ Removed monitor_training.sh"
fi

if [ -f "setup_dataset.py" ]; then
    rm setup_dataset.py
    echo "✓ Removed setup_dataset.py"
fi

# Remove duplicate per_class_accuracy.csv if evaluation version exists
if [ -f "evaluation/results/metrics_per_class.csv" ] && [ -f "per_class_accuracy.csv" ]; then
    rm per_class_accuracy.csv
    echo "✓ Removed duplicate per_class_accuracy.csv (evaluation version exists)"
fi

# Remove build artifacts
if [ -d "skin_disease_app/build" ]; then
    rm -rf skin_disease_app/build
    echo "✓ Removed skin_disease_app/build/ (can be regenerated)"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Files kept:"
echo "  - best_skindisease_model.pt (your trained model)"
echo "  - evaluation/results/ (all evaluation metrics)"
echo "  - All source code and configuration files"
