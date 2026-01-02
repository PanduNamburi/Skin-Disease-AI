#!/bin/bash
set -e

echo "🔧 Building SkinSense AI application..."

# Ensure Git LFS is installed and pull LFS files
echo "📥 Downloading Git LFS files..."
if command -v git-lfs &> /dev/null; then
    git lfs install
    git lfs pull
    echo "✅ Git LFS files downloaded"
else
    echo "⚠️  Git LFS not found, skipping LFS download"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p static
mkdir -p media/uploaded_images

echo "✅ Build preparation complete"

