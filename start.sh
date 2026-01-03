#!/bin/bash
set -e

echo "🚀 Starting SkinSense AI application..."

# Try to pull Git LFS files if Git LFS is available
if command -v git-lfs &> /dev/null; then
    echo "📥 Checking for Git LFS files..."
    git lfs pull || echo "⚠️  Git LFS pull failed, continuing..."
fi

# Create necessary directories
mkdir -p models
mkdir -p static
mkdir -p media/uploaded_images

# Download models from cloud storage if they don't exist (optional)
# Set ENABLE_MODEL_DOWNLOAD=true to enable this feature
if [ "$ENABLE_MODEL_DOWNLOAD" = "true" ]; then
    echo "📥 Downloading models from cloud storage..."
    python download_models.py || echo "⚠️  Model download failed, continuing..."
fi

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput || echo "⚠️  Static files collection failed, continuing..."

# Run migrations
echo "🔄 Running database migrations..."
python manage.py migrate --noinput || echo "⚠️  Migrations failed, continuing..."

# Start the server
echo "✅ Starting Gunicorn server..."
exec gunicorn skindisease_project.wsgi --log-file - --bind 0.0.0.0:$PORT

