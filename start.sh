#!/bin/bash
# Don't use set -e to allow graceful error handling

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

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput || echo "⚠️  Static files collection failed, continuing..."

# Run migrations
echo "🔄 Running database migrations..."
python manage.py migrate --noinput || echo "⚠️  Migrations failed, continuing..."

# Start the server
echo "✅ Starting Gunicorn server..."
# Increased timeout to handle PyTorch imports (PyTorch is heavy)
# Using 1 worker to minimize memory usage on free tier
# Removed --preload as it can cause issues with heavy imports
exec gunicorn skindisease_project.wsgi \
    --log-file - \
    --bind 0.0.0.0:$PORT \
    --timeout 180 \
    --workers 1 \
    --worker-class sync \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --graceful-timeout 30

