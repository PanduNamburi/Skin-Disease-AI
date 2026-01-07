# Render.com Troubleshooting Guide

## 502 Bad Gateway Error

If you're seeing "HTTP ERROR 502", it means the application is crashing or not starting properly.

### Common Causes:

1. **Memory Issues**: PyTorch is very memory-intensive. The free tier has limited memory (512MB).
2. **Worker Timeouts**: Heavy imports (PyTorch/torchvision) can cause worker timeouts.
3. **Import Errors**: Module import failures during startup.

### Solutions:

#### Option 1: Check Render Logs
1. Go to https://dashboard.render.com
2. Select your service
3. Click on **Logs** tab
4. Look for error messages

#### Option 2: Increase Timeout (Already Done)
The `start.sh` has been updated with:
- Timeout: 180 seconds (increased from 30s)
- 1 worker (to save memory)
- Graceful timeout settings

#### Option 3: Set Environment Variables
Make sure these are set in Render Dashboard → Environment:
- `ALLOWED_HOSTS`: `skinsense-ai-nf4p.onrender.com,.onrender.com,localhost,127.0.0.1`
- `DEBUG`: `False`
- `SECRET_KEY`: (generate a secure key)

#### Option 4: Upgrade Plan
The free tier has limited resources. Consider upgrading to:
- **Standard Plan**: More memory (2GB) and better performance
- Cost: ~$7/month

#### Option 5: Optimize for Free Tier
If staying on free tier:
1. Models are loaded lazily (only when needed)
2. Use lighter models if possible
3. Consider using CPU-only PyTorch builds

### Quick Fixes:

1. **Redeploy**: Sometimes a fresh deployment helps
   - Go to Render Dashboard
   - Click **Manual Deploy** → **Deploy latest commit**

2. **Check Build Logs**: Make sure build completed successfully
   - Look for any errors during `pip install` or `collectstatic`

3. **Verify Port**: Make sure `$PORT` environment variable is set (should be automatic)

### If Still Not Working:

1. Check if the service is actually running:
   - Look for "Your service is live 🎉" in logs
   - Check if port 10000 is detected

2. Verify all dependencies installed:
   - Check build logs for any failed package installations

3. Test locally first:
   ```bash
   python manage.py runserver
   ```
   If it works locally but not on Render, it's likely a memory/resource issue.

### Memory Optimization Tips:

1. **Lazy Loading**: Models are already loaded lazily (only when prediction is made)
2. **Model Size**: Consider using smaller models (ResNet18 instead of ResNet101)
3. **Remove Unused Imports**: Clean up any unused dependencies

### Contact Render Support:

If none of the above works, contact Render support with:
- Service logs
- Error messages
- Service configuration
