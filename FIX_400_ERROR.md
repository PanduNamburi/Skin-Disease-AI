# Fix for 400 Bad Request Error on Render

## Problem
The application is returning "Bad Request (400)" errors because Django's `ALLOWED_HOSTS` setting doesn't include the Render domain.

## Solution

You have two options:

### Option 1: Set Environment Variable in Render Dashboard (Recommended)

1. Go to your Render dashboard: https://dashboard.render.com
2. Select your service: `skinsense-ai`
3. Go to **Environment** tab
4. Add a new environment variable:
   - **Key**: `ALLOWED_HOSTS`
   - **Value**: `skinsense-ai-nf4p.onrender.com,.onrender.com,localhost,127.0.0.1`
5. Save and redeploy

### Option 2: Update settings.py directly

The `settings.py` file has been updated to include the Render domain by default. If you haven't pushed the changes yet:

1. The fix is already in `skindisease_project/settings.py`
2. Commit and push the changes:
   ```bash
   git add skindisease_project/settings.py
   git commit -m "Fix ALLOWED_HOSTS for Render deployment"
   git push
   ```
3. Render will automatically redeploy

## What was changed?

The `ALLOWED_HOSTS` setting now defaults to:
```python
ALLOWED_HOSTS = [
    'skinsense-ai-nf4p.onrender.com',
    '.onrender.com',  # Allow all Render subdomains
    'localhost',
    '127.0.0.1',
]
```

This ensures that:
- Your specific Render domain is allowed
- All Render subdomains are allowed (for future changes)
- Local development still works

## Additional Notes

- The CSRF_TRUSTED_ORIGINS has also been configured for production
- DEBUG is set to False by default (can be overridden with environment variable)
- After deploying, wait a few minutes for the service to restart

## Verify the Fix

After deploying, check:
1. The service logs should show successful requests (200 status codes)
2. Visit https://skinsense-ai-nf4p.onrender.com - it should load without 400 errors
3. Check the browser console for any other errors
