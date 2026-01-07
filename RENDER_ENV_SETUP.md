# Render.com Environment Variables Setup

## Critical: Set ALLOWED_HOSTS Environment Variable

To fix the "DisallowedHost" error, you **MUST** set the `ALLOWED_HOSTS` environment variable in your Render dashboard.

### Steps:

1. Go to https://dashboard.render.com
2. Select your service: **skinsense-ai**
3. Click on **Environment** tab (in the left sidebar)
4. Click **Add Environment Variable**
5. Add the following:

   **Key:** `ALLOWED_HOSTS`
   
   **Value:** `skinsense-ai-nf4p.onrender.com,.onrender.com,localhost,127.0.0.1`

6. Click **Save Changes**
7. Render will automatically redeploy your service

### Why This is Needed:

- The code checks for `ALLOWED_HOSTS` environment variable first
- If not set, it falls back to default values
- Setting it explicitly ensures it works correctly in production
- This is the recommended Django production practice

### Optional: Other Environment Variables

You can also set these for better security:

- **SECRET_KEY**: Generate a secure key:
  ```bash
  python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
  ```

- **DEBUG**: Set to `False` for production (default is already False)

### After Setting Environment Variables:

1. Wait 2-3 minutes for automatic redeployment
2. Check the logs to ensure no errors
3. Visit https://skinsense-ai-nf4p.onrender.com

## Memory Issue Note

If you're still experiencing worker timeout/memory issues after fixing ALLOWED_HOSTS, it's because PyTorch is memory-intensive. Consider:

1. Upgrading to Render's **Standard** plan (more memory)
2. Or implementing lazy loading of PyTorch models (only load when needed)
