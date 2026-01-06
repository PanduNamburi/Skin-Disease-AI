# 🚀 Deploy Django Backend to Render - Step by Step

## ✅ Pre-Deployment Checklist

Your project is already configured with:
- ✅ `render.yaml` - Render configuration file
- ✅ `start.sh` - Startup script with Gunicorn
- ✅ `requirements.txt` - All dependencies
- ✅ Production-ready settings (using environment variables)

## 📋 Step 1: Sign Up for Render

1. Go to **[render.com](https://render.com)**
2. Click **"Get Started for Free"**
3. Sign up with your **GitHub account** (recommended - easier to connect repos)
4. Verify your email if prompted

## 📋 Step 2: Connect Your Repository

1. After logging in, click **"New +"** button (top right)
2. Select **"Web Service"**
3. You'll see a list of your GitHub repositories
4. Find and click **"PanduNamburi/Skin-Disease-AI"**
5. Click **"Connect"**

## 📋 Step 3: Configure Your Service

Render will auto-detect your `render.yaml` file, but verify these settings:

### Basic Settings:
- **Name**: `skinsense-ai` (or your preferred name)
- **Region**: Choose closest to you (e.g., `Oregon (US West)`)
- **Branch**: `main`
- **Root Directory**: Leave empty (uses root)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
- **Start Command**: `bash start.sh`

### Environment Variables (IMPORTANT):

Click **"Advanced"** → **"Environment Variables"** and add:

| Key | Value | Description |
|-----|-------|-------------|
| `PYTHON_VERSION` | `3.11.0` | Python version to use (must be major.minor.patch) |
| `SECRET_KEY` | `[Generate below]` | Django secret key (REQUIRED) |
| `DEBUG` | `False` | Set to False for production |
| `ALLOWED_HOSTS` | `your-app.onrender.com` | Your Render app URL |

**Generate SECRET_KEY:**
Run this in your terminal:
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Copy the output and use it as your `SECRET_KEY` value.

### Plan Selection:
- **Starter** (Free) - Good for testing, sleeps after 15 min inactivity
- **Standard** ($7/month) - Always on, better for production

## 📋 Step 4: Create and Deploy

1. Scroll down and click **"Create Web Service"**
2. Render will start building your application
3. **First build takes 5-10 minutes** (installing PyTorch and dependencies)
4. Watch the build logs in real-time

## 📋 Step 5: Monitor Deployment

### Build Process:
1. ✅ Clones your repository
2. ✅ Installs Python dependencies (PyTorch takes time)
3. ✅ Pulls Git LFS files (your model files)
4. ✅ Collects static files
5. ✅ Runs database migrations
6. ✅ Starts Gunicorn server

### Success Indicators:
- ✅ Build status: "Live" (green)
- ✅ Logs show: "Starting Gunicorn server..."
- ✅ No error messages in logs

## 📋 Step 6: Get Your App URL

Once deployed, you'll see:
- **Your app URL**: `https://skinsense-ai.onrender.com` (or your custom name)
- Status: **"Live"** (green indicator)

## 🧪 Step 7: Test Your Deployment

### Test API Endpoints:

```bash
# Test home page
curl https://your-app.onrender.com/

# Test API endpoint
curl https://your-app.onrender.com/api/predict/
```

### Test in Browser:
- Open: `https://your-app.onrender.com`
- You should see your Django app homepage

## 🔧 Step 8: Update Flutter App

After deployment, update your Flutter app to use the Render URL:

1. Open `skin_disease_app/lib/config.dart`
2. Update the cloud URL:

```dart
class ApiConfig {
  static const String localBaseUrl = 'http://localhost:8000';
  static const String cloudBaseUrl = 'https://skinsense-ai.onrender.com'; // Your Render URL
  static const bool useCloud = true; // Set to true for production

  static String get baseUrl => useCloud ? cloudBaseUrl : localBaseUrl;
  // ... rest of the code
}
```

3. Rebuild your Flutter app:
```bash
cd skin_disease_app
flutter build apk --release
```

## 🐛 Troubleshooting

### Build Fails - "Image too large"
- **Solution**: Render has 10GB limit, should be enough
- Check `.dockerignore` excludes unnecessary files

### Build Fails - "Module not found"
- **Check**: All dependencies in `requirements.txt`
- **Verify**: Python version matches (3.11)

### Models Not Loading
- **Check**: Git LFS files are in repository
- **Verify**: Run `git lfs ls-files` locally
- **Solution**: Models should auto-download via Git LFS

### App Crashes on Startup
- **Check logs**: Service → "Logs" tab
- **Common issues**:
  - Missing `SECRET_KEY` environment variable
  - Database migration errors
  - Model files not found
  - Port configuration issues

### Slow Cold Starts (Free Tier)
- **Issue**: Free tier services sleep after 15 minutes
- **Solution**: 
  - Upgrade to paid plan ($7/month)
  - Or use [UptimeRobot](https://uptimerobot.com) to ping every 5 minutes

### "Invalid SECRET_KEY" Error
- **Solution**: Make sure `SECRET_KEY` environment variable is set
- Generate new key and update in Render dashboard

### CORS Errors
- **Check**: `ALLOWED_HOSTS` includes your Render URL
- **Verify**: `CORS_ALLOW_ALL_ORIGINS = True` in settings (for development)

## 📊 Monitoring Your App

### View Logs:
1. Go to your service dashboard
2. Click **"Logs"** tab
3. See real-time application logs

### View Metrics:
1. Click **"Metrics"** tab
2. Monitor CPU, Memory, Request rate

## 🔄 Automatic Deployments

Render automatically deploys when you:
- ✅ Push to `main` branch
- ✅ Merge pull requests to `main`

To disable auto-deploy:
- Go to **Settings** → **Auto-Deploy** → Toggle off

## 💰 Pricing

### Free Tier (Starter Plan)
- ✅ 750 hours/month (enough for 24/7)
- ✅ 512 MB RAM
- ⚠️ Services sleep after 15 min inactivity
- ⚠️ Slower cold starts (~30 seconds)

### Paid Tier (Standard Plan - $7/month)
- ✅ Always on (no sleeping)
- ✅ 512 MB RAM
- ✅ Faster performance
- ✅ Better for production

## ✅ Deployment Checklist

- [ ] Render account created
- [ ] GitHub repository connected
- [ ] Web service created
- [ ] Environment variables set (SECRET_KEY, DEBUG, ALLOWED_HOSTS)
- [ ] Build completed successfully
- [ ] App is "Live" (green status)
- [ ] Tested API endpoint
- [ ] Updated Flutter app with new URL
- [ ] Mobile app tested with deployed backend

## 🎉 Success!

Your Django backend is now live at: `https://your-app.onrender.com`

### Next Steps:
1. Test all API endpoints
2. Update Flutter app configuration
3. Build and test mobile app
4. Share your app! 🚀

## 🆘 Need Help?

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Support**: Click "Support" in Render dashboard
- **Community**: [render.com/community](https://community.render.com)

---

**Your app URL will be**: `https://skinsense-ai.onrender.com` (or your custom name)

Good luck with your deployment! 🎊
