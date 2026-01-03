# 🚀 Deploy SkinSense AI to Render.com

## Why Render.com?
- ✅ **10 GB image limit** (vs Railway's 4 GB) - Perfect for your ML models
- ✅ **Free tier available** - Great for testing
- ✅ **Simple setup** - Similar to Railway
- ✅ **Automatic deployments** - From GitHub
- ✅ **Git LFS support** - Your models will work

---

## 📋 Prerequisites

1. **GitHub account** with your repository: `https://github.com/PanduNamburi/Skin-Disease-AI`
2. **Render.com account** - Sign up at [render.com](https://render.com) (free)

---

## 🎯 Step-by-Step Deployment

### Step 1: Sign Up for Render

1. Go to [render.com](https://render.com)
2. Click **"Get Started for Free"**
3. Sign up with your **GitHub account** (recommended) or email
4. Verify your email if needed

### Step 2: Create a New Web Service

1. Once logged in, click **"New +"** button (top right)
2. Select **"Web Service"**
3. You'll see a list of your GitHub repositories
4. Find and click **"PanduNamburi/Skin-Disease-AI"**
5. Click **"Connect"**

### Step 3: Configure the Service

Render will auto-detect your `render.yaml` file, but you can also configure manually:

**Basic Settings:**
- **Name**: `skinsense-ai` (or any name you like)
- **Region**: Choose closest to you (e.g., `Oregon (US West)`)
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave empty (uses root)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
- **Start Command**: `bash start.sh`

**Environment Variables:**
Click **"Advanced"** → **"Environment Variables"** and add:
- `PYTHON_VERSION` = `3.11`
- `PORT` = (auto-set by Render, don't change)

**Plan:**
- Select **"Starter"** (Free tier) for testing
- Upgrade to **"Standard"** ($7/month) for production

### Step 4: Deploy

1. Scroll down and click **"Create Web Service"**
2. Render will start building your application
3. **First build takes 5-10 minutes** (installing dependencies)
4. You can watch the build logs in real-time

### Step 5: Wait for Deployment

- Build process:
  1. Clones your repository
  2. Installs Python dependencies (this takes time with PyTorch)
  3. Pulls Git LFS files (your models)
  4. Collects static files
  5. Runs migrations
  6. Starts Gunicorn server

- **Build will succeed** if:
  - All dependencies install correctly
  - Git LFS files are pulled successfully
  - No errors in `start.sh`

### Step 6: Get Your App URL

Once deployed, you'll see:
- **Your app URL**: `https://skinsense-ai.onrender.com` (or similar)
- Status: **"Live"** (green)

---

## 🔧 Update Flutter App

After deployment, update your Flutter app to use the new URL:

1. Open `skin_disease_app/lib/config.dart`
2. Update the cloud URL:

```dart
class ApiConfig {
  static const String localBaseUrl = 'http://192.168.1.26:8000';
  static const String cloudBaseUrl = 'https://skinsense-ai.onrender.com'; // Your Render URL
  static const bool useCloud = true; // Set to true for production

  static String get baseUrl {
    return useCloud ? cloudBaseUrl : localBaseUrl;
  }
  // ... rest of the code
}
```

3. Rebuild your Flutter app:
```bash
cd skin_disease_app
flutter build apk --release
```

---

## 🐛 Troubleshooting

### Build Fails with "Image too large"
- **Solution**: Render has 10GB limit, which should be enough
- If still failing, check `.dockerignore` excludes Flutter app

### Models Not Loading
- **Check**: Git LFS files are in your repository
- **Verify**: Run `git lfs ls-files` locally
- **Solution**: Models should auto-download via Git LFS

### App Crashes on Startup
- **Check logs**: Click on your service → "Logs" tab
- **Common issues**:
  - Missing environment variables
  - Database migration errors
  - Model files not found

### Slow Cold Starts (Free Tier)
- **Issue**: Free tier services sleep after 15 minutes of inactivity
- **Solution**: 
  - Upgrade to paid plan ($7/month)
  - Or use a service like [UptimeRobot](https://uptimerobot.com) to ping your app every 5 minutes

### Port Issues
- Render automatically sets `PORT` environment variable
- Your `start.sh` uses `$PORT` - this is correct
- Don't hardcode port numbers

---

## 📊 Monitoring

### View Logs
1. Go to your service dashboard
2. Click **"Logs"** tab
3. See real-time application logs

### View Metrics
1. Click **"Metrics"** tab
2. See CPU, Memory, Request rate
3. Monitor performance

---

## 🔄 Automatic Deployments

Render automatically deploys when you:
- Push to `main` branch
- Merge pull requests to `main`

To disable auto-deploy:
- Go to **Settings** → **Auto-Deploy** → Toggle off

---

## 💰 Pricing

### Free Tier (Starter Plan)
- ✅ 750 hours/month (enough for 24/7)
- ✅ 512 MB RAM
- ⚠️ Services sleep after 15 min inactivity
- ⚠️ Slower cold starts

### Paid Tier (Standard Plan - $7/month)
- ✅ Always on (no sleeping)
- ✅ 512 MB RAM
- ✅ Faster performance
- ✅ Better for production

---

## ✅ Success Checklist

- [ ] Render account created
- [ ] GitHub repository connected
- [ ] Web service created
- [ ] Build completed successfully
- [ ] App is "Live" (green status)
- [ ] Tested API endpoint: `https://your-app.onrender.com/api/predict/`
- [ ] Updated Flutter app with new URL
- [ ] Mobile app tested with deployed backend

---

## 🆘 Need Help?

1. **Render Docs**: [render.com/docs](https://render.com/docs)
2. **Support**: Click "Support" in Render dashboard
3. **Community**: [render.com/community](https://community.render.com)

---

## 🎉 Next Steps

Once deployed:
1. Test your API: `curl https://your-app.onrender.com/api/predict/`
2. Update Flutter app with new URL
3. Build and test APK on your phone
4. Share your app! 🚀

---

**Your app will be live at**: `https://skinsense-ai.onrender.com` (or your custom name)

Good luck! 🎊

