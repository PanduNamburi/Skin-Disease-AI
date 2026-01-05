# ⚡ Quick Deployment Guide - Render.com

## 🎯 5-Minute Setup

### 1. Sign Up
- Go to [render.com](https://render.com)
- Click "Get Started for Free"
- Sign up with GitHub (recommended)

### 2. Deploy
- Click "New +" → "Web Service"
- Select your repo: `PanduNamburi/Skin-Disease-AI`
- Click "Connect"

### 3. Configure (Auto-detected from `render.yaml`)
- **Name**: `skinsense-ai`
- **Plan**: `Starter` (Free)
- Click "Create Web Service"

### 4. Wait
- Build takes 5-10 minutes
- Watch logs in real-time
- Status will turn "Live" when ready

### 5. Get URL
- Your app: `https://skinsense-ai.onrender.com`
- Copy this URL

### 6. Update Flutter App
Edit `skin_disease_app/lib/config.dart`:
```dart
static const String cloudBaseUrl = 'https://skinsense-ai.onrender.com';
static const bool useCloud = true;
```

### 7. Rebuild APK
```bash
cd skin_disease_app
flutter build apk --release
```

## ✅ Done!

Your app is now live and accessible from anywhere! 🎉

---

## 🔍 Test Your Deployment

Open in browser:
```
https://skinsense-ai.onrender.com
```

Test API:
```bash
curl https://skinsense-ai.onrender.com/api/predict/
```

---

## 📱 Full Guide

See `RENDER_DEPLOYMENT.md` for detailed instructions and troubleshooting.

