# 🚀 Deploy to Vercel - Step by Step Guide

## Prerequisites
- GitHub account
- Vercel account (free)
- Your project files ready

## Step 1: Prepare Your Project

### 1.1 Create the API Directory Structure
Your project should have this structure:
```
anu project/
├── api/
│   └── index.py          # Main Flask app for Vercel
├── requirements.txt      # Python dependencies
├── vercel.json          # Vercel configuration
└── DEPLOYMENT.md        # This guide
```

### 1.2 Files Already Created
✅ `vercel.json` - Vercel configuration  
✅ `requirements.txt` - Python dependencies  
✅ `api/index.py` - Flask app optimized for Vercel  

## Step 2: Push to GitHub

### 2.1 Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Blog Content Tagger"
```

### 2.2 Create GitHub Repository
1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Name it: `blog-content-tagger`
4. Make it public
5. Don't initialize with README (you already have files)

### 2.3 Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/blog-content-tagger.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Vercel

### 3.1 Sign Up for Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up with your GitHub account
3. Connect your GitHub account

### 3.2 Import Project
1. Click "New Project" on Vercel dashboard
2. Import from GitHub
3. Select your `blog-content-tagger` repository
4. Click "Import"

### 3.3 Configure Deployment
Vercel will auto-detect your settings:
- **Framework Preset**: Other
- **Root Directory**: `./`
- **Build Command**: (leave empty)
- **Output Directory**: (leave empty)
- **Install Command**: `pip install -r requirements.txt`

### 3.4 Deploy
1. Click "Deploy"
2. Wait for deployment (2-3 minutes)
3. Get your live URL!

## Step 4: Test Your Deployment

### 4.1 Access Your App
Your app will be available at:
```
https://blog-content-tagger.vercel.app
```

### 4.2 Test Features
- ✅ Home page loads
- ✅ Content analysis works
- ✅ Tag generation works
- ✅ API endpoints respond

## Step 5: Custom Domain (Optional)

### 5.1 Add Custom Domain
1. Go to your Vercel project dashboard
2. Click "Settings" → "Domains"
3. Add your custom domain
4. Update DNS records as instructed

## 🎉 Deployment Complete!

### Your Live URLs:
- **Main App**: `https://blog-content-tagger.vercel.app`
- **API Endpoint**: `https://blog-content-tagger.vercel.app/api/analyze`

### Features Available:
- ✅ Interactive web interface
- ✅ Content analysis and tagging
- ✅ API endpoints for integration
- ✅ Responsive design
- ✅ Real-time processing

### API Usage Example:
```bash
curl -X POST https://blog-content-tagger.vercel.app/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"title": "My Blog Post", "content": "Your content here"}'
```

## 🔧 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check `requirements.txt` has all dependencies
   - Ensure `api/index.py` is in correct location

2. **NLTK Data Issues**
   - The app downloads NLTK data automatically on first run
   - First request might be slow (downloading data)

3. **Memory Issues**
   - Vercel free tier has memory limits
   - Consider upgrading if needed

4. **Cold Start**
   - First request after inactivity might be slow
   - Subsequent requests will be fast

### Support:
- Check Vercel logs in dashboard
- Review deployment logs for errors
- Test locally first: `python api/index.py`

---

**🎊 Congratulations! Your Automated Blog Content Tagger is now live on the internet!**
