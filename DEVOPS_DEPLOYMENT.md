# 🚀 DevOps Deployment Guide: Dorsal Hand Vein Auth System

**Stack:** Docker · GitHub Actions CI/CD · Railway.app (Free Cloud)

---

## 📁 Complete File Structure (After Implementation)

```
project-root/
│
├── 🆕 railway.json               ← Tells Railway HOW to build & run the app
├── 🆕 docker-compose.yml         ← One-command local development
├── 🆕 .gitignore                 ← Excludes large training CSVs from Git
│
├── .streamlit/
│   └── 🆕 config.toml            ← Streamlit dark theme + headless cloud config
│
├── deployment/
│   ├── Dockerfile                ← Container build instructions (updated)
│   └── .dockerignore             ← Files excluded from Docker image
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml             ← GitHub Actions: auto-lint + Docker build
│
├── create_csv/
│   ├── Key01.csv  ✅ committed   ← App needs this at runtime
│   ├── Key02.csv  ✅ committed   ← App needs this at runtime
│   ├── Xtrain.csv ❌ gitignored  ← 36 MB — not needed by app, excluded
│   └── full_data.csv ❌ ignored  ← 29 MB — not needed by app, excluded
│
└── requirements.txt              ← Uses opencv-python-headless (server-safe)
```

---

## 🐳 Step 1: Test Locally with Docker

Verify everything works inside a container before pushing to Railway.

### Option A — Plain Docker (Quick verification)

```bash
# Build (run from project root)
docker build -f deployment/Dockerfile -t vein-auth .

# Run
docker run -p 8501:8501 vein-auth
```

### Option B — Docker Compose (Recommended for local dev)

```bash
# Build image and start
docker-compose up --build

# Start in background
docker-compose up -d

# Stream live logs
docker-compose logs -f

# Stop
docker-compose down
```

Visit → **`http://localhost:8501`** ✅

> ⏳ The app takes **~30–60 seconds** to load on first start because it trains
> the KFA model over all 85 subjects. This is expected — it will be the same
> on Railway. Subsequent uploads/verifications are instant.

---

## 🔄 Step 2: CI/CD Pipeline (GitHub Actions)

The file `.github/workflows/ci-cd.yml` is already configured.

Every `git push` to `main` automatically runs **two sequential jobs**:

```
git push origin main
       │
       ▼
┌─────────────────────────────────────────┐
│  Job 1: Lint & Test  (ubuntu-latest)    │
│  • pip install (cached for speed)       │
│  • flake8 syntax check                  │
│  • import smoke test                    │
└────────────┬────────────────────────────┘
             │ PASS only
             ▼
┌─────────────────────────────────────────┐
│  Job 2: Docker Build  (ubuntu-latest)   │
│  • docker buildx (layer-cached)         │
│  • builds full image from Dockerfile    │
│  • confirms image is 100% valid         │
└─────────────────────────────────────────┘
             │ PASS only
             ▼
     ✅ Code verified. Railway will auto-deploy.
```

You can watch runs live at:
`https://github.com/YOUR_USERNAME/YOUR_REPO/actions`

---

## ☁️ Step 3: Deploy to Railway.app

Railway reads `railway.json` to know exactly how to build and serve the app.
Every `git push` triggers an **automatic rolling redeploy** — zero manual steps.

---

### 3.1 Push Your Code to GitHub First

Open your terminal in the project folder and run:

```bash
# Stage all files (large CSVs are excluded by .gitignore automatically)
git add .

# Commit everything
git commit -m "feat: add Railway + Docker deployment setup"

# Push to GitHub — this also triggers the GitHub Actions CI/CD pipeline
git push origin main
```

> ✅ GitHub Actions will run automatically. Check the **Actions** tab to confirm
> both jobs pass before proceeding to Railway.

---

### 3.2 Create Your Railway Account

1. Go to **[railway.app](https://railway.app)**
2. Click **"Login"** → **"Continue with GitHub"**
3. Authorize Railway *(allow access to your repos)*
4. You'll land on your **Railway Dashboard**

> No credit card required. You get **$5 free credit/month** automatically.

---

### 3.3 Create a New Project on Railway

1. Click **"+ New Project"**
2. Select **"Deploy from GitHub repo"**
3. Search for your repo:
   ```
   Dorsal-Hand-Vein-Based-Cancellable-Biometric-Authentication-System
   ```
4. Click **"Deploy Now"**

Railway will start building immediately. Click the service box to watch **build logs in real time**.

---

### 3.4 Set the Port & Generate Your Public URL

1. Click on your service (the box that appeared)
2. Go to **"Settings"** tab
3. Scroll to **"Networking"** section
4. Set **"Custom Port"**: `8501`
5. Click **"Generate Domain"**

Your live URL will look like:
```
https://dorsal-hand-vein-auth-production.up.railway.app
```

---

### 3.5 Add Environment Variables

Go to the **"Variables"** tab and add these one by one:

| Variable Name | Value |
|---|---|
| `STREAMLIT_SERVER_PORT` | `8501` |
| `STREAMLIT_SERVER_HEADLESS` | `true` |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` |

After adding all of them, click **"Deploy"** to apply.

---

### 3.6 Wait for Build & Verify

| Phase | What's happening | Time |
|---|---|---|
| **Building** | Railway pulls your Docker image from GitHub | ~2–3 min |
| **Starting** | App container starts, KFA model trains on 85 subjects | ~1–2 min |
| **Healthy** | `/_stcore/health` returns OK — app is ready | ✅ |

Once the deployment shows **"Active"**, click your domain URL and you should see:

> *"Loading System & Training KFA Model… This may take a minute."*

Wait ~60 seconds → auth system is live 🎉

---

### 3.7 Test End-to-End ✅

1. Open your Railway URL
2. Upload any vein image from `sample dataset/veinpattern/s1/`
3. Select **Subject 1** in the dropdown
4. Click **"Verify Identity"**
5. You should get **"ACCESS GRANTED"** with a distance metric

---

### 3.8 Every Future Update is Automatic

```
You edit code locally
       │
       ▼
git add . && git commit -m "fix: ..." && git push origin main
       │
       ├──▶ GitHub Actions (Lint + Docker verify) ──▶ ✅ pass
       │
       └──▶ Railway detects new commit ──▶ rebuilds image ──▶ rolling deploy
                                                                      │
                                                              Zero downtime ✅
```

---

### 3.9 Monitor Usage & Stay Free

1. Railway Dashboard → your profile icon → **"Usage"**
2. Track CPU hours, RAM, and network

**Cost estimate for this app:**

| State | Cost/hour | $5 credit lasts |
|---|---|---|
| Active (training + serving) | ~$0.008/hr | ~625 hours |
| Idle (no traffic) | ~$0.003/hr | ~1,666 hours |

> 💡 **Pro Tip:** When you're not demoing, go to **Settings → "Suspend Service"**.
> This pauses billing. Resume with one click when needed.

---

## ⚠️ Data Persistence Reference

| File | In Git? | In Docker Image? | Lost on restart? |
|---|---|---|---|
| `create_csv/Key01.csv` | ✅ Yes | ✅ Yes | ❌ No |
| `create_csv/Key02.csv` | ✅ Yes | ✅ Yes | ❌ No |
| `sample dataset/` (85 subjects) | ✅ Yes | ✅ Yes | ❌ No |
| `create_csv/Xtrain.csv` | ❌ gitignored | ❌ No (not needed) | — |
| Files written at runtime | — | — | ✅ Yes — lost |

The keys and dataset are committed to Git, so they're baked into the Docker image Railway builds. The app never needs to regenerate them.

---

## 📋 Quick Reference Commands

```bash
# ── Local Docker ──────────────────────────────────────────────────────
docker build -f deployment/Dockerfile -t vein-auth .    # Build image
docker run -p 8501:8501 vein-auth                        # Run

# ── Docker Compose ────────────────────────────────────────────────────
docker-compose up --build      # Build + start (first time)
docker-compose up -d           # Start in background
docker-compose logs -f         # Live logs
docker-compose down            # Stop all

# ── Git (triggers CI/CD + Railway auto-deploy) ────────────────────────
git add .
git commit -m "feat: describe your change"
git push origin main
```

---

## 🔗 Links

| Resource | URL |
|---|---|
| **Railway Dashboard** | https://railway.app/dashboard |
| **Railway Docs — Docker** | https://docs.railway.app/guides/dockerfiles |
| **Railway Pricing** | https://railway.app/pricing |
| **GitHub Actions** | `https://github.com/YOUR_USERNAME/YOUR_REPO/actions` |
| **App Health Check** | `https://YOUR_APP.up.railway.app/_stcore/health` |
