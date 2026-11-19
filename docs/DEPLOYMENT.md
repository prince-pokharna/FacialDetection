# Production Deployment Guide (Zero-Cost Free-Tier Architecture)

This guide documents the end-to-end production deployment configuration for the **Face Recognition Attendance System**.

---

## 1. Production Architecture Overview

```mermaid
flowchart LR
    Browser["Client Browser (Webcam HTTPS)"]
    Vercel["Frontend: Vercel / Netlify (HTTPS CDN)"]
    HF["Backend API: Hugging Face Spaces / Render (Docker CPU)"]
    NeonDB[("Database: Neon / Supabase (Postgres 16 + pgvector)")]

    Browser -->|HTTPS / WSS| Vercel
    Vercel -->|REST API / WebSocket| HF
    HF -->|Connection Pool| NeonDB
```

- **Frontend**: **Vercel** (Free tier, automatic HTTPS mandatory for `navigator.mediaDevices.getUserMedia`).
- **Backend API & ML Engine**: **Hugging Face Spaces** (Docker SDK, 2 vCPU, 16 GB RAM free) OR **Render.com** (Web Service).
- **Database**: **Neon Serverless Postgres** or **Supabase** with `pgvector` enabled (Free tier, 500 MB storage).

---

## 2. Environment Variables Matrix

### Backend Environment Variables (`.env`)

| Variable | Description | Example / Recommended Value |
| :--- | :--- | :--- |
| `APP_NAME` | Display name of system | `Face Recognition Attendance System` |
| `ENVIRONMENT` | Runtime environment | `production` |
| `DEBUG` | Enable debug traceback | `false` |
| `PORT` | HTTP port to bind | `8000` (or `7860` for Hugging Face Spaces) |
| `SECRET_KEY` | JWT signing secret | Generate with `openssl rand -hex 32` |
| `DATABASE_URL` | PostgreSQL connection string with pgvector | `postgresql://user:pass@ep-cool-db.neon.tech/attendance_db?sslmode=require` |
| `CORS_ORIGINS` | Permitted frontend URLs (comma-separated) | `https://your-app.vercel.app,http://localhost:5173` |
| `FAR_THRESHOLD` | Operating threshold (FAR=1%) | `0.42` |
| `LIVENESS_THRESHOLD` | Passive anti-spoof threshold | `0.50` |
| `COOLDOWN_SECONDS` | Duplicate mark cooldown per person | `30` |

### Frontend Environment Variables (`frontend/.env.production`)

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `VITE_API_URL` | URL of deployed FastAPI backend | `https://prince-pokharna-facial-attendance.hf.space` |

---

## 3. Step-by-Step Deployment Instructions

### Step 3.1: Provision PostgreSQL + pgvector (Neon / Supabase)
1. Sign up at [neon.tech](https://neon.tech) and create a project named `facial-attendance`.
2. In the Neon SQL Console, enable `pgvector`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Copy the connection string: `postgresql://[user]:[password]@[host]/[database]?sslmode=require`.

---

### Step 3.2: Deploy Backend to Hugging Face Spaces (Free 2-vCPU Docker)
1. Log in to [Hugging Face](https://huggingface.co) and create a **New Space**.
   - Space Name: `facial-attendance-api`
   - License: `mit`
   - Space SDK: **Docker** (Blank)
   - Hardware: **CPU Basic (2 vCPU, 16 GB RAM) - Free**
2. In Space Settings -> **Variables and Secrets**, add:
   - Secret `DATABASE_URL`: Your Neon PostgreSQL connection string.
   - Secret `SECRET_KEY`: A random 32-character hex key.
   - Variable `CORS_ORIGINS`: `*` or your Vercel URL.
   - Variable `PORT`: `7860`
3. Push the repository to the Hugging Face Space:
   ```bash
   git remote add space https://huggingface.co/spaces/[your-username]/facial-attendance-api
   git push space main
   ```
   *The container automatically compiles OpenCV dependencies, downloads ONNX weights, and starts Uvicorn.*

---

### Step 3.3: Deploy Frontend to Vercel (HTTPS Required for Webcam)
1. Install Vercel CLI or link GitHub repo at [vercel.com](https://vercel.com):
   ```bash
   cd frontend
   npm run build
   vercel
   ```
2. In Vercel Project Settings -> **Environment Variables**:
   - Set `VITE_API_URL`: `https://[your-username]-facial-attendance-api.hf.space`
3. Deploy to production:
   ```bash
   vercel --prod
   ```
4. Access your live HTTPS URL (e.g. `https://facial-attendance.vercel.app`). Your browser will prompt for camera access.

---

## 4. Local Deployment with Docker Compose

To run the full stack locally (API + PostgreSQL with pgvector + React Nginx):
```bash
# 1. Build and run containers
docker compose up --build -d

# 2. Check running services
docker compose ps

# 3. Access endpoints:
# Frontend UI: http://localhost
# Backend API: http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

---

## 5. Post-Deployment Verification (Smoke Test)

Run the automated smoke test script against your live deployment:
```bash
python deploy/smoke_test.py --url https://your-backend-api.hf.space
```

The script will verify:
- `GET /health` connectivity.
- JWT authentication.
- Identity registration with privacy consent.
- Multi-image biometric embedding extraction.
- Real-time recognition & automated attendance logging.
- Biometric erasure / data deletion cleanup.
