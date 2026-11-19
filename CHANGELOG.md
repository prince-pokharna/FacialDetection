# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 0 Complete**:
  - `docs/AUDIT.md`: Measured baseline metrics (60.39% val accuracy on MobileNetV2, not 95%+), cataloged legacy bugs and vulnerabilities.
  - `docs/ARCHITECTURE.md`: Open-set Detect -> Align -> Embed -> Match target architecture.
  - Root `.gitignore` and restructuring into `/ml`, `/backend`, `/frontend`, `/docs`, `/tests`, `/deploy`, `/legacy`.
- **Phase 1 Complete (Open-Set ML Engine)**:
  - `ml/src/detector.py`: Deep face detection and canonical 5-point landmark alignment (112x112 ArcFace coordinate space) via OpenCV YuNet (SCRFD class).
  - `ml/src/embedder.py`: 512-dimensional ArcFace/SFace feature extractor producing L2-normalized unit vectors with CPU latency < 15ms.
  - `ml/src/anti_spoof.py`: Passive multi-feature liveness (chrominance gamut, Fourier moiré patterns, texture variance, specular reflection) + active challenge.
  - `ml/src/quality.py`: Biometric enrollment quality gate checking sharpness, brightness, resolution, and head pose.
  - `ml/src/matcher.py`: Cosine distance matcher with unknown rejection and rolling temporal smoothing window.
  - `ml/train_arcface.py`: PyTorch ArcMarginProduct loss training with cosine LR and identity-based split.
  - `ml/export_onnx.py`: ONNX export with INT8 dynamic quantization and parity verification test (< 1e-4 tolerance).
  - `ml/evaluate.py`: Benchmark suite reporting honest ROC/AUC, EER, Rank-1/5, open-set unknown rejection, and saving plots to `docs/results/`.
  - `ml/train_colab.ipynb`: Ready-to-run Colab/Kaggle GPU training notebook.
- **Phase 2 Complete (FastAPI Backend)**:
  - Modular FastAPI application architecture (`backend/app/api`, `core`, `db`, `models`, `schemas`, `services`).
  - JWT Authentication with role enforcement (Admin, Instructor) and bcrypt password hashing.
  - Multi-image biometric enrollment endpoint (`/people/{id}/enroll`) with quality feedback.
  - Privacy-compliant biometric erasure endpoint (`DELETE /people/{id}`) for Right to be Forgotten.
  - Attendance management (`/attendance`, `/attendance/export.csv`, `/attendance/export.xlsx`).
  - Low-latency real-time video stream WebSocket (`/ws/stream`) with temporal smoothing and cooldown logic.
  - System observability: Prometheus metrics (`/metrics`), health check (`/health`), and model info (`/model/info`).
  - Comprehensive Pytest test suite with 82% coverage across 21 unit & integration tests.
- **Phase 3 Complete (React + Vite + TypeScript + Three.js Frontend)**:
  - Modern web application with responsive Tailwind CSS design and Dark/Light theme toggle.
  - Procedural Three.js 3D particle face visualizer (`Hero3D.tsx`) reacting to scan states (idle, scanning, verified, unknown) with reduced-motion support.
  - Live Attendance Kiosk (`LiveKiosk.tsx`): webcam stream capture via `getUserMedia`, real-time canvas bounding box overlays, anti-spoof badge, and audio chime.
  - Guided Multi-Angle Enrollment Wizard (`EnrollmentWizard.tsx`): 3-step capture with live quality gating and mandatory biometric consent.
  - Analytics and Admin Dashboard (`AdminDashboard.tsx`): daily attendance logs, Recharts/SVG metrics, CSV/Excel export, and identity deletion.
  - Vitest component testing and Playwright E2E configuration with mocked media stream devices.
- **Phase 4 Complete (Privacy & Security)**:
  - `docs/PRIVACY.md`: Comprehensive policy detailing DPDP Act 2023, GDPR, and VGGFace2 non-commercial academic research licensing limits.
  - Enforced embedding-only storage (no raw photo persistence).
  - Explicit biometric consent tracking with ISO timestamps.
  - Cascading biometric erasure endpoint ("Right to be Forgotten").
  - Frontend npm vulnerability audit resolved to 0 vulnerabilities.
- **Phase 5 Complete (Containers, CI/CD, Deployment)**:
  - Multi-stage Dockerfiles: `deploy/Dockerfile.backend` (slim Python 3.11 with OpenCV & ONNX Runtime) and `deploy/Dockerfile.frontend` (Node 20 + Nginx).
  - Root `docker-compose.yml` orchestrating API, PostgreSQL + pgvector, and React web app.
  - GitHub Actions CI/CD workflows (`.github/workflows/ci.yml`, `.github/workflows/deploy.yml`).
  - One-page zero-cost deployment guide `docs/DEPLOYMENT.md` for Hugging Face Spaces, Render, Neon, and Vercel.
  - Automated post-deploy smoke test script `deploy/smoke_test.py` passing all 7 checks.
