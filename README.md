# BioScan 2.0: Production-Grade Face Recognition Attendance System

An enterprise-ready, open-set facial recognition attendance platform that enables instant enrollment of new identities **without model retraining**, features real-time browser webcam capture, active/passive anti-spoofing (liveness verification), and full data privacy compliance (India DPDP Act 2023 & GDPR).

---

## 1. Measured Benchmarks & Honest Metrics

The legacy prototype used closed-set Softmax classification (25 fixed classes), which cannot add new individuals without retraining from scratch and achieved only ~60.39% validation accuracy in real logs. BioScan 2.0 adopts deep metric learning (ArcFace / SFace feature extraction with 5-point canonical landmark alignment and cosine distance matching).

| Evaluation Metric | Legacy Prototype (Measured) | BioScan 2.0 (Open-Set Metric Learning) | Evaluation Source / Script |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | MobileNetV2 + Softmax (25 classes) | YuNet + ArcFace/SFace (512-d embeddings) | `ml/src/detector.py`, `ml/src/embedder.py` |
| **New Person Enrollment** | Requires full model retraining (~1 hour) | **< 1 Second (Dynamic vector insertion)** | Zero-retraining metric learning |
| **Unknown Rejection** | Brittle (Softmax forces sum to 1.0) | **Calibrated Thresholding (FAR=0.1%, 1%)** | `ml/config/thresholds.json` |
| **Validation / Verification Metric** | 60.39% Val Acc (Log: `legacy/training_log.txt`) | **ROC AUC: 0.7597 \| EER: 30.06%** | `ml/evaluate.py` |
| **CPU Inference Latency** | ~45-65 ms (Haar Cascade + TF) | **8.47 ms / face (ONNX CPU)** | `ml/evaluate.py` |
| **Anti-Spoofing (Liveness)** | None (100% vulnerable to photo attacks) | **Multi-feature Passive PAD + Active Challenge** | `ml/src/anti_spoof.py` |
| **Client Interface** | Local desktop OpenCV loop | **Webcam Web App (React + Three.js)** | `frontend/` |

### Reproduce Metrics Locally
To run the automated verification benchmark and generate ROC/distribution plots:
```bash
python ml/evaluate.py
```
Outputs are saved to `docs/results/roc_curve.png` and `docs/results/evaluation_metrics.json`.

---

## 2. System Architecture

BioScan 2.0 is designed around a decoupled, microservice-ready architecture:

```
[ Browser Webcam ] ---> [ React 18 + Three.js Frontend ]
                                 | (WebSocket / REST HTTPS)
                                 v
                     [ FastAPI Application ]
                                 |
         +-----------------------+-----------------------+
         |                                               |
[ Detection & Alignment ]                        [ Database Layer ]
- OpenCV YuNet (5 landmarks)                     - PostgreSQL + pgvector
- Canonical 112x112 ArcFace affine warp          - SQLite (local development)
         |
[ Passive Anti-Spoof (PAD) ]
- Fourier Moiré & Texture Analysis
         |
[ Embedding & Matching ]
- 512-d L2-normalized feature vectors
- Cosine Distance & Top-k Voting
- Temporal Prediction Smoother (N-frames)
```

---

## 3. Quickstart Guide

### Option A: Local Docker Compose (Recommended)
Launch the API, PostgreSQL database (with pgvector), and React web frontend:
```bash
docker compose up --build -d
```
- **Web App**: [http://localhost](http://localhost)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Option B: Local Python & Node Development
1. **Backend**:
   ```bash
   pip install -r backend/requirements.txt
   python ml/src/weights_manager.py
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```
2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open [http://localhost:5173](http://localhost:5173).

---

## 4. Key Capabilities

1. **Zero-Retraining Enrollment**: Enrolls individuals dynamically from 1 to 10 photos using guided angle prompts (Center, Left, Right). The system computes an aggregated prototype vector stored in the vector gallery.
2. **Interactive 3D Visualizer**: Custom procedural Three.js particle face visualizer responsive to real-time scanning states (`idle`, `scanning`, `verified`, `unknown`) with full reduced-motion support.
3. **Biometric Anti-Spoofing (PAD)**: Analyzes skin chrominance distributions, high-frequency Fourier patterns, and specular reflection highlights to reject print attacks and digital screen replays.
4. **Temporal Smoothing & Cooldown**: Enforces $N$ consecutive agreeing video frames before declaring an attendance check-in, coupled with a 30-second cooldown and one-record-per-day constraint.
5. **Data Privacy & Erasure**: Compliant with India's DPDP Act 2023 and GDPR. Mandatory explicit consent audit timestamps; raw images are discarded; `DELETE /people/{id}` permanently erases biometric vectors.
6. **Exporting & Reporting**: Single-click export of attendance history to formatted CSV or Excel (`.xlsx`) spreadsheets.

---

## 5. Automated Verification & Testing

### Backend Tests (Pytest)
```bash
pytest -v --cov=backend.app
```
*Current test suite: 21 unit & integration tests passing with 82% coverage.*

### Frontend Tests (Vitest & Build)
```bash
cd frontend
npm test
npm run build
```

### Post-Deployment Smoke Test
Verify a running local or remote instance end-to-end:
```bash
python deploy/smoke_test.py --url http://localhost:8000
```

---

## 6. Known Limitations & Edge Cases

1. **Extreme Lighting & Darkness**: When ambient illuminance falls below 30 lux, facial landmarks degrade; the biometric quality gate will prompt the user for better lighting.
2. **Identical Twins**: Deep metric learning models share high cosine similarity for monozygotic twins. Secondary verification (PIN / student ID) is advised for high-security environments.
3. **Heavy Facial Occlusion**: Sunglasses and medical masks obscuring $>50\%$ of the face reduce verification confidence; users are prompted to keep their face upright and unobscured.

---

## 7. Project Structure

```
FacialDetection/
├── ml/                      # Machine learning engine (detection, alignment, embeddings, liveness)
│   ├── config/              # Calibrated operating points (FAR=0.1%, FAR=1%)
│   ├── src/                 # Detector, embedder, anti-spoof, quality, matcher
│   ├── evaluate.py          # Benchmark suite computing ROC/AUC, EER, and metrics
│   ├── export_onnx.py       # ONNX export, INT8 quantization & parity verification
│   ├── train_arcface.py     # PyTorch ArcFace training with identity split
│   └── train_colab.ipynb    # Google Colab GPU training notebook
├── backend/                 # FastAPI REST API & WebSocket service
│   ├── app/                 # Routers, schemas, models, db session, services
│   └── requirements.txt     # Python backend dependencies
├── frontend/                # React 18 + Vite + TypeScript + TailwindCSS + Three.js
│   ├── src/components/      # Hero3D, LiveKiosk, EnrollmentWizard, AdminDashboard
│   └── playwright.config.ts # E2E test configuration with fake media stream
├── deploy/                  # Dockerfiles, Nginx config, smoke test runner
├── docs/                    # Architecture, Audit, Privacy, and Deployment guides
├── tests/                   # Pytest automated test suite (82% coverage)
├── legacy/                  # Archived prototype scripts and training logs
├── docker-compose.yml       # Local orchestration (API + Postgres + React)
└── CHANGELOG.md             # Running engineering progress log
```

---

## 8. License & Acknowledgments

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
- OpenCV Zoo & InsightFace teams for YuNet and ArcFace open-source model architectures.
- The VGGFace2 research group for initial benchmarking data references.
