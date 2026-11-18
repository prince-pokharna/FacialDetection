# Target System Architecture & Technical Specification

## 1. System Vision & Architecture Paradigm

The modernized Face Recognition Attendance System transitions from a legacy monolithic desktop script to an **open-set, cloud-ready, microservices-based biometric attendance platform**.

The core paradigm is **Deep Metric Learning (Detect -> Align -> Embed -> Match)**:
- **Zero Retraining Enrollment**: New individuals are enrolled dynamically by computing 512-dimensional L2-normalized feature vectors stored in a vector index (`pgvector` or `FAISS`).
- **Open-Set Identification & Rejection**: Cosine similarity against enrolled identities with mathematically calibrated confidence thresholds ($FAR=0.1\%$). Unknown faces are rejected reliably.
- **Biometric Anti-Spoofing**: Passive liveness detection + active blink/head-pose challenge prevents photo/video presentation attacks.
- **Full-Stack Decoupled Architecture**: High-performance FastAPI backend paired with a modern React + TypeScript + Three.js frontend.

---

## 2. High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Client["Client Browser (React 18 + Vite + TS)"]
        UI["Webcam Kiosk UI / Guided Enrollment / Admin"]
        ThreeJS["Three.js / R3F 3D Wireframe Scene"]
        Camera["Browser MediaDevices (getUserMedia)"]
        WSClient["WebSocket Client / REST Client"]
        Camera --> WSClient
    end

    subgraph Gateway["Reverse Proxy & Gateway"]
        Nginx["Nginx / HTTPS Gateway"]
    end

    subgraph Backend["FastAPI Backend (Python 3.11+)"]
        API["REST Endpoints (Auth, People, Attendance)"]
        StreamWS["WebSocket /ws/stream (Real-Time Recognition)"]
        AuthMiddleware["JWT Authentication & RBAC"]
        RateLimiter["Rate Limiter & Input Validator"]
        
        subgraph MLEngine["ML Biometric Pipeline"]
            Detector["Face Detector & 5-pt Landmark (YuNet / SCRFD)"]
            Aligner["Similarity Affine Transform (112x112 px)"]
            Liveness["Anti-Spoofing Engine (MiniFASNet / Passive PAD)"]
            Embedder["ArcFace 512-d Feature Extractor (ONNX CPU)"]
            Matcher["Cosine Distance & Top-k Voter"]
            Temporal["Temporal Prediction Smoother (N-frame window)"]
            
            Detector --> Aligner
            Aligner --> Liveness
            Aligner --> Embedder
            Embedder --> Matcher
            Liveness --> Temporal
            Matcher --> Temporal
        end
    end

    subgraph Storage["Persistence Layer"]
        DB[(PostgreSQL + pgvector / SQLite Dev)]
        Prometheus["Prometheus Metrics (/metrics)"]
    end

    WSClient <-->|WSS / HTTPS| Nginx
    Nginx <--> API
    Nginx <--> StreamWS
    StreamWS --> MLEngine
    API --> MLEngine
    MLEngine --> Temporal
    Temporal --> DB
    API --> DB
    Backend --> Prometheus
```

---

## 3. Machine Learning Pipeline: Detect -> Align -> Embed -> Match

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Camera
    participant UI as Kiosk UI
    participant WS as FastAPI Stream WS
    participant Det as Face Detector
    participant Align as Landmark Aligner
    participant PAD as Anti-Spoof (Liveness)
    participant Arc as ArcFace Embedder
    participant DB as Vector Store (pgvector)
    participant Smooth as Temporal Smoother

    User->>UI: Streams webcam frame (5 fps)
    UI->>WS: Sends JPEG frame binary / base64
    WS->>Det: Run face detection (BBox + 5 Landmarks)
    alt No Face Detected
        Det-->>WS: 0 faces detected
        WS-->>UI: Return "No Face" status
    else Face Detected
        Det->>Align: Standard 5-point affine warp to 112x112
        Align->>PAD: Evaluate spoof probability
        alt Spoof Detected (Probability > 0.5)
            PAD-->>WS: Fake / Spoof Detected
            WS-->>UI: Badge: "Spoof Detected - Real Face Required"
        else Live Human
            Align->>Arc: 512-d Feature Extraction (L2 normalized)
            Arc->>DB: Cosine Distance Search against enrolled vectors
            DB-->>Matcher: Top-1 match (ID, Cosine Similarity)
            alt Similarity >= Threshold (e.g. 0.45)
                Matcher->>Smooth: Push (PersonID, Score, Live=True)
                Smooth->>DB: Check Cooldown (30s) & Date Check
                alt Not Marked Today & Cooldown Clear
                    Smooth->>DB: Insert Attendance Record
                    Smooth-->>WS: Attendance Marked: "Welcome, {PersonName}!"
                    WS-->>UI: Success Toast + Audio Chime + BBox Green
                else Already Marked
                    WS-->>UI: Already Marked Today (Cooldown Active)
                end
            else Similarity < Threshold
                WS-->>UI: Badge: "Unknown Person" (BBox Yellow)
            end
        end
    end
```

---

## 4. Component Design Specifications

### 4.1 ML Biometric Engine (`/ml`)
1. **Face Detection & Alignment**:
   - Model: OpenCV YuNet / SCRFD with 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner).
   - Alignment: 5-point similarity transformation warps the face into canonical 112x112 ArcFace coordinate space.
2. **ArcFace Embedding Generation**:
   - Deep residual backbone (ResNet50 / MobileFaceNet) pre-trained with Additive Angular Margin loss.
   - Generates $\mathbf{v} \in \mathbb{R}^{512}$ where $\|\mathbf{v}\|_2 = 1.0$.
   - Quantized to ONNX INT8 for CPU-optimized inference ($<150$ ms latency).
3. **Passive Liveness / Anti-Spoofing (PAD)**:
   - Evaluates texture, reflection, moiré patterns, and depth cues to distinguish genuine 3D facial skin from 2D printouts and high-definition screens.
4. **Matching & Calibration**:
   - Cosine Similarity: $\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$.
   - Calibrated Operating Points:
     - High Security ($FAR = 0.1\%$): Cosine Threshold $\approx 0.50$
     - Balanced ($FAR = 1.0\%$): Cosine Threshold $\approx 0.42$

### 4.2 Backend Architecture (`/backend`)
- **Framework**: FastAPI (async Python 3.11+)
- **Authentication**: JWT tokens, argon2 password hashing, role-based access control (`admin`, `instructor`).
- **Database & Vector Search**:
  - Production: PostgreSQL 16 + `pgvector` extension for cosine distance index (`<->` operator).
  - Development / Fallback: SQLite with in-memory cosine ranking.
- **Endpoints**:
  - `POST /auth/login`, `POST /auth/register`
  - `GET /people`, `POST /people`, `POST /people/{id}/enroll`, `DELETE /people/{id}`
  - `POST /recognize`: Single frame recognition API.
  - `POST /attendance/mark`, `GET /attendance`, `GET /attendance/export.csv`, `GET /attendance/export.xlsx`
  - `WebSocket /ws/stream`: High-efficiency streaming webcam endpoint.
  - `GET /health`, `GET /metrics` (Prometheus), `GET /model/info`

### 4.3 Frontend Architecture (`/frontend`)
- **Stack**: React 18, Vite, TypeScript, TailwindCSS, Lucide-react, Canvas / WebGL.
- **3D Hero Visualizer**: Three.js + React Three Fiber interactive particle face-mesh responding to real-time kiosk state (Idle = floating cyan grid; Scanning = pulsing blue particles; Success = green burst; Unknown = yellow alert).
- **Core Views**:
  - **Live Kiosk**: Real-time webcam canvas feed with bounding box overlay, FPS indicator, liveness badge, audio chime.
  - **Guided Enrollment Wizard**: Step-by-step angle capture (Front, Left, Right) with live quality assessment (sharpness, size, illumination) and explicit consent confirmation.
  - **Admin Analytics Dashboard**: Daily attendance summary, Recharts visualizations, filtering, CSV/Excel downloads, and student/employee management.

### 4.4 DevOps & Deployment (`/deploy`, `.github/workflows`)
- **Docker Compose**: Orchestrates `api` (FastAPI), `db` (Postgres + pgvector), and `web` (React + Nginx).
- **Continuous Integration**: Ruff linting, mypy type verification, pytest execution with coverage check, and Docker build tests.
- **Zero-Cost Deployment Topology**:
  - Backend: Hugging Face Spaces (Docker SDK) or Render/Railway.
  - Database: Neon / Supabase serverless Postgres with pgvector.
  - Frontend: Vercel or Netlify with HTTPS (mandatory for `getUserMedia` camera permissions).
