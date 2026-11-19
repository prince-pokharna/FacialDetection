# Codebase & Architecture Audit Report

**Date**: November 2026  
**Auditor**: ML & DevOps Engineering  
**Repository**: [github.com/prince-pokharna/FacialDetection](https://github.com/prince-pokharna/FacialDetection)

---

## 1. Executive Summary

An in-depth technical audit of the legacy repository was conducted to assess machine learning performance, software architecture, security, and production readiness. 

The original codebase was constructed as a proof-of-concept desktop prototype using a closed-set classification model. While functional as an initial experimental script, it suffers from severe architectural bottlenecks that prevent real-world or production deployment—specifically: **an inability to enroll new individuals without retraining**, **vulnerability to photo/screen spoofing**, **unverified and conflicting accuracy claims**, **weak Haar Cascade detection**, and **absence of tests, APIs, containerization, or web interfaces**.

---

## 2. Real Measured Metrics vs. Published Claims

| Metric / Attribute | README Claims | Measured Reality in Repo | Source / Evidence |
| :--- | :--- | :--- | :--- |
| **Model Backbone** | EfficientNetB0 (4.0M params) | MobileNetV2 | `legacy/model_train.py` line 4, 96 |
| **Claimed Accuracy** | 75%+ up to 95%+ | **60.39% Best / 59.74% Final** | `legacy/training_log.txt` lines 16035-16037 |
| **Referenced Scripts** | `model_train_improved.py` | Missing from tree; contents merged into `model_train.py` | `legacy/training_log.txt`, commit history |
| **Input Resolution** | 128x128 px vs 160x160 px | Conflicted: `preprocess.py` uses 128x128, `model_train.py` uses 160x160 | `legacy/preprocess.py:16`, `legacy/model_train.py:15` |
| **Dataset Identities** | 25 identities | VGGFace2 25-identity subset | `legacy/labels.json` |
| **Detection Backend** | "State of the art" | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) | `legacy/webcam_integration.py:81` |
| **Liveness Check** | Claimed in roadmap | None implemented (photo presentation attack succeeds 100%) | `legacy/webcam_integration.py` |

### Key Findings from Training Logs (`training_log.txt`)
- The model trained for two phases: Phase 1 (Transfer learning feature extraction) and Phase 2 (Fine-tuning with low learning rate `5e-6`).
- Convergence plateaued at **60.39% validation accuracy** across the 25 classes.
- Per-class recall varied wildly, dropping as low as **29.63%** on class `n000115` and **30.77%** on class `n000501`.
- The claim of "75%+ accuracy" or "95%+ accuracy" was aspirational and cannot be reproduced with the legacy pipeline.

---

## 3. Detailed Technical Deficiencies

### 3.1 Machine Learning Flaws
1. **Closed-Set Softmax Classification**:
   - The network outputs a 25-dimensional Softmax probability vector.
   - Enrolling a 26th individual requires regenerating dataset directories, adjusting the output layer, and fully retraining the neural network.
   - Softmax enforces $\sum p_i = 1.0$. When presented with an unseen person, the model is forced to assign 100% of the probability mass across the 25 known classes, generating confident false positives.
2. **Haar Cascade Detection Weakness**:
   - Haar Cascade detection is brittle under lighting changes, in-plane and out-of-plane head rotations ($>15^\circ$), partial occlusions (glasses, masks, hands), and facial expressions.
   - Faces are cropped with arbitrary bounding boxes without 5-point facial landmark alignment (eyes, nose, mouth corners), causing severe spatial misalignment in feature space.
3. **No Presentation Attack Detection (PAD / Liveness)**:
   - Any static photograph on paper or a smartphone display held to the camera is recognized as a live human and immediately logged into the database.

### 3.2 Software Engineering & Architectural Flaws
1. **Monolithic Script Execution**:
   - `webcam_integration.py` couples frame capture (`cv2.VideoCapture`), face detection, model forward pass, UI rendering (`cv2.imshow`), and SQLite writes in a single-threaded blocking loop.
   - Frame rate degrades under inference delay. Any SQLite lock blocks camera capture.
2. **Missing Client-Server Architecture**:
   - The system is locked to a local desktop GUI window. Remote browser access via webcam (`getUserMedia`) is impossible.
3. **Absence of Automated Verification**:
   - Zero unit tests, integration tests, or regression tests.
   - No CI/CD workflow to validate code quality or prevent breaking changes.

### 3.3 Security, Privacy & Compliance Risks
1. **Biometric Privacy Violation**:
   - Storing facial data without explicit user consent violates major data privacy frameworks (including India's Digital Personal Data Protection Act 2023, Section 6, and EU GDPR Article 9).
   - No audit trail or mechanism for biometric data deletion ("Right to be Forgotten").
2. **VGGFace2 Dataset Licensing**:
   - VGGFace2 is licensed strictly for non-commercial research purposes. Including or shipping raw VGGFace2 images in production artifacts violates licensing terms.
3. **Hardcoded Secrets and File Paths**:
   - Database paths, camera IDs, and configurations were hardcoded directly in script bodies without environment variable encapsulation.

---

## 4. Remediation Strategy

To achieve a production-grade system, the entire architecture will be overhauled:
1. **Open-Set Deep Metric Learning**: Migrate to **ArcFace (Additive Angular Margin Loss)** producing 512-dimensional L2-normalized embeddings, coupled with SCRFD / YuNet 5-point landmark alignment to 112x112 px. Enrollment takes $<1$ second via dynamic vector insertion into pgvector/FAISS without any retraining.
2. **Anti-Spoofing Pipeline**: Introduce passive liveness detection (MiniFASNet / Silent-Face-Anti-Spoofing) with active challenge support (FaceMesh blink/head-rotation detection).
3. **Modern Microservice Architecture**:
   - **Backend**: FastAPI with async SQLAlchemy 2.0, JWT authentication, pgvector/SQLite storage, rate limiting, and WebSocket frame streaming.
   - **Frontend**: React 18 + Vite + TypeScript + TailwindCSS, responsive UI, 3D interactive hero scene via Three.js / React Three Fiber, and camera capture.
4. **DevOps & Testing**: Multi-stage Dockerfiles, docker-compose orchestration, GitHub Actions CI/CD pipelines, and automated smoke testing.
