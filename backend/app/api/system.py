"""
System Health, Prometheus Metrics, and Model Metadata Endpoints.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session
from sqlalchemy import text
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from backend.app.db.session import get_db
from backend.app.services.biometric_service import biometric_service
from backend.app.core.config import settings

router = APIRouter(tags=["System"])

# Prometheus Metrics
INSPECTION_REQUESTS = Counter("face_recognition_requests_total", "Total recognition requests processed")
FACES_DETECTED_COUNT = Counter("face_recognition_faces_detected_total", "Total faces detected")
ATTENDANCE_MARKED_COUNT = Counter("attendance_marked_total", "Total attendance markings succeeded")
INFERENCE_LATENCY = Histogram("face_recognition_latency_seconds", "Inference latency per frame")


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """System health verification and database status."""
    db_healthy = False
    try:
        db.execute(text("SELECT 1"))
        db_healthy = True
    except Exception:
        db_healthy = False

    return {
        "status": "healthy" if db_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "database_connected": db_healthy,
        "enrolled_gallery_size": len(biometric_service.matcher.gallery),
    }


@router.get("/metrics")
def get_metrics():
    """Prometheus exposition format metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/model/info")
def get_model_info():
    """Active biometric model architecture and calibration metadata."""
    return {
        "architecture": "Open-Set Metric Learning (Detect -> Align -> Embed -> Match)",
        "detector": "OpenCV YuNet (5-point canonical ArcFace landmark alignment)",
        "embedder": "SFace / ArcFace Deep Residual Backbone",
        "embedding_dimensions": biometric_service.embedder.embedding_dim,
        "liveness_method": "Multi-feature Passive PAD (Texture + Moiré Fourier + Chroma)",
        "thresholds": {
            "cosine_threshold": biometric_service.matcher.threshold,
            "liveness_threshold": settings.LIVENESS_THRESHOLD,
            "smoothing_window": settings.SMOOTHING_WINDOW,
            "cooldown_seconds": settings.COOLDOWN_SECONDS,
        },
        "gallery_identities_loaded": len(biometric_service.matcher.gallery),
    }
