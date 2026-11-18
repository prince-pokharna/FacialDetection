"""
ML Package for Production-Grade Open-Set Face Recognition Attendance System.
Components:
  - Face Detection & 5-point landmark alignment (YuNet / SCRFD)
  - 512-d ArcFace feature embedding with L2 normalization
  - Anti-spoofing & liveness verification (Passive PAD + Active Challenge)
  - Biometric quality gating (sharpness, pose, illumination)
  - Open-set cosine matching & temporal smoothing
"""

from ml.src.detector import FaceDetector, DetectedFace
from ml.src.embedder import FaceEmbedder
from ml.src.anti_spoof import AntiSpoofDetector, LivenessResult
from ml.src.quality import BiometricQualityEvaluator, QualityResult
from ml.src.matcher import FaceMatcher, MatchResult, TemporalSmoother

__all__ = [
    "FaceDetector",
    "DetectedFace",
    "FaceEmbedder",
    "AntiSpoofDetector",
    "LivenessResult",
    "BiometricQualityEvaluator",
    "QualityResult",
    "FaceMatcher",
    "MatchResult",
    "TemporalSmoother",
]
