"""
Biometric Service orchestrating face detection, liveness, embedding extraction, and matching.
"""

import base64
import json
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.person import Person, BiometricEmbedding
from ml.src.detector import FaceDetector, DetectedFace
from ml.src.embedder import FaceEmbedder
from ml.src.anti_spoof import AntiSpoofDetector, LivenessResult
from ml.src.quality import BiometricQualityEvaluator, QualityResult
from ml.src.matcher import FaceMatcher, MatchResult, TemporalSmoother

logger = logging.getLogger("biometric_service")


class BiometricService:
    """
    Central biometric management engine for the FastAPI application.
    """

    def __init__(self):
        logger.info("Initializing Biometric Engine...")
        self.detector = FaceDetector(conf_threshold=0.60)
        self.embedder = FaceEmbedder()
        self.anti_spoof = AntiSpoofDetector(liveness_threshold=settings.LIVENESS_THRESHOLD)
        self.quality = BiometricQualityEvaluator()
        self.matcher = FaceMatcher(default_threshold=settings.FAR_THRESHOLD)
        self.smoother = TemporalSmoother(
            window_size=settings.SMOOTHING_WINDOW,
            min_consecutive_agreements=settings.CONSECUTIVE_AGREEMENTS,
            cooldown_seconds=settings.COOLDOWN_SECONDS,
        )
        self._gallery_loaded = False

    def load_gallery(self, db: Session):
        """Preload all enrolled embeddings from the database into the matcher gallery."""
        people = db.query(Person).filter(Person.is_active == True).all()
        loaded_count = 0
        for person in people:
            embeddings = []
            for emb_record in person.embeddings:
                try:
                    vec = np.array(json.loads(emb_record.embedding_data), dtype=np.float32)
                    if vec.shape == (512,):
                        embeddings.append(vec)
                except Exception as exc:
                    logger.warning("Failed parsing embedding for person %s: %s", person.id, exc)

            if embeddings:
                self.matcher.enroll_person(person.id, person.full_name, embeddings)
                loaded_count += 1

        self._gallery_loaded = True
        logger.info("Biometric gallery loaded %d people from database.", loaded_count)

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert raw image bytes to OpenCV BGR numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def decode_base64_image(self, base64_str: str) -> Optional[np.ndarray]:
        """Convert base64 string (with or without data:image/... prefix) to OpenCV BGR array."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        try:
            image_bytes = base64.b64decode(base64_str)
            return self.decode_image_bytes(image_bytes)
        except Exception as exc:
            logger.warning("Base64 decoding failed: %s", exc)
            return None

    def process_enrollment_images(
        self,
        images_bytes_list: List[bytes],
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Process a batch of enrollment images with quality gating.
        Returns:
            Tuple of (valid_embeddings_list, feedback_records_list)
        """
        valid_embeddings = []
        feedbacks = []

        for idx, img_bytes in enumerate(images_bytes_list):
            img_bgr = self.decode_image_bytes(img_bytes)
            if img_bgr is None:
                feedbacks.append({
                    "image_index": idx,
                    "is_valid": False,
                    "quality_score": 0.0,
                    "feedback": ["Corrupted or unreadable image format"]
                })
                continue

            faces = self.detector.detect(img_bgr)
            if not faces:
                feedbacks.append({
                    "image_index": idx,
                    "is_valid": False,
                    "quality_score": 0.0,
                    "feedback": ["No face detected in this image"]
                })
                continue

            if len(faces) > 1:
                feedbacks.append({
                    "image_index": idx,
                    "is_valid": False,
                    "quality_score": 0.3,
                    "feedback": ["Multiple faces detected. Ensure only one person is in the frame."]
                })
                continue

            face = faces[0]
            # Biometric quality check
            q_res = self.quality.evaluate(face.aligned_face, face.landmarks, face.bbox)
            if not q_res.is_valid:
                feedbacks.append({
                    "image_index": idx,
                    "is_valid": False,
                    "quality_score": q_res.overall_score,
                    "feedback": q_res.feedback,
                })
                continue

            # Extract 512-d feature embedding
            embedding = self.embedder.embed(face, full_image=img_bgr)
            valid_embeddings.append(embedding)

            feedbacks.append({
                "image_index": idx,
                "is_valid": True,
                "quality_score": q_res.overall_score,
                "feedback": ["Quality Good - Enrolled Successfully"],
            })

        return valid_embeddings, feedbacks

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Process a single kiosk/webcam frame:
        Detection -> 5-pt Alignment -> Anti-Spoof Liveness -> Embedding -> Open-Set Match.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        detected_faces = self.detector.detect(frame_bgr)
        results = []

        for face in detected_faces:
            # 1. Evaluate Anti-Spoof Liveness
            liveness = self.anti_spoof.evaluate_passive(face.aligned_face)

            # 2. Extract Embedding
            embedding = self.embedder.embed(face, full_image=frame_bgr)

            # 3. Match against Gallery
            match = self.matcher.match(embedding, threshold=threshold)

            results.append({
                "bbox": face.bbox,
                "detection_score": round(face.score, 3),
                "is_live": liveness.is_live,
                "liveness_score": liveness.liveness_score,
                "spoof_type": liveness.spoof_type,
                "is_known": match.is_known,
                "person_id": match.identity_id,
                "person_name": match.name,
                "confidence": match.confidence,
                "match_object": match,
            })

        return results


biometric_service = BiometricService()
