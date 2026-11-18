"""
Face Detection and 5-Point Landmark Alignment using YuNet (SCRFD/RetinaFace class).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ml.src.weights_manager import get_weights_path

# Standard 112x112 ArcFace canonical landmark coordinates (5 points)
ARCFACE_REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # Right eye
        [73.5318, 51.5014],  # Left eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Right mouth corner
        [70.7299, 92.2041],  # Left mouth corner
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    """Represents a single detected face with spatial landmarks and aligned crop."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    landmarks: np.ndarray  # Shape: (5, 2)
    aligned_face: np.ndarray  # Shape: (112, 112, 3), BGR uint8
    raw_face_array: np.ndarray  # Direct output vector from YuNet [x, y, w, h, x1, y1, ...]


class FaceDetector:
    """
    High-performance face detector utilizing OpenCV YuNet with 5-point facial landmark alignment.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ):
        if model_path is None:
            self.model_path = str(get_weights_path("yunet"))
        else:
            self.model_path = model_path

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        # Initialize YuNet detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in an image (BGR format).
        Returns a list of DetectedFace objects, sorted by score descending.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))

        _, raw_faces = self.detector.detect(image)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        results: List[DetectedFace] = []
        for face_data in raw_faces:
            x, y, box_w, box_h = face_data[0:4].astype(int)
            score = float(face_data[14])

            # Ensure bounding box is within frame boundaries
            x = max(0, x)
            y = max(0, y)
            box_w = min(box_w, w - x)
            box_h = min(box_h, h - y)

            if box_w <= 10 or box_h <= 10:
                continue

            # Extract 5 landmarks (x, y pairs from index 4 to 13)
            landmarks = face_data[4:14].reshape((5, 2)).astype(np.float32)

            # Similarity transform alignment to 112x112 canonical coordinates
            aligned = self.align_face(image, landmarks)

            results.append(
                DetectedFace(
                    bbox=(int(x), int(y), int(box_w), int(box_h)),
                    score=score,
                    landmarks=landmarks,
                    aligned_face=aligned,
                    raw_face_array=face_data,
                )
            )

        # Sort by detection confidence
        results.sort(key=lambda f: f.score, reverse=True)
        return results

    @staticmethod
    def align_face(image: np.ndarray, landmarks: np.ndarray, target_size: int = 112) -> np.ndarray:
        """
        Perform 5-point similarity transformation alignment to canonical ArcFace space (112x112).
        """
        assert landmarks.shape == (5, 2), "Landmarks must be shape (5, 2)"
        
        # Estimate similarity transform using least-squares affine partial
        trans_matrix, inliers = cv2.estimateAffinePartial2D(
            landmarks,
            ARCFACE_REFERENCE_LANDMARKS,
            method=cv2.LMEDS
        )

        if trans_matrix is None:
            # Fallback to simple crop if transform estimation fails
            x_min, y_min = np.min(landmarks, axis=0).astype(int)
            x_max, y_max = np.max(landmarks, axis=0).astype(int)
            pad_w = int((x_max - x_min) * 0.3)
            pad_h = int((y_max - y_min) * 0.3)
            
            x1 = max(0, x_min - pad_w)
            y1 = max(0, y_min - pad_h)
            x2 = min(image.shape[1], x_max + pad_w)
            y2 = min(image.shape[0], y_max + pad_h)
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return cv2.resize(image, (target_size, target_size))
            return cv2.resize(crop, (target_size, target_size))

        aligned = cv2.warpAffine(
            image,
            trans_matrix,
            (target_size, target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned
