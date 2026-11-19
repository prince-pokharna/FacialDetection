"""
Biometric Quality Gate for Enrollment and Recognition Frames.
Ensures images meet minimum requirements for sharpness, illumination, resolution, and pose.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class QualityResult:
    is_valid: bool
    overall_score: float  # [0.0, 1.0]
    sharpness: float
    brightness: float
    resolution: Tuple[int, int]
    pose_ok: bool
    feedback: List[str]


class BiometricQualityEvaluator:
    """
    Evaluates face crops before enrollment to prevent noisy or degraded vectors in the database.
    """

    def __init__(
        self,
        min_sharpness: float = 65.0,
        min_width: int = 60,
        min_height: int = 60,
        min_brightness: float = 40.0,
        max_brightness: float = 230.0,
        max_yaw_angle: float = 30.0,
    ):
        self.min_sharpness = min_sharpness
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_yaw_angle = max_yaw_angle

    def evaluate(
        self,
        face_bgr: np.ndarray,
        landmarks: np.ndarray = None,
        bbox: Tuple[int, int, int, int] = None,
    ) -> QualityResult:
        feedback = []
        if face_bgr is None or face_bgr.size == 0:
            return QualityResult(False, 0.0, 0.0, (0, 0), False, ["No face image provided"])

        h, w = face_bgr.shape[:2]
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness via Laplacian Variance
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            feedback.append(f"Image too blurry ({sharpness:.1f} < {self.min_sharpness:.1f})")

        # 2. Illumination / Brightness & Contrast
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.min_brightness:
            feedback.append("Lighting too dark; move into better light")
        elif mean_brightness > self.max_brightness:
            feedback.append("Image overexposed / too bright")

        # 3. Resolution Check
        eval_w = bbox[2] if bbox else w
        eval_h = bbox[3] if bbox else h
        if eval_w < self.min_width or eval_h < self.min_height:
            feedback.append(f"Face too small ({eval_w}x{eval_h} < {self.min_width}x{self.min_height})")

        # 4. Pose estimation from 5 landmarks (Right Eye, Left Eye, Nose, Right Mouth, Left Mouth)
        pose_ok = True
        if landmarks is not None and landmarks.shape == (5, 2):
            r_eye = landmarks[0]
            l_eye = landmarks[1]
            nose = landmarks[2]

            # Eye distance vs nose symmetry indicates yaw
            dist_r = float(np.linalg.norm(nose - r_eye))
            dist_l = float(np.linalg.norm(nose - l_eye))
            total_dist = dist_r + dist_l
            if total_dist > 1e-4:
                yaw_ratio = abs(dist_r - dist_l) / total_dist
                if yaw_ratio > 0.45:
                    pose_ok = False
                    feedback.append("Face turned too far to the side; please face the camera directly")

            # Eye tilt angle (roll)
            dx = float(l_eye[0] - r_eye[0])
            dy = float(l_eye[1] - r_eye[1])
            roll_angle = abs(np.degrees(np.arctan2(dy, dx)))
            if roll_angle > 25.0 and (180.0 - roll_angle) > 25.0:
                pose_ok = False
                feedback.append("Please keep your head upright")

        # Compute normalized score
        s_score = min(1.0, sharpness / 200.0)
        b_score = 1.0 - abs(mean_brightness - 128.0) / 128.0
        r_score = min(1.0, (eval_w * eval_h) / (120.0 * 120.0))
        p_score = 1.0 if pose_ok else 0.4

        overall = float(np.clip(0.35 * s_score + 0.25 * b_score + 0.20 * r_score + 0.20 * p_score, 0.0, 1.0))
        is_valid = len(feedback) == 0

        if is_valid:
            feedback.append("Quality Good")

        return QualityResult(
            is_valid=is_valid,
            overall_score=round(overall, 3),
            sharpness=round(sharpness, 1),
            brightness=round(mean_brightness, 1),
            resolution=(eval_w, eval_h),
            pose_ok=pose_ok,
            feedback=feedback,
        )
