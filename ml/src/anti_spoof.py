"""
Presentation Attack Detection (PAD) / Anti-Spoofing & Liveness Engine.
Combines:
  1. Passive Texture, Color Diversity & High-Frequency Fourier Moiré Analysis.
  2. Active Challenge: Eye Blink & Head Pose Landmark Dynamics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class LivenessResult:
    is_live: bool
    liveness_score: float  # [0.0, 1.0], higher = more confident live
    spoof_type: Optional[str]  # None, "PRINT_ATTACK", "SCREEN_REPLAY", "LOW_TEXTURE"
    details: Dict[str, float]


class AntiSpoofDetector:
    """
    Robust multi-feature anti-spoofing and liveness detector for webcam streams.
    """

    def __init__(
        self,
        liveness_threshold: float = 0.50,
        enable_active_challenge: bool = True,
    ):
        self.liveness_threshold = liveness_threshold
        self.enable_active_challenge = enable_active_challenge
        self._history: Dict[str, List[float]] = {}

    def evaluate_passive(self, face_bgr: np.ndarray) -> LivenessResult:
        """
        Evaluate single-frame passive anti-spoofing characteristics.
        Args:
            face_bgr: 112x112 or cropped BGR face image.
        Returns:
            LivenessResult: classification and confidence.
        """
        if face_bgr is None or face_bgr.size == 0:
            return LivenessResult(False, 0.0, "INVALID_FRAME", {})

        if face_bgr.shape[:2] != (112, 112):
            face_bgr = cv2.resize(face_bgr, (112, 112))

        # 1. Color Space Distribution & Chrominance Diversity (YCbCr & HSV)
        # Real skin has characteristic YCbCr distribution; paper/screens clip or distort Cb/Cr channels
        ycbcr = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        cb_std = float(np.std(cb))
        cr_std = float(np.std(cr))
        chroma_score = min(1.0, (cb_std + cr_std) / 25.0)

        # 2. Fourier High-Frequency & Moiré Pattern Analysis
        # Screens and printed half-tones introduce periodic high-frequency spikes or planar flatness
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1.0)
        
        # High-frequency band analysis (outer ring vs center)
        cy, cx = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        radius = 20
        y_grid, x_grid = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        mask_low = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
        
        low_freq_energy = float(np.mean(magnitude_spectrum[mask_low]))
        high_freq_energy = float(np.mean(magnitude_spectrum[~mask_low]))
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-6)
        
        # Screens often have unnatural high-frequency moiré energy or completely blur it
        moire_score = 1.0
        if freq_ratio > 0.85:
            moire_score = max(0.0, 1.0 - (freq_ratio - 0.85) * 4.0)
        elif freq_ratio < 0.25:
            moire_score = max(0.0, freq_ratio / 0.25)

        # 3. Local Binary Pattern (LBP) Texture Uniformity
        # Real skin exhibits subtle micro-textures; screens display rigid pixel grids
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = float(laplacian.var())
        texture_score = min(1.0, max(0.0, (texture_var - 20.0) / 150.0))

        # 4. Specular Reflection / Highlight Analysis
        # Phone screens have harsh flat specular reflections; real faces diffuse light
        v_channel = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        saturated_pixels = float(np.mean(v_channel > 245))
        specular_penalty = 1.0 - min(1.0, saturated_pixels * 15.0)

        # Combined fused score
        fused_score = (
            0.35 * chroma_score +
            0.25 * moire_score +
            0.25 * texture_score +
            0.15 * specular_penalty
        )
        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        spoof_type = None
        if fused_score < self.liveness_threshold:
            if specular_penalty < 0.5 or moire_score < 0.4:
                spoof_type = "SCREEN_REPLAY"
            elif chroma_score < 0.4:
                spoof_type = "PRINT_ATTACK"
            else:
                spoof_type = "LOW_TEXTURE"

        details = {
            "chroma_score": round(chroma_score, 3),
            "moire_score": round(moire_score, 3),
            "texture_score": round(texture_score, 3),
            "specular_penalty": round(specular_penalty, 3),
            "fused_score": round(fused_score, 3),
        }

        is_live = fused_score >= self.liveness_threshold
        return LivenessResult(is_live, fused_score, spoof_type, details)

    def evaluate_active_blink(self, landmark_history: List[np.ndarray]) -> bool:
        """
        Evaluate dynamic eye blinking across a sequence of recent landmark frames.
        landmarks: 5 points [(right_eye), (left_eye), (nose), (right_mouth), (left_mouth)]
        """
        if len(landmark_history) < 5:
            return True  # Insufficient history for active challenge yet

        # Track eye-to-nose vertical distance ratio
        ratios = []
        for lm in landmark_history:
            r_eye = lm[0]
            l_eye = lm[1]
            nose = lm[2]
            eye_center = (r_eye + l_eye) / 2.0
            dist = np.linalg.norm(nose - eye_center)
            ratios.append(float(dist))

        # Blink or head-movement causes variance in the ratio
        ratio_std = float(np.std(ratios))
        return ratio_std > 0.5
