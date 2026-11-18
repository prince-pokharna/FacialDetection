"""
Unit Tests for Biometric Components.
"""

import numpy as np
import pytest

from ml.src.detector import FaceDetector
from ml.src.embedder import FaceEmbedder
from ml.src.anti_spoof import AntiSpoofDetector
from ml.src.quality import BiometricQualityEvaluator
from ml.src.matcher import FaceMatcher, TemporalSmoother, MatchResult


def test_embedder_l2_normalization():
    embedder = FaceEmbedder()
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    vec = embedder.embed(dummy_face)
    assert vec.shape == (embedder.embedding_dim,)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_matcher_enroll_and_match():
    matcher = FaceMatcher(default_threshold=0.40)

    # Enroll Person A
    vec_a1 = np.random.randn(128).astype(np.float32)
    vec_a1 /= np.linalg.norm(vec_a1)
    vec_a2 = vec_a1 + np.random.randn(128).astype(np.float32) * 0.01
    vec_a2 /= np.linalg.norm(vec_a2)

    matcher.enroll_person("id-alice", "Alice", [vec_a1, vec_a2])
    assert "id-alice" in matcher.gallery

    # Query with matching vector
    query_vec = vec_a1 + np.random.randn(128).astype(np.float32) * 0.01
    query_vec /= np.linalg.norm(query_vec)
    result = matcher.match(query_vec)

    assert result.is_known is True
    assert result.identity_id == "id-alice"
    assert result.name == "Alice"
    assert result.confidence > 0.80


def test_matcher_rejects_unknown():
    matcher = FaceMatcher(default_threshold=0.50)

    # Enroll Person A
    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0  # Unit vector along axis 0
    matcher.enroll_person("id-alice", "Alice", [vec_a])

    # Query with orthogonal vector (completely different identity)
    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[100] = 1.0  # Unit vector along axis 100
    result = matcher.match(vec_b)

    assert result.is_known is False
    assert result.identity_id is None
    assert result.name == "Unknown"
    assert result.confidence < 0.10


def test_quality_evaluator_catches_blur():
    evaluator = BiometricQualityEvaluator(min_sharpness=60.0)
    # Uniform/blurred image has zero Laplacian variance
    blurred_img = np.ones((112, 112, 3), dtype=np.uint8) * 128
    res = evaluator.evaluate(blurred_img)
    assert res.is_valid is False
    assert any("blurry" in msg for msg in res.feedback)


def test_temporal_smoother_consecutive_agreement():
    smoother = TemporalSmoother(window_size=5, min_consecutive_agreements=3, cooldown_seconds=30)

    match = MatchResult(
        is_known=True,
        identity_id="person-1",
        name="Charlie",
        confidence=0.85,
        distance=0.15,
        threshold_used=0.42,
        top_candidates=[],
    )

    # Frame 1: accumulating
    can_mark1, _, status1 = smoother.push_prediction(match, is_live=True)
    assert can_mark1 is False

    # Frame 2: accumulating
    can_mark2, _, status2 = smoother.push_prediction(match, is_live=True)
    assert can_mark2 is False

    # Frame 3: 3 consecutive agreements reached -> mark!
    can_mark3, pid, status3 = smoother.push_prediction(match, is_live=True)
    assert can_mark3 is True
    assert pid == "person-1"
    assert status3 == "ATTENDANCE_MARKED"

    # Frame 4: immediately after -> already marked today
    can_mark4, _, status4 = smoother.push_prediction(match, is_live=True)
    assert can_mark4 is False
    assert status4 == "ALREADY_MARKED_TODAY"


def test_anti_spoof_rejects_zero_texture():
    anti_spoof = AntiSpoofDetector(liveness_threshold=0.50)
    # Flat grey image with no skin chrominance and no texture
    flat_frame = np.ones((112, 112, 3), dtype=np.uint8) * 128
    result = anti_spoof.evaluate_passive(flat_frame)
    assert result.is_live is False
    assert result.spoof_type is not None
