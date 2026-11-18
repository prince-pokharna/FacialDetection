"""
Open-Set Biometric Matcher and Temporal Prediction Smoother.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.json"


@dataclass
class MatchResult:
    is_known: bool
    identity_id: Optional[str]
    name: Optional[str]
    confidence: float  # Cosine similarity in [0.0, 1.0]
    distance: float  # Cosine distance = 1.0 - confidence
    threshold_used: float
    top_candidates: List[Tuple[str, float]]  # [(person_id, score), ...]


class FaceMatcher:
    """
    Open-set biometric vector matcher with cosine distance ranking and unknown rejection.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        default_threshold: float = 0.42,
    ):
        self.config_path = config_path or CONFIG_PATH
        self.threshold = default_threshold
        self._load_config()

        # In-memory vector gallery: {person_id: {"name": str, "embedding": np.ndarray (512,)}}
        self.gallery: Dict[str, Dict] = {}

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Use calibrated threshold for FAR=1% by default
                    self.threshold = float(data.get("operating_points", {}).get("FAR_1_percent", {}).get("threshold", self.threshold))
            except Exception:
                pass

    def enroll_person(self, person_id: str, name: str, embeddings: List[np.ndarray]):
        """
        Enroll a person from 1 or more embedding vectors.
        Aggregates multiple vectors by computing the normalized centroid.
        """
        if not embeddings:
            raise ValueError("At least one embedding vector is required for enrollment.")

        stacked = np.stack(embeddings, axis=0)  # Shape: (M, 512)
        mean_vec = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            normalized_prototype = (mean_vec / norm).astype(np.float32)
        else:
            normalized_prototype = mean_vec.astype(np.float32)

        self.gallery[person_id] = {
            "name": name,
            "embedding": normalized_prototype,
            "sample_count": len(embeddings),
        }

    def remove_person(self, person_id: str) -> bool:
        """Remove a person from the gallery (biometric erasure)."""
        return self.gallery.pop(person_id, None) is not None

    def match(self, query_embedding: np.ndarray, threshold: Optional[float] = None) -> MatchResult:
        """
        Match a query embedding against the gallery.
        Returns MatchResult with unknown rejection if maximum similarity < threshold.
        """
        th = threshold if threshold is not None else self.threshold

        if not self.gallery:
            return MatchResult(
                is_known=False,
                identity_id=None,
                name=None,
                confidence=0.0,
                distance=1.0,
                threshold_used=th,
                top_candidates=[],
            )

        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 1e-8:
            q = (query_embedding / query_norm).astype(np.float32)
        else:
            q = query_embedding.astype(np.float32)

        candidates = []
        for pid, data in self.gallery.items():
            db_emb = data["embedding"]
            similarity = float(np.dot(q, db_emb))  # Cosine similarity
            candidates.append((pid, data["name"], similarity))

        # Sort descending by similarity
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_id, best_name, best_score = candidates[0]

        is_match = best_score >= th
        top_k = [(c[0], round(c[2], 4)) for c in candidates[:5]]

        return MatchResult(
            is_known=is_match,
            identity_id=best_id if is_match else None,
            name=best_name if is_match else "Unknown",
            confidence=round(best_score, 4),
            distance=round(1.0 - best_score, 4),
            threshold_used=th,
            top_candidates=top_k,
        )


class TemporalSmoother:
    """
    Temporal prediction smoothing over a rolling N-frame window.
    Enforces per-person cooldown (e.g. 30 seconds) and single attendance per session/day.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_consecutive_agreements: int = 3,
        cooldown_seconds: int = 30,
    ):
        self.window_size = window_size
        self.min_consecutive_agreements = min_consecutive_agreements
        self.cooldown_seconds = cooldown_seconds

        self._window: deque = deque(maxlen=window_size)
        self._last_marked: Dict[str, datetime] = {}
        self._marked_today: set = set()
        self._last_reset_date = date.today()

    def _check_day_reset(self):
        today = date.today()
        if today != self._last_reset_date:
            self._marked_today.clear()
            self._last_reset_date = today

    def push_prediction(self, match: MatchResult, is_live: bool) -> Tuple[bool, Optional[str], str]:
        """
        Push new frame prediction into the rolling window.
        Returns:
            Tuple[can_mark, person_id, status_message]
        """
        self._check_day_reset()

        if not is_live:
            self._window.append(None)
            return False, None, "SPOOF_REJECTED"

        if not match.is_known:
            self._window.append(None)
            return False, None, "UNKNOWN_PERSON"

        self._window.append(match.identity_id)

        # Check for consecutive agreement in window
        recent = list(self._window)
        if len(recent) < self.min_consecutive_agreements:
            return False, match.identity_id, "ACCUMULATING_FRAMES"

        tail = recent[-self.min_consecutive_agreements:]
        if all(item == match.identity_id for item in tail):
            pid = match.identity_id
            now = datetime.now()

            # Date check: one record per day
            if pid in self._marked_today:
                return False, pid, "ALREADY_MARKED_TODAY"

            # Cooldown check
            if pid in self._last_marked:
                elapsed = (now - self._last_marked[pid]).total_seconds()
                if elapsed < self.cooldown_seconds:
                    return False, pid, f"COOLDOWN_ACTIVE_{int(self.cooldown_seconds - elapsed)}s"

            # Successfully marked
            self._last_marked[pid] = now
            self._marked_today.add(pid)
            return True, pid, "ATTENDANCE_MARKED"

        return False, match.identity_id, "PREDICTION_FLUCTUATING"
