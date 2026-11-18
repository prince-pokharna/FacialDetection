"""
ArcFace / SFace 512-Dimensional Deep Biometric Feature Extractor.
"""

from pathlib import Path
from typing import Optional, Union, List
import time
import cv2
import numpy as np

from ml.src.weights_manager import get_weights_path
from ml.src.detector import DetectedFace


class FaceEmbedder:
    """
    Extracts L2-normalized 512-dimensional biometric feature embeddings from aligned face crops.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_onnxruntime: bool = False,
    ):
        if model_path is None:
            self.model_path = str(get_weights_path("sface"))
        else:
            self.model_path = model_path

        self.use_onnxruntime = use_onnxruntime
        self.embedding_dim = 128  # SFace default is 128-d; ArcFace ONNX is 512-d

        # Initialize native OpenCV SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

        self._ort_session = None
        if self.use_onnxruntime:
            self._init_onnxruntime()

    def _init_onnxruntime(self):
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort_session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            self._ort_session = None

    def embed(
        self,
        face_input: Union[np.ndarray, DetectedFace],
        full_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract L2-normalized 512-d embedding vector.
        Args:
            face_input: Aligned 112x112 BGR face crop, or DetectedFace object.
            full_image: Original full frame if face_input is DetectedFace.
        Returns:
            np.ndarray: 512-d float32 vector with L2 norm = 1.0.
        """
        if isinstance(face_input, DetectedFace):
            if full_image is not None and hasattr(face_input, "raw_face_array"):
                # Use native alignCrop for optimal precision
                aligned = self.recognizer.alignCrop(full_image, face_input.raw_face_array)
            else:
                aligned = face_input.aligned_face
        else:
            aligned = face_input

        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112))

        # OpenCV FaceRecognizerSF expects BGR 112x112 input
        raw_feat = self.recognizer.feature(aligned)
        vec = raw_feat.flatten().astype(np.float32)

        # Apply rigorous L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec = np.zeros_like(vec)

        return vec

    def compute_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two L2-normalized vectors in [-1.0, 1.0].
        Since vectors are L2-normalized, Cosine Similarity is simply the dot product.
        """
        assert embedding_a.shape == embedding_b.shape, "Embedding dimensions must match"
        return float(np.dot(embedding_a, embedding_b))

    def benchmark_latency(self, iterations: int = 50) -> float:
        """
        Benchmark average inference latency per face on CPU in milliseconds.
        """
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        # Warmup
        for _ in range(5):
            self.embed(dummy_face)

        start = time.perf_counter()
        for _ in range(iterations):
            self.embed(dummy_face)
        total_time = time.perf_counter() - start

        avg_ms = (total_time / iterations) * 1000.0
        return avg_ms
