"""
Weights Manager for automated retrieval and validation of ONNX biometric models.
"""

import os
import urllib.request
import logging
from pathlib import Path

logger = logging.getLogger("weights_manager")

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

MODEL_CONFIGS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "urls": [
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        "min_size": 200_000,
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "urls": [
            "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
        "min_size": 35_000_000,
    },
}


def get_weights_path(model_key: str) -> Path:
    """
    Ensure the requested model weight file is available locally, downloading if necessary.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_key]
    target_path = WEIGHTS_DIR / config["filename"]

    if target_path.exists() and target_path.stat().st_size >= config["min_size"]:
        return target_path

    # If partial/corrupted, delete first
    if target_path.exists():
        target_path.unlink()

    last_error = None
    for url in config["urls"]:
        logger.info("Downloading %s from %s...", model_key, url)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, "wb") as out_file:
                chunk_size = 256 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            
            if target_path.stat().st_size >= config["min_size"]:
                logger.info("Downloaded %s successfully (%d bytes)", config["filename"], target_path.stat().st_size)
                return target_path
            else:
                logger.warning("Downloaded file too small (%d bytes), trying next URL...", target_path.stat().st_size)
                target_path.unlink()
        except Exception as exc:
            logger.warning("Failed from %s: %s", url, exc)
            last_error = exc
            if target_path.exists():
                target_path.unlink()

    raise RuntimeError(f"Could not download {model_key} model. Last error: {last_error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p1 = get_weights_path("yunet")
    print(f"[OK] YuNet ready: {p1} ({p1.stat().st_size} bytes)")
    p2 = get_weights_path("sface")
    print(f"[OK] SFace ready: {p2} ({p2.stat().st_size} bytes)")
