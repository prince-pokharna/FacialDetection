"""
Recognition API Tests.
"""

import base64
import cv2
import numpy as np


def test_recognize_empty_frame(client):
    # Black blank frame
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")

    response = client.post(
        "/recognize",
        json={"image_base64": b64_str, "session_name": "TestSession", "auto_mark": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["faces_detected"] == 0
    assert data["attendance_marked"] is False


def test_recognize_invalid_base64(client):
    response = client.post(
        "/recognize",
        json={"image_base64": "not_a_valid_base64_string", "session_name": "TestSession"},
    )
    assert response.status_code == 400
