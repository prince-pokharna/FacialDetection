"""
WebSocket Streaming Endpoint for Low-Latency Real-Time Attendance Kiosks.
"""

import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from backend.app.db.session import SessionLocal
from backend.app.services.biometric_service import biometric_service
from backend.app.services.attendance_service import attendance_service
from backend.app.api.system import (
    INSPECTION_REQUESTS,
    FACES_DETECTED_COUNT,
    ATTENDANCE_MARKED_COUNT,
    INFERENCE_LATENCY,
)

logger = logging.getLogger("websocket_stream")
router = APIRouter(tags=["Stream"])


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for high-frequency video stream frame processing.
    Expects JSON: {"frame": "<base64_jpeg>", "session": "Main"}
    Returns JSON with detected bounding boxes, identity tags, liveness status, and mark events.
    """
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/stream")
    db: Session = SessionLocal()

    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                msg = json.loads(text_data)
                frame_b64 = msg.get("frame", "")
                session_name = msg.get("session", "Main")
            except Exception:
                frame_b64 = text_data
                session_name = "Main"

            if not frame_b64:
                continue

            t0 = time.perf_counter()
            INSPECTION_REQUESTS.inc()

            frame_bgr = biometric_service.decode_base64_image(frame_b64)
            if frame_bgr is None:
                await websocket.send_json({"error": "Invalid frame data"})
                continue

            detections = biometric_service.process_frame(frame_bgr)
            FACES_DETECTED_COUNT.inc(len(detections))

            payload_faces = []
            attendance_event = None

            for det in detections:
                face_data = {
                    "bbox": det["bbox"],
                    "detection_score": det["detection_score"],
                    "is_live": det["is_live"],
                    "liveness_score": det["liveness_score"],
                    "spoof_type": det["spoof_type"],
                    "is_known": det["is_known"],
                    "person_id": det["person_id"],
                    "person_name": det["person_name"],
                    "confidence": det["confidence"],
                }
                payload_faces.append(face_data)

                # Process temporal smoothing
                match_obj = det["match_object"]
                can_mark, pid, smooth_status = biometric_service.smoother.push_prediction(
                    match=match_obj,
                    is_live=det["is_live"],
                )

                face_data["smooth_status"] = smooth_status

                if can_mark and pid:
                    success, record, log_msg = attendance_service.mark_attendance(
                        db=db,
                        person_id=pid,
                        confidence=det["confidence"],
                        liveness_score=det["liveness_score"],
                        session_name=session_name,
                        marked_by="KIOSK_WEBSOCKET",
                    )
                    if success:
                        ATTENDANCE_MARKED_COUNT.inc()
                        attendance_event = {
                            "event": "ATTENDANCE_SUCCESS",
                            "person_id": pid,
                            "person_name": det["person_name"],
                            "confidence": det["confidence"],
                            "timestamp": record.timestamp.isoformat(),
                            "message": f"Welcome, {det['person_name']}! Attendance recorded.",
                        }

            latency_ms = (time.perf_counter() - t0) * 1000.0
            INFERENCE_LATENCY.observe(latency_ms / 1000.0)

            response = {
                "faces_detected": len(payload_faces),
                "faces": payload_faces,
                "latency_ms": round(latency_ms, 1),
                "attendance_event": attendance_event,
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from /ws/stream")
    except Exception as exc:
        logger.error("WebSocket stream error: %s", exc)
    finally:
        db.close()
