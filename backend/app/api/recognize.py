"""
Real-Time Face Recognition Endpoint.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.attendance import (
    FrameRecognitionRequest,
    FrameRecognitionResponse,
    FaceDetectionOut,
)
from backend.app.services.biometric_service import biometric_service
from backend.app.services.attendance_service import attendance_service

router = APIRouter(tags=["Recognition"])


@router.post("/recognize", response_model=FrameRecognitionResponse)
def recognize_frame(
    request: FrameRecognitionRequest,
    db: Session = Depends(get_db),
):
    """
    Recognize faces in a single video frame (base64).
    Performs detection, anti-spoof liveness verification, ArcFace matching,
    and optional automated attendance logging.
    """
    frame_bgr = biometric_service.decode_base64_image(request.image_base64)
    if frame_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data",
        )

    detections = biometric_service.process_frame(frame_bgr)
    if not detections:
        return FrameRecognitionResponse(
            faces_detected=0,
            faces=[],
            attendance_marked=False,
            marked_person_name=None,
            message="No face detected in frame",
        )

    faces_out = []
    attendance_marked = False
    marked_name = None
    status_message = "Scan completed"

    for det in detections:
        faces_out.append(
            FaceDetectionOut(
                bbox=det["bbox"],
                detection_score=det["detection_score"],
                is_live=det["is_live"],
                liveness_score=det["liveness_score"],
                spoof_type=det["spoof_type"],
                is_known=det["is_known"],
                person_id=det["person_id"],
                person_name=det["person_name"],
                confidence=det["confidence"],
            )
        )

        # Automated attendance marking
        if request.auto_mark and det["is_known"] and det["is_live"] and not attendance_marked:
            success, record, msg = attendance_service.mark_attendance(
                db=db,
                person_id=det["person_id"],
                confidence=det["confidence"],
                liveness_score=det["liveness_score"],
                session_name=request.session_name,
                marked_by="KIOSK_STREAM",
            )
            if success:
                attendance_marked = True
                marked_name = det["person_name"]
                status_message = f"Attendance marked for {marked_name}"
            else:
                status_message = msg

    return FrameRecognitionResponse(
        faces_detected=len(faces_out),
        faces=faces_out,
        attendance_marked=attendance_marked,
        marked_person_name=marked_name,
        message=status_message,
    )
