"""
Attendance and Real-Time Recognition Schemas.
"""

from datetime import datetime, date
from typing import List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class AttendanceRecordOut(BaseModel):
    id: str
    person_id: str
    person_name: str
    confidence: float
    liveness_score: float
    session_name: str
    date: date
    timestamp: datetime
    marked_by: str

    model_config = ConfigDict(from_attributes=True)


class FaceDetectionOut(BaseModel):
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    detection_score: float
    is_live: bool
    liveness_score: float
    spoof_type: Optional[str]
    is_known: bool
    person_id: Optional[str]
    person_name: Optional[str]
    confidence: float


class FrameRecognitionRequest(BaseModel):
    image_base64: str = Field(..., description="JPEG/PNG image encoded as base64 string")
    session_name: str = Field("Main", description="Attendance session name")
    auto_mark: bool = Field(True, description="Whether to mark attendance automatically if recognized")


class FrameRecognitionResponse(BaseModel):
    faces_detected: int
    faces: List[FaceDetectionOut]
    attendance_marked: bool
    marked_person_name: Optional[str]
    message: str
