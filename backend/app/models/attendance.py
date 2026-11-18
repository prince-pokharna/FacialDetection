"""
Attendance Record Model.
"""

from datetime import datetime, date
import uuid
from sqlalchemy import Column, String, Float, DateTime, Date, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.db.base import Base


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id = Column(String(36), ForeignKey("people.id", ondelete="CASCADE"), nullable=False, index=True)
    person_name = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    liveness_score = Column(Float, nullable=False)
    session_name = Column(String(100), default="Main", nullable=False)
    date = Column(Date, default=date.today, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    marked_by = Column(String(100), default="SYSTEM_KIOSK", nullable=False)

    person = relationship("Person", back_populates="attendance_records")
