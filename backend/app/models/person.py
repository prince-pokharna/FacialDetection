"""
Person and Biometric Embedding Models with privacy consent tracking.
"""

from datetime import datetime
import uuid
from sqlalchemy import Column, String, Boolean, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.db.base import Base


class Person(Base):
    __tablename__ = "people"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    external_id = Column(String(100), unique=True, index=True, nullable=True)  # e.g. Student/Employee ID
    full_name = Column(String(255), index=True, nullable=False)
    department = Column(String(255), nullable=True)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_timestamp = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    embeddings = relationship("BiometricEmbedding", back_populates="person", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="person", cascade="all, delete-orphan")


class BiometricEmbedding(Base):
    __tablename__ = "biometric_embeddings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id = Column(String(36), ForeignKey("people.id", ondelete="CASCADE"), nullable=False, index=True)
    # Stored as JSON string or base64 binary vector for cross-platform compatibility (SQLite + Postgres)
    embedding_data = Column(Text, nullable=False)
    quality_score = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="embeddings")
