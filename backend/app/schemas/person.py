"""
Person and Biometric Enrollment Schemas.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PersonCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255)
    external_id: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=255)
    consent_given: bool = Field(..., description="Explicit biometric consent must be granted")


class PersonUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    external_id: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None


class PersonOut(BaseModel):
    id: str
    external_id: Optional[str]
    full_name: str
    department: Optional[str]
    consent_given: bool
    consent_timestamp: Optional[datetime]
    is_active: bool
    created_at: datetime
    enrolled_embeddings_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class EnrollmentFeedback(BaseModel):
    image_index: int
    is_valid: bool
    quality_score: float
    feedback: List[str]


class EnrollmentResponse(BaseModel):
    person_id: str
    full_name: str
    total_images_processed: int
    valid_embeddings_stored: int
    feedbacks: List[EnrollmentFeedback]
    message: str
