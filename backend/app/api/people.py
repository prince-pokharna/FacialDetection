"""
People Management and Biometric Enrollment Endpoints.
"""

from datetime import datetime
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user, require_admin
from backend.app.db.session import get_db
from backend.app.models.person import Person, BiometricEmbedding
from backend.app.models.user import User
from backend.app.schemas.person import PersonCreate, PersonOut, PersonUpdate, EnrollmentResponse
from backend.app.services.biometric_service import biometric_service

router = APIRouter(prefix="/people", tags=["People & Enrollment"])


@router.get("", response_model=List[PersonOut])
def list_people(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all registered people and their biometric status."""
    query = db.query(Person).filter(Person.is_active == True)
    if search:
        query = query.filter(Person.full_name.ilike(f"%{search}%"))
    
    people = query.offset(skip).limit(limit).all()
    results = []
    for p in people:
        p_dict = {
            "id": p.id,
            "external_id": p.external_id,
            "full_name": p.full_name,
            "department": p.department,
            "consent_given": p.consent_given,
            "consent_timestamp": p.consent_timestamp,
            "is_active": p.is_active,
            "created_at": p.created_at,
            "enrolled_embeddings_count": len(p.embeddings),
        }
        results.append(PersonOut(**p_dict))
    return results


@router.post("", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
def create_person(
    person_in: PersonCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a new identity with explicit biometric consent."""
    if not person_in.consent_given:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Biometric consent is mandatory for enrollment under privacy regulations.",
        )

    if person_in.external_id:
        existing = db.query(Person).filter(Person.external_id == person_in.external_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Person with ID '{person_in.external_id}' already exists.",
            )

    person = Person(
        full_name=person_in.full_name,
        external_id=person_in.external_id,
        department=person_in.department,
        consent_given=True,
        consent_timestamp=datetime.utcnow(),
        is_active=True,
    )
    db.add(person)
    db.commit()
    db.refresh(person)

    return PersonOut(
        id=person.id,
        external_id=person.external_id,
        full_name=person.full_name,
        department=person.department,
        consent_given=person.consent_given,
        consent_timestamp=person.consent_timestamp,
        is_active=person.is_active,
        created_at=person.created_at,
        enrolled_embeddings_count=0,
    )


@router.get("/{person_id}", response_model=PersonOut)
def get_person(
    person_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve details for a specific individual."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")

    return PersonOut(
        id=person.id,
        external_id=person.external_id,
        full_name=person.full_name,
        department=person.department,
        consent_given=person.consent_given,
        consent_timestamp=person.consent_timestamp,
        is_active=person.is_active,
        created_at=person.created_at,
        enrolled_embeddings_count=len(person.embeddings),
    )


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Biometric Erasure / Right to be Forgotten:
    Completely erases person metadata, all 512-d embeddings, and removes from the vector gallery.
    """
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")

    # Remove from in-memory biometric matcher gallery
    biometric_service.matcher.remove_person(person.id)

    db.delete(person)
    db.commit()
    return None


@router.post("/{person_id}/enroll", response_model=EnrollmentResponse)
async def enroll_person_faces(
    person_id: str,
    images: List[UploadFile] = File(..., description="1 to 10 facial images from multiple angles"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Multi-image biometric enrollment with automated quality gating.
    Computes 512-d ArcFace embeddings and registers the person dynamically without retraining.
    """
    person = db.query(Person).filter(Person.id == person_id, Person.is_active == True).first()
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found or inactive")

    if not person.consent_given:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot enroll biometric data without explicit recorded consent.",
        )

    # Read image bytes
    raw_images_bytes = []
    for f in images:
        content = await f.read()
        raw_images_bytes.append(content)

    valid_embeddings, feedbacks = biometric_service.process_enrollment_images(raw_images_bytes)

    if not valid_embeddings:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "All provided enrollment images failed quality checks.",
                "feedbacks": feedbacks,
            }
        )

    # Save embeddings to database
    for emb, fb in zip(valid_embeddings, [f for f in feedbacks if f["is_valid"]]):
        emb_record = BiometricEmbedding(
            person_id=person.id,
            embedding_data=json.dumps(emb.tolist()),
            quality_score=fb["quality_score"],
            created_at=datetime.utcnow(),
        )
        db.add(emb_record)

    db.commit()

    # Retrieve all embeddings for this person to build updated prototype
    all_embeddings = []
    for r in person.embeddings:
        all_embeddings.append(np.array(json.loads(r.embedding_data), dtype=np.float32))

    # Enroll in memory matcher gallery
    biometric_service.matcher.enroll_person(person.id, person.full_name, all_embeddings)

    return EnrollmentResponse(
        person_id=person.id,
        full_name=person.full_name,
        total_images_processed=len(raw_images_bytes),
        valid_embeddings_stored=len(valid_embeddings),
        feedbacks=feedbacks,
        message=f"Successfully enrolled {len(valid_embeddings)} high-quality facial representations.",
    )
