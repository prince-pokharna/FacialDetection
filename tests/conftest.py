"""
Pytest Test Fixtures and Application Harness.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set test environment
os.environ["DATABASE_URL"] = "sqlite:///./test_attendance.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-unit-testing-32-chars-long"

from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.core.security import create_access_token, get_password_hash
from backend.app.models.user import User

TEST_DB_URL = "sqlite:///./test_attendance.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_attendance.db"):
        try:
            os.remove("./test_attendance.db")
        except Exception:
            pass


@pytest.fixture
def db_session():
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin_token(db_session) -> str:
    # Ensure admin user in test db
    admin = db_session.query(User).filter(User.email == "testadmin@attendance.ai").first()
    if not admin:
        admin = User(
            email="testadmin@attendance.ai",
            hashed_password=get_password_hash("AdminPass123"),
            full_name="Test Administrator",
            role="admin",
            is_active=True,
        )
        db_session.add(admin)
        db_session.commit()
        db_session.refresh(admin)

    return create_access_token({"sub": admin.id, "role": "admin", "email": admin.email})


@pytest.fixture
def instructor_token(db_session) -> str:
    instructor = db_session.query(User).filter(User.email == "instructor@attendance.ai").first()
    if not instructor:
        instructor = User(
            email="instructor@attendance.ai",
            hashed_password=get_password_hash("TeacherPass123"),
            full_name="Test Instructor",
            role="instructor",
            is_active=True,
        )
        db_session.add(instructor)
        db_session.commit()
        db_session.refresh(instructor)

    return create_access_token({"sub": instructor.id, "role": "instructor", "email": instructor.email})


@pytest.fixture
def sample_face_image_bytes() -> bytes:
    """Generate a clean synthetic test face image with eyes, nose, and mouth."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200  # light skin tone background
    # Eyes
    cv2.circle(img, (70, 80), 8, (50, 50, 50), -1)
    cv2.circle(img, (130, 80), 8, (50, 50, 50), -1)
    # Nose
    cv2.circle(img, (100, 110), 6, (80, 80, 80), -1)
    # Mouth
    cv2.line(img, (75, 145), (125, 145), (60, 60, 60), 3)

    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()
