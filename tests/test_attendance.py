"""
Attendance History and Export Tests.
"""

from datetime import date
from backend.app.services.attendance_service import attendance_service
from backend.app.models.person import Person


def test_attendance_mark_and_export(client, db_session, instructor_token):
    headers = {"Authorization": f"Bearer {instructor_token}"}

    # Create person
    person = Person(
        full_name="Attendance Export Student",
        external_id="EXP001",
        department="Mathematics",
        consent_given=True,
    )
    db_session.add(person)
    db_session.commit()
    db_session.refresh(person)

    # Mark attendance
    success, record, msg = attendance_service.mark_attendance(
        db=db_session,
        person_id=person.id,
        confidence=0.88,
        liveness_score=0.92,
        session_name="MorningLecture",
        marked_by="TEST_HARNESS",
    )
    assert success is True
    assert record is not None

    # Duplicate mark same day should be rejected
    dup_success, _, dup_msg = attendance_service.mark_attendance(
        db=db_session,
        person_id=person.id,
        confidence=0.88,
        liveness_score=0.92,
        session_name="MorningLecture",
    )
    assert dup_success is False
    assert "already marked" in dup_msg

    # Query attendance via API
    resp = client.get("/attendance", headers=headers)
    assert resp.status_code == 200
    records = resp.json()
    assert len(records) >= 1

    # Export CSV
    csv_resp = client.get("/attendance/export.csv", headers=headers)
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "Attendance Export Student" in csv_resp.text

    # Export Excel (.xlsx)
    xlsx_resp = client.get("/attendance/export.xlsx", headers=headers)
    assert xlsx_resp.status_code == 200
    assert "application/vnd.openxmlformats" in xlsx_resp.headers["content-type"]
    assert len(xlsx_resp.content) > 1000
