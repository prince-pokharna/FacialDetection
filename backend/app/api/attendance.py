"""
Attendance Query and Export Endpoints.
"""

from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.schemas.attendance import AttendanceRecordOut
from backend.app.services.attendance_service import attendance_service

router = APIRouter(prefix="/attendance", tags=["Attendance Records"])


@router.get("", response_model=List[AttendanceRecordOut])
def get_attendance_records(
    target_date: Optional[date] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    person_id: Optional[str] = Query(None, description="Filter by person UUID"),
    session_name: Optional[str] = Query(None, description="Filter by session name"),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve filtered attendance history."""
    records = attendance_service.get_records(
        db=db,
        target_date=target_date,
        person_id=person_id,
        session_name=session_name,
        limit=limit,
        offset=skip,
    )
    return records


@router.get("/export.csv")
def export_attendance_csv(
    target_date: Optional[date] = Query(None),
    person_id: Optional[str] = Query(None),
    session_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export attendance records to a downloadable CSV spreadsheet."""
    records = attendance_service.get_records(
        db=db,
        target_date=target_date,
        person_id=person_id,
        session_name=session_name,
        limit=5000,
    )
    csv_data = attendance_service.export_csv(records)
    filename = f"attendance_{target_date or date.today().isoformat()}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export.xlsx")
def export_attendance_excel(
    target_date: Optional[date] = Query(None),
    person_id: Optional[str] = Query(None),
    session_name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export attendance records to a formatted Excel workbook (.xlsx)."""
    records = attendance_service.get_records(
        db=db,
        target_date=target_date,
        person_id=person_id,
        session_name=session_name,
        limit=5000,
    )
    excel_bytes = attendance_service.export_excel(records)
    filename = f"attendance_{target_date or date.today().isoformat()}.xlsx"
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
