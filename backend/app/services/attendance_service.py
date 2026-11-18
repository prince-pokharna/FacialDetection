"""
Attendance management service: recording, filtering, and export.
"""

from datetime import datetime, date
import io
import csv
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

from backend.app.models.attendance import AttendanceRecord
from backend.app.models.person import Person


class AttendanceService:
    """
    Manages attendance recording and report exports.
    """

    @staticmethod
    def mark_attendance(
        db: Session,
        person_id: str,
        confidence: float,
        liveness_score: float,
        session_name: str = "Main",
        marked_by: str = "SYSTEM_KIOSK",
    ) -> Tuple[bool, Optional[AttendanceRecord], str]:
        """
        Record attendance if not already recorded today for this session.
        """
        person = db.query(Person).filter(Person.id == person_id, Person.is_active == True).first()
        if not person:
            return False, None, "Person not found or inactive"

        today = date.today()
        # Check if already marked today in this session
        existing = db.query(AttendanceRecord).filter(
            AttendanceRecord.person_id == person_id,
            AttendanceRecord.date == today,
            AttendanceRecord.session_name == session_name,
        ).first()

        if existing:
            return False, existing, "Attendance already marked for today"

        record = AttendanceRecord(
            person_id=person.id,
            person_name=person.full_name,
            confidence=confidence,
            liveness_score=liveness_score,
            session_name=session_name,
            date=today,
            timestamp=datetime.utcnow(),
            marked_by=marked_by,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return True, record, "Attendance marked successfully"

    @staticmethod
    def get_records(
        db: Session,
        target_date: Optional[date] = None,
        person_id: Optional[str] = None,
        session_name: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[AttendanceRecord]:
        query = db.query(AttendanceRecord)
        if target_date:
            query = query.filter(AttendanceRecord.date == target_date)
        if person_id:
            query = query.filter(AttendanceRecord.person_id == person_id)
        if session_name:
            query = query.filter(AttendanceRecord.session_name == session_name)

        return query.order_by(desc(AttendanceRecord.timestamp)).offset(offset).limit(limit).all()

    @staticmethod
    def export_csv(records: List[AttendanceRecord]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Person ID", "Person Name", "Confidence", "Liveness Score", "Session", "Date", "Timestamp (UTC)", "Marked By"])
        for r in records:
            writer.writerow([
                r.id,
                r.person_id,
                r.person_name,
                f"{r.confidence:.4f}",
                f"{r.liveness_score:.4f}",
                r.session_name,
                r.date.isoformat(),
                r.timestamp.isoformat(),
                r.marked_by,
            ])
        return output.getvalue()

    @staticmethod
    def export_excel(records: List[AttendanceRecord]) -> bytes:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Attendance Log"

        # Headers
        headers = ["ID", "Person ID", "Person Name", "Confidence", "Liveness Score", "Session", "Date", "Timestamp (UTC)", "Marked By"]
        ws.append(headers)

        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        for r in records:
            ws.append([
                r.id,
                r.person_id,
                r.person_name,
                round(r.confidence, 4),
                round(r.liveness_score, 4),
                r.session_name,
                r.date.isoformat(),
                r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                r.marked_by,
            ])

        # Auto-adjust column widths
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

        out = io.BytesIO()
        wb.save(out)
        return out.getvalue()


attendance_service = AttendanceService()
