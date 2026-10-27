
from datetime import date, datetime, time, timedelta

from db import get_connection


def get_active_doctors():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                id,
                doctor_name AS name,
                specialty AS specialization
            FROM doctors
            WHERE active = 1
            ORDER BY specialty, doctor_name
        """)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def get_available_slots(doctor_id: int, selected_date: date):
    if selected_date < date.today():
        raise ValueError("Cannot check slots for a past date.")

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT id, doctor_name AS name
            FROM doctors
            WHERE id = %s AND active = 1
        """, (doctor_id,))

        doctor = cursor.fetchone()

        if not doctor:
            return None

        cursor.execute("""
            SELECT start_time, end_time, slot_duration_minutes
            FROM doctor_schedules
            WHERE doctor_id = %s
              AND day_of_week = %s
              AND active = 1
            ORDER BY start_time
        """, (doctor_id, selected_date.weekday()))

        schedules = cursor.fetchall()

        cursor.execute("""
            SELECT appointment_time
            FROM appointments
            WHERE doctor_id = %s
              AND appointment_date = %s
              AND status = 'booked'
        """, (doctor_id, selected_date))

        booked_times = {
            _to_time(row["appointment_time"])
            for row in cursor.fetchall()
        }

        now = datetime.now()
        available = []

        for schedule in schedules:
            start = _to_time(schedule["start_time"])
            end = _to_time(schedule["end_time"])
            duration = int(schedule["slot_duration_minutes"])

            current = datetime.combine(selected_date, start)
            closing = datetime.combine(selected_date, end)

            while current + timedelta(minutes=duration) <= closing:
                slot_time = current.time()

                if slot_time not in booked_times and current > now:
                    available.append(slot_time.strftime("%H:%M"))

                current += timedelta(minutes=duration)

        return {
            "doctor_id": doctor["id"],
            "doctor_name": doctor["name"],
            "date": selected_date.isoformat(),
            "slots": available,
        }

    finally:
        cursor.close()
        conn.close()


def _to_time(value) -> time:
    """Convert MySQL TIME values to Python datetime.time."""
    if isinstance(value, time):
        return value

    if isinstance(value, timedelta):
        total_seconds = int(value.total_seconds())

        if not 0 <= total_seconds < 24 * 60 * 60:
            raise ValueError(f"Invalid schedule time: {value}")

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return time(hours, minutes, seconds)

    if isinstance(value, str):
        return time.fromisoformat(value)

    raise TypeError(
        f"Unsupported MySQL time value: {value!r} "
        f"({type(value).__name__})"
    )
