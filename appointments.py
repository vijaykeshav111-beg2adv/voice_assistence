from datetime import date, time

from mysql.connector import IntegrityError

from db import get_connection


def _validate_appointment(appointment_date: date, appointment_time: time) -> None:
    if appointment_date < date.today():
        raise ValueError("Appointment date cannot be in the past.")

    if appointment_time.second or appointment_time.microsecond:
        raise ValueError("Appointment time must use whole minutes.")


def _format_time(value) -> str:
    """Format a MySQL TIME value for API and voice responses."""
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    return str(value)


def book_appointment(
    patient_name: str,
    phone: str,
    doctor_name: str,
    appointment_date: date,
    appointment_time: time,
):
    patient_name = patient_name.strip()
    phone = phone.strip()
    doctor_name = doctor_name.strip()

    if not patient_name or not doctor_name:
        raise ValueError("Patient name and doctor name are required.")
    if not phone or len(phone) > 20:
        raise ValueError("A valid phone number is required.")

    _validate_appointment(appointment_date, appointment_time)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()

        # TiDB doctors schema uses doctor_name, specialty, and active.
        cursor.execute(
            """
            SELECT id, doctor_name AS name
            FROM doctors
            WHERE doctor_name = %s AND active = 1
            """,
            (doctor_name,),
        )
        doctor = cursor.fetchone()
        if not doctor:
            raise ValueError("The selected doctor is not active.")

        # Register the patient, or reuse the existing unique phone number.
        cursor.execute(
            """
            INSERT INTO patients (name, phone)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id)
            """,
            (patient_name, phone),
        )
        patient_id = cursor.lastrowid

        cursor.execute(
            """
            SELECT id, name, phone
            FROM patients
            WHERE id = %s
            """,
            (patient_id,),
        )
        patient = cursor.fetchone()
        if not patient:
            raise RuntimeError("Could not retrieve the patient record.")

        # Avoid booking a slot that is already booked.
        cursor.execute(
            """
            SELECT id
            FROM appointments
            WHERE doctor_id = %s
              AND appointment_date = %s
              AND appointment_time = %s
              AND status = 'booked'
            FOR UPDATE
            """,
            (doctor["id"], appointment_date, appointment_time),
        )
        if cursor.fetchone():
            raise ValueError(
                "That doctor already has a booking at this time. "
                "Please choose another slot."
            )

        cursor.execute(
            """
            INSERT INTO appointments (
                patient_id,
                doctor_id,
                appointment_date,
                appointment_time,
                status
            )
            VALUES (%s, %s, %s, %s, 'booked')
            """,
            (patient_id, doctor["id"], appointment_date, appointment_time),
        )

        appointment_id = cursor.lastrowid
        conn.commit()

        return {
            "id": appointment_id,
            "patient_id": patient_id,
            "patient_name": patient["name"],
            "phone": patient["phone"],
            "doctor_id": doctor["id"],
            "doctor_name": doctor["name"],
            "appointment_date": appointment_date.isoformat(),
            "appointment_time": appointment_time.strftime("%H:%M:%S"),
            "status": "booked",
        }

    except IntegrityError as exc:
        if conn.in_transaction:
            conn.rollback()

        if exc.errno == 1062:
            raise ValueError(
                "This appointment slot has just been booked. "
                "Please choose another available time."
            ) from exc
        raise

    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


def get_appointment(appointment_id: int, phone: str):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                a.id,
                a.patient_id,
                a.doctor_id,
                p.name AS patient_name,
                p.phone,
                d.doctor_name AS doctor_name,
                a.appointment_date,
                a.appointment_time,
                a.status
            FROM appointments AS a
            JOIN patients AS p ON p.id = a.patient_id
            JOIN doctors AS d ON d.id = a.doctor_id
            WHERE a.id = %s AND p.phone = %s
            """,
            (appointment_id, phone.strip()),
        )

        row = cursor.fetchone()
        if row:
            row["appointment_date"] = row["appointment_date"].isoformat()
            row["appointment_time"] = _format_time(row["appointment_time"])
        return row

    finally:
        cursor.close()
        conn.close()


def cancel_appointment(appointment_id: int, phone: str):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            UPDATE appointments AS a
            JOIN patients AS p ON p.id = a.patient_id
            SET a.status = 'cancelled'
            WHERE a.id = %s
              AND p.phone = %s
              AND a.status = 'booked'
            """,
            (appointment_id, phone.strip()),
        )
        cancelled = cursor.rowcount == 1
        conn.commit()
        return cancelled

    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


def reschedule_appointment(
    appointment_id: int,
    phone: str,
    new_date: date,
    new_time: time,
):
    _validate_appointment(new_date, new_time)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()
        cursor.execute(
            """
            SELECT a.id, a.doctor_id
            FROM appointments AS a
            JOIN patients AS p ON p.id = a.patient_id
            WHERE a.id = %s
              AND p.phone = %s
              AND a.status = 'booked'
            FOR UPDATE
            """,
            (appointment_id, phone.strip()),
        )
        current = cursor.fetchone()

        if not current:
            conn.rollback()
            return None

        # Do not move an appointment into a slot already booked for this doctor.
        cursor.execute(
            """
            SELECT id
            FROM appointments
            WHERE doctor_id = %s
              AND appointment_date = %s
              AND appointment_time = %s
              AND status = 'booked'
              AND id <> %s
            FOR UPDATE
            """,
            (current["doctor_id"], new_date, new_time, appointment_id),
        )
        if cursor.fetchone():
            raise ValueError(
                "That doctor already has a booking at this time. "
                "Please choose another slot."
            )

        cursor.execute(
            """
            UPDATE appointments
            SET appointment_date = %s,
                appointment_time = %s
            WHERE id = %s AND status = 'booked'
            """,
            (new_date, new_time, appointment_id),
        )
        conn.commit()

        return {
            "id": appointment_id,
            "doctor_id": current["doctor_id"],
            "appointment_date": new_date.isoformat(),
            "appointment_time": new_time.strftime("%H:%M:%S"),
            "status": "booked",
        }

    except IntegrityError as exc:
        if conn.in_transaction:
            conn.rollback()
        if exc.errno == 1062:
            raise ValueError(
                "That doctor already has a booking at this time. "
                "Please choose another slot."
            ) from exc
        raise

    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()
