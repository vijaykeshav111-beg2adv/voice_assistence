
from datetime import date, time
from appointments import (
    book_appointment,
    get_appointment,
    cancel_appointment,
)

# Use a dedicated test phone number.
patient_name = "Test Patient"
phone = "9876501234"
doctor_name = "Dr Aditi Sharma"
appointment_date = date(2026, 11, 2)
appointment_time = time(19, 30)

appointment_id = None

try:
    print("1. Creating test appointment...")

    result = book_appointment(
        patient_name=patient_name,
        phone=phone,
        doctor_name=doctor_name,
        appointment_date=appointment_date,
        appointment_time=appointment_time,
    )

    appointment_id = result["id"]
    print("BOOKING SUCCESS:")
    print(result)

    print("\n2. Retrieving appointment...")
    retrieved = get_appointment(appointment_id, phone)
    print("RETRIEVED:", retrieved)

    if not retrieved:
        raise RuntimeError("Appointment could not be retrieved.")

    print("\n3. Cancelling test appointment...")
    cancelled = cancel_appointment(appointment_id, phone)
    print("CANCELLED:", cancelled)

    if not cancelled:
        raise RuntimeError("Test appointment was not cancelled.")

    print("\nPASS: booking, retrieval, and cancellation work.")

except Exception:
    print("\nTEST FAILED:")
    raise
