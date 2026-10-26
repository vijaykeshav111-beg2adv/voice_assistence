
import logging
import traceback
from datetime import date

from db import get_connection
from slots import get_active_doctors, get_available_slots

logging.basicConfig(level=logging.INFO)

test_date = date(2026, 11, 2)

print("\n=== 1. Test MySQL connection ===")
try:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DATABASE(), VERSION()")
    print(cursor.fetchone())
    cursor.close()
    conn.close()
except Exception:
    traceback.print_exc()
    raise SystemExit("MySQL connection test failed.")

print("\n=== 2. Load active doctors ===")
try:
    doctors = get_active_doctors()
    print(doctors)
except Exception:
    traceback.print_exc()
    raise SystemExit("Doctor query failed.")

print("\n=== 3. Check appointment slots ===")
for doctor in doctors:
    print(
        f"\nTesting doctor: {doctor['name']}, "
        f"ID={doctor['id']}, date={test_date}"
    )
    try:
        result = get_available_slots(
            int(doctor["id"]),
            test_date,
        )
        print("RESULT:", result)
    except Exception:
        traceback.print_exc()

print("\n=== Diagnostic complete ===")
