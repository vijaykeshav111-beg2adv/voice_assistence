
from db import get_connection


def main():
    connection = None

    try:
        connection = get_connection()

        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE(), VERSION()")
        database, version = cursor.fetchone()

        cursor.execute("SHOW TABLES LIKE 'appointments'")
        table = cursor.fetchone()

        print("Database connection: SUCCESS")
        print("Database:", database)
        print("MySQL version:", version)
        print("Appointments table exists:", bool(table))

        cursor.close()

    except Exception as exc:
        print("Database connection: FAILED")
        print(type(exc).__name__ + ":", exc)

    finally:
        if connection and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    main()
