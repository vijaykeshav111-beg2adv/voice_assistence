
import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Create a new connection to the application database."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "voice_assistant_db"),
        user=os.getenv("DB_USER", "voice_bot_user"),
        password=os.getenv("DB_PASSWORD"),
        connection_timeout=3,
    )
