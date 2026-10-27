
import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()
def get_connection():
    """Create a secure connection to TiDB Cloud."""
    config = {
        "host": os.environ["DB_HOST"],
        "port": int(os.getenv("DB_PORT", "4000")),
        "database": os.getenv("DB_NAME", "voice_assistant_db"),
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "connection_timeout": 10,
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
    }

    ca_path = os.getenv("DB_SSL_CA")
    if ca_path:
        config["ssl_ca"] = ca_path

    return mysql.connector.connect(**config)
