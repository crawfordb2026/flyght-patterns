# Python/src/db-pipeline/config.py

import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'fly_ml_db'),
    'user': os.getenv('DB_USER', 'crawfordbarnett'),  # Default to macOS username for Homebrew PostgreSQL
    'password': os.getenv('DB_PASSWORD', '')  # Empty password for Homebrew PostgreSQL (trust auth)
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

USE_DATABASE = os.getenv('USE_DATABASE', 'true').lower() == 'true'