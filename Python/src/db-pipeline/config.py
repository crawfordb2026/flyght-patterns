# Python/src/db-pipeline/config.py

import os
import sys

# Detect platform for smart defaults
_is_windows = sys.platform.startswith('win')
_is_macos = sys.platform == 'darwin'

# Determine default database user based on platform
# Windows: typically uses 'postgres' user
# macOS (Homebrew): typically uses your system username
if _is_windows:
    _default_user = 'postgres'
elif _is_macos:
    # Try to get the current username (Homebrew PostgreSQL default)
    try:
        _default_user = os.getenv('USER') or os.getenv('USERNAME')
        if not _default_user and hasattr(os, 'getlogin'):
            try:
                _default_user = os.getlogin()
            except OSError:
                pass
        _default_user = _default_user or 'postgres'
    except Exception:
        _default_user = 'postgres'
else:
    # Linux or other Unix-like systems
    _default_user = os.getenv('USER') or 'postgres'

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'fly_ml_db'),
    # User: Auto-detects based on platform, can be overridden with DB_USER env var
    # Windows: defaults to 'postgres'
    # macOS: defaults to your system username (Homebrew convention)
    'user': os.getenv('DB_USER', _default_user),
    # Password: Must be set via DB_PASSWORD environment variable
    # Windows: Required - set the password you used during PostgreSQL installation
    # macOS: Often empty (trust authentication with Homebrew)
    'password': os.getenv('DB_PASSWORD', '')
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

USE_DATABASE = os.getenv('USE_DATABASE', 'true').lower() == 'true'