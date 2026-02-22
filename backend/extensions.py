# =============================================================================
# AgriLite-Hybrid Backend
# extensions.py - Flask Extensions Initialization
# 
# This module initializes Flask extensions without the app instance to prevent
# circular imports. Extensions are initialized with the app in the factory.
# =============================================================================

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# =============================================================================
# Database ORM
# SQLAlchemy provides ORM capabilities for database operations
# =============================================================================
db = SQLAlchemy()

# =============================================================================
# Database Migrations
# Alembic-based migrations for schema version control
# =============================================================================
migrate = Migrate()

# =============================================================================
# JWT Authentication
# Handles token generation, verification, and user identity
# =============================================================================
jwt = JWTManager()

# =============================================================================
# Password Hashing
# Bcrypt for secure password hashing and verification
# =============================================================================
bcrypt = Bcrypt()

# =============================================================================
# Cross-Origin Resource Sharing
# Enables frontend to communicate with backend from different origins
# =============================================================================
cors = CORS()

# =============================================================================
# Rate Limiting
# Protects API endpoints from abuse and DDoS attacks
# =============================================================================
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)
