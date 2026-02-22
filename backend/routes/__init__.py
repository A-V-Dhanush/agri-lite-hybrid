# =============================================================================
# AgriLite-Hybrid Backend
# routes/__init__.py - Routes Package
# 
# This package contains all API route blueprints organized by feature.
# =============================================================================

from .auth import auth_bp
from .crops import crops_bp
from .predict import predict_bp
from .history import history_bp
from .admin import admin_bp

__all__ = [
    'auth_bp',
    'crops_bp',
    'predict_bp',
    'history_bp',
    'admin_bp'
]
