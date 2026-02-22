# =============================================================================
# AgriLite-Hybrid Backend
# routes/admin.py - Admin Routes
# 
# Administrative endpoints for user management, analytics,
# and system monitoring. Requires admin role.
# =============================================================================

from datetime import datetime, timedelta, timezone
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required

from extensions import db, limiter
from models import User, DetectionHistory
from utils import admin_required, success_response, error_response
from decorators import paginated_response, handle_db_errors

# Create blueprint
admin_bp = Blueprint('admin', __name__)


# =============================================================================
# Get All Users
# =============================================================================

@admin_bp.route('/users', methods=['GET'])
@admin_required
@limiter.limit("30 per minute")
@paginated_response(default_per_page=50)
def get_all_users(page, per_page):
    """
    Get list of all registered users (admin only).
    
    Query Parameters:
        page (int): Page number
        per_page (int): Items per page
        search (str): Search by email or name
        role (str): Filter by role
        active (bool): Filter by active status
        
    Returns:
        200: Paginated list of users
    """
    query = User.query
    
    # Search filter
    search = request.args.get('search', '').strip()
    if search:
        query = query.filter(
            db.or_(
                User.email.ilike(f'%{search}%'),
                User.first_name.ilike(f'%{search}%'),
                User.last_name.ilike(f'%{search}%')
            )
        )
    
    # Role filter
    role = request.args.get('role', '').strip()
    if role:
        query = query.filter(User.role == role)
    
    # Active status filter
    active = request.args.get('active')
    if active is not None:
        is_active = active.lower() == 'true'
        query = query.filter(User.is_active == is_active)
    
    # Order by creation date
    query = query.order_by(User.created_at.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    users = [user.to_dict() for user in pagination.items]
    
    return jsonify({
        'success': True,
        'data': {
            'users': users,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'total': pagination.total,
            'pages': pagination.pages
        }
    }), 200


# =============================================================================
# Get Single User
# =============================================================================

@admin_bp.route('/users/<int:user_id>', methods=['GET'])
@admin_required
@limiter.limit("60 per minute")
def get_user(user_id):
    """
    Get detailed information about a specific user.
    
    Args:
        user_id: User ID
        
    Returns:
        200: User details with detection statistics
        404: User not found
    """
    user = db.session.get(User, user_id)
    
    if not user:
        return error_response('User not found', status_code=404)
    
    # Get user's detection stats
    detection_count = DetectionHistory.query.filter_by(user_id=user_id).count()
    
    user_data = user.to_dict()
    user_data['detection_count'] = detection_count
    
    return jsonify({
        'success': True,
        'data': user_data
    }), 200


# =============================================================================
# Update User Status
# =============================================================================

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@admin_required
@limiter.limit("30 per minute")
@handle_db_errors
def update_user(user_id):
    """
    Update user account status or role.
    
    Args:
        user_id: User ID
        
    Request Body:
        is_active (bool): Account active status
        role (str): User role (user, admin)
        
    Returns:
        200: User updated
        404: User not found
    """
    user = db.session.get(User, user_id)
    
    if not user:
        return error_response('User not found', status_code=404)
    
    data = request.get_json() or {}
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
    
    if 'role' in data and data['role'] in ['user', 'admin']:
        user.role = data['role']
    
    db.session.commit()
    
    current_app.logger.info(f"User {user_id} updated by admin")
    
    return success_response(
        data=user.to_dict(),
        message='User updated successfully'
    )


# =============================================================================
# Delete User
# =============================================================================

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
@limiter.limit("10 per minute")
@handle_db_errors
def delete_user(user_id):
    """
    Delete a user account and all associated data.
    
    Args:
        user_id: User ID
        
    Returns:
        200: User deleted
        404: User not found
        400: Cannot delete admin
    """
    user = db.session.get(User, user_id)
    
    if not user:
        return error_response('User not found', status_code=404)
    
    # Prevent deleting admin accounts
    if user.role == 'admin':
        return error_response(
            'Cannot delete admin accounts',
            status_code=400
        )
    
    # Delete user (cascades to detection history)
    db.session.delete(user)
    db.session.commit()
    
    current_app.logger.info(f"User {user_id} deleted by admin")
    
    return success_response(message='User deleted successfully')


# =============================================================================
# System Analytics
# =============================================================================

@admin_bp.route('/analytics', methods=['GET'])
@admin_required
@limiter.limit("30 per minute")
def get_analytics():
    """
    Get system-wide analytics and statistics.
    
    Query Parameters:
        days (int): Number of days to analyze (default: 30)
        
    Returns:
        200: Analytics data including user and detection statistics
    """
    days = request.args.get('days', 30, type=int)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # User statistics
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    new_users = User.query.filter(User.created_at >= cutoff_date).count()
    
    # Detection statistics
    total_detections = DetectionHistory.query.count()
    recent_detections = DetectionHistory.query.filter(
        DetectionHistory.created_at >= cutoff_date
    ).count()
    
    # Detections by crop
    crops_query = db.session.query(
        DetectionHistory.crop,
        db.func.count(DetectionHistory.id)
    ).group_by(DetectionHistory.crop).all()
    
    crops_stats = {crop: count for crop, count in crops_query}
    
    # Detections by severity
    severity_query = db.session.query(
        DetectionHistory.severity,
        db.func.count(DetectionHistory.id)
    ).group_by(DetectionHistory.severity).all()
    
    severity_stats = {severity: count for severity, count in severity_query}
    
    # Top diseases (excluding healthy)
    disease_query = db.session.query(
        DetectionHistory.disease,
        db.func.count(DetectionHistory.id).label('count')
    ).filter(DetectionHistory.disease != 'Healthy')\
     .group_by(DetectionHistory.disease)\
     .order_by(db.desc('count'))\
     .limit(10).all()
    
    top_diseases = [{'disease': d, 'count': c} for d, c in disease_query]
    
    # Daily detections for the period
    daily_query = db.session.query(
        db.func.date(DetectionHistory.created_at).label('date'),
        db.func.count(DetectionHistory.id).label('count')
    ).filter(DetectionHistory.created_at >= cutoff_date)\
     .group_by(db.func.date(DetectionHistory.created_at))\
     .order_by('date').all()
    
    daily_stats = [{'date': str(d), 'count': c} for d, c in daily_query]
    
    return jsonify({
        'success': True,
        'data': {
            'period_days': days,
            'users': {
                'total': total_users,
                'active': active_users,
                'new': new_users
            },
            'detections': {
                'total': total_detections,
                'recent': recent_detections,
                'by_crop': crops_stats,
                'by_severity': severity_stats,
                'top_diseases': top_diseases,
                'daily': daily_stats
            }
        }
    }), 200


# =============================================================================
# Get All Detections (Admin View)
# =============================================================================

@admin_bp.route('/detections', methods=['GET'])
@admin_required
@limiter.limit("30 per minute")
@paginated_response(default_per_page=50)
def get_all_detections(page, per_page):
    """
    Get all detection history across all users.
    
    Query Parameters:
        page, per_page: Pagination
        crop: Filter by crop
        severity: Filter by severity
        user_id: Filter by user
        
    Returns:
        200: Paginated list of all detections
    """
    query = DetectionHistory.query
    
    # Filters
    crop = request.args.get('crop', '').lower().strip()
    if crop:
        query = query.filter(DetectionHistory.crop == crop)
    
    severity = request.args.get('severity', '').lower().strip()
    if severity and severity in ['mild', 'medium', 'severe']:
        query = query.filter(DetectionHistory.severity == severity)
    
    user_id = request.args.get('user_id', type=int)
    if user_id:
        query = query.filter(DetectionHistory.user_id == user_id)
    
    # Order by most recent
    query = query.order_by(DetectionHistory.created_at.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    items = [item.to_dict(include_images=False) for item in pagination.items]
    
    return jsonify({
        'success': True,
        'data': {
            'items': items,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'total': pagination.total,
            'pages': pagination.pages
        }
    }), 200


# =============================================================================
# System Health Check (Admin)
# =============================================================================

@admin_bp.route('/health', methods=['GET'])
@admin_required
@limiter.limit("60 per minute")
def admin_health_check():
    """
    Detailed system health check for administrators.
    
    Returns:
        200: Detailed health status including database, model, and memory
    """
    import sys
    
    # Database health
    db_healthy = False
    db_message = 'unknown'
    try:
        db.session.execute(db.text('SELECT 1'))
        db.session.commit()
        db_healthy = True
        db_message = 'connected'
    except Exception as e:
        db_message = str(e)
        db.session.rollback()
    
    # Model status
    ml_service = current_app.config.get('ML_SERVICE')
    model_loaded = ml_service.is_loaded() if ml_service else False
    
    # Basic stats
    user_count = User.query.count()
    detection_count = DetectionHistory.query.count()
    
    return jsonify({
        'success': True,
        'data': {
            'status': 'healthy' if db_healthy else 'degraded',
            'database': {
                'status': db_message,
                'healthy': db_healthy
            },
            'model': {
                'loaded': model_loaded
            },
            'stats': {
                'total_users': user_count,
                'total_detections': detection_count
            },
            'python_version': sys.version,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }), 200
