# =============================================================================
# AgriLite-Hybrid Backend
# routes/history.py - Detection History Routes
# 
# Handles user detection history including saving, retrieving,
# and managing past disease detection results.
# =============================================================================

import json
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

from extensions import db, limiter
from models import DetectionHistory
from utils import success_response, error_response, paginated_response
from decorators import paginated_response as paginate, handle_db_errors

# Create blueprint
history_bp = Blueprint('history', __name__)


# =============================================================================
# Get User's Detection History
# =============================================================================

@history_bp.route('/', methods=['GET'])
@jwt_required()
@limiter.limit("60 per minute")
@paginate(default_per_page=20, max_per_page=100)
def get_history(page, per_page):
    """
    Get current user's detection history with pagination.
    
    Headers:
        Authorization: Bearer <access_token>
        
    Query Parameters:
        page (int): Page number (default: 1)
        per_page (int): Items per page (default: 20, max: 100)
        crop (str): Filter by crop type (optional)
        disease (str): Filter by disease (optional)
        severity (str): Filter by severity (optional)
        
    Returns:
        200: Paginated list of detection history
        401: Not authenticated
    """
    current_user_id = int(get_jwt_identity())
    
    # Build query with filters
    query = DetectionHistory.query.filter_by(user_id=current_user_id)
    
    # Apply optional filters
    crop_filter = request.args.get('crop', '').lower().strip()
    if crop_filter:
        query = query.filter(DetectionHistory.crop == crop_filter)
    
    disease_filter = request.args.get('disease', '').strip()
    if disease_filter:
        query = query.filter(DetectionHistory.disease.ilike(f'%{disease_filter}%'))
    
    severity_filter = request.args.get('severity', '').lower().strip()
    if severity_filter and severity_filter in ['mild', 'medium', 'severe']:
        query = query.filter(DetectionHistory.severity == severity_filter)
    
    # Order by most recent first
    query = query.order_by(DetectionHistory.created_at.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Serialize results (without images by default for performance)
    items = [item.to_dict(include_images=False) for item in pagination.items]
    
    return jsonify({
        'success': True,
        'data': {
            'items': items,
            'page': pagination.page,
            'per_page': pagination.per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    }), 200


# =============================================================================
# Get Single Detection Detail
# =============================================================================

@history_bp.route('/<int:detection_id>', methods=['GET'])
@jwt_required()
@limiter.limit("60 per minute")
def get_detection(detection_id):
    """
    Get detailed information about a specific detection.
    
    Args:
        detection_id: ID of the detection record
        
    Query Parameters:
        include_images (bool): Include base64 image data (default: true)
        
    Returns:
        200: Detection details with images
        404: Detection not found
        403: Not authorized to view this detection
    """
    current_user_id = int(get_jwt_identity())
    
    detection = DetectionHistory.query.get(detection_id)
    
    if not detection:
        return error_response('Detection not found', status_code=404)
    
    # Check ownership
    if detection.user_id != current_user_id:
        return error_response(
            'Not authorized to view this detection',
            status_code=403
        )
    
    # Include images by default
    include_images = request.args.get('include_images', 'true').lower() == 'true'
    
    return jsonify({
        'success': True,
        'data': detection.to_dict(include_images=include_images)
    }), 200


# =============================================================================
# Save New Detection to History
# =============================================================================

@history_bp.route('/', methods=['POST'])
@jwt_required()
@limiter.limit("30 per minute")
@handle_db_errors
def save_detection():
    """
    Save a detection result to user's history.
    
    Request Body:
        crop (str): Crop type - required
        disease (str): Detected disease - required
        severity (str): Severity level - required
        confidence (float): Confidence percentage - required
        treatment (list): Treatment recommendations - optional
        original_image (str): Base64 original image - optional
        heatmap_image (str): Base64 heatmap image - optional
        temperature (float): Environmental temperature - optional
        humidity (float): Environmental humidity - optional
        environmental_risk (str): Risk assessment - optional
        notes (str): User notes - optional
        
    Returns:
        201: Detection saved successfully
        400: Validation error
    """
    current_user_id = int(get_jwt_identity())
    data = request.get_json()
    
    if not data:
        return error_response('Request body is required', status_code=400)
    
    # Validate required fields
    required_fields = ['crop', 'disease', 'severity', 'confidence']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return error_response(
            'Missing required fields',
            details={'missing': missing},
            status_code=400
        )
    
    # Create detection record
    detection = DetectionHistory(
        user_id=current_user_id,
        crop=data['crop'].lower().strip(),
        disease=data['disease'],
        severity=data['severity'].lower().strip(),
        confidence=float(data['confidence']),
        treatment=json.dumps(data.get('treatment', [])),
        original_image=data.get('original_image', ''),
        heatmap_image=data.get('heatmap_image', ''),
        temperature=data.get('temperature'),
        humidity=data.get('humidity'),
        environmental_risk=data.get('environmental_risk'),
        device_type=data.get('device_type', 'web'),
        location=data.get('location'),
        notes=data.get('notes')
    )
    
    db.session.add(detection)
    db.session.commit()
    
    current_app.logger.info(
        f"Detection saved: ID={detection.id}, user={current_user_id}, "
        f"crop={detection.crop}, disease={detection.disease}"
    )
    
    return jsonify({
        'success': True,
        'message': 'Detection saved successfully',
        'data': detection.to_dict(include_images=False)
    }), 201


# =============================================================================
# Update Detection Notes
# =============================================================================

@history_bp.route('/<int:detection_id>', methods=['PUT'])
@jwt_required()
@limiter.limit("30 per minute")
@handle_db_errors
def update_detection(detection_id):
    """
    Update notes for a detection record.
    
    Args:
        detection_id: ID of the detection record
        
    Request Body:
        notes (str): Updated notes
        
    Returns:
        200: Detection updated
        404: Detection not found
        403: Not authorized
    """
    current_user_id = int(get_jwt_identity())
    
    detection = DetectionHistory.query.get(detection_id)
    
    if not detection:
        return error_response('Detection not found', status_code=404)
    
    if detection.user_id != current_user_id:
        return error_response('Not authorized', status_code=403)
    
    data = request.get_json()
    if data and 'notes' in data:
        detection.notes = data['notes']
        db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Detection updated',
        'data': detection.to_dict(include_images=False)
    }), 200


# =============================================================================
# Delete Detection
# =============================================================================

@history_bp.route('/<int:detection_id>', methods=['DELETE'])
@jwt_required()
@limiter.limit("30 per minute")
@handle_db_errors
def delete_detection(detection_id):
    """
    Delete a detection record from history.
    
    Args:
        detection_id: ID of the detection record
        
    Returns:
        200: Detection deleted
        404: Detection not found
        403: Not authorized
    """
    current_user_id = int(get_jwt_identity())
    
    detection = DetectionHistory.query.get(detection_id)
    
    if not detection:
        return error_response('Detection not found', status_code=404)
    
    if detection.user_id != current_user_id:
        return error_response('Not authorized', status_code=403)
    
    db.session.delete(detection)
    db.session.commit()
    
    current_app.logger.info(f"Detection deleted: ID={detection_id}, user={current_user_id}")
    
    return jsonify({
        'success': True,
        'message': 'Detection deleted successfully'
    }), 200


# =============================================================================
# Get History Statistics
# =============================================================================

@history_bp.route('/stats', methods=['GET'])
@jwt_required()
@limiter.limit("60 per minute")
def get_history_stats():
    """
    Get statistics for user's detection history.
    
    Returns:
        200: Statistics including counts by crop, disease, severity
    """
    current_user_id = int(get_jwt_identity())
    
    # Total detections
    total = DetectionHistory.query.filter_by(user_id=current_user_id).count()
    
    # Count by crop
    crops_query = db.session.query(
        DetectionHistory.crop,
        db.func.count(DetectionHistory.id)
    ).filter_by(user_id=current_user_id).group_by(DetectionHistory.crop).all()
    
    crops_stats = {crop: count for crop, count in crops_query}
    
    # Count by severity
    severity_query = db.session.query(
        DetectionHistory.severity,
        db.func.count(DetectionHistory.id)
    ).filter_by(user_id=current_user_id).group_by(DetectionHistory.severity).all()
    
    severity_stats = {severity: count for severity, count in severity_query}
    
    # Most common diseases
    disease_query = db.session.query(
        DetectionHistory.disease,
        db.func.count(DetectionHistory.id).label('count')
    ).filter_by(user_id=current_user_id)\
     .filter(DetectionHistory.disease != 'Healthy')\
     .group_by(DetectionHistory.disease)\
     .order_by(db.desc('count'))\
     .limit(5).all()
    
    top_diseases = [{'disease': d, 'count': c} for d, c in disease_query]
    
    return jsonify({
        'success': True,
        'data': {
            'total_detections': total,
            'by_crop': crops_stats,
            'by_severity': severity_stats,
            'top_diseases': top_diseases,
            'healthy_count': DetectionHistory.query.filter_by(
                user_id=current_user_id,
                disease='Healthy'
            ).count()
        }
    }), 200
