# =============================================================================
# AgriLite-Hybrid Backend
# routes/auth.py - Authentication Routes
# 
# Handles user registration, login, logout, and profile management.
# Uses JWT tokens for stateless authentication.
# =============================================================================

from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token, 
    create_refresh_token,
    jwt_required, 
    get_jwt_identity,
    get_jwt
)

from extensions import db, bcrypt, limiter
from models import User
from utils import validate_email, validate_password, success_response, error_response
from decorators import validate_json, handle_db_errors

# Create blueprint
auth_bp = Blueprint('auth', __name__)


# =============================================================================
# User Registration
# =============================================================================

@auth_bp.route('/register', methods=['POST'])
@limiter.limit("5 per minute")
@handle_db_errors
def register():
    """
    Register a new user account.
    
    Request Body:
        email (str): User's email address (required)
        password (str): User's password (required, min 8 chars)
        first_name (str): User's first name (optional)
        last_name (str): User's last name (optional)
        phone (str): User's phone number (optional)
        
    Returns:
        201: User registered successfully with tokens
        400: Validation error
        409: Email already exists
    """
    data = request.get_json()
    
    # Validate required fields
    if not data:
        return error_response('Request body is required', status_code=400)
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    phone = data.get('phone', '').strip()
    
    # Validate email
    if not email:
        return error_response('Email is required', status_code=400)
    if not validate_email(email):
        return error_response('Invalid email format', status_code=400)
    
    # Check if email already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return error_response('Email already registered', status_code=409)
    
    # Validate password
    if not password:
        return error_response('Password is required', status_code=400)
    is_valid, password_message = validate_password(password)
    if not is_valid:
        return error_response(password_message, status_code=400)
    
    # Create password hash
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # Create new user
    user = User(
        email=email,
        password_hash=password_hash,
        first_name=first_name or None,
        last_name=last_name or None,
        phone=phone or None,
        is_active=True,
        role='user'
    )
    
    db.session.add(user)
    db.session.commit()
    
    # Generate tokens
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    
    current_app.logger.info(f"New user registered: {email}")
    
    return jsonify({
        'success': True,
        'message': 'Registration successful',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict()
    }), 201


# =============================================================================
# User Login
# =============================================================================

@auth_bp.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
@handle_db_errors
def login():
    """
    Authenticate user and return JWT tokens.
    
    Request Body:
        email (str): User's email address
        password (str): User's password
        
    Returns:
        200: Login successful with tokens
        400: Missing credentials
        401: Invalid credentials
        403: Account inactive
    """
    data = request.get_json()
    
    if not data:
        return error_response('Request body is required', status_code=400)
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    # Validate input
    if not email or not password:
        return error_response('Email and password are required', status_code=400)
    
    # Find user
    user = User.query.filter_by(email=email).first()
    
    if not user:
        return error_response('Invalid email or password', status_code=401)
    
    # Check password
    if not bcrypt.check_password_hash(user.password_hash, password):
        return error_response('Invalid email or password', status_code=401)
    
    # Check if account is active
    if not user.is_active:
        return error_response('Account is deactivated. Please contact support.', status_code=403)
    
    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.session.commit()
    
    # Generate tokens
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    
    current_app.logger.info(f"User logged in: {email}")
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict()
    }), 200


# =============================================================================
# Token Refresh
# =============================================================================

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """
    Refresh access token using refresh token.
    
    Headers:
        Authorization: Bearer <refresh_token>
        
    Returns:
        200: New access token
        401: Invalid or expired refresh token
    """
    current_user_id = get_jwt_identity()
    
    # Verify user still exists and is active
    user = db.session.get(User, int(current_user_id))
    if not user or not user.is_active:
        return error_response('User not found or inactive', status_code=401)
    
    # Generate new access token
    access_token = create_access_token(identity=current_user_id)
    
    return jsonify({
        'success': True,
        'access_token': access_token
    }), 200


# =============================================================================
# Get Current User Profile
# =============================================================================

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Get current user's profile information.
    
    Headers:
        Authorization: Bearer <access_token>
        
    Returns:
        200: User profile data
        401: Not authenticated
        404: User not found
    """
    current_user_id = get_jwt_identity()
    user = db.session.get(User, int(current_user_id))
    
    if not user:
        return error_response('User not found', status_code=404)
    
    return success_response(data=user.to_dict())


# =============================================================================
# Update User Profile
# =============================================================================

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
@handle_db_errors
def update_profile():
    """
    Update current user's profile information.
    
    Headers:
        Authorization: Bearer <access_token>
        
    Request Body:
        first_name (str): New first name (optional)
        last_name (str): New last name (optional)
        phone (str): New phone number (optional)
        
    Returns:
        200: Profile updated successfully
        400: Validation error
        404: User not found
    """
    current_user_id = get_jwt_identity()
    user = db.session.get(User, int(current_user_id))
    
    if not user:
        return error_response('User not found', status_code=404)
    
    data = request.get_json()
    if not data:
        return error_response('Request body is required', status_code=400)
    
    # Update fields if provided
    if 'first_name' in data:
        user.first_name = data['first_name'].strip() or None
    if 'last_name' in data:
        user.last_name = data['last_name'].strip() or None
    if 'phone' in data:
        user.phone = data['phone'].strip() or None
    
    db.session.commit()
    
    return success_response(
        data=user.to_dict(),
        message='Profile updated successfully'
    )


# =============================================================================
# Change Password
# =============================================================================

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
@handle_db_errors
def change_password():
    """
    Change current user's password.
    
    Headers:
        Authorization: Bearer <access_token>
        
    Request Body:
        current_password (str): Current password
        new_password (str): New password
        
    Returns:
        200: Password changed successfully
        400: Validation error
        401: Current password incorrect
    """
    current_user_id = get_jwt_identity()
    user = db.session.get(User, int(current_user_id))
    
    if not user:
        return error_response('User not found', status_code=404)
    
    data = request.get_json()
    if not data:
        return error_response('Request body is required', status_code=400)
    
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')
    
    # Verify current password
    if not bcrypt.check_password_hash(user.password_hash, current_password):
        return error_response('Current password is incorrect', status_code=401)
    
    # Validate new password
    is_valid, password_message = validate_password(new_password)
    if not is_valid:
        return error_response(password_message, status_code=400)
    
    # Update password
    user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
    db.session.commit()
    
    current_app.logger.info(f"Password changed for user: {user.email}")
    
    return success_response(message='Password changed successfully')


# =============================================================================
# Logout (optional - for token blacklisting)
# =============================================================================

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    Logout user (client should discard tokens).
    
    Note: JWT tokens are stateless. This endpoint is for client-side cleanup.
    For true logout, implement token blacklisting with Redis.
    
    Returns:
        200: Logout successful
    """
    return success_response(message='Logged out successfully')
