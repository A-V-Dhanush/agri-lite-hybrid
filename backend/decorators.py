# =============================================================================
# AgriLite-Hybrid Backend
# decorators.py - Reusable Decorators
# 
# Custom decorators for common functionality including pagination,
# validation, error handling, and database operations.
# =============================================================================

from functools import wraps
from flask import request, jsonify, current_app
from extensions import db


def paginated_response(default_per_page=20, max_per_page=100):
    """
    Handle pagination parameters for list endpoints.
    
    Extracts 'page' and 'per_page' from query parameters and validates them.
    Passes validated values to the decorated function as keyword arguments.
    
    Args:
        default_per_page: Default items per page if not specified
        max_per_page: Maximum allowed items per page
        
    Usage:
        @posts_bp.route('/', methods=['GET'])
        @paginated_response(default_per_page=20)
        def get_posts(page, per_page):
            posts = Post.query.paginate(page=page, per_page=per_page)
            return jsonify({...})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract pagination parameters
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', default_per_page, type=int)
            
            # Validate page number
            if page < 1:
                return jsonify({
                    'error': 'Page number must be greater than 0'
                }), 400
                
            # Validate and cap per_page
            if per_page < 1:
                per_page = default_per_page
            if per_page > max_per_page:
                per_page = max_per_page
                
            # Add pagination params to kwargs
            kwargs['page'] = page
            kwargs['per_page'] = per_page
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_json(*required_fields):
    """
    Validate that request contains JSON body with required fields.
    
    Checks Content-Type header, parses JSON body, and validates
    that all required fields are present. Passes validated data
    to the decorated function as 'data' keyword argument.
    
    Args:
        *required_fields: Variable number of required field names
        
    Usage:
        @auth_bp.route('/register', methods=['POST'])
        @validate_json('email', 'password', 'first_name')
        def register(data):
            email = data['email']
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check Content-Type header
            if not request.is_json:
                return jsonify({
                    'error': 'Content-Type must be application/json'
                }), 400
                
            # Parse JSON body
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({
                    'error': 'Invalid JSON or empty request body'
                }), 400
                
            # Check for required fields
            missing_fields = [
                field for field in required_fields
                if field not in data or data[field] is None
            ]
            
            if missing_fields:
                return jsonify({
                    'error': 'Missing required fields',
                    'missing_fields': missing_fields
                }), 400
                
            # Add validated data to kwargs
            kwargs['data'] = data
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def handle_db_errors(f):
    """
    Handle common database errors and return appropriate responses.
    
    Catches IntegrityError, OperationalError, and general exceptions.
    Automatically rolls back the session on error.
    
    Usage:
        @posts_bp.route('/', methods=['POST'])
        @handle_db_errors
        def create_post():
            post = Post(...)
            db.session.add(post)
            db.session.commit()
            return jsonify(post.to_dict()), 201
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
            
        except db.exc.IntegrityError as e:
            db.session.rollback()
            current_app.logger.error(f"Database integrity error: {e}")
            
            # Parse error message for common constraints
            error_msg = str(e.orig).lower()
            if 'unique' in error_msg or 'duplicate' in error_msg:
                return jsonify({
                    'error': 'A record with this value already exists',
                    'type': 'duplicate_entry'
                }), 409
            elif 'foreign key' in error_msg:
                return jsonify({
                    'error': 'Referenced record does not exist',
                    'type': 'foreign_key_violation'
                }), 400
            else:
                return jsonify({
                    'error': 'Database constraint violation',
                    'type': 'integrity_error'
                }), 400
                
        except db.exc.OperationalError as e:
            db.session.rollback()
            current_app.logger.error(f"Database operational error: {e}")
            return jsonify({
                'error': 'Database is temporarily unavailable',
                'type': 'database_error'
            }), 503
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Unexpected error: {e}")
            return jsonify({
                'error': 'An unexpected error occurred',
                'type': 'internal_error'
            }), 500
            
    return decorated_function


def validate_file_upload(required=True, allowed_extensions=None, max_size_mb=16):
    """
    Validate file upload in request.
    
    Checks for file presence, extension validation, and file size.
    Passes the file object to the decorated function.
    
    Args:
        required: Whether file upload is required
        allowed_extensions: Set of allowed file extensions (uses config if None)
        max_size_mb: Maximum file size in megabytes
        
    Usage:
        @predict_bp.route('/analyze', methods=['POST'])
        @validate_file_upload(allowed_extensions={'jpg', 'jpeg', 'png'})
        def analyze_image(file):
            # file is the uploaded FileStorage object
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if file is in request
            if 'image' not in request.files and 'file' not in request.files:
                if required:
                    return jsonify({
                        'error': 'No image file provided',
                        'details': 'Upload an image using "image" or "file" field'
                    }), 400
                kwargs['file'] = None
                return f(*args, **kwargs)
                
            # Get file (try both common field names)
            file = request.files.get('image') or request.files.get('file')
            
            # Check if file was actually selected
            if file.filename == '':
                if required:
                    return jsonify({
                        'error': 'No file selected'
                    }), 400
                kwargs['file'] = None
                return f(*args, **kwargs)
                
            # Get allowed extensions
            extensions = allowed_extensions or current_app.config.get(
                'ALLOWED_EXTENSIONS',
                {'png', 'jpg', 'jpeg', 'gif', 'webp'}
            )
            
            # Validate extension
            if not ('.' in file.filename and 
                    file.filename.rsplit('.', 1)[1].lower() in extensions):
                return jsonify({
                    'error': 'Invalid file type',
                    'allowed_extensions': list(extensions)
                }), 400
                
            # Check file size (read content length or check actual size)
            content_length = request.content_length
            if content_length and content_length > max_size_mb * 1024 * 1024:
                return jsonify({
                    'error': f'File too large. Maximum size is {max_size_mb}MB'
                }), 413
                
            # Add file to kwargs
            kwargs['file'] = file
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def log_request(f):
    """
    Log incoming request details for debugging and monitoring.
    
    Logs method, path, remote address, and response status code.
    
    Usage:
        @api_bp.route('/endpoint')
        @log_request
        def my_endpoint():
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Log request details
        current_app.logger.info(
            f"Request: {request.method} {request.path} "
            f"from {request.remote_addr}"
        )
        
        # Execute function
        response = f(*args, **kwargs)
        
        # Log response status
        if isinstance(response, tuple):
            status_code = response[1] if len(response) > 1 else 200
        else:
            status_code = getattr(response, 'status_code', 200)
            
        current_app.logger.info(
            f"Response: {status_code} for {request.method} {request.path}"
        )
        
        return response
    return decorated_function


def rate_limit_key_user():
    """
    Custom rate limit key function that uses user ID if authenticated.
    
    Falls back to IP address for unauthenticated requests.
    
    Usage:
        @limiter.limit("10 per minute", key_func=rate_limit_key_user)
        def my_endpoint():
            ...
    """
    from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
    
    try:
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
        if user_id:
            return f"user:{user_id}"
    except:
        pass
        
    return request.remote_addr
