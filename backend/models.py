# =============================================================================
# AgriLite-Hybrid Backend
# models.py - Database Models
# 
# SQLAlchemy ORM models for the application database.
# Includes User model for authentication and DetectionHistory for storing results.
# =============================================================================

from datetime import datetime, timezone
from extensions import db


class User(db.Model):
    """
    User model for authentication and authorization.
    
    Stores user credentials, profile information, and account status.
    Related to DetectionHistory through one-to-many relationship.
    """
    __tablename__ = 'users'
    
    # Primary Key
    id = db.Column(db.Integer, primary_key=True)
    
    # Authentication Fields
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Profile Fields
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    
    # Account Status
    is_active = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    
    # Timestamps
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    detection_history = db.relationship(
        'DetectionHistory',
        backref='user',
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    
    def to_dict(self, include_email=True):
        """
        Serialize user object to dictionary for API responses.
        
        Args:
            include_email: Whether to include email in response (privacy)
            
        Returns:
            dict: User data dictionary
        """
        data = {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f"{self.first_name or ''} {self.last_name or ''}".strip(),
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        
        if include_email:
            data['email'] = self.email
            
        return data
    
    def __repr__(self):
        return f'<User {self.email}>'


class DetectionHistory(db.Model):
    """
    Detection History model for storing plant disease predictions.
    
    Stores all prediction results including disease, severity, confidence,
    treatment recommendations, and optional environmental data.
    """
    __tablename__ = 'detection_history'
    
    # Primary Key
    id = db.Column(db.Integer, primary_key=True)
    
    # Foreign Key to User (optional - allows guest detections)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=True,
        index=True
    )
    
    # Crop Information
    crop = db.Column(db.String(50), nullable=False, index=True)
    
    # Prediction Results
    disease = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(20), nullable=False)  # mild, medium, severe
    confidence = db.Column(db.Float, nullable=False)
    
    # Treatment stored as JSON string
    treatment = db.Column(db.Text, nullable=True)
    
    # Image Data (stored as base64 or file path)
    original_image = db.Column(db.Text, nullable=True)  # Base64 or path
    heatmap_image = db.Column(db.Text, nullable=True)   # Base64 Grad-CAM
    
    # Environmental Data (optional - from DHT22 sensor)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    environmental_risk = db.Column(db.String(50), nullable=True)
    
    # Metadata
    device_type = db.Column(db.String(50), default='web')  # web, mobile, edge
    location = db.Column(db.String(200), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    
    # Indexes for common queries
    __table_args__ = (
        db.Index('idx_history_user_created', 'user_id', 'created_at'),
        db.Index('idx_history_crop_disease', 'crop', 'disease'),
    )
    
    def to_dict(self, include_images=False):
        """
        Serialize detection history to dictionary for API responses.
        
        Args:
            include_images: Whether to include base64 image data (large)
            
        Returns:
            dict: Detection history data dictionary
        """
        import json
        
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'crop': self.crop,
            'disease': self.disease,
            'severity': self.severity,
            'confidence': self.confidence,
            'treatment': json.loads(self.treatment) if self.treatment else [],
            'temperature': self.temperature,
            'humidity': self.humidity,
            'environmental_risk': self.environmental_risk,
            'device_type': self.device_type,
            'location': self.location,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
        if include_images:
            data['original_image'] = self.original_image
            data['heatmap_image'] = self.heatmap_image
            
        return data
    
    def __repr__(self):
        return f'<DetectionHistory {self.id}: {self.crop} - {self.disease}>'


class Crop(db.Model):
    """
    Crop model for storing crop metadata.
    
    Stores information about supported crops including diseases,
    images, and descriptions. Can be extended for dynamic crop management.
    """
    __tablename__ = 'crops'
    
    # Primary Key
    id = db.Column(db.Integer, primary_key=True)
    
    # Crop Information
    name = db.Column(db.String(50), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(255), nullable=True)
    
    # Diseases stored as JSON string (list of disease names)
    diseases = db.Column(db.Text, nullable=True)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )
    
    def to_dict(self):
        """
        Serialize crop to dictionary for API responses.
        
        Returns:
            dict: Crop data dictionary
        """
        import json
        
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'image_url': self.image_url,
            'diseases': json.loads(self.diseases) if self.diseases else [],
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f'<Crop {self.name}>'
