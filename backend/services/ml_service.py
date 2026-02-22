# =============================================================================
# AgriLite-Hybrid Backend
# services/ml_service.py - Machine Learning Service
# 
# Handles model loading, image preprocessing, inference, and
# Grad-CAM heatmap generation for plant disease detection.
# =============================================================================

import os
import io
import json
import base64
import random
import logging
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


class MLService:
    """
    Machine Learning Service for plant disease detection.
    
    This service handles:
    - Model loading (Keras .h5 or TensorFlow Lite)
    - Image preprocessing
    - Disease prediction inference
    - Grad-CAM heatmap generation
    - Treatment recommendation lookup
    
    When the actual model is not available, it provides placeholder
    predictions for development and testing purposes.
    """
    
    # Image preprocessing constants
    IMAGE_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML service.
        
        Args:
            model_path: Path to the models directory (optional)
        """
        self.model = None
        self.model_loaded = False
        self.use_tflite = False
        self.tflite_interpreter = None
        
        # Set model path
        if model_path:
            self.model_path = model_path
        else:
            # Default path relative to backend directory
            self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '..',
                'models'
            )
        
        # Load class labels and treatments
        self.class_labels = self._load_class_labels()
        self.treatments = self.class_labels.get('treatments', {})
        self.crops = self.class_labels.get('crops', {})
        
        # Try to load the model
        self._load_model()
    
    def _load_class_labels(self) -> Dict:
        """
        Load class labels and treatment data from JSON file.
        
        Returns:
            Dictionary containing crops, diseases, and treatments
        """
        labels_path = os.path.join(self.model_path, 'class_labels.json')
        
        try:
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load class_labels.json: {e}")
        
        # Return default labels if file not found
        return self._get_default_labels()
    
    def _get_default_labels(self) -> Dict:
        """
        Get default class labels for fallback.
        
        Returns:
            Default labels dictionary
        """
        return {
            'crops': {
                'brinjal': {
                    'diseases': ['Cercospora Leaf Spot', 'Little Leaf Disease', 
                                'Phomopsis Blight', 'Healthy']
                },
                'okra': {
                    'diseases': ['Yellow Vein Mosaic', 'Powdery Mildew',
                                'Leaf Curl Disease', 'Healthy']
                },
                'tomato': {
                    'diseases': ['Early Blight', 'Late Blight', 'Bacterial Spot',
                                'Septoria Leaf Spot', 'Healthy']
                },
                'chilli': {
                    'diseases': ['Leaf Curl Virus', 'Powdery Mildew',
                                'Anthracnose', 'Bacterial Leaf Spot', 'Healthy']
                }
            },
            'severity_levels': ['mild', 'medium', 'severe'],
            'treatments': {}
        }
    
    def _load_model(self) -> bool:
        """
        Load the trained model (Keras or TFLite).
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Try loading Keras model first
        h5_path = os.path.join(self.model_path, 'agrilite_hybrid.h5')
        keras_path = os.path.join(self.model_path, 'agrilite_hybrid.keras')
        tflite_path = os.path.join(self.model_path, 'agrilite_hybrid.tflite')
        
        # Try Keras/H5 model
        for model_file in [h5_path, keras_path]:
            if os.path.exists(model_file):
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_file)
                    self.model_loaded = True
                    logger.info(f"Keras model loaded from {model_file}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load Keras model: {e}")
        
        # Try TFLite model
        if os.path.exists(tflite_path):
            try:
                import tflite_runtime.interpreter as tflite
                self.tflite_interpreter = tflite.Interpreter(model_path=tflite_path)
                self.tflite_interpreter.allocate_tensors()
                self.use_tflite = True
                self.model_loaded = True
                logger.info(f"TFLite model loaded from {tflite_path}")
                return True
            except ImportError:
                try:
                    import tensorflow as tf
                    self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
                    self.tflite_interpreter.allocate_tensors()
                    self.use_tflite = True
                    self.model_loaded = True
                    logger.info(f"TFLite model loaded using TensorFlow")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load TFLite model: {e}")
        
        logger.warning("No model file found - using placeholder predictions")
        return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loaded
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image array of shape (1, 224, 224, 3)
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Apply ImageNet normalization
        img_array = (img_array - self.MEAN) / self.STD
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(
        self,
        image_data: bytes,
        crop: str,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run disease prediction on an image.
        
        Args:
            image_data: Raw image bytes
            crop: Crop type (brinjal, okra, tomato, chilli)
            temperature: Optional environmental temperature (°C)
            humidity: Optional environmental humidity (%)
            
        Returns:
            Dictionary containing prediction results:
            - disease: Predicted disease name
            - severity: Severity level (mild, medium, severe)
            - confidence: Prediction confidence percentage
            - treatment: List of treatment recommendations
            - heatmap_base64: Grad-CAM heatmap as base64
            - original_image_base64: Original image as base64
            - environmental_risk: Risk assessment based on environment
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Get diseases for this crop
            crop_diseases = self.crops.get(crop, {}).get('diseases', [])
            if not crop_diseases:
                crop_diseases = self._get_default_labels()['crops'].get(crop, {}).get('diseases', [])
            
            # Run inference
            if self.model_loaded and not self.use_tflite:
                # Keras model prediction
                predictions = self.model.predict(processed_image, verbose=0)
                disease_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][disease_idx]) * 100
                
                # Map to disease name
                if disease_idx < len(crop_diseases):
                    disease = crop_diseases[disease_idx]
                else:
                    disease = crop_diseases[0] if crop_diseases else 'Unknown'
                    
            elif self.model_loaded and self.use_tflite:
                # TFLite prediction
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()
                
                self.tflite_interpreter.set_tensor(
                    input_details[0]['index'],
                    processed_image.astype(np.float32)
                )
                self.tflite_interpreter.invoke()
                
                predictions = self.tflite_interpreter.get_tensor(output_details[0]['index'])
                disease_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][disease_idx]) * 100
                
                if disease_idx < len(crop_diseases):
                    disease = crop_diseases[disease_idx]
                else:
                    disease = crop_diseases[0] if crop_diseases else 'Unknown'
            else:
                # Placeholder prediction for development
                disease, confidence = self._placeholder_prediction(crop, crop_diseases)
            
            # Determine severity based on confidence and disease type
            severity = self._determine_severity(disease, confidence)
            
            # Get treatment recommendations
            treatment = self.get_treatment(disease, severity)
            
            # Generate heatmap
            heatmap_base64 = self._generate_heatmap(image_data, processed_image)
            
            # Convert original image to base64
            original_base64 = self._image_to_base64(image_data)
            
            # Calculate environmental risk
            environmental_risk = self._assess_environmental_risk(
                temperature, humidity, disease
            )
            
            return {
                'success': True,
                'crop': crop,
                'disease': disease,
                'severity': severity,
                'confidence': round(confidence, 1),
                'treatment': treatment,
                'heatmap_base64': heatmap_base64,
                'original_image_base64': original_base64,
                'environmental_risk': environmental_risk,
                'model_used': 'keras' if (self.model_loaded and not self.use_tflite) else (
                    'tflite' if self.use_tflite else 'placeholder'
                )
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _placeholder_prediction(
        self,
        crop: str,
        diseases: List[str]
    ) -> tuple:
        """
        Generate placeholder prediction for development.
        
        Args:
            crop: Crop type
            diseases: List of possible diseases
            
        Returns:
            Tuple of (disease_name, confidence)
        """
        # Weighted random selection (healthy should be less common for demo)
        weights = []
        for d in diseases:
            if d.lower() == 'healthy':
                weights.append(0.2)  # 20% chance healthy
            else:
                weights.append(0.8 / (len(diseases) - 1))  # Split remaining among diseases
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        disease = random.choices(diseases, weights=weights, k=1)[0]
        
        # Generate realistic confidence
        if disease.lower() == 'healthy':
            confidence = random.uniform(85, 99)
        else:
            confidence = random.uniform(75, 98)
        
        logger.info(f"Placeholder prediction: {disease} ({confidence:.1f}%)")
        
        return disease, confidence
    
    def _determine_severity(self, disease: str, confidence: float) -> str:
        """
        Determine disease severity based on prediction.
        
        Args:
            disease: Disease name
            confidence: Prediction confidence
            
        Returns:
            Severity level (mild, medium, severe)
        """
        if disease.lower() == 'healthy':
            return 'mild'  # Not really applicable, but needed for consistency
        
        # Use confidence as a rough indicator plus randomization for demo
        # In real implementation, this would be a separate model output
        base_severity = random.random()
        
        if base_severity < 0.33:
            return 'mild'
        elif base_severity < 0.66:
            return 'medium'
        else:
            return 'severe'
    
    def get_treatment(self, disease: str, severity: str) -> List[str]:
        """
        Get treatment recommendations for a disease.
        
        Args:
            disease: Disease name
            severity: Severity level
            
        Returns:
            List of treatment recommendations
        """
        # Look up treatment in loaded data
        disease_treatments = self.treatments.get(disease, {})
        
        if disease_treatments:
            return disease_treatments.get(severity, disease_treatments.get('medium', []))
        
        # Default treatments if not found
        if disease.lower() == 'healthy':
            return [
                "No treatment needed - plant appears healthy",
                "Continue regular maintenance and monitoring",
                "Ensure proper watering and fertilization"
            ]
        
        return [
            "Consult with a local agricultural extension officer",
            "Remove affected plant parts if possible",
            "Ensure proper plant spacing for air circulation",
            "Consider appropriate fungicide or pesticide treatment"
        ]
    
    def _generate_heatmap(
        self,
        original_data: bytes,
        processed_image: np.ndarray
    ) -> str:
        """
        Generate Grad-CAM heatmap for prediction visualization.
        
        Args:
            original_data: Original image bytes
            processed_image: Preprocessed image array
            
        Returns:
            Base64 encoded heatmap overlay image
        """
        try:
            # Load original image
            original = Image.open(io.BytesIO(original_data))
            if original.mode != 'RGB':
                original = original.convert('RGB')
            original = original.resize(self.IMAGE_SIZE)
            original_array = np.array(original)
            
            if self.model_loaded and self.model is not None and not self.use_tflite:
                # Real Grad-CAM with Keras model
                heatmap = self._compute_gradcam(processed_image)
            else:
                # Placeholder heatmap for development
                heatmap = self._generate_placeholder_heatmap()
            
            # Create colored heatmap
            import cv2
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay on original image
            overlay = cv2.addWeighted(
                original_array, 0.6,
                heatmap_colored, 0.4,
                0
            )
            
            # Convert to base64
            overlay_image = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_str}"
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            # Return original image as fallback
            return self._image_to_base64(original_data)
    
    def _compute_gradcam(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Grad-CAM heatmap using the model.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Heatmap array of shape (224, 224)
        """
        try:
            import tensorflow as tf
            
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                return self._generate_placeholder_heatmap()
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]
            
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature maps
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
            
            # ReLU and normalize
            heatmap = tf.nn.relu(heatmap)
            heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
            
            # Resize to image size
            heatmap = tf.image.resize(
                heatmap[..., tf.newaxis],
                self.IMAGE_SIZE
            )
            
            return heatmap.numpy().squeeze()
            
        except Exception as e:
            logger.warning(f"Grad-CAM computation failed: {e}")
            return self._generate_placeholder_heatmap()
    
    def _generate_placeholder_heatmap(self) -> np.ndarray:
        """
        Generate a placeholder heatmap for development.
        
        Creates a realistic-looking heatmap with random but
        structured hot spots.
        
        Returns:
            Heatmap array of shape (224, 224)
        """
        # Create base heatmap with gaussian blobs
        heatmap = np.zeros((224, 224), dtype=np.float32)
        
        # Add 2-4 random hot spots
        num_spots = random.randint(2, 4)
        
        for _ in range(num_spots):
            # Random center position (avoiding edges)
            cx = random.randint(40, 184)
            cy = random.randint(40, 184)
            
            # Random size
            size = random.randint(30, 60)
            
            # Create gaussian blob
            y, x = np.ogrid[:224, :224]
            blob = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * size**2))
            
            # Add to heatmap with random intensity
            intensity = random.uniform(0.5, 1.0)
            heatmap = np.maximum(heatmap, blob * intensity)
        
        # Normalize
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap
    
    def _image_to_base64(self, image_data: bytes) -> str:
        """
        Convert image bytes to base64 string.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Base64 encoded image with data URI prefix
        """
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for efficient transfer
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    
    def _assess_environmental_risk(
        self,
        temperature: Optional[float],
        humidity: Optional[float],
        disease: str
    ) -> str:
        """
        Assess environmental risk based on temperature and humidity.
        
        Args:
            temperature: Temperature in °C (optional)
            humidity: Humidity in % (optional)
            disease: Detected disease
            
        Returns:
            Risk level string (normal, elevated, high)
        """
        if temperature is None and humidity is None:
            return 'unknown'
        
        risk_factors = []
        
        if temperature is not None:
            if temperature > 35:
                risk_factors.append('high_temp')
            elif temperature < 10:
                risk_factors.append('low_temp')
        
        if humidity is not None:
            if humidity > 85:
                risk_factors.append('high_humidity')
            elif humidity < 30:
                risk_factors.append('low_humidity')
        
        # High humidity increases fungal disease risk
        fungal_diseases = [
            'Powdery Mildew', 'Cercospora Leaf Spot', 
            'Phomopsis Blight', 'Early Blight', 'Late Blight',
            'Septoria Leaf Spot', 'Anthracnose'
        ]
        
        if disease in fungal_diseases and 'high_humidity' in risk_factors:
            return 'high'
        
        if len(risk_factors) >= 2:
            return 'high'
        elif len(risk_factors) == 1:
            return 'elevated'
        
        return 'normal'
