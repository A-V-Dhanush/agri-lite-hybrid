"""
Brinjal (Eggplant) Disease Detection Model Training Script
==========================================================
Architecture: EfficientNetV2-B0 with CBAM Attention
Dataset: Eggplant Disease Recognition Dataset (7 classes)
Classes: Healthy Leaf, Insect Pest Disease, Leaf Spot Disease, 
         Mosaic Virus Disease, Small Leaf Disease, White Mold Disease, Wilt Disease

Author: AgriLite Hybrid Project
Date: February 2026
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall

# Sklearn for evaluation
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score
)

# For Grad-CAM
import cv2

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class Config:
    """Training configuration parameters"""
    
    # Dataset paths - UPDATE THESE TO YOUR LOCAL PATHS
    # Option 1: If you have train/val/test split already
    # DATA_DIR = "data/brinjal"
    # TRAIN_DIR = os.path.join(DATA_DIR, "train")
    # VAL_DIR = os.path.join(DATA_DIR, "val")
    # TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Option 2: Use the Kaggle dataset structure (Augmented Images Version 02)
    # This script will auto-split if needed
    DATA_DIR = r"D:\rts project\agri-lite-hybrid\DataSets\eggplant\Eggplant Disease Recognition Dataset\Augmented Images (Version 02)\Augmented Images (Version 02)"
    
    # Model parameters
    MODEL_NAME = "brinjal_efficientnetv2b0_cbam"
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 7
    
    # Class names (will be auto-detected from folders)
    CLASS_NAMES = [
        "Healthy Leaf",
        "Insect Pest Disease", 
        "Leaf Spot Disease",
        "Mosaic Virus Disease",
        "Small Leaf Disease",
        "White Mold Disease",
        "Wilt Disease"
    ]
    
    # Training parameters
    BATCH_SIZE = 32  # Adjust based on GPU memory (use 16 if OOM)
    EPOCHS = 100
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Data split (only used if auto-splitting)
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Callbacks
    EARLY_STOP_PATIENCE = 10
    LR_REDUCE_PATIENCE = 5
    LR_REDUCE_FACTOR = 0.5
    
    # Output directories
    OUTPUT_DIR = "outputs/brinjal"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    
    # Create directories
    @classmethod
    def create_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.PLOT_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"Output directories created at: {cls.OUTPUT_DIR}")

# =============================================================================
# 3. DATA LOADING & AUGMENTATION
# =============================================================================

def create_data_generators(config):
    """
    Create data generators with augmentation for training.
    
    Augmentation strategy:
    - Rotation: ±30°
    - Horizontal/Vertical flip
    - Brightness: ±20%
    - Zoom: 0.8-1.2
    - Shear: ±10°
    - Width/Height shift: ±20%
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation (will split again for test)
    )
    
    # Validation/Test data generator (no augmentation, only rescale)
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    print(f"\nLoading data from: {config.DATA_DIR}")
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        config.DATA_DIR,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    val_generator = val_test_datagen.flow_from_directory(
        config.DATA_DIR,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Get class names from generator
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    num_classes = len(class_names)
    
    print(f"\nDataset Statistics:")
    print(f"  - Training samples: {train_generator.samples}")
    print(f"  - Validation samples: {val_generator.samples}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    
    # Save class labels to JSON
    labels_path = os.path.join(config.OUTPUT_DIR, "class_labels.json")
    with open(labels_path, 'w') as f:
        json.dump({
            'class_indices': class_indices,
            'class_names': class_names,
            'num_classes': num_classes
        }, f, indent=2)
    print(f"  - Class labels saved to: {labels_path}")
    
    return train_generator, val_generator, class_names, num_classes


def create_split_generators(config, train_dir, val_dir, test_dir):
    """
    Alternative: Create generators from pre-split directories.
    Use this if you have separate train/val/test folders.
    """
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    return train_generator, val_generator, test_generator, class_names, num_classes


# =============================================================================
# 4. CBAM ATTENTION MODULE
# =============================================================================

class ChannelAttention(layers.Layer):
    """
    Channel Attention Module (CAM) - Part of CBAM
    
    Applies attention across channels using both average and max pooling.
    Reference: CBAM: Convolutional Block Attention Module (Woo et al., 2018)
    """
    
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        # Shared MLP
        self.fc1 = layers.Dense(
            self.channels // self.reduction_ratio, 
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True
        )
        self.fc2 = layers.Dense(
            self.channels,
            kernel_initializer='he_normal',
            use_bias=True
        )
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Average pooling path
        avg_out = self.avg_pool(inputs)
        avg_out = self.fc1(avg_out)
        avg_out = self.fc2(avg_out)
        
        # Max pooling path
        max_out = self.max_pool(inputs)
        max_out = self.fc1(max_out)
        max_out = self.fc2(max_out)
        
        # Combine and apply sigmoid
        attention = tf.nn.sigmoid(avg_out + max_out)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])
        
        return inputs * attention
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio
        })
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module (SAM) - Part of CBAM
    
    Applies attention across spatial dimensions using channel-wise pooling.
    """
    
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Channel-wise average pooling
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        
        # Channel-wise max pooling
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate and convolve
        concat = tf.concat([avg_out, max_out], axis=-1)
        attention = self.conv(concat)
        
        return inputs * attention
    
    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM)
    
    Combines Channel Attention and Spatial Attention sequentially.
    Improves feature representation by focusing on important channels
    and spatial locations.
    """
    
    def __init__(self, channels, reduction_ratio=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.channel_attention = ChannelAttention(
            self.channels, 
            self.reduction_ratio
        )
        self.spatial_attention = SpatialAttention(self.kernel_size)
        super(CBAM, self).build(input_shape)
        
    def call(self, inputs):
        # Apply channel attention first
        x = self.channel_attention(inputs)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x
    
    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config


# =============================================================================
# 5. MODEL ARCHITECTURE
# =============================================================================

def build_efficientnet_cbam_model(input_shape, num_classes, use_cbam=True):
    """
    Build EfficientNetV2-B0 model with CBAM attention.
    
    Architecture:
    - EfficientNetV2-B0 base (ImageNet pretrained, frozen initially)
    - CBAM attention module after base features
    - Global Average Pooling
    - Dense layers with dropout
    - Softmax classification head
    
    Args:
        input_shape: Tuple (height, width, channels)
        num_classes: Number of output classes
        use_cbam: Whether to add CBAM attention (default: True)
    
    Returns:
        Keras Model
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_layer')
    
    # Base model: EfficientNetV2-B0 with ImageNet weights
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None  # We'll add our own pooling after CBAM
    )
    
    # Freeze base model layers initially (for transfer learning)
    base_model.trainable = False
    
    # Get base model output
    x = base_model.output
    
    # Add CBAM attention module
    if use_cbam:
        # Get number of channels from base model output
        num_channels = x.shape[-1]  # 1280 for EfficientNetV2-B0
        x = CBAM(channels=num_channels, reduction_ratio=16, kernel_size=7, name='cbam')(x)
        print(f"CBAM attention added with {num_channels} channels")
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    # Dense layer
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='EfficientNetV2B0_CBAM')
    
    return model, base_model


def unfreeze_model(model, base_model, num_layers_to_unfreeze=20):
    """
    Unfreeze the last N layers of the base model for fine-tuning.
    
    Args:
        model: Full model
        base_model: Base EfficientNet model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    """
    
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze all layers except the last N
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Count trainable layers
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    non_trainable_count = sum([1 for layer in model.layers if not layer.trainable])
    
    print(f"\nModel unfrozen for fine-tuning:")
    print(f"  - Trainable layers: {trainable_count}")
    print(f"  - Non-trainable layers: {non_trainable_count}")
    
    return model


# =============================================================================
# 6. CUSTOM METRICS
# =============================================================================

class F1Score(tf.keras.metrics.Metric):
    """Custom F1 Score metric for Keras (macro average)"""
    
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision = Precision()
        self.recall = Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


# =============================================================================
# 7. TRAINING FUNCTIONS
# =============================================================================

def get_callbacks(config, phase='initial'):
    """
    Create training callbacks.
    
    Args:
        config: Configuration object
        phase: 'initial' or 'finetune'
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Model checkpoint (save best model)
        ModelCheckpoint(
            filepath=os.path.join(
                config.MODEL_DIR, 
                f'{config.MODEL_NAME}_{phase}_best.keras'
            ),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Learning rate scheduler
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, f'{phase}_{timestamp}'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def compile_model(model, learning_rate, num_classes):
    """Compile model with optimizer, loss, and metrics."""
    
    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=Config.WEIGHT_DECAY
        ),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    
    return model


def train_model(model, train_gen, val_gen, config, phase='initial', epochs=None):
    """
    Train the model.
    
    Args:
        model: Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        config: Configuration object
        phase: Training phase ('initial' or 'finetune')
        epochs: Number of epochs (default: config.EPOCHS)
    """
    
    if epochs is None:
        epochs = config.EPOCHS
    
    print(f"\n{'='*60}")
    print(f"Starting {phase} training for {epochs} epochs...")
    print(f"{'='*60}")
    
    callbacks = get_callbacks(config, phase)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# =============================================================================
# 8. EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, test_gen, class_names, config):
    """
    Evaluate model on test set and generate reports.
    
    Args:
        model: Trained model
        test_gen: Test data generator
        class_names: List of class names
        config: Configuration object
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("\n" + "="*60)
    print("Evaluating model on test/validation set...")
    print("="*60)
    
    # Get predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Get class labels in correct order
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Classification report
    print("\n📊 Classification Report:")
    print("-" * 60)
    report = classification_report(
        y_true, y_pred, 
        target_names=class_labels,
        digits=4
    )
    print(report)
    
    # Save classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        output_dict=True
    )
    
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nClassification report saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix - Brinjal Disease Detection', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'classification_report': report_dict
    }
    
    print(f"\n📈 Summary Metrics:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1-Score (Macro): {f1_macro:.4f}")
    print(f"  - F1-Score (Weighted): {f1_weighted:.4f}")
    
    return metrics


def plot_training_history(history, config, phase=''):
    """Plot and save training history curves."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History - Brinjal EfficientNetV2-B0 + CBAM ({phase})', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(config.PLOT_DIR, f"training_history_{phase}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {plot_path}")


# =============================================================================
# 9. MODEL EXPORT & TFLITE CONVERSION
# =============================================================================

def save_model(model, config):
    """Save model in multiple formats."""
    
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)
    
    # Save Keras model (.keras format - recommended)
    keras_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final.keras")
    model.save(keras_path)
    print(f"✓ Keras model saved: {keras_path}")
    
    # Save H5 format (legacy, but widely compatible)
    h5_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final.h5")
    model.save(h5_path, save_format='h5')
    print(f"✓ H5 model saved: {h5_path}")
    
    # Save SavedModel format (TF serving compatible)
    savedmodel_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_savedmodel")
    model.export(savedmodel_path)
    print(f"✓ SavedModel saved: {savedmodel_path}")
    
    return keras_path, h5_path, savedmodel_path


def convert_to_tflite(model, config, quantize=True):
    """
    Convert model to TFLite format for edge deployment.
    
    Args:
        model: Trained Keras model
        config: Configuration object
        quantize: Whether to apply INT8 quantization (default: True)
    
    Returns:
        Paths to TFLite models
    """
    
    print("\n" + "="*60)
    print("Converting to TFLite...")
    print("="*60)
    
    # Convert to TFLite (float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"✓ TFLite (float32) saved: {tflite_path} ({tflite_size:.2f} MB)")
    
    # Convert with INT8 quantization (for Raspberry Pi)
    if quantize:
        converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quant.target_spec.supported_types = [tf.int8]
        
        # For full INT8, we need representative dataset
        # This is a simple version without representative data
        tflite_quant_model = converter_quant.convert()
        
        tflite_quant_path = os.path.join(
            config.MODEL_DIR, 
            f"{config.MODEL_NAME}_int8.tflite"
        )
        with open(tflite_quant_path, 'wb') as f:
            f.write(tflite_quant_model)
        
        tflite_quant_size = os.path.getsize(tflite_quant_path) / (1024 * 1024)
        print(f"✓ TFLite (INT8) saved: {tflite_quant_path} ({tflite_quant_size:.2f} MB)")
        
        return tflite_path, tflite_quant_path
    
    return tflite_path, None


def create_representative_dataset(data_gen, num_samples=100):
    """
    Create representative dataset generator for full INT8 quantization.
    """
    def representative_data_gen():
        data_gen.reset()
        for i, (images, _) in enumerate(data_gen):
            if i >= num_samples // data_gen.batch_size:
                break
            for image in images:
                yield [np.expand_dims(image.astype(np.float32), axis=0)]
    
    return representative_data_gen


def convert_to_tflite_full_int8(model, data_gen, config):
    """
    Convert to fully quantized INT8 TFLite with representative dataset.
    This provides best performance on Raspberry Pi.
    """
    
    print("\n" + "="*60)
    print("Converting to fully quantized INT8 TFLite...")
    print("="*60)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = create_representative_dataset(data_gen)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    try:
        tflite_quant_model = converter.convert()
        
        tflite_path = os.path.join(
            config.MODEL_DIR,
            f"{config.MODEL_NAME}_full_int8.tflite"
        )
        with open(tflite_path, 'wb') as f:
            f.write(tflite_quant_model)
        
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"✓ TFLite (full INT8) saved: {tflite_path} ({size_mb:.2f} MB)")
        
        return tflite_path
    except Exception as e:
        print(f"⚠ Full INT8 conversion failed: {e}")
        print("  Falling back to default quantization...")
        return None


# =============================================================================
# 10. GRAD-CAM VISUALIZATION
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Generates visual explanations for CNN predictions.
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Name of conv layer to visualize (default: last conv layer)
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
                    layer_name = layer.name
                    break
                # For EfficientNet, look for the top conv layer
                if 'top_conv' in layer.name or 'block' in layer.name:
                    if hasattr(layer, 'output'):
                        layer_name = layer.name
                        break
        
        if layer_name is None:
            # For EfficientNetV2, use the output before global pooling
            layer_name = model.layers[-6].name  # Approximate position
        
        self.layer_name = layer_name
        print(f"Grad-CAM using layer: {layer_name}")
        
        # Create gradient model
        self.grad_model = Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(layer_name).output,
                model.output
            ]
        )
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Input image (preprocessed, shape: 1,H,W,C)
            class_idx: Target class index (default: predicted class)
            eps: Small constant for numerical stability
        
        Returns:
            Heatmap normalized to [0, 1]
        """
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap
            image: Original image (0-255 range, RGB)
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            Superimposed image
        """
        
        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        superimposed = heatmap * alpha + image * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed


def generate_gradcam_examples(model, data_gen, class_names, config, num_examples=1):
    """
    Generate Grad-CAM visualizations for each class.
    
    Args:
        model: Trained model
        data_gen: Data generator
        class_names: List of class names
        config: Configuration object
        num_examples: Examples per class
    """
    
    print("\n" + "="*60)
    print("Generating Grad-CAM visualizations...")
    print("="*60)
    
    # Initialize Grad-CAM
    # For EfficientNetV2-B0, we'll use the CBAM layer or last conv before pooling
    try:
        gradcam = GradCAM(model)
    except Exception as e:
        print(f"⚠ Grad-CAM initialization failed: {e}")
        # Try with a specific layer
        gradcam = GradCAM(model, layer_name='top_conv')
    
    # Get class indices
    idx_to_class = {v: k for k, v in data_gen.class_indices.items()}
    
    # Collect one example per class
    class_examples = {i: [] for i in range(len(class_names))}
    
    data_gen.reset()
    for images, labels in data_gen:
        for img, label in zip(images, labels):
            class_idx = np.argmax(label)
            if len(class_examples[class_idx]) < num_examples:
                class_examples[class_idx].append(img)
        
        # Check if we have enough examples
        if all(len(v) >= num_examples for v in class_examples.values()):
            break
    
    # Generate visualizations
    fig, axes = plt.subplots(len(class_names), 3, figsize=(12, 4 * len(class_names)))
    
    for class_idx, images in class_examples.items():
        if not images:
            continue
        
        img = images[0]
        class_name = idx_to_class.get(class_idx, f"Class {class_idx}")
        
        # Prepare image for model
        img_array = np.expand_dims(img, axis=0)
        
        # Get prediction
        pred_probs = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred_probs[0])
        pred_conf = pred_probs[0][pred_class]
        pred_name = idx_to_class.get(pred_class, f"Class {pred_class}")
        
        # Compute Grad-CAM heatmap
        heatmap = gradcam.compute_heatmap(img_array, class_idx)
        
        # Convert image to display format (0-255)
        img_display = (img * 255).astype(np.uint8)
        
        # Create overlay
        overlay = gradcam.overlay_heatmap(heatmap, img_display)
        
        # Plot
        row = class_idx
        
        # Original image
        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f'Original: {class_name}', fontsize=10)
        axes[row, 0].axis('off')
        
        # Heatmap
        axes[row, 1].imshow(heatmap, cmap='jet')
        axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[row, 1].axis('off')
        
        # Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f'Pred: {pred_name} ({pred_conf:.2%})', fontsize=10)
        axes[row, 2].axis('off')
    
    plt.suptitle('Grad-CAM Visualizations - Brinjal Disease Detection', fontsize=14)
    plt.tight_layout()
    
    gradcam_path = os.path.join(config.PLOT_DIR, "gradcam_examples.png")
    plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Grad-CAM visualizations saved: {gradcam_path}")


# =============================================================================
# 11. MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline for Brinjal disease detection model."""
    
    print("="*70)
    print("🍆 BRINJAL (EGGPLANT) DISEASE DETECTION MODEL TRAINING")
    print("   Architecture: EfficientNetV2-B0 + CBAM Attention")
    print("="*70)
    
    # Initialize configuration
    config = Config()
    config.create_dirs()
    
    # =========================================================================
    # PHASE 1: Data Loading
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Loading and Preparing Data")
    print("="*60)
    
    train_gen, val_gen, class_names, num_classes = create_data_generators(config)
    
    # Update config with detected classes
    config.NUM_CLASSES = num_classes
    config.CLASS_NAMES = class_names
    
    # =========================================================================
    # PHASE 2: Build Model
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Building Model")
    print("="*60)
    
    model, base_model = build_efficientnet_cbam_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=num_classes,
        use_cbam=True
    )
    
    # Print model summary
    model.summary()
    
    # =========================================================================
    # PHASE 3: Initial Training (Frozen Base)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Initial Training (Base Frozen)")
    print("="*60)
    
    model = compile_model(model, config.INITIAL_LR, num_classes)
    
    history_initial = train_model(
        model, train_gen, val_gen, config,
        phase='initial',
        epochs=20  # Initial phase with frozen base
    )
    
    plot_training_history(history_initial, config, phase='initial')
    
    # =========================================================================
    # PHASE 4: Fine-tuning (Unfrozen Base)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 4: Fine-tuning (Base Unfrozen)")
    print("="*60)
    
    # Unfreeze last 30 layers of base model
    model = unfreeze_model(model, base_model, num_layers_to_unfreeze=30)
    
    # Recompile with lower learning rate
    model = compile_model(model, config.INITIAL_LR / 10, num_classes)
    
    history_finetune = train_model(
        model, train_gen, val_gen, config,
        phase='finetune',
        epochs=config.EPOCHS - 20
    )
    
    plot_training_history(history_finetune, config, phase='finetune')
    
    # =========================================================================
    # PHASE 5: Evaluation
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 5: Model Evaluation")
    print("="*60)
    
    metrics = evaluate_model(model, val_gen, class_names, config)
    
    # Save metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # =========================================================================
    # PHASE 6: Save Models
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 6: Saving Models")
    print("="*60)
    
    keras_path, h5_path, savedmodel_path = save_model(model, config)
    
    # =========================================================================
    # PHASE 7: TFLite Conversion
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 7: TFLite Conversion")
    print("="*60)
    
    tflite_path, tflite_quant_path = convert_to_tflite(model, config, quantize=True)
    
    # Full INT8 quantization with representative dataset
    tflite_full_int8_path = convert_to_tflite_full_int8(model, val_gen, config)
    
    # =========================================================================
    # PHASE 8: Grad-CAM Visualization
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 8: Grad-CAM Visualization")
    print("="*60)
    
    generate_gradcam_examples(model, val_gen, class_names, config)
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Output Directory: {config.OUTPUT_DIR}")
    print(f"\n📊 Final Results:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"   - F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"\n📦 Saved Models:")
    print(f"   - Keras: {keras_path}")
    print(f"   - H5: {h5_path}")
    print(f"   - TFLite: {tflite_path}")
    if tflite_quant_path:
        print(f"   - TFLite (INT8): {tflite_quant_path}")
    if tflite_full_int8_path:
        print(f"   - TFLite (Full INT8): {tflite_full_int8_path}")
    print(f"\n📈 Training plots saved in: {config.PLOT_DIR}")
    print(f"📝 Logs saved in: {config.LOG_DIR}")
    print("\n" + "="*70)
    
    return model, metrics


# =============================================================================
# 12. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run training
    model, metrics = main()
