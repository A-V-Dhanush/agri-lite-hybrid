"""
Tomato Disease Detection Model Training Script
===============================================
Architecture: EfficientNetV2-B0 (Transfer Learning)
Dataset: Tomato Leaves Dataset from Kaggle (~25k images)
Classes: 11 classes (Healthy + 10 diseases)
    - Bacterial_spot
    - Early_blight
    - Late_blight
    - Leaf_Mold
    - Septoria_leaf_spot
    - Spider_mites Two-spotted_spider_mite
    - Target_Spot
    - Tomato_Yellow_Leaf_Curl_Virus
    - Tomato_mosaic_virus
    - healthy
    - powdery_mildew

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

# GPU Configuration
from gpu_config import setup_gpu, get_strategy

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Setup GPU with optimal settings
print("\n" + "="*70)
print("🔧 CONFIGURING GPU FOR TOMATO MODEL TRAINING")
print("="*70)
gpu_info = setup_gpu(
    memory_growth=True,
    mixed_precision=True,
    verbose=True
)

print(f"\nTensorFlow version: {tf.__version__}")
if gpu_info['gpu_available']:
    print(f"✓ Training on GPU: {gpu_info['gpu_names']}")
    print(f"✓ Mixed precision: {gpu_info['mixed_precision']}")
else:
    print("⚠️ Training on CPU (this will be slow)")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Get project root directory (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class Config:
    """Training configuration parameters for Tomato model"""
    
    # Dataset paths - RELATIVE from project root (works on Windows & Linux)
    DATA_DIR = os.path.join(PROJECT_ROOT, "DataSets", "tamota")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "valid")
    
    # Model parameters
    MODEL_NAME = "tomato_efficientnetv2b0"
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 11  # 10 diseases + 1 healthy
    
    # Class names (will be auto-detected from folders)
    CLASS_NAMES = [
        "Bacterial_spot",
        "Early_blight",
        "Late_blight",
        "Leaf_Mold",
        "Septoria_leaf_spot",
        "Spider_mites Two-spotted_spider_mite",
        "Target_Spot",
        "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_mosaic_virus",
        "healthy",
        "powdery_mildew"
    ]
    
    # Training parameters
    BATCH_SIZE = 32  # Adjust based on GPU memory (use 16 if OOM)
    EPOCHS = 100
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Fine-tuning parameters
    INITIAL_EPOCHS = 20  # Epochs with frozen base
    FINE_TUNE_EPOCHS = 80  # Additional epochs for fine-tuning
    FINE_TUNE_LR = 1e-5  # Lower learning rate for fine-tuning
    
    # Callbacks
    EARLY_STOP_PATIENCE = 10
    LR_REDUCE_PATIENCE = 5
    LR_REDUCE_FACTOR = 0.5
    
    # Output directories
    OUTPUT_DIR = "outputs/tomato"
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
    
    Augmentation strategy (as per specification):
    - Rotation: ±30°
    - Horizontal/Vertical flip
    - Brightness: ±20%
    - Zoom: 0.8-1.2
    - Shear: ±10°
    - Width/Height shift: ±20%
    """
    
    # Training data generator with heavy augmentation
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
    
    # Validation data generator (no augmentation, only rescale)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print(f"\nLoading training data from: {config.TRAIN_DIR}")
    print(f"Loading validation data from: {config.VAL_DIR}")
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    # Get class names from generator
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    num_classes = len(class_names)
    
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"  - Training samples: {train_generator.samples}")
    print(f"  - Validation samples: {val_generator.samples}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Steps per epoch (train): {len(train_generator)}")
    print(f"  - Steps per epoch (val): {len(val_generator)}")
    print(f"\nClass distribution:")
    for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
        print(f"    {idx}: {class_name}")
    
    # Save class labels to JSON
    labels_path = os.path.join(config.OUTPUT_DIR, "class_labels.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(labels_path, 'w') as f:
        json.dump({
            'class_indices': class_indices,
            'class_names': class_names,
            'num_classes': num_classes,
            'crop': 'tomato'
        }, f, indent=2)
    print(f"\n✓ Class labels saved to: {labels_path}")
    
    return train_generator, val_generator, class_names, num_classes


def visualize_augmentations(train_generator, config, num_samples=5):
    """Visualize augmented training samples."""
    
    print("\n📷 Generating augmentation examples...")
    
    # Get one batch
    images, labels = next(train_generator)
    
    # Get class names
    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(images[i])
        class_idx = np.argmax(labels[i])
        axes[i].set_title(f'{idx_to_class[class_idx][:15]}...', fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('Sample Augmented Training Images', fontsize=12)
    plt.tight_layout()
    
    aug_path = os.path.join(config.PLOT_DIR, "augmentation_samples.png")
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plt.savefig(aug_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Augmentation samples saved to: {aug_path}")


# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================

def build_efficientnet_model(input_shape, num_classes):
    """
    Build EfficientNetV2-B0 model for tomato disease classification.
    
    Architecture:
    - EfficientNetV2-B0 base (ImageNet pretrained, frozen initially)
    - Global Average Pooling
    - Dense layers with dropout for regularization
    - Softmax classification head
    
    Args:
        input_shape: Tuple (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Keras Model, Base Model (for fine-tuning)
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_layer')
    
    # Base model: EfficientNetV2-B0 with ImageNet weights
    # Note: EfficientNetV2B0 expects inputs in [0, 255] range by default,
    # but we're using rescale=1./255, so we need include_preprocessing=False
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None,
        include_preprocessing=False  # We handle preprocessing ourselves
    )
    
    # Freeze base model layers initially (for transfer learning)
    base_model.trainable = False
    
    # Get base model output
    x = base_model.output
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    # Dense layer with batch normalization
    x = layers.Dense(512, name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Second dense layer
    x = layers.Dense(256, name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(0.2, name='dropout_3')(x)
    
    # Output layer (use float32 for numerical stability with mixed precision)
    outputs = layers.Dense(
        num_classes, 
        activation='softmax', 
        dtype='float32',
        name='output'
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='TomatoEfficientNetV2B0')
    
    # Print model info
    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n{'='*60}")
    print("Model Architecture Summary:")
    print(f"{'='*60}")
    print(f"  - Base model: EfficientNetV2-B0")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Output classes: {num_classes}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Non-trainable parameters: {non_trainable_params:,}")
    
    return model, base_model


def unfreeze_model(model, base_model, num_layers_to_unfreeze=30):
    """
    Unfreeze the last N layers of the base model for fine-tuning.
    
    For EfficientNetV2-B0, unfreezing the last 30-50 layers works well.
    
    Args:
        model: Full model
        base_model: Base EfficientNet model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    """
    
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze all layers except the last N
    total_layers = len(base_model.layers)
    freeze_until = total_layers - num_layers_to_unfreeze
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
            # Keep batch normalization layers frozen for stability
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    
    # Count trainable layers
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    
    print(f"\n{'='*60}")
    print("Model Unfrozen for Fine-tuning:")
    print(f"{'='*60}")
    print(f"  - Total base model layers: {total_layers}")
    print(f"  - Layers frozen: {freeze_until}")
    print(f"  - Layers unfrozen: {num_layers_to_unfreeze}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model


# =============================================================================
# 5. TRAINING FUNCTIONS
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
        # Early stopping - monitor validation loss
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Model checkpoint - save best model based on val_accuracy
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


def compile_model(model, learning_rate, config):
    """Compile model with optimizer, loss, and metrics."""
    
    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=config.WEIGHT_DECAY
        ),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    
    print(f"\n✓ Model compiled with:")
    print(f"  - Optimizer: AdamW (lr={learning_rate}, wd={config.WEIGHT_DECAY})")
    print(f"  - Loss: CategoricalCrossentropy (label_smoothing=0.1)")
    print(f"  - Metrics: accuracy, precision, recall")
    
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
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting {phase.upper()} training for {epochs} epochs...")
    print(f"{'='*70}")
    
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
# 6. EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, test_gen, class_names, config):
    """
    Evaluate model on test/validation set and generate reports.
    
    Args:
        model: Trained model
        test_gen: Test/validation data generator
        class_names: List of class names
        config: Configuration object
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("\n" + "="*70)
    print("📊 EVALUATING MODEL")
    print("="*70)
    
    # Get predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Get class labels in correct order
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Print overall metrics
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    report = classification_report(
        y_true, y_pred, 
        target_names=class_labels,
        digits=4
    )
    print(report)
    
    # Save classification report as JSON
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        output_dict=True
    )
    
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n✓ Classification report saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix - Tomato Disease Detection\nEfficientNetV2-B0', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Calculate summary metrics
    accuracy = np.mean(y_pred == y_true)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'classification_report': report_dict
    }
    
    print(f"\n{'='*60}")
    print("📈 SUMMARY METRICS:")
    print(f"{'='*60}")
    print(f"  ✓ Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✓ F1-Score (Macro):   {f1_macro:.4f}")
    print(f"  ✓ F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"{'='*60}")
    
    return metrics


def plot_training_history(history, config, phase=''):
    """Plot and save training history curves."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#2196F3')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#FF5722')
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, color='#2196F3')
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#FF5722')
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2, color='#4CAF50')
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2, color='#9C27B0')
        axes[1, 0].set_title('Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2, color='#4CAF50')
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2, color='#9C27B0')
        axes[1, 1].set_title('Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle(f'Training History - Tomato EfficientNetV2-B0 ({phase})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(config.PLOT_DIR, f"training_history_{phase}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history plot saved to: {plot_path}")


# =============================================================================
# 7. MODEL EXPORT & TFLITE CONVERSION
# =============================================================================

def save_model(model, config):
    """Save model in multiple formats."""
    
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    
    # Save Keras model (.keras format - recommended for TF 2.x)
    keras_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final.keras")
    model.save(keras_path)
    print(f"✓ Keras model saved: {keras_path}")
    
    # Save H5 format (legacy, but widely compatible)
    h5_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final.h5")
    model.save(h5_path, save_format='h5')
    print(f"✓ H5 model saved: {h5_path}")
    
    # Save SavedModel format (TensorFlow Serving compatible)
    savedmodel_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_savedmodel")
    model.export(savedmodel_path)
    print(f"✓ SavedModel saved: {savedmodel_path}")
    
    # Get model size
    keras_size = os.path.getsize(keras_path) / (1024 * 1024)
    h5_size = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"\n  Model sizes:")
    print(f"    - Keras: {keras_size:.2f} MB")
    print(f"    - H5: {h5_size:.2f} MB")
    
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
    print("📱 CONVERTING TO TFLITE")
    print("="*60)
    
    # Convert to TFLite (float32 - highest accuracy)
    print("\n1. Converting to TFLite (float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"✓ TFLite (float32) saved: {tflite_path}")
    print(f"  Size: {tflite_size:.2f} MB")
    
    # Convert with float16 quantization (good balance)
    print("\n2. Converting to TFLite (float16)...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    
    tflite_fp16_model = converter_fp16.convert()
    
    tflite_fp16_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_fp16.tflite")
    with open(tflite_fp16_path, 'wb') as f:
        f.write(tflite_fp16_model)
    
    tflite_fp16_size = os.path.getsize(tflite_fp16_path) / (1024 * 1024)
    print(f"✓ TFLite (float16) saved: {tflite_fp16_path}")
    print(f"  Size: {tflite_fp16_size:.2f} MB")
    
    # Convert with INT8 quantization (smallest, for Raspberry Pi)
    if quantize:
        print("\n3. Converting to TFLite (INT8 dynamic range)...")
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_int8_model = converter_int8.convert()
        
        tflite_int8_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_int8.tflite")
        with open(tflite_int8_path, 'wb') as f:
            f.write(tflite_int8_model)
        
        tflite_int8_size = os.path.getsize(tflite_int8_path) / (1024 * 1024)
        print(f"✓ TFLite (INT8) saved: {tflite_int8_path}")
        print(f"  Size: {tflite_int8_size:.2f} MB")
        
        print(f"\n📊 Size comparison:")
        print(f"  - Original (float32): {tflite_size:.2f} MB")
        print(f"  - Float16: {tflite_fp16_size:.2f} MB ({(1-tflite_fp16_size/tflite_size)*100:.1f}% smaller)")
        print(f"  - INT8: {tflite_int8_size:.2f} MB ({(1-tflite_int8_size/tflite_size)*100:.1f}% smaller)")
        
        return tflite_path, tflite_fp16_path, tflite_int8_path
    
    return tflite_path, tflite_fp16_path, None


def create_representative_dataset(data_gen, num_samples=200):
    """
    Create representative dataset generator for full INT8 quantization.
    """
    def representative_data_gen():
        data_gen.reset()
        count = 0
        for images, _ in data_gen:
            for image in images:
                if count >= num_samples:
                    return
                yield [np.expand_dims(image.astype(np.float32), axis=0)]
                count += 1
    
    return representative_data_gen


def convert_to_tflite_full_int8(model, data_gen, config):
    """
    Convert to fully quantized INT8 TFLite with representative dataset.
    This provides the best performance on Raspberry Pi and other edge devices.
    """
    
    print("\n4. Converting to TFLite (Full INT8 with representative dataset)...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = create_representative_dataset(data_gen, num_samples=300)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    try:
        tflite_full_int8_model = converter.convert()
        
        tflite_path = os.path.join(
            config.MODEL_DIR,
            f"{config.MODEL_NAME}_full_int8.tflite"
        )
        with open(tflite_path, 'wb') as f:
            f.write(tflite_full_int8_model)
        
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"✓ TFLite (Full INT8) saved: {tflite_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return tflite_path
    except Exception as e:
        print(f"⚠ Full INT8 conversion failed: {e}")
        print("  This can happen with some model architectures.")
        print("  The dynamic range INT8 model can still be used for deployment.")
        return None


# =============================================================================
# 8. GRAD-CAM VISUALIZATION
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Generates visual explanations for CNN predictions by highlighting
    regions of the input image that are most important for the prediction.
    
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
        
        # Find appropriate layer for Grad-CAM
        if layer_name is None:
            # For EfficientNetV2, find the last conv layer before GAP
            for layer in reversed(model.layers):
                if 'top_conv' in layer.name.lower():
                    layer_name = layer.name
                    break
                elif isinstance(layer, layers.Conv2D):
                    layer_name = layer.name
                    break
                elif 'block' in layer.name.lower() and hasattr(layer, 'output'):
                    layer_name = layer.name
        
        if layer_name is None:
            # Fallback: use the layer before global average pooling
            for i, layer in enumerate(model.layers):
                if 'global_avg_pool' in layer.name:
                    layer_name = model.layers[i-1].name
                    break
        
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
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        superimposed = heatmap_colored * alpha + image * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed, heatmap


def generate_gradcam_examples(model, data_gen, class_names, config, num_examples=1):
    """
    Generate Grad-CAM visualizations for each class.
    
    Shows which regions of the leaf image the model focuses on
    when making disease predictions.
    """
    
    print("\n" + "="*60)
    print("🔍 GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*60)
    
    # Initialize Grad-CAM
    try:
        gradcam = GradCAM(model)
    except Exception as e:
        print(f"⚠ Grad-CAM initialization failed: {e}")
        return
    
    # Get class indices
    idx_to_class = {v: k for k, v in data_gen.class_indices.items()}
    num_classes = len(idx_to_class)
    
    # Collect one example per class
    class_examples = {i: [] for i in range(num_classes)}
    
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
    num_rows = (num_classes + 2) // 3  # 3 classes per row
    fig, axes = plt.subplots(num_classes, 3, figsize=(12, 4 * num_classes))
    
    for class_idx in range(num_classes):
        images = class_examples.get(class_idx, [])
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
        try:
            heatmap = gradcam.compute_heatmap(img_array, class_idx)
            
            # Convert image to display format (0-255)
            img_display = (img * 255).astype(np.uint8)
            
            # Create overlay
            overlay, heatmap_resized = gradcam.overlay_heatmap(heatmap, img_display)
            
            # Plot
            row = class_idx
            
            # Original image
            axes[row, 0].imshow(img_display)
            axes[row, 0].set_title(f'True: {class_name[:20]}', fontsize=9)
            axes[row, 0].axis('off')
            
            # Heatmap
            axes[row, 1].imshow(heatmap_resized, cmap='jet')
            axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=9)
            axes[row, 1].axis('off')
            
            # Overlay
            axes[row, 2].imshow(overlay)
            pred_status = "✓" if pred_class == class_idx else "✗"
            axes[row, 2].set_title(f'{pred_status} Pred: {pred_name[:15]} ({pred_conf:.1%})', fontsize=9)
            axes[row, 2].axis('off')
            
        except Exception as e:
            print(f"  ⚠ Failed for class {class_name}: {e}")
            continue
    
    plt.suptitle('Grad-CAM Visualizations - Tomato Disease Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    gradcam_path = os.path.join(config.PLOT_DIR, "gradcam_examples.png")
    plt.savefig(gradcam_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Grad-CAM visualizations saved: {gradcam_path}")


def show_sample_predictions(model, data_gen, config, num_samples=16):
    """
    Show sample predictions with confidence scores.
    """
    
    print("\n📷 Generating sample predictions...")
    
    # Get samples
    data_gen.reset()
    images, labels = next(data_gen)
    
    # Get class names
    idx_to_class = {v: k for k, v in data_gen.class_indices.items()}
    
    # Get predictions
    predictions = model.predict(images[:num_samples], verbose=0)
    
    # Plot
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    
    for i in range(min(num_samples, rows * cols)):
        row = i // cols
        col = i % cols
        
        img = images[i]
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(predictions[i])
        confidence = predictions[i][pred_idx]
        
        true_name = idx_to_class[true_idx]
        pred_name = idx_to_class[pred_idx]
        
        # Display image
        axes[row, col].imshow(img)
        
        # Color title based on correctness
        color = 'green' if true_idx == pred_idx else 'red'
        axes[row, col].set_title(
            f'True: {true_name[:15]}\nPred: {pred_name[:15]} ({confidence:.1%})',
            fontsize=8,
            color=color
        )
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Predictions - Tomato Disease Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    pred_path = os.path.join(config.PLOT_DIR, "sample_predictions.png")
    plt.savefig(pred_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample predictions saved: {pred_path}")


# =============================================================================
# 9. MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline for Tomato disease detection model."""
    
    print("\n" + "="*70)
    print("🍅 TOMATO DISEASE DETECTION MODEL TRAINING")
    print("   Architecture: EfficientNetV2-B0")
    print("   Dataset: ~25k images, 11 classes")
    print("="*70)
    
    # Initialize configuration
    config = Config()
    config.create_dirs()
    
    # =========================================================================
    # PHASE 1: Data Loading
    # =========================================================================
    print("\n" + "="*70)
    print("📁 PHASE 1: Loading and Preparing Data")
    print("="*70)
    
    train_gen, val_gen, class_names, num_classes = create_data_generators(config)
    
    # Update config with detected classes
    config.NUM_CLASSES = num_classes
    config.CLASS_NAMES = class_names
    
    # Visualize augmentations
    visualize_augmentations(train_gen, config)
    
    # =========================================================================
    # PHASE 2: Build Model
    # =========================================================================
    print("\n" + "="*70)
    print("🏗️ PHASE 2: Building Model")
    print("="*70)
    
    model, base_model = build_efficientnet_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=num_classes
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary(show_trainable=True, expand_nested=False)
    
    # =========================================================================
    # PHASE 3: Initial Training (Frozen Base)
    # =========================================================================
    print("\n" + "="*70)
    print("🎯 PHASE 3: Initial Training (Base Model Frozen)")
    print("="*70)
    
    model = compile_model(model, config.INITIAL_LR, config)
    
    history_initial = train_model(
        model, train_gen, val_gen, config,
        phase='initial',
        epochs=config.INITIAL_EPOCHS
    )
    
    plot_training_history(history_initial, config, phase='initial')
    
    # =========================================================================
    # PHASE 4: Fine-tuning (Unfrozen Base)
    # =========================================================================
    print("\n" + "="*70)
    print("🔧 PHASE 4: Fine-tuning (Base Model Partially Unfrozen)")
    print("="*70)
    
    # Unfreeze last 40 layers of base model
    model = unfreeze_model(model, base_model, num_layers_to_unfreeze=40)
    
    # Recompile with lower learning rate
    model = compile_model(model, config.FINE_TUNE_LR, config)
    
    history_finetune = train_model(
        model, train_gen, val_gen, config,
        phase='finetune',
        epochs=config.FINE_TUNE_EPOCHS
    )
    
    plot_training_history(history_finetune, config, phase='finetune')
    
    # =========================================================================
    # PHASE 5: Evaluation
    # =========================================================================
    print("\n" + "="*70)
    print("📊 PHASE 5: Model Evaluation")
    print("="*70)
    
    metrics = evaluate_model(model, val_gen, class_names, config)
    
    # Save metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Show sample predictions
    show_sample_predictions(model, val_gen, config)
    
    # =========================================================================
    # PHASE 6: Save Models
    # =========================================================================
    print("\n" + "="*70)
    print("💾 PHASE 6: Saving Models")
    print("="*70)
    
    keras_path, h5_path, savedmodel_path = save_model(model, config)
    
    # =========================================================================
    # PHASE 7: TFLite Conversion
    # =========================================================================
    print("\n" + "="*70)
    print("📱 PHASE 7: TFLite Conversion for Edge Deployment")
    print("="*70)
    
    tflite_paths = convert_to_tflite(model, config, quantize=True)
    
    # Full INT8 quantization with representative dataset
    tflite_full_int8_path = convert_to_tflite_full_int8(model, val_gen, config)
    
    # =========================================================================
    # PHASE 8: Grad-CAM Visualization
    # =========================================================================
    print("\n" + "="*70)
    print("🔍 PHASE 8: Grad-CAM Visualization")
    print("="*70)
    
    generate_gradcam_examples(model, val_gen, class_names, config)
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Directory: {config.OUTPUT_DIR}")
    
    print(f"\n📊 Final Results:")
    print(f"   ✓ Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   ✓ F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"   ✓ F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    print(f"\n📦 Saved Models:")
    print(f"   ✓ Keras: {keras_path}")
    print(f"   ✓ H5: {h5_path}")
    for path in tflite_paths:
        if path:
            print(f"   ✓ TFLite: {path}")
    if tflite_full_int8_path:
        print(f"   ✓ TFLite (Full INT8): {tflite_full_int8_path}")
    
    print(f"\n📈 Plots saved in: {config.PLOT_DIR}")
    print(f"📝 Logs saved in: {config.LOG_DIR}")
    
    print("\n" + "="*70)
    print("🍅 Tomato model ready for deployment!")
    print("="*70)
    
    return model, metrics


# =============================================================================
# 10. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🖥️ GPU(s) detected: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
        # Enable memory growth to prevent OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠️ No GPU detected. Training will be slower on CPU.")
    
    # Run training
    model, metrics = main()
