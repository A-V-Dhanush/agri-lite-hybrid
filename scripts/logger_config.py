"""
Logging Configuration for AgriLite-Hybrid
==========================================
Centralized logging setup for training scripts.

Features:
- Console and file logging
- Rotating log files
- Timestamped logs
- Different log levels for console vs file
- Training progress tracking

Author: AgriLite Hybrid Project
Date: February 2026
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name='agrilite',
    log_dir='logs',
    log_file=None,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
):
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log filename (auto-generated if None)
        console_level: Logging level for console
        file_level: Logging level for file
        max_bytes: Max size per log file
        backup_count: Number of backup files to keep
    
    Returns:
        logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    log_filepath = log_path / log_file
    
    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = ColoredFormatter(
        '%(levelname)s | %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Logging initialized - Console: {logging.getLevelName(console_level)}, "
                f"File: {logging.getLevelName(file_level)}")
    logger.info(f"Log file: {log_filepath}")
    
    return logger


class TrainingLogger:
    """
    Custom logger for tracking training progress.
    
    Provides methods for logging epochs, metrics, and model information.
    """
    
    def __init__(self, logger, log_dir='logs'):
        self.logger = logger
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_file = self.log_dir / f'training_metrics_{timestamp}.csv'
        
        # Initialize metrics CSV
        with open(self.metrics_file, 'w') as f:
            f.write('epoch,phase,loss,accuracy,val_loss,val_accuracy,lr,duration_sec\n')
    
    def log_section(self, title, char='=', width=70):
        """Log a section divider."""
        self.logger.info('')
        self.logger.info(char * width)
        self.logger.info(title.center(width))
        self.logger.info(char * width)
    
    def log_subsection(self, title, char='-', width=60):
        """Log a subsection divider."""
        self.logger.info('')
        self.logger.info(char * width)
        self.logger.info(title)
        self.logger.info(char * width)
    
    def log_config(self, config):
        """Log configuration parameters."""
        self.log_section('CONFIGURATION')
        
        for attr in dir(config):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(config, attr)
                if not callable(value):
                    self.logger.info(f"  {attr}: {value}")
    
    def log_dataset_info(self, train_samples, val_samples, test_samples=None, 
                         num_classes=None, class_names=None):
        """Log dataset information."""
        self.log_subsection('Dataset Information')
        self.logger.info(f"  Training samples: {train_samples:,}")
        self.logger.info(f"  Validation samples: {val_samples:,}")
        if test_samples:
            self.logger.info(f"  Test samples: {test_samples:,}")
        if num_classes:
            self.logger.info(f"  Number of classes: {num_classes}")
        
        if class_names:
            self.logger.info(f"\n  Classes:")
            for i, name in enumerate(class_names[:10]):  # Show first 10
                self.logger.info(f"    {i}: {name}")
            if len(class_names) > 10:
                self.logger.info(f"    ... and {len(class_names) - 10} more")
    
    def log_model_info(self, model):
        """Log model architecture information."""
        self.log_subsection('Model Architecture')
        
        total_params = model.count_params()
        trainable_params = sum([tf.reduce_prod(v.shape).numpy() 
                               for v in model.trainable_variables])
        
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        self.logger.info(f"  Model size (float32): ~{total_params * 4 / (1024**2):.1f} MB")
        self.logger.info(f"  Model size (INT8): ~{total_params / (1024**2):.1f} MB")
    
    def log_epoch_start(self, epoch, total_epochs, phase='training'):
        """Log epoch start."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Epoch {epoch}/{total_epochs} - {phase.upper()}")
        self.logger.info(f"{'='*70}")
    
    def log_epoch_end(self, epoch, history, phase='initial', duration=None):
        """Log epoch end with metrics."""
        metrics = {
            'loss': history.get('loss', [0])[-1],
            'accuracy': history.get('accuracy', [0])[-1],
            'val_loss': history.get('val_loss', [0])[-1],
            'val_accuracy': history.get('val_accuracy', [0])[-1],
        }
        
        lr = history.get('lr', [0])[-1] if 'lr' in history else 0
        
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Loss: {metrics['loss']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Val Loss: {metrics['val_loss']:.4f} | Val Accuracy: {metrics['val_accuracy']:.4f}")
        if lr:
            self.logger.info(f"  Learning Rate: {lr:.2e}")
        if duration:
            self.logger.info(f"  Duration: {duration:.1f}s ({duration/60:.1f}min)")
        
        # Save to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{phase},{metrics['loss']:.6f},{metrics['accuracy']:.6f},"
                   f"{metrics['val_loss']:.6f},{metrics['val_accuracy']:.6f},"
                   f"{lr:.8f},{duration if duration else 0:.2f}\n")
    
    def log_training_complete(self, total_time, best_accuracy):
        """Log training completion."""
        self.log_section('TRAINING COMPLETE')
        self.logger.info(f"  Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"  Best validation accuracy: {best_accuracy:.4f}")
        self.logger.info(f"  Metrics saved to: {self.metrics_file}")
    
    def log_evaluation(self, metrics):
        """Log evaluation metrics."""
        self.log_section('EVALUATION RESULTS')
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                self.logger.info(f"\n  {key}:")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        self.logger.info(f"    {k}: {v:.4f}")
                    else:
                        self.logger.info(f"    {k}: {v}")
            elif isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_file_saved(self, filepath, description="File"):
        """Log file save."""
        self.logger.info(f"  ✓ {description} saved: {filepath}")
    
    def log_error(self, error, context=""):
        """Log error with context."""
        self.logger.error(f"ERROR in {context}: {str(error)}")
        import traceback
        self.logger.debug(traceback.format_exc())


# Import tensorflow here to avoid circular imports
try:
    import tensorflow as tf
except ImportError:
    tf = None


# Create default logger for module-level use
def get_logger(name='agrilite'):
    """Get or create logger."""
    return logging.getLogger(name)


if __name__ == "__main__":
    # Test logging
    logger = setup_logger('test', log_dir='logs/test')
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test training logger
    train_logger = TrainingLogger(logger)
    train_logger.log_section("TEST SECTION")
    train_logger.log_subsection("Test Subsection")
    
    print("\nLog files created in logs/test/")
