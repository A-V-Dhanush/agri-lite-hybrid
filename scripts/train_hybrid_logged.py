"""
AgriLite-Hybrid Training Wrapper with Comprehensive Logging
============================================================
This wrapper runs train_hybrid.py with full logging support.

Features:
- Redirects all output to log files
- Timestamps all operations
- Captures errors and stack traces
- Creates detailed training logs for debugging

Usage:
    python train_hybrid_logged.py

Logs will be saved to: logs/hybrid/
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import subprocess
import logging
from logger_config import setup_logger, TrainingLogger

# Initialize logger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path('logs/hybrid')
log_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(
    name='hybrid_training',
    log_dir=str(log_dir),
    log_file=f'training_{timestamp}.log',
    console_level=logging.INFO,
    file_level=logging.DEBUG
)

train_logger = TrainingLogger(logger, log_dir=str(log_dir))


class TeeOutput:
    """Redirect stdout/stderr to both console and logger."""
    
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = []
        
    def write(self, message):
        if message and message.strip():
            # Log the message
            self.logger.log(self.level, message.rstrip())
            # Also write to original stdout
            sys.__stdout__.write(message)
            
    def flush(self):
        sys.__stdout__.flush()


def main():
    """Run training with full logging."""
    
    train_logger.log_section("AGRILITE-HYBRID TRAINING SESSION", char='=', width=80)
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log directory: {log_dir.absolute()}")
    
    # Check if train_hybrid.py exists
    script_path = Path('scripts/train_hybrid.py')
    if not script_path.exists():
        logger.error(f"Training script not found: {script_path}")
        return 1
    
    logger.info(f"Training script: {script_path.absolute()}")
    
    # Redirect output
    train_logger.log_section("STARTING TRAINING", char='-', width=80)
    
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Create tee outputs
        stdout_tee = TeeOutput(logger, level=logging.INFO)
        stderr_tee = TeeOutput(logger, level=logging.ERROR)
        
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        
        # Import and run training
        logger.info("Importing training module...")
        
        # Add scripts to path
        sys.path.insert(0, 'scripts')
        
        # Import training script
        import train_hybrid
        
        # Run main function
        logger.info("Starting training pipeline...")
        start_time = datetime.now()
        
        model, metrics = train_hybrid.main()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Restore outputs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Log completion
        train_logger.log_training_complete(
            total_time=duration,
            best_accuracy=metrics.get('accuracy', 0.0)
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {duration/3600:.2f} hours")
        logger.info(f"Final accuracy: {metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"Logs saved to: {log_dir.absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.warning("\n" + "="*80)
        logger.warning("TRAINING INTERRUPTED BY USER")
        logger.warning("="*80)
        return 130
        
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error("\n" + "="*80)
        logger.error("TRAINING FAILED WITH ERROR")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
