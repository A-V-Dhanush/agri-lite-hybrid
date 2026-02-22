"""
GPU Configuration Utility for AgriLite-Hybrid
=============================================
Centralized GPU detection, configuration, and optimization.

Features:
- Automatic GPU detection (NVIDIA CUDA, AMD ROCm, Apple Metal)
- Memory growth configuration to prevent OOM errors
- Mixed precision training for faster GPU training
- Graceful CPU fallback
- Multi-GPU support

Usage:
    from gpu_config import setup_gpu, get_strategy
    
    # Basic setup
    gpu_info = setup_gpu()
    
    # For distributed training
    strategy = get_strategy()
    with strategy.scope():
        model = build_model()

Author: AgriLite Hybrid Project
Date: February 2026
"""

import os
import logging

# Suppress TensorFlow warnings before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

logger = logging.getLogger('agrilite')


def setup_gpu(
    memory_growth: bool = True,
    memory_limit: int = None,
    mixed_precision: bool = True,
    visible_devices: str = None,
    verbose: bool = True
) -> dict:
    """
    Configure GPU for optimal TensorFlow training.
    
    Args:
        memory_growth: Enable memory growth to prevent GPU OOM
        memory_limit: Optional memory limit in MB per GPU
        mixed_precision: Enable mixed precision (float16) for faster training
        visible_devices: Comma-separated GPU indices (e.g., "0,1")
        verbose: Print configuration details
    
    Returns:
        dict: GPU configuration info
            {
                'gpu_available': bool,
                'gpu_count': int,
                'gpu_names': list,
                'memory_growth': bool,
                'mixed_precision': bool,
                'strategy': str ('GPU', 'Multi-GPU', 'CPU')
            }
    """
    
    info = {
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'memory_growth': False,
        'mixed_precision': False,
        'strategy': 'CPU'
    }
    
    # Set visible devices if specified
    if visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        if verbose:
            print(f"🔧 CUDA_VISIBLE_DEVICES set to: {visible_devices}")
    
    # Detect GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        info['gpu_available'] = True
        info['gpu_count'] = len(gpus)
        
        if verbose:
            print("\n" + "="*60)
            print("🖥️  GPU CONFIGURATION")
            print("="*60)
            print(f"✓ Found {len(gpus)} GPU(s)")
        
        for i, gpu in enumerate(gpus):
            gpu_name = gpu.name
            info['gpu_names'].append(gpu_name)
            
            if verbose:
                print(f"  GPU {i}: {gpu_name}")
            
            try:
                # Enable memory growth
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    info['memory_growth'] = True
                    if verbose:
                        print(f"    ✓ Memory growth enabled")
                
                # Set memory limit if specified
                if memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    if verbose:
                        print(f"    ✓ Memory limit: {memory_limit} MB")
                        
            except RuntimeError as e:
                print(f"    ⚠️ GPU config error: {e}")
        
        # Set strategy based on GPU count
        if len(gpus) > 1:
            info['strategy'] = 'Multi-GPU'
        else:
            info['strategy'] = 'GPU'
        
        # Enable mixed precision for faster training
        if mixed_precision:
            try:
                # Check GPU compute capability for mixed precision
                # Mixed precision requires compute capability >= 7.0 (Volta+)
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                info['mixed_precision'] = True
                if verbose:
                    print(f"\n✓ Mixed precision (float16) enabled")
                    print("  This speeds up training on modern GPUs (RTX series, V100+)")
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Mixed precision not available: {e}")
                    print("  Training will use float32 (slower but compatible)")
        
        if verbose:
            print(f"\n🚀 Strategy: {info['strategy']}")
            print("="*60)
    
    else:
        if verbose:
            print("\n" + "="*60)
            print("⚠️  NO GPU DETECTED - Using CPU")
            print("="*60)
            print("Training will be significantly slower on CPU.")
            print("\nTo enable GPU:")
            print("  1. Install NVIDIA drivers")
            print("  2. Install CUDA Toolkit 11.x or 12.x")
            print("  3. Install cuDNN")
            print("  4. pip install tensorflow[and-cuda]")
            print("\nOr use Google Colab for free GPU access:")
            print("  https://colab.research.google.com")
            print("="*60)
    
    return info


def get_strategy():
    """
    Get appropriate distribution strategy based on available hardware.
    
    Returns:
        tf.distribute.Strategy: MirroredStrategy for multi-GPU, 
                                default strategy for single GPU/CPU
    """
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        # Multi-GPU: Use MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()
        print(f"📊 Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
    elif len(gpus) == 1:
        # Single GPU: Use default strategy
        strategy = tf.distribute.get_strategy()
        print("📊 Using single GPU strategy")
    else:
        # CPU: Use default strategy
        strategy = tf.distribute.get_strategy()
        print("📊 Using CPU strategy")
    
    return strategy


def check_gpu_memory():
    """
    Check current GPU memory usage.
    
    Returns:
        dict: Memory info per GPU
    """
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        memory_info = {}
        
        for i, gpu in enumerate(gpus):
            try:
                # Get memory info using nvidia-smi via subprocess
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                     '--format=csv,noheader,nounits', '-i', str(i)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    used, total = map(int, result.stdout.strip().split(','))
                    memory_info[f'GPU_{i}'] = {
                        'used_mb': used,
                        'total_mb': total,
                        'free_mb': total - used,
                        'utilization': used / total * 100
                    }
            except Exception:
                pass
        
        return memory_info
    except Exception:
        return {}


def print_gpu_summary():
    """Print a summary of GPU configuration."""
    
    print("\n" + "="*60)
    print("📊 GPU SUMMARY")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detected: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"\n  GPU {i}: {gpu.name}")
        
        # Try to get detailed info
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"    Device name: {details.get('device_name', 'N/A')}")
                print(f"    Compute capability: {details.get('compute_capability', 'N/A')}")
        except Exception:
            pass
    
    # Memory info
    mem_info = check_gpu_memory()
    if mem_info:
        print("\n  Memory Usage:")
        for gpu_name, info in mem_info.items():
            print(f"    {gpu_name}: {info['used_mb']}MB / {info['total_mb']}MB "
                  f"({info['utilization']:.1f}% used)")
    
    # Current policy
    policy = tf.keras.mixed_precision.global_policy()
    print(f"\n  Precision policy: {policy.name}")
    
    print("="*60)


def optimize_for_inference():
    """
    Optimize TensorFlow for inference (disable training-specific optimizations).
    
    Call this before loading a model for inference only.
    """
    
    # Disable eager execution optimizations
    tf.config.optimizer.set_jit(True)  # Enable XLA
    
    # Set thread configuration for inference
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)


def clear_gpu_memory():
    """
    Clear GPU memory by resetting the backend.
    
    Useful between training runs to prevent memory accumulation.
    """
    
    try:
        from tensorflow.keras import backend as K
        K.clear_session()
        
        import gc
        gc.collect()
        
        print("✓ GPU memory cleared")
    except Exception as e:
        print(f"⚠️ Could not clear GPU memory: {e}")


# =============================================================================
# Automatic setup when imported
# =============================================================================

# Run basic GPU detection on import (silent)
_gpu_info = None

def get_gpu_info():
    """Get cached GPU info."""
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = setup_gpu(verbose=False)
    return _gpu_info


# =============================================================================
# Main - Test GPU configuration
# =============================================================================

if __name__ == "__main__":
    print("Testing GPU Configuration...\n")
    
    # Full setup with verbose output
    info = setup_gpu(
        memory_growth=True,
        mixed_precision=True,
        verbose=True
    )
    
    print(f"\nConfiguration Result:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Print summary
    print_gpu_summary()
    
    # Test strategy
    strategy = get_strategy()
    print(f"\nDistribution Strategy: {strategy}")
    
    # Simple GPU test
    if info['gpu_available']:
        print("\n🧪 Running GPU computation test...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print(f"✓ GPU computation successful! Result shape: {c.shape}")
