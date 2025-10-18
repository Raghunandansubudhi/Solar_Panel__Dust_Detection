#!/usr/bin/env python3
"""
GPU Verification Script for Solar Panel Dust Detection
Verifies CUDA installation and GPU availability for TensorFlow
"""

import tensorflow as tf
import sys

def verify_gpu_setup():
    """Verify GPU setup and CUDA installation"""
    
    print("=" * 60)
    print("🔍 GPU VERIFICATION FOR SOLAR PANEL DUST DETECTION")
    print("=" * 60)
    
    # Check TensorFlow version
    print(f"📦 TensorFlow version: {tf.__version__}")
    
    # Check CUDA availability
    print(f"🔧 CUDA available: {tf.test.is_built_with_cuda()}")
    
    # List physical devices
    print(f"🖥️  Physical devices: {tf.config.list_physical_devices()}")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"🎮 GPUs detected: {gpus}")
    
    if len(gpus) == 0:
        print("\n❌ NO GPU DETECTED!")
        print("📋 To enable GPU acceleration, install:")
        print("   1. CUDA 12.x (https://developer.nvidia.com/cuda-downloads)")
        print("   2. cuDNN 8.9 (https://developer.nvidia.com/cudnn)")
        print("   3. Restart your computer after installation")
        print("   4. Verify with: nvidia-smi")
        return False
    
    # Test GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✅ GPU computation test passed: {c.numpy()}")
        
        print("\n🎉 GPU SETUP VERIFIED SUCCESSFULLY!")
        print("🚀 Ready for GPU-accelerated training!")
        return True
        
    except Exception as e:
        print(f"\n❌ GPU setup error: {e}")
        print("📋 Troubleshooting:")
        print("   1. Check CUDA installation: nvidia-smi")
        print("   2. Verify TensorFlow GPU support")
        print("   3. Restart Python environment")
        return False

def get_gpu_info():
    """Get detailed GPU information"""
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n📊 NVIDIA GPU Information:")
            print("-" * 40)
            print(result.stdout)
        else:
            print("⚠️  nvidia-smi not available")
    except FileNotFoundError:
        print("⚠️  NVIDIA drivers not installed or nvidia-smi not found")

if __name__ == "__main__":
    success = verify_gpu_setup()
    get_gpu_info()
    
    if not success:
        print("\n⚠️  Proceeding with CPU-only mode...")
        print("   Training will be slower but will still work.")
    
    sys.exit(0 if success else 1)
