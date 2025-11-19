# Module 1.2: CUDA Setup and Verification

## Learning Objectives

By the end of this lesson, you will be able to:

1. Verify CUDA installation and driver compatibility
2. Understand GPU compute capability and its implications
3. Use nvidia-smi and torch.cuda for GPU monitoring
4. Diagnose common GPU detection and configuration issues
5. Configure multi-GPU systems for vLLM
6. Test GPU performance and memory capabilities

**Estimated Time:** 1.5-2 hours

---

## Table of Contents

1. [Understanding CUDA Architecture](#understanding-cuda-architecture)
2. [CUDA Installation Verification](#cuda-installation-verification)
3. [GPU Detection and Properties](#gpu-detection-and-properties)
4. [Compute Capability Deep Dive](#compute-capability-deep-dive)
5. [CUDA Toolkit vs Driver](#cuda-toolkit-vs-driver)
6. [PyTorch CUDA Integration](#pytorch-cuda-integration)
7. [Multi-GPU Configuration](#multi-gpu-configuration)
8. [Performance Testing](#performance-testing)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Practical Exercises](#practical-exercises)
11. [Key Takeaways](#key-takeaways)

---

## Understanding CUDA Architecture

### The CUDA Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (vLLM, PyTorch, TensorFlow, etc.)              │
├─────────────────────────────────────────────────────────────┤
│                 CUDA Toolkit (nvcc, libraries)              │
│              - cuBLAS, cuDNN, NCCL, etc.                    │
├─────────────────────────────────────────────────────────────┤
│                    CUDA Driver API                           │
│              (libcuda.so - version 525.x, etc.)             │
├─────────────────────────────────────────────────────────────┤
│                    GPU Hardware                              │
│         (A100, V100, RTX series with CUDA cores)            │
└─────────────────────────────────────────────────────────────┘
```

**Key Concepts:**

1. **CUDA Driver**: Low-level interface to GPU hardware (bundled with NVIDIA driver)
2. **CUDA Toolkit**: Development tools and libraries (nvcc compiler, cuBLAS, etc.)
3. **CUDA Runtime**: High-level API for CUDA programming
4. **Compute Capability**: GPU's feature set and capabilities

---

## CUDA Installation Verification

### Step 1: Check NVIDIA Driver

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Expected output:
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA A100-SXM4-40GB          Off | 00000000:00:04.0 Off |                    0 |
# | N/A   30C    P0              44W / 400W |      0MiB / 40960MiB |      0%      Default |
# |                                         |                      |             Disabled |
# +-----------------------------------------+----------------------+----------------------+
```

**Key Information from nvidia-smi:**
- **Driver Version**: NVIDIA driver version (e.g., 535.129.03)
- **CUDA Version**: Maximum CUDA version supported by this driver (e.g., 12.2)
- **GPU Name**: GPU model (e.g., A100-SXM4-40GB)
- **Memory**: Total and used GPU memory
- **Power**: Current power consumption and limit
- **Temperature**: GPU temperature in Celsius

### Step 2: Check CUDA Toolkit

```bash
# Check if CUDA toolkit is installed
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Tue_Aug_15_22:02:13_PDT_2023
# Cuda compilation tools, release 12.2, V12.2.140
# Build cuda_12.2.r12.2/compiler.33191640_0

# Alternative: Check CUDA installation directory
ls /usr/local/cuda/

# Check specific libraries
ls /usr/local/cuda/lib64/ | grep -E "cublas|cudnn|nccl"
```

### Step 3: Verify CUDA Environment Variables

```bash
# Check CUDA-related environment variables
echo $CUDA_HOME
echo $CUDA_PATH
echo $LD_LIBRARY_PATH

# Recommended settings (add to ~/.bashrc):
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Comprehensive CUDA Verification Script

```bash
#!/bin/bash
# cuda_verification.sh - Comprehensive CUDA setup check

echo "======================================================================"
echo "                    CUDA Installation Verification                    "
echo "======================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check 1: NVIDIA Driver
echo "1. NVIDIA Driver Check"
echo "-------------------------------------------------------------------"
if command_exists nvidia-smi; then
    echo -e "${GREEN}✓${NC} nvidia-smi found"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   Driver Version: $DRIVER_VERSION"
else
    echo -e "${RED}✗${NC} nvidia-smi not found. Install NVIDIA driver."
    exit 1
fi
echo ""

# Check 2: CUDA Toolkit
echo "2. CUDA Toolkit Check"
echo "-------------------------------------------------------------------"
if command_exists nvcc; then
    echo -e "${GREEN}✓${NC} nvcc (CUDA compiler) found"
    nvcc --version | grep "release"
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
    echo "   CUDA Toolkit Version: $CUDA_VERSION"
else
    echo -e "${YELLOW}⚠${NC} nvcc not found. CUDA toolkit may not be installed."
    echo "   (This is OK if using PyTorch's bundled CUDA runtime)"
fi
echo ""

# Check 3: CUDA Environment Variables
echo "3. Environment Variables Check"
echo "-------------------------------------------------------------------"
if [ -n "$CUDA_HOME" ]; then
    echo -e "${GREEN}✓${NC} CUDA_HOME is set: $CUDA_HOME"
else
    echo -e "${YELLOW}⚠${NC} CUDA_HOME not set (optional but recommended)"
fi

if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
    echo -e "${YELLOW}⚠${NC} LD_LIBRARY_PATH not set"
fi
echo ""

# Check 4: GPU Information
echo "4. GPU Information"
echo "-------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,power.limit --format=csv
echo ""

# Check 5: CUDA Libraries
echo "5. CUDA Libraries Check"
echo "-------------------------------------------------------------------"
CUDA_LIB_PATH="/usr/local/cuda/lib64"
if [ -d "$CUDA_LIB_PATH" ]; then
    echo "   Checking $CUDA_LIB_PATH"

    # Check for key libraries
    for lib in libcudart.so libcublas.so libcudnn.so libnccl.so; do
        if ls $CUDA_LIB_PATH/$lib* 1> /dev/null 2>&1; then
            echo -e "   ${GREEN}✓${NC} $lib found"
        else
            echo -e "   ${YELLOW}⚠${NC} $lib not found"
        fi
    done
else
    echo -e "   ${YELLOW}⚠${NC} $CUDA_LIB_PATH not found"
fi
echo ""

# Check 6: Compute Capability
echo "6. Compute Capability"
echo "-------------------------------------------------------------------"
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "   Compute Capability: $COMPUTE_CAP"

# Interpret compute capability
case ${COMPUTE_CAP%.*} in
    7)
        echo "   Generation: Volta/Turing (V100, T4, RTX 20xx)"
        echo "   vLLM Support: ✓ Supported"
        ;;
    8)
        echo "   Generation: Ampere/Ada (A100, RTX 30xx/40xx)"
        echo "   vLLM Support: ✓ Fully Supported (recommended)"
        ;;
    9)
        echo "   Generation: Hopper (H100)"
        echo "   vLLM Support: ✓ Fully Supported (best performance)"
        ;;
    *)
        echo -e "   ${YELLOW}⚠${NC} Compute capability < 7.0 may not be supported"
        ;;
esac
echo ""

# Check 7: GPU Memory
echo "7. GPU Memory Analysis"
echo "-------------------------------------------------------------------"
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "   Total GPU Memory: ${TOTAL_MEM} MiB"

if [ "$TOTAL_MEM" -lt 8000 ]; then
    echo -e "   ${YELLOW}⚠${NC} Less than 8GB. Limited to small models (< 7B parameters)"
elif [ "$TOTAL_MEM" -lt 16000 ]; then
    echo -e "   ${GREEN}✓${NC} 8-16GB. Can run models up to ~13B parameters"
elif [ "$TOTAL_MEM" -lt 40000 ]; then
    echo -e "   ${GREEN}✓${NC} 16-40GB. Can run models up to ~30B parameters"
else
    echo -e "   ${GREEN}✓✓${NC} 40GB+. Can run large models (70B+ with optimization)"
fi
echo ""

echo "======================================================================"
echo "                    Verification Complete                             "
echo "======================================================================"
```

Save as `cuda_verification.sh` and run:
```bash
chmod +x cuda_verification.sh
./cuda_verification.sh
```

---

## GPU Detection and Properties

### Using nvidia-smi Advanced Queries

```bash
# Query all GPU properties
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit,compute_cap --format=csv

# Monitor GPU in real-time (updates every second)
nvidia-smi dmon -s pucvmet

# Watch GPU utilization
watch -n 1 nvidia-smi

# Query specific GPU (multi-GPU systems)
nvidia-smi -i 0 --query-gpu=name,memory.used --format=csv

# List all processes using GPU
nvidia-smi pmon

# Detailed GPU information
nvidia-smi -q
```

### PyTorch GPU Detection

```python
# gpu_detection.py
"""
Comprehensive GPU detection and information using PyTorch.
Run: python gpu_detection.py
"""

import torch
import sys

def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_cuda_availability():
    """Check if CUDA is available in PyTorch"""
    print_separator("CUDA Availability")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\n❌ CUDA is not available!")
        print("Possible reasons:")
        print("  1. PyTorch was installed without CUDA support")
        print("  2. NVIDIA drivers are not installed")
        print("  3. No NVIDIA GPU present")
        return False

    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")

    return True

def list_gpus():
    """List all available GPUs"""
    print_separator("Available GPUs")

    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")

        # Memory info
        mem_info = torch.cuda.mem_get_info(i)
        free_mem_gb = mem_info[0] / (1024**3)
        total_mem_gb = mem_info[1] / (1024**3)
        used_mem_gb = total_mem_gb - free_mem_gb

        print(f"Total Memory: {total_mem_gb:.2f} GB")
        print(f"Free Memory: {free_mem_gb:.2f} GB")
        print(f"Used Memory: {used_mem_gb:.2f} GB")

        # Additional properties
        props = torch.cuda.get_device_properties(i)
        print(f"Multi-Processor Count: {props.multi_processor_count}")
        print(f"Total Memory (detailed): {props.total_memory / (1024**3):.2f} GB")
        print(f"Major.Minor: {props.major}.{props.minor}")

def test_gpu_operations():
    """Test basic GPU operations"""
    print_separator("GPU Operations Test")

    device = torch.device("cuda:0")
    print(f"Testing on device: {device}")

    try:
        # Create tensors on GPU
        print("\n1. Creating tensors on GPU...")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        print("   ✓ Tensors created successfully")

        # Perform computation
        print("2. Performing matrix multiplication...")
        c = torch.matmul(a, b)
        print("   ✓ Computation successful")

        # Synchronize
        print("3. Synchronizing CUDA...")
        torch.cuda.synchronize()
        print("   ✓ Synchronization successful")

        # Check memory usage
        print("4. Checking memory allocation...")
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"   Allocated: {allocated:.2f} MB")
        print(f"   Reserved: {reserved:.2f} MB")

        print("\n✅ All GPU operations successful!")
        return True

    except Exception as e:
        print(f"\n❌ GPU operation failed: {e}")
        return False

def check_memory_bandwidth():
    """Estimate GPU memory bandwidth"""
    print_separator("Memory Bandwidth Test")

    device = torch.device("cuda:0")

    # Size of data to transfer (100 MB)
    size = 100 * 1024 * 1024 // 4  # 100MB in float32 elements

    # Create data
    data = torch.randn(size, device='cpu', dtype=torch.float32)

    # Test host to device transfer
    print("Testing Host -> Device transfer...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    data_gpu = data.to(device)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end)
    bandwidth_gb = (size * 4 / (1024**3)) / (time_ms / 1000)

    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Bandwidth: {bandwidth_gb:.2f} GB/s")

    # Test device to host transfer
    print("\nTesting Device -> Host transfer...")
    start.record()
    data_cpu = data_gpu.to('cpu')
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end)
    bandwidth_gb = (size * 4 / (1024**3)) / (time_ms / 1000)

    print(f"  Time: {time_ms:.2f} ms")
    print(f"  Bandwidth: {bandwidth_gb:.2f} GB/s")

def main():
    print("=" * 70)
    print("          PyTorch CUDA Detection and Testing")
    print("=" * 70)

    # Check CUDA availability
    if not check_cuda_availability():
        sys.exit(1)

    # List all GPUs
    list_gpus()

    # Test GPU operations
    if not test_gpu_operations():
        sys.exit(1)

    # Test memory bandwidth
    check_memory_bandwidth()

    print("\n" + "=" * 70)
    print("  All tests completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

---

## Compute Capability Deep Dive

### What is Compute Capability?

Compute capability defines the features supported by a CUDA-capable GPU. It consists of a major and minor version number (e.g., 8.0 for A100).

### Compute Capability Versions

```
┌─────────────────────────────────────────────────────────────────────┐
│ Compute   │ Generation │ GPUs                    │ Key Features      │
│ Capability│            │                         │                   │
├───────────┼────────────┼─────────────────────────┼───────────────────┤
│ 7.0       │ Volta      │ V100, Titan V          │ Tensor Cores v1   │
│ 7.5       │ Turing     │ T4, RTX 20xx, GTX 16xx │ INT8/INT4 Tensor  │
│ 8.0       │ Ampere     │ A100, A30              │ Tensor Cores v3   │
│ 8.6       │ Ampere     │ A10, A40, RTX 30xx     │ TF32, BF16        │
│ 8.9       │ Ada        │ RTX 40xx, L4           │ FP8 Tensor Cores  │
│ 9.0       │ Hopper     │ H100, H200             │ Transformer Engine│
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Support by Compute Capability

```python
# check_compute_features.py
"""Check which features are supported by your GPU"""

import torch

def check_compute_capability_features():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Get compute capability
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"{major}.{minor}"

    print(f"Compute Capability: {cc}")
    print("\nSupported Features:")
    print("-" * 50)

    # Check Tensor Cores
    if major >= 7:
        print("✓ Tensor Cores (FP16 acceleration)")
    else:
        print("✗ No Tensor Cores")

    # Check TF32
    if (major == 8 and minor >= 0) or major > 8:
        print("✓ TF32 (19-bit precision for training)")
    else:
        print("✗ No TF32 support")

    # Check BF16
    if (major == 8 and minor >= 0) or major > 8:
        print("✓ BF16 (bfloat16 format)")
    else:
        print("✗ No BF16 support")

    # Check FP8
    if (major == 8 and minor >= 9) or major >= 9:
        print("✓ FP8 (8-bit floating point)")
    else:
        print("✗ No FP8 support")

    # Check INT8 Tensor Cores
    if major >= 7 and minor >= 5:
        print("✓ INT8 Tensor Cores (quantization)")
    else:
        print("✗ No INT8 Tensor Core support")

    # Check async copy
    if major >= 8:
        print("✓ Asynchronous copy (memory optimization)")
    else:
        print("✗ No async copy")

    # Recommendations for vLLM
    print("\nvLLM Recommendations:")
    print("-" * 50)
    if major >= 8:
        print("✓✓ Excellent - Full vLLM feature support")
        print("  - Use FP16 or BF16 for best performance")
        print("  - Enable Tensor Cores for attention")
    elif major == 7:
        print("✓ Good - vLLM supported with some limitations")
        print("  - Use FP16 for Tensor Core acceleration")
    else:
        print("⚠ Limited - vLLM may not run or be slow")

if __name__ == "__main__":
    check_compute_capability_features()
```

---

## CUDA Toolkit vs Driver

### Understanding the Difference

```
┌──────────────────────────────────────────────────────────────┐
│                      CUDA Driver                             │
│  - Installed with NVIDIA driver                              │
│  - Backward compatible (newer driver supports older CUDA)    │
│  - Version shown in nvidia-smi                               │
│  - Location: /usr/lib/x86_64-linux-gnu/libcuda.so           │
└──────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              │ Uses
                              │
┌──────────────────────────────────────────────────────────────┐
│                     CUDA Toolkit                             │
│  - Separate installation (optional for vLLM)                 │
│  - Contains nvcc compiler and libraries                      │
│  - Version checked with nvcc --version                       │
│  - Location: /usr/local/cuda/                                │
└──────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              │ Uses
                              │
┌──────────────────────────────────────────────────────────────┐
│                   PyTorch / vLLM                             │
│  - Can bundle its own CUDA runtime                           │
│  - Doesn't require CUDA toolkit for inference                │
│  - Requires toolkit only for building from source            │
└──────────────────────────────────────────────────────────────┘
```

### Version Compatibility

```bash
# Check driver's maximum supported CUDA version
nvidia-smi | grep "CUDA Version"
# Output: CUDA Version: 12.2

# This means:
# - Driver supports CUDA 12.2 and all earlier versions
# - You can use PyTorch with CUDA 11.8, 12.0, 12.1, 12.2
# - You CANNOT use PyTorch with CUDA 12.3+ (would need driver update)

# Check PyTorch's CUDA version
python -c "import torch; print(torch.version.cuda)"
# Output: 12.1

# As long as: PyTorch CUDA version <= Driver CUDA version
# Everything will work!
```

---

## PyTorch CUDA Integration

### Verifying PyTorch CUDA Build

```python
# pytorch_cuda_check.py

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

# Check if PyTorch was built with CUDA
if torch.cuda.is_available():
    print("\n✓ PyTorch is built with CUDA support")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  Current GPU: {torch.cuda.current_device()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("\n✗ PyTorch is NOT built with CUDA support")
    print("  Reinstall with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
```

### Common PyTorch CUDA Issues

**Issue 1: PyTorch not built with CUDA**

```bash
# Symptom
python -c "import torch; print(torch.cuda.is_available())"
# Output: False (but you have an NVIDIA GPU)

# Solution: Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Issue 2: CUDA version mismatch**

```bash
# Symptom
RuntimeError: CUDA error: no kernel image is available for execution on the device

# Solution: Match PyTorch CUDA version with driver
# If driver supports CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# If driver supports CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Multi-GPU Configuration

### Detecting Multiple GPUs

```python
# multi_gpu_check.py

import torch

def list_all_gpus():
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s):\n")

    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")

        # Get memory info
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)
        print(f"  Memory: {total_memory:.2f} GB")

        # Get compute capability
        major, minor = torch.cuda.get_device_capability(i)
        print(f"  Compute Capability: {major}.{minor}")

        # Check memory usage
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        print(f"  Free Memory: {mem_free / (1024**3):.2f} GB")
        print()

if __name__ == "__main__":
    list_all_gpus()
```

### Setting Visible GPUs

```bash
# Make only specific GPUs visible to application
export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0    # Use only GPU 0
export CUDA_VISIBLE_DEVICES=""   # Use no GPUs (CPU only)

# Run vLLM with specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python your_script.py
```

---

## Performance Testing

### GPU Compute Test

```python
# gpu_benchmark.py
"""Simple GPU performance benchmark"""

import torch
import time

def benchmark_matmul(size=8192, iterations=100):
    """Benchmark matrix multiplication"""
    device = torch.device("cuda:0")

    # Create random matrices
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.time()

    # Calculate TFLOPS
    total_ops = iterations * (2 * size**3)  # Multiply-add operations
    time_taken = end - start
    tflops = (total_ops / time_taken) / 1e12

    print(f"Matrix size: {size}x{size}")
    print(f"Iterations: {iterations}")
    print(f"Time: {time_taken:.2f} seconds")
    print(f"Performance: {tflops:.2f} TFLOPS")

    return tflops

if __name__ == "__main__":
    print("GPU Compute Benchmark")
    print("=" * 50)
    benchmark_matmul()
```

---

## Common Issues and Solutions

### Issue 1: "CUDA out of memory"

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Get memory summary
print(torch.cuda.memory_summary())
```

### Issue 2: GPU not detected

```bash
# Check if GPU is visible to OS
lspci | grep -i nvidia

# Reload NVIDIA driver
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm
```

---

## Practical Exercises

### Exercise 1.2.1: CUDA Verification
Run all verification scripts and document your system configuration.

### Exercise 1.2.2: Multi-GPU Setup
If you have multiple GPUs, configure them and test vLLM with tensor parallelism.

### Exercise 1.2.3: Performance Baseline
Run the GPU benchmark and record your results.

---

## Key Takeaways

1. **CUDA driver** and **CUDA toolkit** are different
2. **Compute capability** determines GPU features
3. **PyTorch** can bundle its own CUDA runtime
4. **nvidia-smi** is essential for GPU monitoring
5. Always verify **GPU detection** before running vLLM

---

## Next Steps

Next lesson: **03_environment_configuration.md** - Python environments and dependency management.

---

**Estimated Completion Time:** 1.5-2 hours
**Difficulty:** Intermediate
**Prerequisites:** Linux basics, Python installation
