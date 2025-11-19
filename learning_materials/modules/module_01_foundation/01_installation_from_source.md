# Module 1.1: Installing vLLM from Source

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the differences between pip installation and building from source
2. Successfully build vLLM from source on a Linux system with CUDA
3. Verify the installation and troubleshoot common build errors
4. Configure build options for optimal performance
5. Set up a development environment for vLLM modification and debugging

**Estimated Time:** 2-3 hours

---

## Table of Contents

1. [Why Build from Source?](#why-build-from-source)
2. [Prerequisites](#prerequisites)
3. [Installation Methods Comparison](#installation-methods-comparison)
4. [Step-by-Step Build Process](#step-by-step-build-process)
5. [Build Configuration Options](#build-configuration-options)
6. [Verification and Testing](#verification-and-testing)
7. [Common Build Errors](#common-build-errors)
8. [Development Mode Setup](#development-mode-setup)
9. [Practical Exercises](#practical-exercises)
10. [Key Takeaways](#key-takeaways)

---

## Why Build from Source?

Building vLLM from source provides several advantages over pip installation:

### Advantages

1. **Latest Features**: Access cutting-edge features before they're released to PyPI
2. **Custom Modifications**: Modify core functionality for research or specific use cases
3. **Debugging**: Set breakpoints and trace through C++/CUDA code
4. **Optimization**: Compile with specific optimization flags for your hardware
5. **Understanding**: Learn the build system and project structure intimately

### Disadvantages

1. **Time**: Initial build can take 15-45 minutes depending on hardware
2. **Complexity**: More dependencies and potential failure points
3. **Maintenance**: Need to rebuild after pulling updates
4. **Disk Space**: Requires ~5GB for source code and build artifacts

### When to Use Each Method

```
┌─────────────────────────────────────────────────────────────┐
│                    Installation Decision Tree                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Need to modify vLLM code? ───────────────────► BUILD SOURCE │
│          │                                                    │
│          NO                                                   │
│          │                                                    │
│          ▼                                                    │
│  Need latest unreleased features? ─────────────► BUILD SOURCE │
│          │                                                    │
│          NO                                                   │
│          │                                                    │
│          ▼                                                    │
│  Want to understand internals deeply? ─────────► BUILD SOURCE │
│          │                                                    │
│          NO                                                   │
│          │                                                    │
│          ▼                                                    │
│  Just want to use vLLM for inference? ─────────► PIP INSTALL │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Rocky Linux 8+, or similar)
- **NOT officially supported**: macOS, Windows (use WSL2)

**Hardware:**
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (V100, T4, A100, H100, etc.)
  - Check your GPU: `nvidia-smi --query-gpu=compute_cap --format=csv`
- **CPU**: Multi-core recommended (build parallelization)
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Disk**: 10GB free space for build

**Software:**
- Python 3.8 - 3.11 (3.9 or 3.10 recommended)
- CUDA 11.8 or 12.1+ (check with `nvcc --version`)
- GCC 7+ or Clang 10+ (check with `gcc --version`)
- Git

### Quick Prerequisites Check

```bash
#!/bin/bash
# prerequisites_check.sh - Run this to verify your system is ready

echo "=== vLLM Build Prerequisites Check ==="
echo ""

# Check Python version
echo "Python version:"
python3 --version
echo ""

# Check CUDA
echo "CUDA version:"
nvcc --version 2>/dev/null || echo "CUDA not found in PATH"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Check GCC
echo "GCC version:"
gcc --version | head -1
echo ""

# Check Git
echo "Git version:"
git --version
echo ""

# Check available disk space
echo "Disk space in current directory:"
df -h . | tail -1
echo ""

# Check available RAM
echo "Available RAM:"
free -h | grep "Mem:" | awk '{print $7 " available"}'
echo ""

echo "=== End of Prerequisites Check ==="
```

---

## Installation Methods Comparison

### Method 1: Pip Install (Quick Start)

```bash
# Simple pip installation
pip install vllm

# With specific CUDA version
pip install vllm-cuda12  # For CUDA 12.x
```

**Pros:**
- Fast (1-5 minutes)
- Minimal dependencies
- Stable releases
- Works for most use cases

**Cons:**
- Can't modify source code
- Delayed access to new features
- Limited debugging capabilities

### Method 2: Build from Source (Full Control)

```bash
# Clone repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install build dependencies
pip install -e .
```

**Pros:**
- Full source code access
- Latest features
- Customization options
- Deep debugging

**Cons:**
- Longer installation time
- More complex troubleshooting
- Requires build tools

### Method 3: Docker (Containerized)

```bash
# Pull official image
docker pull vllm/vllm-openai:latest

# Run container
docker run --gpus all vllm/vllm-openai:latest
```

**Pros:**
- Reproducible environment
- No local dependency conflicts
- Easy deployment

**Cons:**
- Docker overhead
- Less flexible for development
- Larger disk footprint

---

## Step-by-Step Build Process

### Step 1: Set Up Python Environment

```bash
# Create a fresh virtual environment (recommended)
python3 -m venv vllm-env
source vllm-env/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Verify Python version
python --version  # Should be 3.8-3.11
```

**Why virtual environment?**
- Isolates dependencies
- Prevents conflicts with system packages
- Easy to delete and recreate

### Step 2: Clone vLLM Repository

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Check current branch
git branch

# (Optional) Switch to a specific version
git checkout v0.6.0  # Or any stable tag

# View recent commits
git log --oneline -10
```

**Repository Structure Overview:**
```
vllm/
├── vllm/              # Main Python package
│   ├── engine/        # Core inference engine
│   ├── worker/        # GPU worker processes
│   ├── model_executor/# Model execution logic
│   └── entrypoints/   # API servers (OpenAI compatible, etc.)
├── csrc/              # C++ and CUDA source code
│   ├── attention/     # Attention kernel implementations
│   ├── cache_kernels/ # KV cache operations
│   └── quantization/  # Quantization kernels
├── CMakeLists.txt     # CMake build configuration
├── setup.py           # Python package setup
└── requirements*.txt  # Dependency specifications
```

### Step 3: Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements-common.txt
pip install -r requirements-cuda.txt

# Install development dependencies (optional, for development)
pip install -r requirements-dev.txt
```

**Key Dependencies:**
- **torch**: PyTorch framework (auto-installs correct CUDA version)
- **transformers**: HuggingFace model loading
- **xformers**: Memory-efficient attention operations
- **ray**: Distributed execution framework
- **numpy**, **pydantic**: Core utilities

### Step 4: Build and Install vLLM

```bash
# Install in editable mode (recommended for development)
pip install -e .

# This command:
# 1. Compiles C++ and CUDA kernels using CMake
# 2. Builds Python extension modules
# 3. Installs vLLM package in editable mode
```

**Build Process Overview:**
```
┌──────────────────────────────────────────────────────────┐
│                  vLLM Build Pipeline                      │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  setup.py                                                │
│     │                                                     │
│     ├──► CMake Configuration                            │
│     │      └─► Detect CUDA, Compute Capability          │
│     │                                                     │
│     ├──► Compile C++ Sources (csrc/)                    │
│     │      ├─► Attention kernels                         │
│     │      ├─► Cache management kernels                  │
│     │      └─► Custom CUDA operations                    │
│     │                                                     │
│     ├──► Build Python Extensions (.so files)            │
│     │                                                     │
│     └──► Install Package                                 │
│            └─► Create importable vllm module             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Expected Build Output:**
```
Building wheels for collected packages: vllm
  Building editable for vllm (pyproject.toml) ...
  -- The CUDA compiler identification is NVIDIA 12.1.105
  -- Detecting CUDA compile features
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
  -- Found Python3: /path/to/python (found version "3.10.12")
  -- Found CUDA: /usr/local/cuda (found version "12.1")
  -- Building for compute capabilities: 80;86;89;90
  ...
  [100%] Built target vllm
  Successfully built vllm
```

**Build Time Expectations:**
- First build: 15-45 minutes (depending on CPU cores)
- Incremental builds: 1-10 minutes (only changed files)
- Parallel build jobs automatically detected from CPU count

### Step 5: Set Environment Variables (Optional)

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence

# Specify CUDA architecture (optional, auto-detected)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # A100, RTX 40xx, H100

# Enable verbose build output for debugging
export VERBOSE=1

# Set number of parallel build jobs
export MAX_JOBS=8  # Adjust based on your CPU cores

# CUDA paths (if not in default location)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Compute Capability Reference:**
```
GPU Model         | Compute Capability | TORCH_CUDA_ARCH_LIST
------------------|--------------------|-----------------------
V100              | 7.0                | 7.0
T4                | 7.5                | 7.5
A10, A10G         | 8.6                | 8.6
A100              | 8.0                | 8.0
RTX 3090          | 8.6                | 8.6
RTX 4090          | 8.9                | 8.9
H100              | 9.0                | 9.0
L4                | 8.9                | 8.9
```

---

## Build Configuration Options

### CMake Build Options

vLLM's build system (in `CMakeLists.txt`) supports several configuration options:

```bash
# View CMake configuration
cat CMakeLists.txt | grep "option("

# Common options (set via environment variables):

# 1. Build with debugging symbols
export CMAKE_BUILD_TYPE=Debug
pip install -e .

# 2. Build with optimizations (default)
export CMAKE_BUILD_TYPE=Release
pip install -e .

# 3. Disable specific CUDA architectures to speed up build
export TORCH_CUDA_ARCH_LIST="8.0"  # Only build for A100

# 4. Use specific compiler
export CC=gcc-11
export CXX=g++-11
export CUDACXX=/usr/local/cuda/bin/nvcc

# 5. Enable/disable features
export VLLM_USE_PRECOMPILED=0  # Force rebuild of kernels
```

### Advanced Build Customization

**File:** `setup.py` (excerpt)

```python
# Located at: /home/user/vllm-learn/setup.py
# This file orchestrates the build process

# Key build parameters you can modify:
# - CUDA architectures
# - Compiler flags
# - Extension modules to build
```

To modify build behavior:

```bash
# Edit setup.py before building
# Example: Add custom compiler flags

# In setup.py, find the ext_modules section and modify
extra_compile_args = {
    "cxx": ["-O3", "-march=native"],  # C++ optimizations
    "nvcc": ["-O3", "--use_fast_math"],  # CUDA optimizations
}
```

---

## Verification and Testing

### Quick Verification

```bash
# Test 1: Import vLLM
python -c "import vllm; print(vllm.__version__)"
# Expected output: 0.6.0 (or your version)

# Test 2: Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected output: CUDA available: True

# Test 3: List compiled extensions
python -c "import vllm._C; print('vLLM extensions loaded successfully')"
# Expected output: vLLM extensions loaded successfully
```

### Comprehensive Test

```python
# test_installation.py
"""
Comprehensive installation verification script.
Save as test_installation.py and run: python test_installation.py
"""

import sys
import torch

def test_basic_import():
    """Test 1: Basic vLLM import"""
    try:
        import vllm
        print(f"✓ vLLM imported successfully (version {vllm.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import vLLM: {e}")
        return False

def test_cuda_availability():
    """Test 2: CUDA availability"""
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Device count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print(f"✗ CUDA is not available")
        return False

def test_vllm_extensions():
    """Test 3: vLLM C++ extensions"""
    try:
        import vllm._C
        print(f"✓ vLLM C++ extensions loaded")
        return True
    except ImportError as e:
        print(f"✗ Failed to load vLLM extensions: {e}")
        return False

def test_simple_inference():
    """Test 4: Simple inference test"""
    try:
        from vllm import LLM, SamplingParams

        # Create a simple model instance (will download if not cached)
        # Using a tiny model for quick testing
        print("  Creating LLM instance (may download model)...")
        llm = LLM(model="facebook/opt-125m", max_model_len=512)

        # Simple inference
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate(prompts, sampling_params)

        print(f"✓ Inference test passed")
        print(f"  - Prompt: {prompts[0]}")
        print(f"  - Output: {outputs[0].outputs[0].text}")
        return True

    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("vLLM Installation Verification")
    print("=" * 60)
    print()

    tests = [
        test_basic_import,
        test_cuda_availability,
        test_vllm_extensions,
        test_simple_inference,
    ]

    results = [test() for test in tests]

    print()
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

Run the verification:

```bash
python test_installation.py
```

---

## Common Build Errors

### Error 1: CUDA Not Found

**Symptom:**
```
CMake Error: CUDA not found
```

**Solution:**
```bash
# Check if CUDA is installed
which nvcc
# If not found:

# Option A: Install CUDA toolkit
# Download from https://developer.nvidia.com/cuda-downloads

# Option B: Add CUDA to PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Error 2: GCC Version Incompatibility

**Symptom:**
```
nvcc fatal: Unsupported gcc version!
```

**Solution:**
```bash
# Check GCC version
gcc --version

# CUDA 12.x requires GCC ≤ 12
# Install compatible GCC if needed
sudo apt install gcc-11 g++-11

# Set environment variables
export CC=gcc-11
export CXX=g++-11

# Rebuild
pip install -e . --force-reinstall --no-cache-dir
```

### Error 3: Out of Memory During Build

**Symptom:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution:**
```bash
# Limit parallel build jobs
export MAX_JOBS=2  # Reduce based on available RAM

# Alternatively, add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Rebuild
pip install -e .
```

### Error 4: PyTorch CUDA Version Mismatch

**Symptom:**
```
RuntimeError: CUDA version mismatch: PyTorch was compiled with CUDA 11.8 but CUDA 12.1 is available
```

**Solution:**
```bash
# Reinstall PyTorch with matching CUDA version
pip uninstall torch

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Rebuild vLLM
pip install -e .
```

### Error 5: Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements-common.txt
pip install -r requirements-cuda.txt

# If still issues, try:
pip install --upgrade transformers accelerate
```

---

## Development Mode Setup

### Editable Installation Benefits

Installing with `-e` flag creates an **editable** installation:

```bash
pip install -e .
```

**Benefits:**
1. **No reinstall needed**: Changes to Python code take effect immediately
2. **Source code access**: IDE can navigate to source files
3. **Debugging**: Set breakpoints in vLLM code
4. **Experimentation**: Modify and test quickly

### IDE Setup (VS Code Example)

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/vllm-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "yapf",
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/csrc",
        "/usr/local/cuda/include"
    ]
}
```

### Rebuilding After Changes

```bash
# Python changes: No rebuild needed (editable install)
# Just reload your Python session

# C++/CUDA changes: Rebuild required
pip install -e . --force-reinstall --no-deps --no-build-isolation

# Faster incremental build (only changed files)
cd build
make -j$(nproc)
cd ..
```

### Pre-commit Hooks (Code Quality)

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Hooks include:
# - Code formatting (yapf)
# - Linting (flake8, mypy)
# - Import sorting (isort)
```

---

## Practical Exercises

### Exercise 1.1: Build vLLM from Source

**Objective:** Successfully build and install vLLM from source.

**Tasks:**
1. Create a new virtual environment
2. Clone the vLLM repository
3. Install dependencies
4. Build vLLM from source
5. Verify the installation

**Success Criteria:**
- [ ] Build completes without errors
- [ ] Can import vllm in Python
- [ ] `test_installation.py` passes all tests

**Time:** 30-45 minutes

### Exercise 1.2: Experiment with Build Options

**Objective:** Understand how build configuration affects the output.

**Tasks:**
1. Build with Debug mode: `export CMAKE_BUILD_TYPE=Debug`
2. Build with specific CUDA architecture only
3. Compare build times and binary sizes
4. Document your findings

**Questions to Answer:**
- How much does build time differ between Debug and Release?
- How does limiting CUDA architectures affect build time?
- What is the size difference of compiled binaries?

**Time:** 20-30 minutes

### Exercise 1.3: Troubleshoot a Build Error

**Objective:** Practice debugging build issues.

**Tasks:**
1. Intentionally cause a build error (e.g., unset CUDA_HOME)
2. Diagnose the error using error messages
3. Fix the error
4. Document the problem and solution

**Time:** 15-20 minutes

### Exercise 1.4: Set Up Development Environment

**Objective:** Configure a productive development setup.

**Tasks:**
1. Install vLLM in editable mode
2. Set up VS Code (or your preferred IDE)
3. Configure code formatting and linting
4. Make a small modification to vLLM code and test

**Suggested Modification:**
- Add a print statement to `vllm/engine/llm_engine.py` in the `__init__` method
- Import vLLM and verify your print statement executes

**Time:** 20-30 minutes

---

## Key Takeaways

1. **Building from source** provides full control and access to latest features, but requires more setup time
2. **Prerequisites** include CUDA, compatible GCC, and Python environment
3. **The build process** compiles C++/CUDA kernels and installs Python package
4. **Editable installation** (`pip install -e .`) is ideal for development
5. **Common errors** usually relate to CUDA/GCC compatibility or missing dependencies
6. **Verification is crucial** - always test your installation thoroughly

---

## Next Steps

In the next lesson, **02_cuda_setup_verification.md**, we'll dive deep into:
- Verifying CUDA installation and GPU detection
- Understanding compute capability
- Testing GPU performance
- Configuring multiple GPUs

---

## Additional Resources

**Official Documentation:**
- [vLLM Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

**Source Code References:**
- `setup.py`: Build configuration
- `CMakeLists.txt`: CMake build rules
- `requirements-cuda.txt`: CUDA-specific dependencies

**Community:**
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [vLLM Discord](https://discord.gg/vllm)

---

**Estimated Completion Time:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** Linux command line, Python basics
