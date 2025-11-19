# Module 1.3: Environment Configuration

## Learning Objectives

By the end of this lesson, you will be able to:

1. Set up isolated Python environments for vLLM development
2. Configure environment variables for optimal vLLM performance
3. Manage dependencies and avoid version conflicts
4. Create reproducible development environments
5. Configure vLLM for different deployment scenarios
6. Troubleshoot environment-related issues

**Estimated Time:** 1.5-2 hours

---

## Table of Contents

1. [Python Environment Management](#python-environment-management)
2. [Virtual Environments](#virtual-environments)
3. [Conda vs venv](#conda-vs-venv)
4. [Environment Variables](#environment-variables)
5. [vLLM Configuration](#vllm-configuration)
6. [Dependency Management](#dependency-management)
7. [Reproducible Environments](#reproducible-environments)
8. [Common Configuration Issues](#common-configuration-issues)
9. [Practical Exercises](#practical-exercises)
10. [Key Takeaways](#key-takeaways)

---

## Python Environment Management

### Why Use Virtual Environments?

```
┌──────────────────────────────────────────────────────────────┐
│                    System Python (Avoid!)                     │
│  /usr/bin/python3                                             │
│  - Shared by all applications                                 │
│  - Package conflicts likely                                   │
│  - May require sudo for installations                         │
│  - System updates can break your code                         │
└──────────────────────────────────────────────────────────────┘
                              vs.
┌──────────────────────────────────────────────────────────────┐
│              Virtual Environment (Recommended!)               │
│  ~/vllm-env/bin/python                                        │
│  - Isolated dependencies                                      │
│  - No package conflicts                                       │
│  - No sudo required                                           │
│  - Can have multiple environments with different versions     │
└──────────────────────────────────────────────────────────────┘
```

**Benefits:**
1. **Isolation**: Each project has its own dependencies
2. **Reproducibility**: Lock exact versions for consistency
3. **Safety**: No accidental system package modifications
4. **Flexibility**: Test different dependency versions easily

---

## Virtual Environments

### Method 1: venv (Built-in Python)

```bash
# Create virtual environment
python3 -m venv vllm-env

# Activate environment
source vllm-env/bin/activate  # Linux/Mac
# vllm-env\Scripts\activate   # Windows

# Verify activation
which python
# Output: /path/to/vllm-env/bin/python

# Check Python version
python --version

# Install packages
pip install vllm

# Deactivate when done
deactivate
```

**Directory Structure:**
```
vllm-env/
├── bin/
│   ├── python -> python3.10
│   ├── pip
│   └── activate
├── include/
├── lib/
│   └── python3.10/
│       └── site-packages/  # All installed packages here
└── pyvenv.cfg
```

### Method 2: virtualenv (Enhanced venv)

```bash
# Install virtualenv
pip install virtualenv

# Create environment with specific Python version
virtualenv -p python3.10 vllm-env

# Or use Python 3.9
virtualenv -p python3.9 vllm-env-py39

# Activate
source vllm-env/bin/activate

# Advanced: Create environment with system packages access
virtualenv --system-site-packages vllm-env
```

### Method 3: pyenv (Python Version Management)

```bash
# Install pyenv (Linux)
curl https://pyenv.run | bash

# Add to ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install specific Python version
pyenv install 3.10.12

# Create virtual environment with that version
pyenv virtualenv 3.10.12 vllm-env

# Activate
pyenv activate vllm-env

# Set as default for directory
cd ~/projects/vllm-project
pyenv local vllm-env  # Creates .python-version file

# List all environments
pyenv versions
```

---

## Conda vs venv

### Comparison Table

```
┌────────────────────────────────────────────────────────────────┐
│ Feature          │ venv/virtualenv │ Conda                     │
├──────────────────┼─────────────────┼───────────────────────────┤
│ Python versions  │ Use installed   │ Can install any version   │
│ Package source   │ PyPI (pip)      │ Conda repos + PyPI        │
│ Non-Python deps  │ No              │ Yes (CUDA, gcc, etc.)     │
│ Disk usage       │ Smaller         │ Larger (includes system)  │
│ Speed            │ Faster          │ Slower (dependency solve) │
│ ML/DS support    │ Good            │ Excellent                 │
│ Reproducibility  │ Good            │ Excellent                 │
└────────────────────────────────────────────────────────────────┘
```

### Using Conda

```bash
# Install Miniconda (lightweight)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment with Python 3.10
conda create -n vllm-env python=3.10

# Activate
conda activate vllm-env

# Install packages from conda-forge
conda install -c conda-forge cudatoolkit=11.8

# Install pip packages
pip install vllm

# Export environment
conda env export > environment.yml

# Create from environment file
conda env create -f environment.yml

# List environments
conda env list

# Remove environment
conda env remove -n vllm-env
```

### Conda environment.yml Example

```yaml
# environment.yml
name: vllm-dev
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - cudatoolkit=11.8
  - numpy
  - pip:
    - vllm
    - transformers>=4.36.0
    - torch>=2.1.0
    - ray>=2.9.0
```

### Recommendation for vLLM

**Use venv/virtualenv if:**
- You have CUDA pre-installed
- You want minimal overhead
- You're comfortable with system package management

**Use Conda if:**
- You need to install CUDA toolkit
- You want everything self-contained
- You work with multiple Python versions regularly
- You're in a data science workflow

---

## Environment Variables

### Critical vLLM Environment Variables

```bash
# ============================================================
# CUDA Configuration
# ============================================================

# CUDA installation path
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

# Add CUDA binaries to PATH
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA libraries to library path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Specify visible GPUs (GPU selection)
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3
export CUDA_VISIBLE_DEVICES=0        # Use only GPU 0
export CUDA_VISIBLE_DEVICES=""       # Force CPU mode

# ============================================================
# vLLM Performance Configuration
# ============================================================

# Worker initialization timeout (seconds)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Logging level
export VLLM_LOGGING_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Trace function calls (debugging)
export VLLM_TRACE_FUNCTION=1

# Configure KV cache dtype
export VLLM_USE_MODELSCOPE=0

# Attention backend selection
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # or XFORMERS, TORCH_SDPA

# ============================================================
# PyTorch Configuration
# ============================================================

# Number of threads for CPU operations
export OMP_NUM_THREADS=8

# PyTorch CUDA memory allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 on Ampere+ GPUs (faster but less precise)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# CUDA architecture list (for building from source)
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# ============================================================
# HuggingFace Configuration
# ============================================================

# Model cache directory
export HF_HOME=/data/huggingface
export TRANSFORMERS_CACHE=/data/huggingface/transformers
export HF_DATASETS_CACHE=/data/huggingface/datasets

# HuggingFace API token (for gated models)
export HF_TOKEN=your_token_here

# Disable telemetry
export HF_HUB_DISABLE_TELEMETRY=1

# ============================================================
# Ray Configuration (for distributed vLLM)
# ============================================================

# Ray temporary directory
export RAY_TMPDIR=/tmp/ray

# Disable Ray usage stats
export RAY_USAGE_STATS_DISABLE=1

# Ray object store memory
export RAY_OBJECT_STORE_MEMORY=10000000000  # 10GB

# ============================================================
# Debugging and Development
# ============================================================

# Enable Python development mode
export PYTHONDONTWRITEBYTECODE=1  # Don't create .pyc files

# Show Python warnings
export PYTHONWARNINGS=default

# Unbuffered Python output
export PYTHONUNBUFFERED=1

# CUDA launch blocking (for debugging, slower!)
export CUDA_LAUNCH_BLOCKING=1

# NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# ============================================================
# Memory and Performance
# ============================================================

# Limit GPU memory fraction
export VLLM_GPU_MEMORY_UTILIZATION=0.9  # Use 90% of GPU memory

# Enable/disable CUDA graphs
export VLLM_USE_CUDA_GRAPH=1

# Swap space for KV cache
export VLLM_SWAP_SPACE=4  # GB
```

### Creating an Environment Configuration File

```bash
# vllm_env.sh - Source this file to set up environment

#!/bin/bash

# CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# vLLM configuration
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# HuggingFace
export HF_HOME=/data/huggingface
export HF_HUB_DISABLE_TELEMETRY=1

# PyTorch
export OMP_NUM_THREADS=$(nproc)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Debugging (comment out for production)
# export CUDA_LAUNCH_BLOCKING=1
# export VLLM_TRACE_FUNCTION=1

echo "vLLM environment configured"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "HuggingFace cache: $HF_HOME"
```

**Usage:**
```bash
# Source the configuration
source vllm_env.sh

# Or add to ~/.bashrc for automatic loading
echo "source ~/vllm_env.sh" >> ~/.bashrc
```

---

## vLLM Configuration

### Configuration Files

vLLM can be configured via:
1. **Environment variables** (shown above)
2. **Command-line arguments** (runtime)
3. **Python API parameters** (programmatic)

### Runtime Configuration Example

```python
# config_example.py
"""
vLLM configuration through Python API
Located: examples/config_example.py
"""

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# Method 1: Direct LLM initialization
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,      # Use 2 GPUs
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_model_len=4096,          # Maximum sequence length
    dtype="float16",             # Use FP16
    trust_remote_code=False,     # Security: don't run remote code
    download_dir="/data/models", # Model download location
    seed=42,                     # Random seed for reproducibility
)

# Method 2: Using EngineArgs (more configurable)
engine_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    tokenizer_mode="auto",
    trust_remote_code=False,
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=None,          # Auto-detect
    gpu_memory_utilization=0.9,
    swap_space=4,                # GB of CPU swap
    enforce_eager=False,         # Use CUDA graphs
    max_num_seqs=256,            # Max concurrent sequences
    max_num_batched_tokens=None, # Auto-detect
)

# Initialize engine
from vllm import EngineArgs, LLMEngine
engine = LLMEngine.from_engine_args(engine_args)
```

### Configuration Profiles

Create different profiles for different scenarios:

```python
# config_profiles.py

# Development Profile (fast iteration, small model)
DEV_CONFIG = {
    "model": "facebook/opt-125m",
    "max_model_len": 512,
    "gpu_memory_utilization": 0.5,
    "enforce_eager": True,  # No CUDA graphs for easier debugging
}

# Production Profile (optimized for throughput)
PROD_CONFIG = {
    "model": "meta-llama/Llama-2-70b-hf",
    "tensor_parallel_size": 4,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.95,
    "enforce_eager": False,  # Use CUDA graphs
    "max_num_seqs": 256,
}

# Memory-constrained Profile
LOW_MEM_CONFIG = {
    "model": "meta-llama/Llama-2-7b-hf",
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.7,
    "swap_space": 8,  # Allow more CPU offloading
}

# Multi-GPU Profile
MULTI_GPU_CONFIG = {
    "model": "meta-llama/Llama-2-13b-hf",
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "max_model_len": 4096,
}

# Usage:
from vllm import LLM

# Load profile
import os
profile = os.getenv("VLLM_PROFILE", "dev").upper()
config_map = {
    "DEV": DEV_CONFIG,
    "PROD": PROD_CONFIG,
    "LOW_MEM": LOW_MEM_CONFIG,
    "MULTI_GPU": MULTI_GPU_CONFIG,
}

llm = LLM(**config_map[profile])
```

---

## Dependency Management

### Managing Requirements

```bash
# requirements.txt - Standard pip format
torch>=2.1.0
transformers>=4.36.0
vllm>=0.2.7
ray>=2.9.0

# Generate from current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# requirements-dev.txt - Development dependencies
-r requirements.txt  # Include base requirements
pytest
black
mypy
ipython
jupyter
```

### Using pip-tools for Pinned Dependencies

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in (high-level dependencies)
cat > requirements.in << EOF
vllm
torch
transformers
EOF

# Generate pinned requirements.txt
pip-compile requirements.in

# This creates requirements.txt with exact versions:
# torch==2.1.2
# transformers==4.36.2
# vllm==0.2.7
# ... and all transitive dependencies

# Install pinned versions
pip-sync requirements.txt

# Update all dependencies
pip-compile --upgrade requirements.in
```

### Poetry for Advanced Dependency Management

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize project
poetry init

# Add dependencies
poetry add vllm
poetry add torch --source pytorch

# Install all dependencies
poetry install

# Create lock file (like package-lock.json)
poetry lock

# Run commands in environment
poetry run python script.py

# Activate shell
poetry shell
```

**pyproject.toml Example:**
```toml
[tool.poetry]
name = "vllm-project"
version = "0.1.0"
description = "vLLM development project"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
vllm = "^0.2.7"
torch = "^2.1.0"
transformers = "^4.36.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
black = "^23.0.0"
mypy = "^1.7.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

---

## Reproducible Environments

### Method 1: Docker

```dockerfile
# Dockerfile for vLLM environment
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/vllm-env
ENV PATH="/opt/vllm-env/bin:$PATH"

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install vLLM from source
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    pip install -e .

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
```

**Build and run:**
```bash
# Build image
docker build -t vllm-dev:latest .

# Run container
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    vllm-dev:latest

# With specific GPU
docker run --gpus '"device=0,1"' -it --rm vllm-dev:latest
```

### Method 2: Environment Export/Import

```bash
# Export environment (pip)
pip freeze > requirements-frozen.txt
python --version > python-version.txt

# Export environment (conda)
conda env export > environment.yml
conda env export --no-builds > environment-no-builds.yml

# Import on another machine (pip)
# 1. Install same Python version
# 2. Create virtual environment
# 3. Install requirements
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install -r requirements-frozen.txt

# Import on another machine (conda)
conda env create -f environment.yml
```

### Method 3: Dev Containers (VS Code)

```json
// .devcontainer/devcontainer.json
{
    "name": "vLLM Development",
    "dockerFile": "../Dockerfile",
    "runArgs": ["--gpus", "all"],
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-toolsai.jupyter"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/opt/vllm-env/bin/python"
            }
        }
    },
    "postCreateCommand": "pip install -e .",
    "remoteUser": "root"
}
```

---

## Common Configuration Issues

### Issue 1: Import Errors After Installation

```python
# Symptom
import vllm
# ImportError: cannot import name 'LLM' from 'vllm'

# Diagnosis
pip show vllm  # Check if installed
python -c "import sys; print(sys.path)"  # Check Python path

# Solution 1: Reinstall in current environment
pip uninstall vllm
pip install vllm

# Solution 2: Check you're in the right environment
which python  # Should point to virtual environment
```

### Issue 2: CUDA Version Mismatch

```bash
# Symptom
RuntimeError: CUDA error: no kernel image available

# Diagnosis
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
nvidia-smi  # Driver CUDA

# Solution: Reinstall PyTorch matching driver
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: Out of Disk Space

```bash
# Check disk usage
du -sh ~/.cache/huggingface  # HuggingFace cache
du -sh ~/.cache/pip          # Pip cache

# Clean caches
pip cache purge
rm -rf ~/.cache/huggingface/hub/*

# Set cache location to larger disk
export HF_HOME=/data/huggingface
export PIP_CACHE_DIR=/data/pip-cache
```

### Issue 4: Permission Denied

```bash
# Symptom
PermissionError: [Errno 13] Permission denied

# Solution: Don't use sudo with pip in virtual env
# Instead:
deactivate
python3 -m venv vllm-env  # Create new environment
source vllm-env/bin/activate
pip install vllm  # No sudo needed
```

---

## Practical Exercises

### Exercise 1.3.1: Environment Setup

**Objective:** Create a production-ready vLLM environment.

**Tasks:**
1. Create a new virtual environment
2. Configure environment variables
3. Install vLLM and dependencies
4. Export environment for reproducibility

**Time:** 30 minutes

### Exercise 1.3.2: Multi-Environment Management

**Objective:** Manage multiple environments for different projects.

**Tasks:**
1. Create three environments:
   - `vllm-dev`: Development with latest vLLM
   - `vllm-prod`: Stable versions
   - `vllm-experimental`: Testing new features
2. Switch between them
3. Document differences

**Time:** 20 minutes

### Exercise 1.3.3: Dependency Conflict Resolution

**Objective:** Practice resolving version conflicts.

**Tasks:**
1. Intentionally create a conflict (e.g., incompatible torch/vllm)
2. Diagnose the issue
3. Resolve using requirements pinning

**Time:** 20 minutes

### Exercise 1.3.4: Docker Environment

**Objective:** Create a containerized vLLM environment.

**Tasks:**
1. Write a Dockerfile for vLLM
2. Build the image
3. Run inference in container
4. Share image with team

**Time:** 30 minutes

---

## Key Takeaways

1. **Always use virtual environments** to isolate dependencies
2. **Configure environment variables** for optimal vLLM performance
3. **Pin dependency versions** for reproducibility
4. **Use Docker** for fully reproducible environments
5. **Set HF_HOME** to manage model cache location
6. **Document your environment** for team collaboration

---

## Quick Reference: Environment Setup Checklist

```bash
# ✅ Quick Setup Checklist

# 1. Create virtual environment
python3 -m venv vllm-env
source vllm-env/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install vLLM
pip install vllm

# 4. Set environment variables
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/data/huggingface
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 5. Verify installation
python -c "import vllm; import torch; print(f'vLLM: {vllm.__version__}, CUDA: {torch.cuda.is_available()}')"

# 6. Create requirements file
pip freeze > requirements.txt

# 7. (Optional) Create activation script
cat > activate_vllm.sh << 'EOF'
source vllm-env/bin/activate
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/data/huggingface
echo "vLLM environment activated"
EOF

chmod +x activate_vllm.sh
```

---

## Next Steps

In the next lesson, **04_project_structure_tour.md**, we'll explore:
- vLLM repository organization
- Key source code directories
- Understanding the codebase architecture
- Navigating source files effectively

---

## Additional Resources

**Documentation:**
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [Docker documentation](https://docs.docker.com/)

**Tools:**
- [pyenv](https://github.com/pyenv/pyenv) - Python version management
- [Poetry](https://python-poetry.org/) - Dependency management
- [pip-tools](https://github.com/jazzband/pip-tools) - Requirements pinning

---

**Estimated Completion Time:** 1.5-2 hours
**Difficulty:** Intermediate
**Prerequisites:** Python basics, command line experience
