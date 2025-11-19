# Module 1.11: Dependency Management

## Learning Objectives

1. Understand vLLM dependencies
2. Manage version conflicts
3. Update dependencies safely
4. Create reproducible environments
5. Handle security updates
6. Troubleshoot dependency issues

**Estimated Time:** 1 hour

---

## Core Dependencies

### Required Dependencies

```
PyTorch >= 2.0.0       # Deep learning framework
transformers >= 4.36.0 # HuggingFace models
ray >= 2.9.0           # Distributed execution
numpy                  # Numerical computing
xformers               # Memory-efficient attention (optional)
```

**Source:** `/home/user/vllm-learn/requirements-common.txt`

### CUDA Dependencies

```
nvidia-cuda-runtime    # CUDA runtime
nvidia-cudnn          # cuDNN for deep learning
nvidia-nccl           # NCCL for multi-GPU communication
```

---

## Version Management

### Check Versions

```python
# check_versions.py
import vllm
import torch
import transformers
import ray

print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Ray: {ray.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

### Update Dependencies

```bash
# Update all dependencies
pip install --upgrade vllm

# Update specific package
pip install --upgrade transformers

# Update PyTorch (be careful!)
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Conflict Resolution

### Common Conflicts

#### PyTorch Version Mismatch

```bash
# Symptom
ERROR: pip's dependency resolver does not currently take into account all the packages

# Solution: Install compatible versions
pip install torch==2.1.0 transformers==4.36.0 vllm==0.2.7
```

#### CUDA Version Mismatch

```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check driver CUDA version
nvidia-smi

# If mismatch, reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Pinned Dependencies

### Create requirements.txt

```bash
# Generate from current environment
pip freeze > requirements-frozen.txt

# Or use pip-compile
pip install pip-tools
pip-compile requirements.in -o requirements.txt
```

### requirements.in Example

```
# requirements.in
vllm>=0.2.7
torch>=2.1.0
transformers>=4.36.0
ray>=2.9.0
```

### Install from Pinned

```bash
pip install -r requirements.txt
```

---

## Development Dependencies

```bash
# requirements-dev.txt
-r requirements.txt  # Include base requirements

# Testing
pytest>=7.0.0
pytest-asyncio

# Code quality
black
flake8
mypy
isort

# Development tools
ipython
jupyterlab
```

---

## Conda Environment

```yaml
# environment.yml
name: vllm-env
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch::pytorch>=2.1.0
  - nvidia::cuda-toolkit=11.8
  - pip
  - pip:
    - vllm>=0.2.7
    - transformers>=4.36.0
```

```bash
# Create environment
conda env create -f environment.yml

# Update environment
conda env update -f environment.yml
```

---

## Security Updates

### Check for Vulnerabilities

```bash
# Install safety
pip install safety

# Check dependencies
safety check

# Or use pip-audit
pip install pip-audit
pip-audit
```

### Update Securely

```bash
# Update only patch versions (safe)
pip install --upgrade vllm --upgrade-strategy only-if-needed

# Full upgrade (risky)
pip install --upgrade vllm
```

---

## Troubleshooting Dependencies

### Dependency Tree

```bash
# Show dependency tree
pip install pipdeptree
pipdeptree

# Show dependencies of specific package
pipdeptree -p vllm
```

### Find Conflicts

```bash
pip check
```

### Clean Install

```bash
# Remove all packages
pip freeze | xargs pip uninstall -y

# Reinstall from requirements
pip install -r requirements.txt
```

---

## Practical Exercises

### Exercise 1.11.1: Version Audit
1. Check all dependency versions
2. Compare with requirements
3. Document any mismatches

**Time:** 15 minutes

### Exercise 1.11.2: Create Pinned Requirements
1. Generate requirements-frozen.txt
2. Create new environment
3. Install from frozen requirements

**Time:** 20 minutes

### Exercise 1.11.3: Resolve Conflict
1. Intentionally create version conflict
2. Diagnose using pip check
3. Resolve and verify

**Time:** 15 minutes

---

## Key Takeaways

1. **PyTorch version** must match CUDA driver
2. **Pin dependencies** for reproducibility
3. **Check for conflicts** regularly with pip check
4. **Update carefully** to avoid breakage
5. **Use virtual environments** always
6. **Monitor security updates** regularly

---

**Next:** 12_baseline_benchmarking.md
**Time:** 1 hour
**Source:** `/home/user/vllm-learn/requirements-common.txt`
