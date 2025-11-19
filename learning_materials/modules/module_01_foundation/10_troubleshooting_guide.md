# Module 1.10: Troubleshooting Guide

## Learning Objectives

1. Diagnose common vLLM issues
2. Fix installation problems
3. Resolve runtime errors
4. Handle memory issues
5. Debug performance problems
6. Find solutions quickly

**Estimated Time:** 1.5 hours

---

## Common Issues Reference

### Installation Issues

#### Issue 1: CUDA Not Found

**Symptoms:**
```
CMake Error: CUDA not found
nvcc: command not found
```

**Solutions:**
```bash
# Check CUDA installation
which nvcc
nvidia-smi

# If not found, install CUDA
# Or set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild
pip install -e . --force-reinstall
```

#### Issue 2: GCC/G++ Version Mismatch

**Symptoms:**
```
nvcc fatal: Unsupported gcc version!
```

**Solutions:**
```bash
# Check GCC version
gcc --version

# CUDA 12.x requires GCC <= 12
# Install compatible version
sudo apt install gcc-11 g++-11

# Use specific version
export CC=gcc-11
export CXX=g++-11

# Rebuild
pip install -e .
```

#### Issue 3: Out of Memory During Build

**Symptoms:**
```
c++: fatal error: Killed signal terminated program
```

**Solutions:**
```bash
# Limit parallel jobs
export MAX_JOBS=2

# Or add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Rebuild
pip install -e .
```

---

## Runtime Issues

### CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions:**

1. **Reduce GPU Memory Utilization:**
```python
llm = LLM(
    model="...",
    gpu_memory_utilization=0.7  # Lower from default 0.9
)
```

2. **Enable Swap Space:**
```python
llm = LLM(
    model="...",
    swap_space=8  # GB of CPU RAM for swapping
)
```

3. **Use Smaller Batch Size:**
```python
# Process prompts in smaller batches
for i in range(0, len(prompts), 10):
    batch = prompts[i:i+10]
    outputs = llm.generate(batch, params)
```

4. **Use Quantization:**
```python
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq"
)
```

5. **Enable Tensor Parallelism:**
```python
llm = LLM(
    model="...",
    tensor_parallel_size=2  # Use 2 GPUs
)
```

### GPU Not Detected

**Symptoms:**
```
RuntimeError: No GPU found
CUDA is not available
```

**Solutions:**
```bash
# Check GPU visibility
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES  # Make all GPUs visible
```

### Model Download Fails

**Symptoms:**
```
OSError: Can't load model from...
Connection timeout
```

**Solutions:**
```bash
# 1. Check internet connection
ping huggingface.co

# 2. Set HuggingFace token (for gated models)
export HF_TOKEN=your_token_here

# 3. Pre-download model
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir /data/models/llama-2-7b

# 4. Use local path
python -c "from vllm import LLM; llm = LLM(model='/data/models/llama-2-7b')"

# 5. Use offline mode
export HF_HUB_OFFLINE=1
```

---

## Performance Issues

### Slow Inference

**Diagnosis:**
```python
# Profile inference
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

start = time.time()
outputs = llm.generate(["Test"] * 100, SamplingParams(max_tokens=50))
duration = time.time() - start

throughput = 100 / duration
print(f"Throughput: {throughput:.2f} req/s")

# Expected: >100 req/s for opt-125m on modern GPU
# If much lower, investigate:
```

**Solutions:**

1. **Check GPU Utilization:**
```bash
nvidia-smi dmon -s pucvmet
# Should see high GPU utilization (>80%) during inference
```

2. **Increase Batch Size:**
```python
# Process more prompts together
outputs = llm.generate(prompts_batch_100, params)
```

3. **Enable CUDA Graphs (default):**
```python
llm = LLM(
    model="...",
    enforce_eager=False  # Use CUDA graphs (faster)
)
```

4. **Check for CPU Bottleneck:**
```python
import psutil
print(f"CPU Usage: {psutil.cpu_percent()}%")
# If 100%, might be tokenization bottleneck
```

### High Latency

**Diagnosis:**
```python
# Measure latency components
import torch
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# Measure total latency
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
outputs = llm.generate(["Test"], SamplingParams(max_tokens=100))
end.record()

torch.cuda.synchronize()
latency = start.elapsed_time(end)
print(f"Latency: {latency:.2f} ms")
```

**Solutions:**

1. **Reduce Max Tokens:**
```python
params = SamplingParams(max_tokens=50)  # Instead of 200
```

2. **Use Greedy Decoding:**
```python
params = SamplingParams(temperature=0.0)  # Faster than sampling
```

3. **Check Network Latency (for API):**
```bash
ping your-vllm-server
# High ping indicates network issue
```

---

## Debugging Workflow

### Step 1: Reproduce Issue

```python
# minimal_repro.py
"""
Minimal reproducible example
"""

from vllm import LLM, SamplingParams

try:
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(["Test"], SamplingParams(max_tokens=10))
    print("Success:", outputs[0].outputs[0].text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### Step 2: Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code
from vllm import LLM
llm = LLM(model="facebook/opt-125m")
```

### Step 3: Check System Resources

```bash
# GPU status
nvidia-smi

# Disk space
df -h

# Memory
free -h

# CPU
top
```

### Step 4: Isolate Component

```python
# Test each component separately

# 1. Test PyTorch CUDA
import torch
print(torch.cuda.is_available())
a = torch.randn(100, 100, device='cuda')
b = torch.randn(100, 100, device='cuda')
c = torch.matmul(a, b)
print("PyTorch CUDA works")

# 2. Test model loading
from vllm import LLM
llm = LLM(model="facebook/opt-125m")
print("Model loaded")

# 3. Test inference
from vllm import SamplingParams
outputs = llm.generate(["Test"], SamplingParams(max_tokens=5))
print("Inference works")
```

---

## Error Messages Reference

### Common Errors

```
Error: RuntimeError: CUDA error: invalid device ordinal
Cause: GPU index out of range
Fix: Check CUDA_VISIBLE_DEVICES and tensor_parallel_size

Error: ValueError: Cannot find a matched tokenizer
Cause: Model path incorrect or tokenizer missing
Fix: Verify model path and files exist

Error: AssertionError: KV cache size must be positive
Cause: Not enough GPU memory for KV cache
Fix: Reduce gpu_memory_utilization or use smaller model

Error: ImportError: cannot import name '_C'
Cause: vLLM C++ extensions not built
Fix: Rebuild with: pip install -e . --force-reinstall

Error: RuntimeError: CUDA driver version is insufficient
Cause: NVIDIA driver too old
Fix: Update NVIDIA driver

Error: torch.cuda.OutOfMemoryError: CUDA out of memory
Cause: Not enough GPU memory
Fix: See CUDA OOM solutions above
```

---

## Getting Help

### Before Asking

1. **Check documentation:** https://docs.vllm.ai
2. **Search GitHub issues:** https://github.com/vllm-project/vllm/issues
3. **Check FAQ:** https://docs.vllm.ai/en/latest/faq.html

### Creating Good Bug Reports

```markdown
**Environment:**
- vLLM version: (run `pip show vllm`)
- Python version:
- PyTorch version:
- CUDA version: (run `nvcc --version`)
- GPU: (run `nvidia-smi`)
- OS:

**Reproducible Example:**
```python
from vllm import LLM
llm = LLM(model="facebook/opt-125m")
# ... minimal code to reproduce issue
```

**Error Message:**
```
Full error traceback here
```

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened
```

---

## Practical Exercises

### Exercise 1.10.1: Diagnose OOM
1. Intentionally cause OOM
2. Use debugging techniques to diagnose
3. Apply fixes and verify

**Time:** 20 minutes

### Exercise 1.10.2: Performance Debug
1. Create slow inference scenario
2. Profile and identify bottleneck
3. Optimize and measure improvement

**Time:** 30 minutes

### Exercise 1.10.3: Create Bug Report
1. Find or create an issue
2. Write detailed bug report
3. Include minimal reproduction

**Time:** 20 minutes

---

## Key Takeaways

1. **Most issues** are CUDA/GPU related
2. **OOM** is most common runtime issue
3. **Logging** helps diagnose problems
4. **Minimal reproduction** essential for debugging
5. **Check system resources** first
6. **Search existing issues** before reporting

---

**Next:** 11_dependency_management.md
**Time:** 1.5 hours
**Resources:** GitHub issues, documentation, Discord
