# Module 1.6: Model Loading and Caching

## Learning Objectives

1. Understand HuggingFace Hub integration
2. Configure model caching effectively
3. Load custom and quantized models
4. Manage disk space for models
5. Use offline models
6. Understand weight formats

**Estimated Time:** 1.5 hours

---

## HuggingFace Hub Integration

### Automatic Downloads

```python
from vllm import LLM

# Automatically downloads from HuggingFace
llm = LLM(model="facebook/opt-125m")

# Process:
# 1. Check cache: ~/.cache/huggingface/
# 2. Download if not found
# 3. Cache for future use
```

### Model Naming

```python
# Format: "organization/model-name"
"meta-llama/Llama-2-7b-hf"
"mistralai/Mistral-7B-v0.1"
"facebook/opt-1.3b"

# Local path
"/data/models/my-model"
```

---

## Cache Configuration

### Default Location

```bash
# Default HuggingFace cache
~/.cache/huggingface/hub/

# Structure:
~/.cache/huggingface/hub/
├── models--meta-llama--Llama-2-7b-hf/
│   ├── snapshots/
│   └── refs/
└── models--facebook--opt-125m/
```

### Custom Cache Location

```bash
# Set environment variable
export HF_HOME=/data/huggingface
export TRANSFORMERS_CACHE=/data/huggingface/transformers

# Or in Python
import os
os.environ["HF_HOME"] = "/data/huggingface"

from vllm import LLM
llm = LLM(model="facebook/opt-125m")
```

### Cache Management

```bash
#!/bin/bash
# Show cache size
du -sh ~/.cache/huggingface

# List models
ls ~/.cache/huggingface/hub/

# Clean specific model
rm -rf ~/.cache/huggingface/hub/models--facebook--opt-125m
```

---

## Weight Formats

### Supported Formats

```
SafeTensors (.safetensors) - Recommended
PyTorch (.bin, .pth) - Standard
GPTQ (.safetensors) - Quantized
AWQ (.pt) - Quantized
```

### Loading Formats

```python
# Auto-detect (default)
llm = LLM(model="model-name", load_format="auto")

# Force SafeTensors
llm = LLM(model="model-name", load_format="safetensors")

# Force PyTorch
llm = LLM(model="model-name", load_format="pt")
```

---

## Custom Models

### Local Path

```python
# Load from directory
llm = LLM(model="/data/models/my-custom-model")

# Required files:
# - config.json
# - model weights (.bin or .safetensors)
# - tokenizer files
```

### Fine-tuned Models

```python
# From HuggingFace
llm = LLM(model="username/my-finetuned-model")

# From local path
llm = LLM(model="/path/to/finetuned")
```

---

## Quantized Models

### GPTQ

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="half"
)
```

### AWQ

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="half"
)
```

---

## Offline Loading

### Pre-download

```bash
# Using huggingface-cli
pip install huggingface_hub[cli]

huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir /data/models/llama-2-7b
```

### Use Offline

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"

llm = LLM(model="/data/models/llama-2-7b")
```

---

## Storage Management

### Monitor Space

```bash
# Check cache size
du -sh ~/.cache/huggingface/

# List by size
du -sh ~/.cache/huggingface/hub/models--* | sort -h
```

### Clean Cache

```python
# clean_cache.py
import shutil
from pathlib import Path

cache = Path.home() / ".cache/huggingface/hub"

# Remove specific model
model_dir = cache / "models--facebook--opt-125m"
if model_dir.exists():
    shutil.rmtree(model_dir)
    print(f"Removed {model_dir}")
```

---

## Practical Exercises

### Exercise 1.6.1: Download and Cache
1. Download opt-125m
2. Find in cache
3. Verify no re-download on second load

### Exercise 1.6.2: Offline Setup
1. Pre-download model
2. Set HF_HUB_OFFLINE=1
3. Load from local path

### Exercise 1.6.3: Quantized Model
1. Load GPTQ model
2. Compare memory vs FP16
3. Benchmark speed

---

## Key Takeaways

1. Models auto-download from HuggingFace Hub
2. Cache location: ~/.cache/huggingface/
3. SafeTensors is preferred format
4. Quantized models save memory
5. Offline loading requires pre-download
6. Monitor and manage cache size

---

**Next:** 07_sampling_parameters.md
**Time:** 1.5 hours
**Source:** `/home/user/vllm-learn/vllm/model_executor/model_loader/`
