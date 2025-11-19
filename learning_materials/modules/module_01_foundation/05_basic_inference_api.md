# Module 1.5: Basic Inference API

## Learning Objectives

By the end of this lesson, you will be able to:

1. Use the vLLM Python API for basic text generation
2. Understand the LLM class and its initialization parameters
3. Configure SamplingParams for different generation strategies
4. Run synchronous and asynchronous inference
5. Handle batch inference efficiently
6. Understand offline vs online inference modes
7. Measure basic inference performance

**Estimated Time:** 1.5-2 hours

---

## Table of Contents

1. [The LLM Class](#the-llm-class)
2. [SamplingParams Configuration](#samplingparams-configuration)
3. [Basic Inference Examples](#basic-inference-examples)
4. [Batch Inference](#batch-inference)
5. [Async Inference](#async-inference)
6. [Offline vs Online Mode](#offline-vs-online-mode)
7. [Request and Response Objects](#request-and-response-objects)
8. [Error Handling](#error-handling)
9. [Practical Exercises](#practical-exercises)
10. [Key Takeaways](#key-takeaways)

---

## The LLM Class

### Initialization

The `LLM` class is the primary interface for offline inference in vLLM.

**Source:** `/home/user/vllm-learn/vllm/__init__.py`

```python
from vllm import LLM, SamplingParams

# Basic initialization
llm = LLM(model="facebook/opt-125m")

# Full initialization with common parameters
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",          # Model name or path
    tokenizer=None,                             # Use same as model
    tokenizer_mode="auto",                      # auto, slow (for debugging)
    trust_remote_code=False,                    # Security: disable remote code
    tensor_parallel_size=1,                     # Number of GPUs for TP
    dtype="auto",                               # auto, float16, bfloat16, float32
    quantization=None,                          # awq, gptq, squeezellm
    revision=None,                              # Model revision/branch
    tokenizer_revision=None,                    # Tokenizer revision
    seed=0,                                     # Random seed
    gpu_memory_utilization=0.9,                 # GPU memory fraction to use
    swap_space=4,                               # CPU swap space in GB
    max_model_len=None,                         # Max sequence length (auto-detect)
    download_dir=None,                          # Model download directory
    load_format="auto",                         # auto, pt, safetensors
    disable_custom_all_reduce=False,            # Use custom all-reduce kernel
    enforce_eager=False,                        # Disable CUDA graphs
    max_num_batched_tokens=None,                # Max tokens per batch
    max_num_seqs=256,                           # Max concurrent sequences
    max_paddings=256,                           # Max padding tokens
    disable_log_stats=False,                    # Disable logging statistics
)
```

### Key Parameters Explained

```python
# ===== Model Selection =====
model="meta-llama/Llama-2-7b-hf"
# - HuggingFace model ID
# - Local path: "/data/models/llama-2-7b"
# - Auto-downloads from HuggingFace Hub

# ===== Parallelism =====
tensor_parallel_size=2
# - Number of GPUs for tensor parallelism
# - Splits model weights across GPUs
# - Required for large models (70B+)

# ===== Memory Management =====
gpu_memory_utilization=0.9
# - Fraction of GPU memory to use (0.0 - 1.0)
# - Lower = more headroom, fewer OOMs
# - Higher = more KV cache, higher throughput

swap_space=4
# - CPU RAM for swapping KV cache (GB)
# - Helps during memory pressure
# - Slight performance penalty

# ===== Precision =====
dtype="auto"
# - auto: FP16 for GPU, FP32 for CPU
# - float16: Half precision (faster, less accurate)
# - bfloat16: Brain float (Ampere+ GPUs)
# - float32: Full precision (slower, more accurate)

# ===== Performance =====
enforce_eager=False
# - False: Use CUDA graphs (faster)
# - True: Eager mode (easier debugging)

max_num_seqs=256
# - Maximum concurrent sequences
# - Higher = better throughput, more memory
```

### Model Loading Process

```python
# What happens during LLM initialization:

# 1. Download model (if not cached)
# 2. Load model weights to GPU
# 3. Initialize KV cache blocks
# 4. Compile/load CUDA kernels
# 5. Create worker processes (if multi-GPU)
# 6. Warm up (optional CUDA graphs)

# Example with progress tracking:
import logging
logging.basicConfig(level=logging.INFO)

llm = LLM(model="facebook/opt-125m")
# INFO: Initializing an LLM engine (v0.x.x) with config: ...
# INFO: # GPU blocks: 1234, # CPU blocks: 567
# INFO: Model weights loaded.
```

---

## SamplingParams Configuration

### Basic Parameters

**Source:** `/home/user/vllm-learn/vllm/sampling_params.py`

```python
from vllm import SamplingParams

# Greedy decoding (deterministic)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
)

# Sampling with temperature
sampling_params = SamplingParams(
    temperature=0.8,        # Randomness (0.0 = greedy, 1.0 = more random)
    top_p=0.95,            # Nucleus sampling
    top_k=50,              # Top-k sampling
    max_tokens=200,        # Maximum tokens to generate
    stop=["</s>", "\n\n"], # Stop sequences
    presence_penalty=0.0,  # Penalize repeated tokens
    frequency_penalty=0.0, # Penalize frequent tokens
    repetition_penalty=1.0,# Alternative repetition penalty
    length_penalty=1.0,    # Favor longer/shorter outputs
    n=1,                   # Number of completions per prompt
    best_of=None,          # Generate n*best_of, return best n
    use_beam_search=False, # Use beam search
    logprobs=None,         # Return log probabilities
    prompt_logprobs=None,  # Return prompt log probabilities
    skip_special_tokens=True,  # Skip special tokens in output
    spaces_between_special_tokens=True,
    min_tokens=0,          # Minimum tokens to generate
)
```

### Sampling Strategies

```python
# Strategy 1: Greedy Decoding (deterministic, best for factual)
greedy = SamplingParams(temperature=0.0, max_tokens=100)

# Strategy 2: Nucleus Sampling (balanced creativity)
nucleus = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

# Strategy 3: Top-K Sampling
topk = SamplingParams(temperature=1.0, top_k=50, max_tokens=200)

# Strategy 4: Beam Search (explore multiple paths)
beam = SamplingParams(
    use_beam_search=True,
    best_of=5,
    temperature=0.0,
    max_tokens=100
)

# Strategy 5: Multiple Completions
multi = SamplingParams(
    n=5,              # Generate 5 completions
    temperature=0.8,
    max_tokens=100
)

# Strategy 6: Constrained Generation
constrained = SamplingParams(
    temperature=0.0,
    max_tokens=50,
    stop=[".", "!", "?"],  # Stop at punctuation
    min_tokens=10,          # At least 10 tokens
)
```

### Parameter Effects Visualization

```
Temperature Effect:
──────────────────────────────────────────────────────
T=0.0  (greedy)    : "The capital of France is Paris."
T=0.3  (focused)   : "The capital of France is Paris."
T=0.7  (balanced)  : "The capital of France is Paris, known for..."
T=1.0  (creative)  : "The capital of France? Well, that's Paris, the..."
T=2.0  (very random): "The capital France is... wait, Paris! Right?"

Top-p (Nucleus) Effect:
──────────────────────────────────────────────────────
p=0.1  (very focused) : Most probable tokens only
p=0.5  (focused)      : Moderate diversity
p=0.9  (diverse)      : Wide range of tokens
p=1.0  (unrestricted) : All tokens considered

Top-k Effect:
──────────────────────────────────────────────────────
k=1    : Equivalent to greedy (best token only)
k=10   : Choose from top 10 tokens
k=50   : More diverse (common setting)
k=100  : Very diverse
```

---

## Basic Inference Examples

### Example 1: Simple Text Generation

```python
# simple_inference.py
"""
Basic text generation with vLLM
File: examples/simple_inference.py
"""

from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="facebook/opt-125m")

# Define prompts
prompts = [
    "The future of AI is",
    "Once upon a time",
    "The meaning of life is",
]

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate
outputs = llm.generate(prompts, sampling_params)

# Process outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("-" * 80)
```

**Output:**
```
Prompt: 'The future of AI is'
Generated: ' bright and full of possibilities. With advances in machine learning...'
────────────────────────────────────────────────────────────────────────────────
Prompt: 'Once upon a time'
Generated: ' in a land far away, there lived a young princess who...'
────────────────────────────────────────────────────────────────────────────────
```

### Example 2: With Detailed Output Inspection

```python
# detailed_output.py
"""
Inspect detailed output from vLLM
"""

from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
    logprobs=5,  # Return top 5 log probabilities
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated text: {output.outputs[0].text}")
    print(f"\nGenerated tokens: {output.outputs[0].token_ids}")
    print(f"Cumulative logprob: {output.outputs[0].cumulative_logprob}")
    print(f"Finish reason: {output.outputs[0].finish_reason}")  # 'stop', 'length'

    # Inspect per-token logprobs
    if output.outputs[0].logprobs:
        print("\nPer-token log probabilities:")
        for token_id, logprobs_dict in zip(
            output.outputs[0].token_ids,
            output.outputs[0].logprobs
        ):
            print(f"  Token {token_id}: {logprobs_dict}")
```

### Example 3: Multiple Completions

```python
# multiple_completions.py
"""
Generate multiple completions for each prompt
"""

from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompts = ["The best way to learn programming is"]

# Generate 5 different completions
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=50,
    n=5,  # Number of completions
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}\n")
    for i, completion in enumerate(output.outputs, 1):
        print(f"Completion {i}:")
        print(f"  {completion.text}")
        print(f"  Logprob: {completion.cumulative_logprob:.2f}")
        print()
```

### Example 4: Stopping Criteria

```python
# stopping_criteria.py
"""
Control generation stopping
"""

from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# Stop at specific sequences
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=200,
    stop=["\n\n", "END", "</s>"],  # Stop sequences
    min_tokens=10,  # Generate at least 10 tokens
)

prompts = ["Write a short paragraph about AI:\n"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
    print(f"\nFinish reason: {output.outputs[0].finish_reason}")
    # 'stop' if stopped by stop sequence
    # 'length' if stopped by max_tokens
```

---

## Batch Inference

### Efficient Batch Processing

```python
# batch_inference.py
"""
Efficient batch processing with vLLM
Demonstrates continuous batching advantage
"""

from vllm import LLM, SamplingParams
import time

llm = LLM(model="facebook/opt-125m")

# Large batch of prompts
prompts = [
    f"Question {i}: What is the meaning of life?"
    for i in range(100)
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50
)

# Measure batch inference time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

print(f"Processed {len(prompts)} prompts in {end - start:.2f} seconds")
print(f"Throughput: {len(prompts) / (end - start):.2f} requests/second")

# vLLM automatically batches and uses continuous batching
# No manual batching required!
```

### Streaming vs Batch

```python
# streaming_vs_batch.py
"""
Compare streaming (online) vs batch (offline) inference
"""

from vllm import LLM, SamplingParams
import time

llm = LLM(model="facebook/opt-125m")
prompts = ["Explain quantum computing in simple terms"] * 10
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

# Batch mode (offline)
start = time.time()
outputs = llm.generate(prompts, sampling_params)
batch_time = time.time() - start

print(f"Batch mode: {batch_time:.2f}s")
print(f"Throughput: {len(prompts) / batch_time:.2f} req/s")

# Note: For streaming (online) mode, use AsyncLLMEngine
# (covered in later modules)
```

---

## Async Inference

### Using AsyncLLMEngine

```python
# async_inference.py
"""
Asynchronous inference for API servers
Source: vllm/engine/async_llm_engine.py
"""

import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def main():
    # Initialize async engine
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        max_model_len=512
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    # Generate asynchronously
    prompts = ["Hello world", "How are you?", "What is AI?"]

    # Method 1: Concurrent requests
    tasks = []
    for i, prompt in enumerate(prompts):
        request_id = f"request-{i}"
        task = engine.generate(prompt, sampling_params, request_id)
        tasks.append(task)

    # Wait for all requests
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"Prompt: {result.prompt}")
        print(f"Output: {result.outputs[0].text}")
        print("-" * 80)

# Run
if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Async Inference

```python
# async_streaming.py
"""
Stream tokens as they're generated
"""

import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def stream_generate(engine, prompt, sampling_params, request_id):
    """Stream tokens from async engine"""
    results_generator = engine.generate(prompt, sampling_params, request_id)

    print(f"Prompt: {prompt}")
    print("Streaming output: ", end="", flush=True)

    async for request_output in results_generator:
        # Get latest token
        text = request_output.outputs[0].text
        print(text, end="", flush=True)

    print("\n" + "-" * 80)

async def main():
    engine_args = AsyncEngineArgs(model="facebook/opt-125m")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

    prompts = [
        "The future of AI",
        "Once upon a time",
    ]

    # Stream each prompt
    tasks = [
        stream_generate(engine, prompt, sampling_params, f"req-{i}")
        for i, prompt in enumerate(prompts)
    ]

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Offline vs Online Mode

### Comparison

```
┌────────────────────────────────────────────────────────────┐
│ Offline Mode (LLM class)                                   │
├────────────────────────────────────────────────────────────┤
│ Use Case: Batch processing, experimentation, benchmarking │
│ API: llm.generate(prompts, sampling_params)               │
│ Returns: All results at once                               │
│ Best for: High throughput, offline workloads              │
│ Example: Processing dataset, batch evaluation             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Online Mode (AsyncLLMEngine)                               │
├────────────────────────────────────────────────────────────┤
│ Use Case: API servers, interactive applications           │
│ API: engine.generate() - async generator                  │
│ Returns: Streaming tokens                                  │
│ Best for: Low latency, real-time applications             │
│ Example: Chatbots, API services                           │
└────────────────────────────────────────────────────────────┘
```

### When to Use Each

```python
# Use Offline Mode when:
# ✓ Processing large batches
# ✓ Running benchmarks
# ✓ Evaluating models
# ✓ Throughput is more important than latency

from vllm import LLM
llm = LLM(model="...")
outputs = llm.generate(prompts, sampling_params)

# Use Online Mode when:
# ✓ Building API servers
# ✓ Interactive applications
# ✓ Streaming responses
# ✓ Latency is critical

from vllm import AsyncLLMEngine
engine = AsyncLLMEngine.from_engine_args(args)
async for output in engine.generate(...):
    # Stream tokens to user
    yield output
```

---

## Request and Response Objects

### Request Output Structure

```python
# Type definitions from vllm/outputs.py

class CompletionOutput:
    """Single completion output"""
    index: int                      # Completion index (for n > 1)
    text: str                       # Generated text
    token_ids: List[int]            # Generated token IDs
    cumulative_logprob: float       # Sum of log probabilities
    logprobs: Optional[List[Dict]]  # Per-token logprobs
    finish_reason: Optional[str]    # 'stop', 'length', 'abort'

class RequestOutput:
    """Output for a single request"""
    request_id: str                 # Unique request ID
    prompt: str                     # Input prompt
    prompt_token_ids: List[int]     # Prompt token IDs
    prompt_logprobs: Optional[List] # Prompt logprobs (if requested)
    outputs: List[CompletionOutput] # List of completions
    finished: bool                  # Is generation finished?
```

### Accessing Output Fields

```python
# access_outputs.py
"""
Access all fields of request output
"""

from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
    logprobs=1,
    n=2,  # Generate 2 completions
)

outputs = llm.generate(["Hello, my name is"], sampling_params)

for request_output in outputs:
    print("=" * 80)
    print(f"Request ID: {request_output.request_id}")
    print(f"Prompt: {request_output.prompt}")
    print(f"Prompt tokens: {request_output.prompt_token_ids}")
    print(f"Finished: {request_output.finished}")
    print()

    for i, completion in enumerate(request_output.outputs):
        print(f"Completion {i + 1}:")
        print(f"  Text: {completion.text}")
        print(f"  Token IDs: {completion.token_ids}")
        print(f"  Cumulative logprob: {completion.cumulative_logprob:.4f}")
        print(f"  Finish reason: {completion.finish_reason}")
        print()
```

---

## Error Handling

### Common Errors

```python
# error_handling.py
"""
Handle common vLLM errors gracefully
"""

from vllm import LLM, SamplingParams
from vllm.utils import is_hip, is_cpu

def safe_initialize_llm(model_name, **kwargs):
    """Initialize LLM with error handling"""
    try:
        llm = LLM(model=model_name, **kwargs)
        print(f"✓ Successfully loaded {model_name}")
        return llm
    except ValueError as e:
        print(f"✗ Invalid configuration: {e}")
        return None
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"✗ Out of GPU memory. Try:")
            print("  - Reduce gpu_memory_utilization")
            print("  - Use smaller model")
            print("  - Enable tensor parallelism")
        else:
            print(f"✗ Runtime error: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def safe_generate(llm, prompts, sampling_params):
    """Generate with error handling"""
    try:
        outputs = llm.generate(prompts, sampling_params)
        return outputs
    except ValueError as e:
        print(f"✗ Invalid parameters: {e}")
        return None
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"✗ OOM during generation. Try smaller batch or max_tokens")
        else:
            print(f"✗ Runtime error during generation: {e}")
        return None

# Usage
llm = safe_initialize_llm(
    "facebook/opt-125m",
    gpu_memory_utilization=0.9
)

if llm:
    outputs = safe_generate(
        llm,
        prompts=["Hello"],
        sampling_params=SamplingParams(max_tokens=100)
    )
```

### Memory Management

```python
# memory_management.py
"""
Monitor and manage GPU memory during inference
"""

import torch
from vllm import LLM, SamplingParams

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Before loading model
print("Before loading model:")
print_gpu_memory()

# Load model
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.7)

print("\nAfter loading model:")
print_gpu_memory()

# Run inference
outputs = llm.generate(["Hello"] * 10, SamplingParams(max_tokens=50))

print("\nAfter inference:")
print_gpu_memory()

# Clean up
del llm
torch.cuda.empty_cache()

print("\nAfter cleanup:")
print_gpu_memory()
```

---

## Practical Exercises

### Exercise 1.5.1: Basic Inference

**Objective:** Run your first vLLM inference.

**Tasks:**
1. Initialize LLM with a small model (opt-125m or gpt2)
2. Generate text for 3 different prompts
3. Try different sampling parameters
4. Measure generation time

**Success Criteria:**
- Successful generation without errors
- Understanding of temperature and top_p effects

**Time:** 20 minutes

### Exercise 1.5.2: Batch Processing

**Objective:** Process a batch of prompts efficiently.

**Tasks:**
1. Create a list of 50+ prompts
2. Run batch inference
3. Measure throughput (requests/second)
4. Compare with processing one-by-one

**Time:** 30 minutes

### Exercise 1.5.3: Multiple Completions

**Objective:** Generate and compare multiple completions.

**Tasks:**
1. Generate 5 completions for a creative prompt
2. Compare diversity at different temperatures
3. Select best completion based on logprob

**Time:** 20 minutes

### Exercise 1.5.4: Custom Stopping

**Objective:** Control generation stopping behavior.

**Tasks:**
1. Use custom stop sequences
2. Set min_tokens and max_tokens
3. Observe different finish_reasons
4. Handle edge cases

**Time:** 20 minutes

---

## Key Takeaways

1. **LLM class** provides simple offline inference API
2. **SamplingParams** controls generation behavior (temperature, top_p, etc.)
3. **Batch inference** is automatic and efficient with vLLM
4. **temperature=0.0** gives deterministic output (greedy)
5. **AsyncLLMEngine** for online/streaming use cases
6. **Always handle errors** gracefully, especially OOM
7. **Monitoring GPU memory** helps prevent issues

---

## Quick Reference

```python
# Minimal working example
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m")
outputs = llm.generate(
    prompts=["Hello, world!"],
    sampling_params=SamplingParams(temperature=0.0, max_tokens=20)
)
print(outputs[0].outputs[0].text)
```

---

## Next Steps

In the next lesson, **06_model_loading_caching.md**, we'll cover:
- HuggingFace Hub integration
- Model caching and storage
- Loading custom models
- Weight formats (safetensors, pytorch)

---

**Estimated Completion Time:** 1.5-2 hours
**Difficulty:** Beginner to Intermediate
**Prerequisites:** Python basics, understanding of LLMs
