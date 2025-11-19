# Performance Tuning for Quantized LLM Inference

## Table of Contents
1. [Introduction](#introduction)
2. [vLLM Configuration Parameters](#vllm-configuration-parameters)
3. [Memory Optimization](#memory-optimization)
4. [Throughput Optimization](#throughput-optimization)
5. [Latency Optimization](#latency-optimization)
6. [Profiling and Debugging](#profiling-and-debugging)

---

## Introduction

### Performance Tuning Goals

Optimizing quantized LLM inference involves balancing three dimensions:

```
Performance Triangle:
    Throughput
      /  \
     /    \
  Latency -- Memory

Goals:
- Maximize throughput (tokens/second)
- Minimize latency (milliseconds)
- Minimize memory footprint (GB)
```

**Common Trade-offs:**
- Higher batch size → Higher throughput, higher latency
- Larger KV cache → Lower latency, higher memory
- More parallelism → Higher throughput, more memory

---

## vLLM Configuration Parameters

### Critical Parameters

```python
from vllm import LLM, SamplingParams

# Comprehensive configuration example
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",

    # GPU memory management
    gpu_memory_utilization=0.9,        # Use 90% of GPU memory
    swap_space=4,                       # 4GB CPU swap space
    max_model_len=2048,                 # Max sequence length

    # Parallelism
    tensor_parallel_size=1,             # Number of GPUs for tensor parallelism
    pipeline_parallel_size=1,           # Pipeline parallelism degree

    # KV cache
    block_size=16,                      # KV cache block size
    max_num_batched_tokens=None,        # Max tokens per iteration
    max_num_seqs=256,                   # Max concurrent sequences

    # Performance
    enable_chunked_prefill=False,       # Chunked prefill for long contexts
    enable_prefix_caching=True,         # Prefix caching
    disable_custom_all_reduce=False,    # Custom all-reduce

    # dtype
    dtype="auto",                       # Data type (auto, float16, bfloat16)
    kv_cache_dtype="auto",              # KV cache dtype

    # Other
    seed=42,
    trust_remote_code=False,
)
```

### Parameter Impact Matrix

| Parameter | Throughput | Latency | Memory |
|-----------|------------|---------|--------|
| `gpu_memory_utilization` ↑ | ↑ | → | ↑ |
| `max_num_seqs` ↑ | ↑ | ↑ | ↑ |
| `max_model_len` ↓ | → | → | ↓ |
| `block_size` ↓ | → | → | ↓ |
| `tensor_parallel_size` ↑ | ↑ | → | → |
| `enable_prefix_caching` | ↑ | ↓ | ↑ |

---

## Memory Optimization

### 1. GPU Memory Utilization

```python
def find_optimal_gpu_memory_utilization(model_name, quantization="awq"):
    """
    Find optimal GPU memory utilization setting.

    Binary search for maximum utilization without OOM.
    """
    low, high = 0.5, 0.99
    optimal = low

    while high - low > 0.05:
        mid = (low + high) / 2

        try:
            print(f"Testing gpu_memory_utilization={mid:.2f}...")

            llm = LLM(
                model=model_name,
                quantization=quantization,
                gpu_memory_utilization=mid,
            )

            # Test with actual load
            prompts = ["Test"] * 100
            params = SamplingParams(max_tokens=10)
            llm.generate(prompts, params)

            # Success - try higher
            optimal = mid
            low = mid

            # Cleanup
            del llm
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # OOM - try lower
            high = mid
            torch.cuda.empty_cache()

    print(f"Optimal gpu_memory_utilization: {optimal:.2f}")
    return optimal


# Example
optimal_util = find_optimal_gpu_memory_utilization("TheBloke/Llama-2-7B-AWQ")
```

### 2. KV Cache Optimization

**KV Cache Size Formula:**
```
KV_cache_size = 2 * num_layers * hidden_dim * num_tokens * precision_bytes
```

For Llama-2-7B with INT8 KV cache:
```
KV = 2 * 32 * 4096 * num_tokens * 1 byte
   = 262 KB per token
   = 268 MB for 1024 tokens per sequence
```

**Optimization:**

```python
def optimize_kv_cache_config(
    model_name,
    target_concurrent_requests,
    avg_context_length,
    avg_generation_length,
):
    """
    Compute optimal KV cache configuration.

    Args:
        model_name: Model to use
        target_concurrent_requests: Desired concurrency
        avg_context_length: Average context length
        avg_generation_length: Average output length

    Returns:
        Recommended configuration
    """
    # Load model to get config
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)

    # Calculate KV cache per token
    kv_per_token_bytes = (
        2 *  # K and V
        config.num_hidden_layers *
        config.hidden_size *
        1  # INT8 = 1 byte (or 2 for FP16)
    )

    # Total tokens needed
    avg_tokens_per_request = avg_context_length + avg_generation_length
    total_tokens = target_concurrent_requests * avg_tokens_per_request

    # Total KV cache memory
    total_kv_cache_gb = (total_tokens * kv_per_token_bytes) / (1024**3)

    # Get available GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Reserve memory for model weights (approximate)
    model_size_gb = config.num_parameters * 0.5 / (1024**3)  # INT4 = 0.5 bytes

    # Available for KV cache
    available_for_kv = gpu_memory_gb * 0.9 - model_size_gb

    # Check if feasible
    if total_kv_cache_gb > available_for_kv:
        max_concurrent = int(available_for_kv / (avg_tokens_per_request * kv_per_token_bytes / (1024**3)))
        print(f"⚠ Cannot support {target_concurrent_requests} concurrent requests")
        print(f"  Max possible: {max_concurrent}")
        target_concurrent_requests = max_concurrent

    # Compute configuration
    max_num_seqs = target_concurrent_requests
    max_model_len = avg_context_length + avg_generation_length

    return {
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.9,
        "kv_cache_memory_gb": total_kv_cache_gb,
    }


# Example
config = optimize_kv_cache_config(
    "TheBloke/Llama-2-7B-AWQ",
    target_concurrent_requests=64,
    avg_context_length=512,
    avg_generation_length=128,
)
print(f"Recommended config: {config}")
```

### 3. Swap Space

```python
# Enable CPU swap for overflow
llm = LLM(
    model="TheBloke/Llama-2-13B-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.9,
    swap_space=8,  # 8GB CPU swap
)

# Trade-off:
# - Allows higher concurrency
# - Slower when swapping occurs
# - Useful for bursty workloads
```

---

## Throughput Optimization

### 1. Batch Size Tuning

```python
def find_optimal_batch_size(llm, test_prompts, max_batch_size=256):
    """
    Find optimal batch size for maximum throughput.

    Args:
        llm: vLLM engine
        test_prompts: List of test prompts
        max_batch_size: Maximum batch size to test

    Returns:
        Optimal batch size and throughput
    """
    import time

    results = []
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_sizes = [b for b in batch_sizes if b <= max_batch_size]

    params = SamplingParams(temperature=0.8, max_tokens=100)

    for batch_size in batch_sizes:
        prompts = test_prompts[:batch_size]

        try:
            # Warmup
            llm.generate(prompts, params)

            # Benchmark
            start = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed = time.perf_counter() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed

            results.append({
                "batch_size": batch_size,
                "throughput": throughput,
                "latency": elapsed / batch_size,
            })

            print(f"Batch size {batch_size:3d}: {throughput:7.2f} tok/s, "
                  f"{elapsed/batch_size*1000:6.2f} ms/req")

        except Exception as e:
            print(f"Batch size {batch_size:3d}: FAILED ({e})")
            break

    # Find optimal
    optimal = max(results, key=lambda x: x["throughput"])
    print(f"\nOptimal batch size: {optimal['batch_size']} "
          f"({optimal['throughput']:.2f} tok/s)")

    return optimal


# Example usage
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")
test_prompts = ["Write a story about " + str(i) for i in range(256)]
optimal = find_optimal_batch_size(llm, test_prompts)
```

### 2. Continuous Batching

vLLM uses continuous batching by default. Tune with:

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",

    # Control batching behavior
    max_num_seqs=128,              # Max sequences in batch
    max_num_batched_tokens=8192,   # Max tokens per iteration
)

# Continuous batching automatically:
# - Adds new requests to batch when space available
# - Removes completed sequences
# - Maximizes GPU utilization
```

### 3. Prefix Caching

```python
# Enable prefix caching for shared prompts
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    enable_prefix_caching=True,
)

# Benefit: Shared prompt prefixes computed once
# Example: Chat with system prompt
system_prompt = "You are a helpful assistant. " * 100  # Long shared prefix

prompts = [
    system_prompt + "Question 1",
    system_prompt + "Question 2",
    system_prompt + "Question 3",
]

# First request: Full computation
# Subsequent requests: Reuse cached system prompt computation
outputs = llm.generate(prompts, params)
```

---

## Latency Optimization

### 1. Time to First Token (TTFT)

**TTFT = Prefill time + Scheduling overhead**

```python
def measure_ttft(llm, prompts, num_runs=10):
    """Measure time to first token."""
    import time

    params = SamplingParams(max_tokens=100, temperature=0.8)
    ttfts = []

    for _ in range(num_runs):
        start = time.perf_counter()

        # Use streaming to detect first token
        outputs = llm.generate(prompts, params, use_streaming=True)

        # Record time to first token
        first_token_time = None
        for output in outputs:
            if output.outputs[0].token_ids:  # First token arrived
                first_token_time = time.perf_counter() - start
                break

        if first_token_time:
            ttfts.append(first_token_time)

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else float('inf')
    print(f"Average TTFT: {avg_ttft*1000:.2f} ms")

    return avg_ttft


# Optimization strategies:
# 1. Reduce max_model_len (shorter context)
# 2. Use chunked prefill for very long contexts
# 3. Smaller batch size for latency-sensitive requests
```

### 2. Chunked Prefill

For long contexts (>4096 tokens):

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    max_model_len=8192,
    enable_chunked_prefill=True,  # Split prefill into chunks
)

# Benefit:
# - Reduces TTFT for long contexts
# - Allows interleaving with decode
# - Better latency for concurrent requests
```

### 3. Speculative Decoding

```python
# Use draft model for speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="awq",
    speculative_model="meta-llama/Llama-2-7b-hf",  # Smaller draft model
    num_speculative_tokens=5,  # Speculate 5 tokens ahead
)

# Benefit:
# - 1.5-2x speedup for long generation
# - Same output quality (acceptance rate dependent)
```

---

## Profiling and Debugging

### 1. Built-in vLLM Stats

```python
from vllm import LLM, SamplingParams

llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")

# Generate with stats
prompts = ["Test prompt"] * 32
params = SamplingParams(max_tokens=100)
outputs = llm.generate(prompts, params)

# Get engine stats
stats = llm.llm_engine.get_stats()

print("Engine Statistics:")
print(f"  Running requests: {stats.num_running_requests}")
print(f"  Waiting requests: {stats.num_waiting_requests}")
print(f"  GPU cache usage:  {stats.gpu_cache_usage*100:.1f}%")
print(f"  CPU cache usage:  {stats.cpu_cache_usage*100:.1f}%")
```

### 2. PyTorch Profiler

```python
import torch.profiler as profiler

llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")
prompts = ["Test"] * 16
params = SamplingParams(max_tokens=50)

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    llm.generate(prompts, params)

# Print summary
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# Export for Chrome trace viewer
prof.export_chrome_trace("vllm_trace.json")
```

### 3. Custom Metrics Collection

```python
class PerformanceMonitor:
    """Monitor vLLM performance metrics."""

    def __init__(self):
        self.metrics = []

    def log_request(self, prompt, output, start_time, end_time):
        """Log metrics for one request."""
        import time

        # Compute metrics
        latency = end_time - start_time
        prompt_tokens = len(prompt.split())  # Approximate
        output_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + output_tokens
        throughput = total_tokens / latency

        # GPU memory
        memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)

        metric = {
            "timestamp": time.time(),
            "latency_ms": latency * 1000,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "throughput_tok_s": throughput,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
        }

        self.metrics.append(metric)

        return metric

    def get_summary(self):
        """Get summary statistics."""
        if not self.metrics:
            return {}

        import numpy as np

        latencies = [m["latency_ms"] for m in self.metrics]
        throughputs = [m["throughput_tok_s"] for m in self.metrics]

        return {
            "num_requests": len(self.metrics),
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_throughput_tok_s": np.mean(throughputs),
            "total_tokens": sum(m["prompt_tokens"] + m["output_tokens"] for m in self.metrics),
        }


# Usage
monitor = PerformanceMonitor()

for prompt in prompts:
    start = time.perf_counter()
    output = llm.generate([prompt], params)[0]
    end = time.perf_counter()

    monitor.log_request(prompt, output, start, end)

summary = monitor.get_summary()
print(f"Performance Summary:")
print(f"  Requests:     {summary['num_requests']}")
print(f"  Avg Latency:  {summary['avg_latency_ms']:.2f} ms")
print(f"  P95 Latency:  {summary['p95_latency_ms']:.2f} ms")
print(f"  Throughput:   {summary['avg_throughput_tok_s']:.2f} tok/s")
```

---

## Summary

### Performance Tuning Checklist

**Memory:**
- ✓ Set `gpu_memory_utilization` to 0.85-0.95
- ✓ Optimize `max_model_len` based on use case
- ✓ Configure swap space for bursty loads
- ✓ Use INT8/FP8 KV cache if available

**Throughput:**
- ✓ Find optimal batch size (typically 32-128)
- ✓ Enable prefix caching for shared prompts
- ✓ Use continuous batching (default in vLLM)
- ✓ Consider tensor parallelism for large models

**Latency:**
- ✓ Reduce batch size for latency-sensitive workloads
- ✓ Use chunked prefill for long contexts
- ✓ Enable speculative decoding if appropriate
- ✓ Optimize TTFT

**Monitoring:**
- ✓ Track throughput, latency, memory continuously
- ✓ Profile with PyTorch profiler
- ✓ Monitor cache hit rates
- ✓ Set up alerts for degradation

---

## References

1. vLLM Performance Tuning Guide: https://docs.vllm.ai/en/latest/performance/
2. "Efficient Memory Management for Large Language Model Serving with PagedAttention"
3. NVIDIA Nsight Systems Documentation

---

**Module 6.9 Complete** • Next: [Batch Size Optimization](10_batch_size_optimization.md)
