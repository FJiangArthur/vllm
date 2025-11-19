# Batch Size Optimization for Quantized LLMs

## Table of Contents
1. [Introduction to Batching](#introduction-to-batching)
2. [Batch Size Impact Analysis](#batch-size-impact-analysis)
3. [Optimal Batch Size Selection](#optimal-batch-size-selection)
4. [Dynamic Batching Strategies](#dynamic-batching-strategies)
5. [Memory-Constrained Optimization](#memory-constrained-optimization)
6. [Production Recommendations](#production-recommendations)

---

## Introduction to Batching

### What is Batch Size?

**Batch size** is the number of sequences processed simultaneously in one forward pass.

```
Single Request:  [Prompt 1] → Model → [Output 1]

Batched (size=4): [Prompt 1]
                  [Prompt 2]  → Model → [Output 1]
                  [Prompt 3]          → [Output 2]
                  [Prompt 4]          → [Output 3]
                                      → [Output 4]
```

**Benefits:**
- Better GPU utilization
- Higher throughput (tokens/second)
- Amortized fixed costs

**Trade-offs:**
- Higher latency per request
- More memory usage
- Potential queueing delays

---

## Batch Size Impact Analysis

### 1. GPU Utilization

```python
import torch
import time
from vllm import LLM, SamplingParams

def measure_gpu_utilization(model_name, batch_sizes=[1, 2, 4, 8, 16, 32, 64]):
    """
    Measure GPU utilization across different batch sizes.

    Args:
        model_name: Model to benchmark
        batch_sizes: List of batch sizes to test

    Returns:
        Utilization metrics per batch size
    """
    llm = LLM(model=model_name, quantization="awq")
    params = SamplingParams(temperature=0.8, max_tokens=100)

    results = []

    for batch_size in batch_sizes:
        # Create prompts
        prompts = [f"Tell me about topic {i}" for i in range(batch_size)]

        # Warmup
        llm.generate(prompts[:min(batch_size, 4)], params)

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = llm.generate(prompts, params)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Metrics
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed
        latency_per_request = elapsed / batch_size
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # GPU utilization (approximate)
        peak_throughput = 450  # Theoretical max for this model
        utilization = (throughput / peak_throughput) * 100

        results.append({
            "batch_size": batch_size,
            "throughput_tok_s": throughput,
            "latency_ms": latency_per_request * 1000,
            "memory_gb": memory_gb,
            "gpu_utilization_pct": utilization,
        })

        print(f"Batch {batch_size:3d}: {throughput:7.2f} tok/s, "
              f"{latency_per_request*1000:6.2f} ms/req, "
              f"{memory_gb:5.2f} GB, "
              f"{utilization:5.1f}% util")

        # Reset memory tracking
        torch.cuda.reset_peak_memory_stats()

    return results


# Example output:
# Batch   1:  120.34 tok/s,  21.47 ms/req,  4.23 GB,  26.7% util
# Batch   2:  234.12 tok/s,  22.15 ms/req,  4.45 GB,  52.0% util
# Batch   4:  421.56 tok/s,  24.83 ms/req,  5.12 GB,  93.7% util
# Batch   8:  697.34 tok/s,  30.21 ms/req,  6.78 GB,  99.2% util <- Saturated
# Batch  16:  701.45 tok/s,  60.12 ms/req,  9.45 GB,  99.5% util <- No gain
```

### 2. Throughput vs Latency Curve

```python
import matplotlib.pyplot as plt

def plot_throughput_latency_curve(results):
    """
    Visualize throughput-latency trade-off.

    Args:
        results: Output from measure_gpu_utilization
    """
    batch_sizes = [r["batch_size"] for r in results]
    throughputs = [r["throughput_tok_s"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput vs batch size
    ax1.plot(batch_sizes, throughputs, marker='o', linewidth=2)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (tokens/s)')
    ax1.set_title('Throughput vs Batch Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Latency vs batch size
    ax2.plot(batch_sizes, latencies, marker='o', linewidth=2, color='orange')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency vs Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig('batch_size_analysis.png', dpi=150)
    plt.show()

    # Find "knee" of curve (best throughput/latency trade-off)
    # Using simple heuristic: max throughput with latency < 50ms
    acceptable = [r for r in results if r["latency_ms"] < 50]
    if acceptable:
        optimal = max(acceptable, key=lambda x: x["throughput_tok_s"])
        print(f"\nRecommended batch size: {optimal['batch_size']}")
        print(f"  Throughput: {optimal['throughput_tok_s']:.2f} tok/s")
        print(f"  Latency:    {optimal['latency_ms']:.2f} ms")
    else:
        print("\nNo batch size meets latency requirement (<50ms)")
```

### 3. Memory Scaling

```python
def analyze_memory_scaling(model_name, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]):
    """
    Analyze how memory scales with batch size.

    Args:
        model_name: Model to analyze
        batch_sizes: Batch sizes to test

    Returns:
        Memory usage analysis
    """
    llm = LLM(model=model_name, quantization="awq", gpu_memory_utilization=0.95)
    params = SamplingParams(max_tokens=512)  # Long generation

    memory_results = []

    for batch_size in batch_sizes:
        prompts = [f"Prompt {i}" for i in range(batch_size)]

        try:
            torch.cuda.reset_peak_memory_stats()

            outputs = llm.generate(prompts, params)

            memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            # Estimate KV cache memory
            # For Llama-2-7B: ~0.26 MB per token
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            estimated_kv_cache = (total_tokens * 0.26) / 1024  # GB

            memory_results.append({
                "batch_size": batch_size,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "estimated_kv_cache_gb": estimated_kv_cache,
            })

            print(f"Batch {batch_size:3d}: "
                  f"Allocated={memory_allocated:.2f} GB, "
                  f"Reserved={memory_reserved:.2f} GB, "
                  f"KV Cache≈{estimated_kv_cache:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print(f"Batch {batch_size:3d}: OUT OF MEMORY")
            break

    return memory_results


# Example analysis
memory_results = analyze_memory_scaling("TheBloke/Llama-2-7B-AWQ")

# Compute memory efficiency
for i, result in enumerate(memory_results[1:], 1):
    prev = memory_results[i-1]
    delta_memory = result["memory_allocated_gb"] - prev["memory_allocated_gb"]
    delta_batch = result["batch_size"] - prev["batch_size"]
    memory_per_seq = delta_memory / delta_batch

    print(f"Batch {prev['batch_size']} → {result['batch_size']}: "
          f"+{delta_memory:.2f} GB ({memory_per_seq:.3f} GB/seq)")
```

---

## Optimal Batch Size Selection

### 1. Single Objective Optimization

#### Maximize Throughput

```python
def find_max_throughput_batch_size(model_name, max_batch_size=256):
    """
    Find batch size that maximizes throughput.

    Args:
        model_name: Model to optimize
        max_batch_size: Maximum batch size to try

    Returns:
        Optimal batch size for throughput
    """
    results = measure_gpu_utilization(
        model_name,
        batch_sizes=[2**i for i in range(int(math.log2(max_batch_size))+1)]
    )

    # Find maximum throughput
    optimal = max(results, key=lambda x: x["throughput_tok_s"])

    print(f"\nMax Throughput Configuration:")
    print(f"  Batch Size:  {optimal['batch_size']}")
    print(f"  Throughput:  {optimal['throughput_tok_s']:.2f} tok/s")
    print(f"  Latency:     {optimal['latency_ms']:.2f} ms")
    print(f"  Memory:      {optimal['memory_gb']:.2f} GB")

    return optimal["batch_size"]
```

#### Minimize Latency

```python
def find_min_latency_batch_size(model_name, max_latency_ms=50):
    """
    Find maximum batch size meeting latency constraint.

    Args:
        model_name: Model to optimize
        max_latency_ms: Maximum acceptable latency

    Returns:
        Optimal batch size for latency
    """
    results = measure_gpu_utilization(model_name)

    # Filter by latency constraint
    acceptable = [r for r in results if r["latency_ms"] <= max_latency_ms]

    if not acceptable:
        print(f"⚠ No batch size meets latency requirement (<{max_latency_ms}ms)")
        print(f"  Minimum latency: {min(r['latency_ms'] for r in results):.2f} ms")
        return 1

    # Among acceptable, choose highest throughput
    optimal = max(acceptable, key=lambda x: x["throughput_tok_s"])

    print(f"\nLatency-Constrained Configuration (<{max_latency_ms}ms):")
    print(f"  Batch Size:  {optimal['batch_size']}")
    print(f"  Throughput:  {optimal['throughput_tok_s']:.2f} tok/s")
    print(f"  Latency:     {optimal['latency_ms']:.2f} ms")

    return optimal["batch_size"]
```

### 2. Multi-Objective Optimization

```python
def find_pareto_optimal_batch_size(results):
    """
    Find Pareto-optimal batch sizes (throughput vs latency).

    A configuration is Pareto-optimal if no other configuration
    has both higher throughput AND lower latency.

    Args:
        results: Batch size benchmark results

    Returns:
        List of Pareto-optimal configurations
    """
    pareto_optimal = []

    for i, r1 in enumerate(results):
        is_dominated = False

        for j, r2 in enumerate(results):
            if i != j:
                # r2 dominates r1 if it has higher throughput AND lower latency
                if (r2["throughput_tok_s"] >= r1["throughput_tok_s"] and
                    r2["latency_ms"] <= r1["latency_ms"] and
                    (r2["throughput_tok_s"] > r1["throughput_tok_s"] or
                     r2["latency_ms"] < r1["latency_ms"])):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_optimal.append(r1)

    # Sort by batch size
    pareto_optimal.sort(key=lambda x: x["batch_size"])

    print(f"\nPareto-Optimal Configurations:")
    print(f"{'Batch':>6} {'Throughput':>12} {'Latency':>10} {'Memory':>8}")
    print("-" * 42)
    for config in pareto_optimal:
        print(f"{config['batch_size']:>6} "
              f"{config['throughput_tok_s']:>12.2f} "
              f"{config['latency_ms']:>10.2f} "
              f"{config['memory_gb']:>8.2f}")

    return pareto_optimal
```

---

## Dynamic Batching Strategies

### 1. Adaptive Batch Size

```python
class AdaptiveBatchSizer:
    """
    Dynamically adjust batch size based on queue length and latency.
    """

    def __init__(
        self,
        min_batch_size=1,
        max_batch_size=64,
        target_latency_ms=50,
        adjustment_interval=10,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.adjustment_interval = adjustment_interval

        self.current_batch_size = min_batch_size
        self.request_count = 0
        self.latency_history = []

    def get_batch_size(self, queue_length):
        """
        Determine batch size based on current conditions.

        Args:
            queue_length: Number of requests waiting

        Returns:
            Recommended batch size
        """
        # If queue is empty, use minimum
        if queue_length == 0:
            return self.min_batch_size

        # Adjust based on latency feedback
        if len(self.latency_history) >= self.adjustment_interval:
            avg_latency = sum(self.latency_history) / len(self.latency_history)

            if avg_latency < self.target_latency_ms * 0.8:
                # Latency is good, increase batch size
                self.current_batch_size = min(
                    int(self.current_batch_size * 1.5),
                    self.max_batch_size,
                    queue_length
                )
            elif avg_latency > self.target_latency_ms:
                # Latency too high, decrease batch size
                self.current_batch_size = max(
                    int(self.current_batch_size * 0.7),
                    self.min_batch_size
                )

            # Reset history
            self.latency_history = []

        # Don't batch more than available
        batch_size = min(self.current_batch_size, queue_length)

        return batch_size

    def record_latency(self, latency_ms):
        """Record observed latency for adaptation."""
        self.latency_history.append(latency_ms)
        self.request_count += 1


# Usage example
adaptive_batcher = AdaptiveBatchSizer(
    min_batch_size=1,
    max_batch_size=64,
    target_latency_ms=50,
)

# Simulated request processing
request_queue = []

while requests_incoming:
    # Get batch size based on queue
    batch_size = adaptive_batcher.get_batch_size(len(request_queue))

    # Process batch
    batch = request_queue[:batch_size]
    request_queue = request_queue[batch_size:]

    start = time.perf_counter()
    outputs = llm.generate(batch, params)
    latency_ms = (time.perf_counter() - start) / len(batch) * 1000

    # Provide feedback
    adaptive_batcher.record_latency(latency_ms)
```

### 2. Priority-Based Batching

```python
class PriorityBatcher:
    """
    Batch requests with priority awareness.
    """

    def __init__(self, max_batch_size=64):
        self.max_batch_size = max_batch_size
        self.high_priority_queue = []
        self.normal_priority_queue = []

    def add_request(self, request, priority="normal"):
        """Add request to appropriate queue."""
        if priority == "high":
            self.high_priority_queue.append(request)
        else:
            self.normal_priority_queue.append(request)

    def get_batch(self):
        """
        Form batch prioritizing high-priority requests.

        Returns:
            Batch of requests
        """
        batch = []

        # Always include high-priority requests first
        while self.high_priority_queue and len(batch) < self.max_batch_size:
            batch.append(self.high_priority_queue.pop(0))

        # Fill remaining slots with normal priority
        while self.normal_priority_queue and len(batch) < self.max_batch_size:
            batch.append(self.normal_priority_queue.pop(0))

        return batch

    def is_empty(self):
        """Check if both queues are empty."""
        return not self.high_priority_queue and not self.normal_priority_queue
```

---

## Memory-Constrained Optimization

### Maximum Batch Size Under Memory Limit

```python
def find_max_batch_under_memory_limit(
    model_name,
    max_memory_gb,
    avg_output_tokens=256,
):
    """
    Find maximum batch size that fits in memory limit.

    Args:
        model_name: Model name
        max_memory_gb: Maximum GPU memory to use (GB)
        avg_output_tokens: Average output length

    Returns:
        Maximum feasible batch size
    """
    from transformers import AutoConfig

    # Get model config
    config = AutoConfig.from_pretrained(model_name)

    # Estimate model weight memory (INT4 quantized)
    model_params = getattr(config, 'num_parameters', None)
    if model_params is None:
        model_params = config.hidden_size * config.num_hidden_layers * 12000  # Approx

    model_memory_gb = model_params * 0.5 / (1024**3)  # INT4 = 0.5 bytes

    # Available for KV cache
    available_for_kv = max_memory_gb - model_memory_gb - 1.0  # 1GB safety margin

    # KV cache per token
    kv_per_token = (
        2 *  # K and V
        config.num_hidden_layers *
        config.hidden_size *
        1  # INT8 KV cache = 1 byte
    ) / (1024**3)  # Convert to GB

    # Max tokens
    max_tokens = available_for_kv / kv_per_token

    # Max sequences (assuming avg_output_tokens per sequence)
    max_batch_size = int(max_tokens / avg_output_tokens)

    print(f"Memory Budget Analysis:")
    print(f"  Total GPU Memory:    {max_memory_gb:.2f} GB")
    print(f"  Model Weights:       {model_memory_gb:.2f} GB")
    print(f"  Available for KV:    {available_for_kv:.2f} GB")
    print(f"  KV per token:        {kv_per_token*1024:.2f} MB")
    print(f"  Max tokens:          {max_tokens:.0f}")
    print(f"  Max batch size:      {max_batch_size}")

    return max_batch_size


# Example
max_batch = find_max_batch_under_memory_limit(
    "TheBloke/Llama-2-7B-AWQ",
    max_memory_gb=40,  # A100 40GB
    avg_output_tokens=256,
)
```

---

## Production Recommendations

### Batch Size Selection Guide

```python
BATCH_SIZE_RECOMMENDATIONS = {
    # Use case: (min, max, recommended)

    # Real-time interactive (chat, code completion)
    "interactive": {
        "min": 1,
        "max": 4,
        "recommended": 2,
        "rationale": "Minimize latency, accept lower throughput"
    },

    # Batch processing (summarization, translation)
    "batch": {
        "min": 32,
        "max": 256,
        "recommended": 64,
        "rationale": "Maximize throughput, latency less critical"
    },

    # Mixed workload (API serving)
    "mixed": {
        "min": 8,
        "max": 64,
        "recommended": 16,
        "rationale": "Balance throughput and latency"
    },

    # Long-form generation (articles, stories)
    "longform": {
        "min": 4,
        "max": 16,
        "recommended": 8,
        "rationale": "Long sequences use more memory"
    },
}


def get_recommended_batch_size(use_case, gpu_memory_gb):
    """
    Get recommended batch size for use case and hardware.

    Args:
        use_case: One of 'interactive', 'batch', 'mixed', 'longform'
        gpu_memory_gb: Available GPU memory

    Returns:
        Recommended batch size
    """
    if use_case not in BATCH_SIZE_RECOMMENDATIONS:
        raise ValueError(f"Unknown use case: {use_case}")

    config = BATCH_SIZE_RECOMMENDATIONS[use_case]
    recommended = config["recommended"]

    # Adjust for memory
    if gpu_memory_gb < 24:  # Small GPU
        recommended = max(config["min"], recommended // 2)
    elif gpu_memory_gb >= 80:  # Large GPU
        recommended = min(config["max"], recommended * 2)

    print(f"Use Case: {use_case}")
    print(f"  Recommended Batch Size: {recommended}")
    print(f"  Rationale: {config['rationale']}")
    print(f"  Valid Range: [{config['min']}, {config['max']}]")

    return recommended
```

### Production Configuration Template

```python
def create_production_config(
    use_case="mixed",
    gpu_memory_gb=40,
    model_name="TheBloke/Llama-2-7B-AWQ",
):
    """
    Create optimized vLLM configuration for production.

    Args:
        use_case: Application use case
        gpu_memory_gb: GPU memory size
        model_name: Model to serve

    Returns:
        vLLM configuration dictionary
    """
    # Get recommended batch size
    batch_size = get_recommended_batch_size(use_case, gpu_memory_gb)

    # Base configuration
    config = {
        "model": model_name,
        "quantization": "awq",
        "dtype": "half",

        # Memory
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,  # Adjust based on use case

        # Batching
        "max_num_seqs": batch_size,
        "max_num_batched_tokens": batch_size * 512,  # Approximate

        # Performance
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,  # Enable for long contexts

        # Monitoring
        "disable_log_stats": False,
    }

    # Use case specific adjustments
    if use_case == "interactive":
        config["max_model_len"] = 1024  # Shorter contexts
        config["enable_chunked_prefill"] = False

    elif use_case == "longform":
        config["max_model_len"] = 4096  # Longer contexts
        config["enable_chunked_prefill"] = True
        config["max_num_seqs"] = batch_size // 2  # Reduce for memory

    elif use_case == "batch":
        config["max_num_seqs"] = batch_size * 2  # Aggressive batching
        config["gpu_memory_utilization"] = 0.95  # Use more memory

    print(f"\nProduction Configuration for '{use_case}':")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config


# Example usage
config = create_production_config(
    use_case="mixed",
    gpu_memory_gb=40,
    model_name="TheBloke/Llama-2-7B-AWQ",
)
```

---

## Summary

### Key Takeaways

1. **Batch Size Impact**: Significantly affects throughput, latency, and memory
2. **Optimal Selection**: Depends on use case and hardware constraints
3. **Dynamic Strategies**: Adapt batch size based on load and latency
4. **Memory Awareness**: Calculate maximum batch size from available memory
5. **Use Case Specific**: Different applications need different batch sizes

### Quick Reference

| Use Case | Batch Size | Priority |
|----------|------------|----------|
| Interactive Chat | 1-4 | Latency |
| API Serving | 8-32 | Balance |
| Batch Processing | 64-256 | Throughput |
| Long-form Gen | 4-16 | Memory |

---

## References

1. "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. vLLM Documentation: Performance Tuning
3. "Orca: A Distributed Serving System for Transformer-Based Generative Models"

---

**Module 6.10 Complete** • Next: [CUDA Graph Optimization](11_cuda_graph_optimization.md)
