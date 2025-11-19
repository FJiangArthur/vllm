# CUDA Graph Optimization for LLM Inference

## Table of Contents
1. [Introduction to CUDA Graphs](#introduction-to-cuda-graphs)
2. [CUDA Graph Basics](#cuda-graph-basics)
3. [CUDA Graphs in vLLM](#cuda-graphs-in-vllm)
4. [Performance Benefits](#performance-benefits)
5. [Implementation Details](#implementation-details)
6. [Best Practices](#best-practices)

---

## Introduction to CUDA Graphs

### What are CUDA Graphs?

**CUDA Graphs** allow recording a sequence of CUDA operations and replaying them with minimal CPU overhead.

**Traditional CUDA:**
```
For each inference:
    CPU: Prepare kernel args
    CPU: Launch kernel 1
    GPU: Execute kernel 1
    CPU: Launch kernel 2
    GPU: Execute kernel 2
    ... (repeated CPU overhead)
```

**With CUDA Graphs:**
```
One-time setup:
    Record graph (kernels + dependencies)

For each inference:
    CPU: Launch entire graph (single API call)
    GPU: Execute all kernels
```

### Why CUDA Graphs Matter for LLMs

**Benefits:**
1. **Reduced CPU Overhead**: ~10-100x lower CPU usage
2. **Lower Latency**: Faster kernel launches
3. **Better GPU Utilization**: Less idle time between kernels
4. **Consistent Performance**: Deterministic execution

**Trade-offs:**
- Fixed computation graph (no dynamic control flow)
- Memory overhead for graph storage
- Limited to constant tensor shapes

---

## CUDA Graph Basics

### Creating a CUDA Graph

```python
import torch

def basic_cuda_graph_example():
    """
    Basic CUDA graph creation and replay.
    """
    # Input tensor
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')

    # Warmup (required before graph capture)
    for _ in range(10):
        z = x @ y
        z = torch.relu(z)

    # Create graph
    graph = torch.cuda.CUDAGraph()

    # Capture graph
    with torch.cuda.graph(graph):
        z = x @ y
        z = torch.relu(z)

    # Replay graph (much faster than regular execution)
    for _ in range(100):
        graph.replay()

    torch.cuda.synchronize()
    print(f"Output shape: {z.shape}")

    return graph
```

### Graph Capture Requirements

**Requirements for graph capture:**
1. All operations must be CUDA ops
2. Tensor shapes must be constant
3. No CPU synchronization points
4. No dynamic control flow (if/while based on tensor values)
5. No memory allocations

**Example of graph-incompatible code:**

```python
# ❌ BAD: Dynamic control flow
def incompatible_with_graphs(x):
    if x.sum() > 0:  # Runtime condition
        return x * 2
    else:
        return x / 2

# ❌ BAD: Dynamic shapes
def incompatible_shapes(x):
    mask = x > 0
    return x[mask]  # Variable output size

# ✓ GOOD: Static computation
def compatible_with_graphs(x):
    return torch.relu(x) * 2 + 1  # No dynamic behavior
```

---

## CUDA Graphs in vLLM

### Automatic CUDA Graph Usage

vLLM automatically uses CUDA graphs for the **decode phase**:

```python
from vllm import LLM, SamplingParams

# vLLM automatically enables CUDA graphs
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    enforce_eager=False,  # False = enable CUDA graphs (default)
)

params = SamplingParams(temperature=0.8, max_tokens=100)
prompts = ["Once upon a time"]

# Prefill: Regular execution
# Decode: CUDA graph execution (automatic)
outputs = llm.generate(prompts, params)
```

### Disabling CUDA Graphs

```python
# Disable CUDA graphs (for debugging or dynamic scenarios)
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    enforce_eager=True,  # True = disable CUDA graphs
)

# Use cases for disabling:
# - Debugging (graphs hide individual kernel timing)
# - Highly dynamic batch sizes
# - Profiling individual operations
```

### Graph Configuration

```python
# vLLM CUDA graph configuration (internal)
# File: vllm/worker/model_runner.py

class CUDAGraphConfig:
    """Configuration for CUDA graph capturing."""

    # Batch sizes to pre-capture graphs for
    graph_capture_sizes = [1, 2, 4, 8, 16, 32, 64]

    # Maximum batch size for graphs
    max_graph_batch_size = 64

    # Memory pool for graph replay
    graph_memory_pool_size = 1024 * 1024 * 1024  # 1GB


# vLLM captures graphs for common batch sizes
# at initialization to avoid runtime overhead
```

---

## Performance Benefits

### Latency Reduction

```python
import torch
import time

def benchmark_cuda_graph_latency():
    """
    Compare latency with and without CUDA graphs.
    """
    # Setup
    model_layer = torch.nn.Linear(4096, 4096).cuda().half()
    x = torch.randn(16, 4096, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(100):
        y = model_layer(x)

    torch.cuda.synchronize()

    # Benchmark without graph
    times_eager = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        y = model_layer(x)
        end.record()

        torch.cuda.synchronize()
        times_eager.append(start.elapsed_time(end))

    # Create graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = model_layer(x)

    # Benchmark with graph
    times_graph = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        graph.replay()
        end.record()

        torch.cuda.synchronize()
        times_graph.append(start.elapsed_time(end))

    # Results
    import numpy as np

    print(f"Latency Comparison (1000 iterations):")
    print(f"  Eager Mode:")
    print(f"    Mean:   {np.mean(times_eager):.3f} ms")
    print(f"    Median: {np.median(times_eager):.3f} ms")
    print(f"    P99:    {np.percentile(times_eager, 99):.3f} ms")
    print(f"  CUDA Graph:")
    print(f"    Mean:   {np.mean(times_graph):.3f} ms")
    print(f"    Median: {np.median(times_graph):.3f} ms")
    print(f"    P99:    {np.percentile(times_graph, 99):.3f} ms")
    print(f"  Speedup: {np.mean(times_eager) / np.mean(times_graph):.2f}x")

# Example output:
# Latency Comparison (1000 iterations):
#   Eager Mode:
#     Mean:   1.234 ms
#     Median: 1.231 ms
#     P99:    1.387 ms
#   CUDA Graph:
#     Mean:   0.987 ms
#     Median: 0.985 ms
#     P99:    1.012 ms
#   Speedup: 1.25x
```

### Throughput Improvement

```python
def benchmark_cuda_graph_throughput(batch_sizes=[1, 4, 8, 16, 32]):
    """
    Measure throughput improvement with CUDA graphs.

    Args:
        batch_sizes: List of batch sizes to test

    Returns:
        Throughput comparison
    """
    from vllm import LLM, SamplingParams

    model_name = "TheBloke/Llama-2-7B-AWQ"
    params = SamplingParams(temperature=0.8, max_tokens=100)

    results = {}

    for enforce_eager in [True, False]:
        mode = "Eager" if enforce_eager else "CUDA Graph"
        print(f"\nBenchmarking: {mode}")

        llm = LLM(
            model=model_name,
            quantization="awq",
            enforce_eager=enforce_eager,
        )

        mode_results = []

        for batch_size in batch_sizes:
            prompts = [f"Prompt {i}" for i in range(batch_size)]

            # Warmup
            llm.generate(prompts[:min(4, batch_size)], params)

            # Benchmark
            start = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed = time.perf_counter() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed

            mode_results.append({
                "batch_size": batch_size,
                "throughput": throughput,
            })

            print(f"  Batch {batch_size:2d}: {throughput:.2f} tok/s")

        results[mode] = mode_results

        # Cleanup
        del llm
        torch.cuda.empty_cache()

    # Print comparison
    print(f"\nSpeedup Summary:")
    print(f"{'Batch':<8} {'Eager':<12} {'Graph':<12} {'Speedup':<8}")
    print("-" * 45)

    for i, batch_size in enumerate(batch_sizes):
        eager_thr = results["Eager"][i]["throughput"]
        graph_thr = results["CUDA Graph"][i]["throughput"]
        speedup = graph_thr / eager_thr

        print(f"{batch_size:<8} {eager_thr:<12.2f} {graph_thr:<12.2f} {speedup:<8.2f}x")

    return results
```

---

## Implementation Details

### Graph Capture in vLLM

```python
# Simplified version of vLLM's graph capture
# File: vllm/worker/model_runner.py

class ModelRunner:
    """Runs the model for inference."""

    def __init__(self, ...):
        self.graphs = {}  # batch_size -> CUDAGraph
        self.graph_memory_pool = None

    def capture_model_graph(self, batch_size):
        """
        Capture CUDA graph for specific batch size.

        Args:
            batch_size: Batch size to capture

        Returns:
            CUDA graph instance
        """
        # Prepare static inputs (fixed shapes)
        input_ids = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device='cuda'
        )

        position_ids = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device='cuda'
        )

        # Create memory pool for graph
        if self.graph_memory_pool is None:
            self.graph_memory_pool = torch.cuda.graph_pool_handle()

        # Warmup
        for _ in range(3):
            _ = self.model(
                input_ids=input_ids,
                positions=position_ids,
                kv_caches=self.kv_caches,
            )

        torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph, pool=self.graph_memory_pool):
            output = self.model(
                input_ids=input_ids,
                positions=position_ids,
                kv_caches=self.kv_caches,
            )

        # Store graph
        self.graphs[batch_size] = {
            'graph': graph,
            'input_ids': input_ids,
            'position_ids': position_ids,
            'output': output,
        }

        print(f"Captured CUDA graph for batch size {batch_size}")

        return graph

    def execute_model_graph(self, batch_size, new_input_ids, new_positions):
        """
        Execute captured graph with new inputs.

        Args:
            batch_size: Batch size
            new_input_ids: New input token IDs
            new_positions: New positions

        Returns:
            Model output
        """
        if batch_size not in self.graphs:
            raise ValueError(f"No graph captured for batch size {batch_size}")

        graph_data = self.graphs[batch_size]

        # Copy new data into graph's input buffers
        graph_data['input_ids'].copy_(new_input_ids)
        graph_data['position_ids'].copy_(new_positions)

        # Replay graph
        graph_data['graph'].replay()

        # Return output
        return graph_data['output']
```

### Multi-Size Graph Management

```python
class MultiSizeGraphManager:
    """
    Manage CUDA graphs for multiple batch sizes.
    """

    def __init__(self, capture_sizes=[1, 2, 4, 8, 16, 32]):
        self.capture_sizes = sorted(capture_sizes)
        self.graphs = {}

    def get_graph_for_batch_size(self, batch_size):
        """
        Get appropriate graph for batch size.

        Uses smallest graph >= batch_size, with padding.

        Args:
            batch_size: Actual batch size

        Returns:
            (graph_size, graph, padding_needed)
        """
        # Find smallest captured size >= batch_size
        for captured_size in self.capture_sizes:
            if captured_size >= batch_size:
                padding = captured_size - batch_size
                return captured_size, self.graphs[captured_size], padding

        # Batch too large for any graph
        return None, None, 0

    def should_use_graph(self, batch_size, threshold=0.5):
        """
        Decide whether to use graph for given batch size.

        Avoid graphs when padding overhead is too high.

        Args:
            batch_size: Actual batch size
            threshold: Maximum acceptable padding ratio

        Returns:
            bool: Whether to use graph
        """
        graph_size, _, padding = self.get_graph_for_batch_size(batch_size)

        if graph_size is None:
            return False

        padding_ratio = padding / graph_size

        # Don't use graph if too much padding
        return padding_ratio <= threshold


# Example usage
graph_manager = MultiSizeGraphManager([1, 4, 8, 16, 32])

# Batch size 6: Use size-8 graph with 2 padding
use_graph = graph_manager.should_use_graph(6)  # True (25% padding)

# Batch size 20: Use size-32 graph with 12 padding
use_graph = graph_manager.should_use_graph(20)  # False (37.5% padding > threshold)
```

---

## Best Practices

### 1. Graph Capture Strategy

```python
# ✓ GOOD: Capture common batch sizes
common_sizes = [1, 2, 4, 8, 16, 32, 64]

# ✓ GOOD: Use power-of-2 sizes (efficient padding)
sizes = [2**i for i in range(7)]  # 1, 2, 4, 8, 16, 32, 64

# ❌ BAD: Too many sizes (memory overhead)
too_many = list(range(1, 129))  # Don't do this

# ❌ BAD: Very large sizes (memory waste)
too_large = [512, 1024]  # Usually unnecessary
```

### 2. Memory Management

```python
def optimize_graph_memory():
    """
    Best practices for CUDA graph memory management.
    """
    # 1. Use memory pool
    memory_pool = torch.cuda.graph_pool_handle()

    # 2. Capture graphs in order (small to large)
    # Helps with memory allocation
    for batch_size in [1, 2, 4, 8, 16]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):
            # ... capture operations ...
            pass

    # 3. Free unused graphs
    # If certain batch sizes are rare, don't keep their graphs
    del graph
    torch.cuda.empty_cache()

    # 4. Monitor memory usage
    print(f"Graph memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### 3. Debugging

```python
def debug_cuda_graph_issues():
    """
    Common CUDA graph issues and solutions.
    """

    # Issue 1: Graph capture fails
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Operations here
            pass
    except RuntimeError as e:
        print(f"Graph capture failed: {e}")
        # Solution: Check for CPU synchronization, dynamic shapes

    # Issue 2: Incorrect results
    # Solution: Ensure proper warmup before capture
    x = torch.randn(10, 10, device='cuda')

    # Warmup (let caching allocator stabilize)
    for _ in range(10):
        y = x * 2

    # Now capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2

    # Issue 3: Performance degradation
    # Solution: Profile to check if graph is actually being used
    # Use Nsight Systems to verify graph replay vs. eager execution
```

### 4. Production Checklist

```python
CUDA_GRAPH_PRODUCTION_CHECKLIST = """
✓ Capture graphs for common batch sizes (1, 4, 8, 16, 32)
✓ Use memory pool for all graphs
✓ Implement fallback to eager mode for unusual sizes
✓ Monitor graph hit rate (% of requests using graphs)
✓ Profile to verify expected speedup
✓ Test with various input patterns
✓ Document which operations are graphed
✓ Set up alerts for graph capture failures
✓ Validate numerical correctness
✓ Benchmark end-to-end latency
"""

def production_graph_config():
    """Production-ready CUDA graph configuration."""
    return {
        "enable_graphs": True,
        "capture_sizes": [1, 2, 4, 8, 16, 32, 64],
        "max_padding_ratio": 0.5,  # Max 50% padding
        "eager_fallback": True,  # Fall back to eager if no suitable graph
        "warmup_iterations": 10,  # Warmup before capture
        "memory_pool_size_gb": 2,  # Reserved for graphs
    }
```

---

## Summary

### Key Takeaways

1. **CUDA Graphs Reduce Overhead**: 10-100x lower CPU overhead
2. **Automatic in vLLM**: Enabled by default for decode phase
3. **Fixed Shapes Required**: Capture for common batch sizes
4. **Memory Trade-off**: Graph storage vs. performance gain
5. **Production Ready**: Use in production for consistent performance

### Performance Impact

**Typical Improvements:**
- **Latency**: 10-25% reduction
- **Throughput**: 15-30% increase
- **CPU Usage**: 50-90% reduction
- **Jitter**: More consistent latency

### When to Use CUDA Graphs

**Use when:**
- Decode phase (token-by-token generation)
- Fixed batch sizes
- Production deployment
- Latency-sensitive applications

**Avoid when:**
- Highly dynamic batch sizes
- Debugging kernel-level issues
- Profiling individual operations
- Prefill phase (shapes vary)

---

## References

1. CUDA Graphs Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
2. PyTorch CUDA Graphs: https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
3. vLLM Source: vllm/worker/model_runner.py
4. "CUDA Graphs for Deep Learning" - NVIDIA Blog

---

**Module 6.11 Complete** • Next: [Production Deployment](12_production_deployment.md)
