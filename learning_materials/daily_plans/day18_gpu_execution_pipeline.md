# Day 18: GPU Execution Pipeline

> **Goal**: Master GPU execution flow, kernel launches, and memory operations in vLLM
> **Time**: 4-6 hours
> **Prerequisites**: Day 17 complete, CUDA basics
> **Deliverables**: Execution timeline diagram, kernel profile analysis, optimization report

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): GPU Execution Flow

**9:00-9:30** - GPU Execution Overview
**9:30-10:30** - Kernel Launch Pipeline
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Memory Operations & Synchronization

### Afternoon Session (2-3 hours): Performance Analysis

**14:00-15:00** - Profiling with Nsight Systems
**15:00-16:00** - Kernel Performance Analysis
**16:00-16:30** - Break
**16:30-17:30** - Hands-On: Optimization Experiments

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Review profiles, document bottlenecks, preview Day 19

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Trace GPU kernel launches in vLLM execution
- [ ] Understand memory copy and synchronization patterns
- [ ] Profile GPU execution with Nsight Systems
- [ ] Identify performance bottlenecks
- [ ] Propose kernel-level optimizations

---

## 📂 Morning: GPU Execution Flow (9:00-12:00)

### Task 1: GPU Execution Overview (30 min)

**High-Level GPU Execution**:

```
CPU (Host) Timeline:
┌────────────────────────────────────────────────────────┐
│ Schedule │ Prepare │ Launch │ Wait │ Process │ Repeat │
│          │ Inputs  │ Kernels│      │ Outputs │        │
└────────────────────────────────────────────────────────┘

GPU (Device) Timeline:
           ┌─────────────────────────────────────────────┐
           │ Attention │ MLP │ Norm │ Sample │ (Idle)   │
           │ Kernel    │     │      │        │          │
           └─────────────────────────────────────────────┘
```

**Key GPU Operations in vLLM**:

```python
# Per-step GPU operations

1. Memory Preparation:
   - Copy input tokens: CPU → GPU
   - Prepare attention metadata
   - Set up block tables

2. Model Execution:
   - Embedding lookup
   - For each layer:
     - Attention kernel (PagedAttention)
     - MLP kernels
     - Normalization kernels
   - Final projection

3. Sampling:
   - Softmax over logits
   - Top-k/top-p filtering
   - Random sampling

4. Memory Updates:
   - Write KV cache (via slot_mapping)
   - Copy outputs: GPU → CPU
   - Block operations (allocate/free)
```

**CUDA Stream Organization**:

```python
# vLLM uses CUDA streams for concurrency

class CUDAExecutor:
    def __init__(self):
        # Main compute stream
        self.compute_stream = torch.cuda.Stream()

        # Memory copy streams
        self.h2d_stream = torch.cuda.Stream()  # Host to Device
        self.d2h_stream = torch.cuda.Stream()  # Device to Host

        # KV cache stream
        self.cache_stream = torch.cuda.Stream()

    def execute_step(self):
        # Concurrent execution using streams

        with torch.cuda.stream(self.h2d_stream):
            # Copy inputs to GPU
            input_tokens_gpu = input_tokens.cuda(non_blocking=True)

        with torch.cuda.stream(self.compute_stream):
            # Wait for input copy
            self.compute_stream.wait_stream(self.h2d_stream)

            # Run model
            logits = self.model(input_tokens_gpu, ...)

        with torch.cuda.stream(self.d2h_stream):
            # Copy outputs back
            self.d2h_stream.wait_stream(self.compute_stream)
            logits_cpu = logits.cpu(non_blocking=True)
```

### Task 2: Kernel Launch Pipeline (60 min)

**File**: `vllm/attention/ops/paged_attn.py`

**Attention Kernel Launch**:

```python
def paged_attention_v1(
    out: torch.Tensor,              # [num_tokens, num_heads, head_size]
    query: torch.Tensor,            # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,        # [num_blocks, num_heads, block_size, head_size]
    value_cache: torch.Tensor,      # [num_blocks, num_heads, block_size, head_size]
    block_tables: torch.Tensor,     # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,     # [num_seqs]
    scale: float,
) -> None:
    """
    Launch PagedAttention kernel.
    """

    # Determine launch configuration
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]

    # Thread organization
    # - Each thread block handles one (token, head) pair
    # - Threads within block collaborate on attention over context
    num_blocks = num_tokens * num_heads
    threads_per_block = 128  # Tuned for attention pattern

    # Launch kernel
    torch.ops._C.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_blocks,
        threads_per_block,
        scale,
    )
```

**Kernel Launch Configuration**:

```cuda
// csrc/attention/attention_kernels.cu

// Launch configuration
template<typename T>
void launch_paged_attention_kernel(
    T* out,
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int* block_tables,
    const int* context_lens,
    int num_heads,
    int head_size,
    int num_tokens
) {
    // Grid dimensions
    dim3 grid(num_tokens, num_heads);  // One block per (token, head)

    // Block dimensions
    dim3 block(128);  // 128 threads per block

    // Shared memory
    int shared_mem_size = head_size * sizeof(T) * 2;  // For Q and partial sums

    // Launch
    paged_attention_kernel<T><<<grid, block, shared_mem_size>>>(
        out,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        head_size,
        scale
    );
}
```

**Kernel Execution Pattern**:

```
Example: Batch with 4 tokens, 8 heads

Grid Layout:
┌─────────────────────────────────────┐
│ Token 0                             │
│ [H0][H1][H2][H3][H4][H5][H6][H7]   │
│                                     │
│ Token 1                             │
│ [H0][H1][H2][H3][H4][H5][H6][H7]   │
│                                     │
│ Token 2                             │
│ [H0][H1][H2][H3][H4][H5][H6][H7]   │
│                                     │
│ Token 3                             │
│ [H0][H1][H2][H3][H4][H5][H6][H7]   │
└─────────────────────────────────────┘

Total thread blocks: 4 × 8 = 32
Threads per block: 128
Total threads: 4,096

Each block computes:
  out[token, head] = attention(
    query[token, head],
    key_cache[blocks_for_token],
    value_cache[blocks_for_token]
  )
```

**Other Key Kernels**:

```python
# 1. Reshape and Cache (write KV cache)
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Write K/V to cache at specified slots.

    Thread organization:
    - One thread per token
    - Writes head_size elements
    """
    torch.ops._C.reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping
    )


# 2. Rotary Embedding (position encoding)
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
) -> None:
    """
    Apply rotary position encoding.

    Thread organization:
    - One thread per (token, head, dim/2)
    - Applies rotation in-place
    """
    torch.ops._C.rotary_embedding(positions, query, key, head_size)


# 3. RMS Norm
def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """
    Root Mean Square Layer Normalization.

    Thread organization:
    - One block per token
    - Threads reduce within token
    """
    torch.ops._C.rms_norm(out, input, weight, epsilon)
```

### Task 3: Memory Operations & Synchronization (60 min)

**Memory Copy Patterns**:

```python
# 1. Input preparation (H2D)
def prepare_inputs_on_gpu(input_tokens_cpu, input_positions_cpu):
    """
    Copy inputs from CPU to GPU.
    """

    # Asynchronous copy
    input_tokens_gpu = input_tokens_cpu.cuda(non_blocking=True)
    input_positions_gpu = input_positions_cpu.cuda(non_blocking=True)

    # Optional: pin CPU memory for faster transfer
    # input_tokens_cpu = input_tokens_cpu.pin_memory()

    return input_tokens_gpu, input_positions_gpu


# 2. KV cache operations (D2D)
def swap_blocks(
    src_blocks: List[int],
    dst_blocks: List[int],
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
) -> None:
    """
    Copy KV cache blocks (GPU ↔ GPU or GPU ↔ CPU).
    """

    # Device to device copy (within GPU)
    if src_cache.device == dst_cache.device:
        for src, dst in zip(src_blocks, dst_blocks):
            # Direct copy
            dst_cache[dst] = src_cache[src].clone()

    # GPU to CPU swap
    else:
        for src, dst in zip(src_blocks, dst_blocks):
            # Synchronous copy (blocking)
            dst_cache[dst] = src_cache[src].cpu()


# 3. Output retrieval (D2H)
def get_sampled_tokens_cpu(logits_gpu):
    """
    Sample tokens on GPU, copy to CPU.
    """

    # Sample on GPU
    sampled_tokens_gpu = torch.argmax(logits_gpu, dim=-1)

    # Copy to CPU
    sampled_tokens_cpu = sampled_tokens_gpu.cpu()

    return sampled_tokens_cpu
```

**Synchronization Points**:

```python
# Explicit synchronization

# 1. Wait for all GPU operations
torch.cuda.synchronize()

# 2. Wait for specific stream
stream = torch.cuda.current_stream()
stream.synchronize()

# 3. Event-based synchronization
event = torch.cuda.Event()
event.record()
event.wait()

# 4. Stream dependencies
stream_a = torch.cuda.Stream()
stream_b = torch.cuda.Stream()

with torch.cuda.stream(stream_a):
    operation_a()

with torch.cuda.stream(stream_b):
    stream_b.wait_stream(stream_a)  # Wait for stream_a
    operation_b()
```

**Memory Bandwidth Analysis**:

```python
#!/usr/bin/env python3
"""
Analyze memory bandwidth usage.
"""

import torch
import time

def measure_memory_bandwidth():
    """Measure achieved memory bandwidth."""

    # GPU specs (example: A100)
    peak_bandwidth_gbps = 1555  # GB/s for A100 40GB

    # Test data
    size = 1024 * 1024 * 1024  # 1GB
    data_cpu = torch.randn(size // 4, dtype=torch.float32)
    data_gpu = torch.randn(size // 4, dtype=torch.float32, device='cuda')

    # Measure H2D
    torch.cuda.synchronize()
    start = time.time()
    data_gpu_copy = data_cpu.cuda()
    torch.cuda.synchronize()
    h2d_time = time.time() - start

    h2d_bandwidth = size / h2d_time / 1e9  # GB/s

    # Measure D2D (copy)
    torch.cuda.synchronize()
    start = time.time()
    data_gpu_copy2 = data_gpu.clone()
    torch.cuda.synchronize()
    d2d_time = time.time() - start

    d2d_bandwidth = size / d2d_time / 1e9  # GB/s

    # Measure D2H
    torch.cuda.synchronize()
    start = time.time()
    data_cpu_copy = data_gpu.cpu()
    torch.cuda.synchronize()
    d2h_time = time.time() - start

    d2h_bandwidth = size / d2h_time / 1e9  # GB/s

    print(f"Memory Bandwidth Analysis")
    print(f"=" * 60)
    print(f"Peak (A100): {peak_bandwidth_gbps} GB/s")
    print(f"\nMeasured:")
    print(f"  H2D: {h2d_bandwidth:.1f} GB/s ({h2d_bandwidth/peak_bandwidth_gbps*100:.1f}% of peak)")
    print(f"  D2D: {d2d_bandwidth:.1f} GB/s ({d2d_bandwidth/peak_bandwidth_gbps*100:.1f}% of peak)")
    print(f"  D2H: {d2h_bandwidth:.1f} GB/s ({d2h_bandwidth/peak_bandwidth_gbps*100:.1f}% of peak)")

measure_memory_bandwidth()
```

---

## 🔬 Afternoon: Performance Analysis (14:00-17:30)

### Task 4: Profiling with Nsight Systems (60 min)

**Profile vLLM Execution**:

```bash
# Basic profiling
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=vllm_profile.qdrep \
    --force-overwrite true \
    python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --max-num-batched-tokens 2048

# Run workload, then stop server (Ctrl+C)

# View profile
nsys-ui vllm_profile.qdrep
```

**NVTX Markers in Code**:

```python
# Add custom markers for profiling

import torch

# Method 1: Context manager
with torch.cuda.nvtx.range("Scheduler"):
    scheduler_outputs = scheduler.schedule()

with torch.cuda.nvtx.range("ModelExecutor"):
    logits = model_executor.execute_model(...)

with torch.cuda.nvtx.range("Sampling"):
    sampled_tokens = sampler.sample(logits)


# Method 2: Manual push/pop
torch.cuda.nvtx.range_push("AttentionKernel")
attention_output = attention_forward(...)
torch.cuda.nvtx.range_pop()
```

**What to Look For in Profile**:

```
Nsight Systems Timeline View:

1. CPU Utilization:
   - Python GIL activity
   - Scheduler overhead
   - Data preparation time

2. GPU Kernels:
   - Kernel duration (should be majority of time)
   - Gaps between kernels (should be minimal)
   - Kernel types (attention, matmul, norm)

3. Memory Operations:
   - H2D/D2H transfers (should be small)
   - D2D copies (for swapping)

4. NVTX Ranges:
   - Time in each component
   - Identify bottlenecks

Ideal profile:
├─ Scheduler (2-5% time)
├─ ModelExecutor (85-90% time)
│  ├─ Attention kernels (40-50%)
│  ├─ MLP kernels (30-40%)
│  └─ Other kernels (10-20%)
└─ Sampling (3-5% time)
```

### Task 5: Kernel Performance Analysis (60 min)

**Using Nsight Compute**:

```bash
# Profile specific kernel
ncu --set full \
    --target-processes all \
    --kernel-name paged_attention_v1 \
    --launch-count 1 \
    -o attention_kernel_profile \
    python inference_test.py

# View report
ncu-ui attention_kernel_profile.ncu-rep
```

**Key Metrics to Check**:

```
Nsight Compute Metrics:

1. Compute Utilization:
   - SM Utilization: % of SMs active
   - Target: > 80%
   - Low → not enough parallelism

2. Memory Utilization:
   - Memory Throughput: GB/s achieved
   - Compare to peak bandwidth
   - Target: > 60% for memory-bound kernels

3. Occupancy:
   - Active warps per SM
   - Target: > 50%
   - Low → register pressure or shared memory limits

4. Memory Access Patterns:
   - Global Load Efficiency: % of coalesced loads
   - Target: > 80%
   - Low → uncoalesced access, poor performance

5. Instruction Mix:
   - FP32/FP16 instructions
   - Memory instructions
   - Should match expected pattern
```

**Bottleneck Identification**:

```python
#!/usr/bin/env python3
"""
Analyze kernel bottlenecks from Nsight Compute metrics.
"""

def analyze_kernel_bottleneck(metrics):
    """
    Determine kernel bottleneck type.

    Args:
        metrics: dict with Nsight Compute metrics

    Returns:
        bottleneck type and recommendations
    """

    sm_util = metrics['sm_utilization']
    mem_util = metrics['memory_utilization']
    occupancy = metrics['occupancy']
    load_efficiency = metrics['load_efficiency']

    print(f"Kernel Bottleneck Analysis")
    print(f"=" * 60)
    print(f"SM Utilization: {sm_util:.1f}%")
    print(f"Memory Utilization: {mem_util:.1f}%")
    print(f"Occupancy: {occupancy:.1f}%")
    print(f"Load Efficiency: {load_efficiency:.1f}%")
    print()

    # Classify bottleneck
    if sm_util < 50:
        print("⚠️  COMPUTE BOUND (Low SM utilization)")
        print("Recommendations:")
        print("  - Increase parallelism (more thread blocks)")
        print("  - Reduce work per thread")
        print("  - Check for warp divergence")

    elif mem_util > 80:
        print("⚠️  MEMORY BOUND (High memory utilization)")
        print("Recommendations:")
        print("  - Optimize memory access patterns")
        print("  - Use shared memory for reuse")
        print("  - Increase compute/memory ratio")

    elif occupancy < 30:
        print("⚠️  OCCUPANCY LIMITED")
        print("Recommendations:")
        print("  - Reduce register usage per thread")
        print("  - Reduce shared memory per block")
        print("  - Adjust block size")

    elif load_efficiency < 60:
        print("⚠️  UNCOALESCED MEMORY ACCESS")
        print("Recommendations:")
        print("  - Align data structures")
        print("  - Access memory contiguously")
        print("  - Consider memory layout transformation")

    else:
        print("✓ WELL OPTIMIZED")
        print("Kernel is utilizing GPU efficiently")

# Example usage
metrics = {
    'sm_utilization': 45.2,
    'memory_utilization': 88.5,
    'occupancy': 62.3,
    'load_efficiency': 78.1,
}

analyze_kernel_bottleneck(metrics)
# Output: MEMORY BOUND
```

### Task 6: Optimization Experiments (60 min)

**Experiment 1: Thread Block Size Tuning**:

```python
#!/usr/bin/env python3
"""
Test different thread block sizes.
"""

import torch
import time

def benchmark_block_size(block_size: int):
    """Benchmark with specific block size."""

    # Create test data
    num_tokens = 1024
    num_heads = 32
    head_size = 128

    query = torch.randn(num_tokens, num_heads, head_size, device='cuda')
    key = torch.randn(num_tokens, num_heads, head_size, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_size, device='cuda')

    # Warmup
    for _ in range(10):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = 100 / elapsed  # iterations/second

    print(f"Block size {block_size}: {throughput:.2f} iter/s")

    return throughput

# Test different block sizes
for block_size in [64, 128, 256, 512]:
    benchmark_block_size(block_size)
```

**Experiment 2: Batch Size Impact**:

```python
def profile_batch_sizes():
    """Profile GPU utilization with different batch sizes."""

    batch_sizes = [1, 4, 16, 64, 256]

    results = []

    for batch_size in batch_sizes:
        # Run inference
        # Measure: kernel time, memory usage, throughput
        result = run_inference(batch_size)
        results.append(result)

    # Analyze
    print(f"\nBatch Size Impact:")
    print(f"=" * 60)
    print(f"{'Batch':>8} {'Kernel (ms)':>12} {'Memory (GB)':>12} {'Throughput':>12}")

    for batch_size, result in zip(batch_sizes, results):
        print(f"{batch_size:>8} {result['kernel_ms']:>12.2f} "
              f"{result['memory_gb']:>12.2f} {result['throughput']:>12.2f}")
```

**Experiment 3: Mixed Precision**:

```python
def compare_precision():
    """Compare FP32 vs FP16 performance."""

    model_fp32 = load_model(dtype=torch.float32)
    model_fp16 = load_model(dtype=torch.float16)

    # Benchmark both
    throughput_fp32 = benchmark(model_fp32)
    throughput_fp16 = benchmark(model_fp16)

    speedup = throughput_fp16 / throughput_fp32

    print(f"\nPrecision Comparison:")
    print(f"  FP32: {throughput_fp32:.2f} tok/s")
    print(f"  FP16: {throughput_fp16:.2f} tok/s")
    print(f"  Speedup: {speedup:.2f}x")
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **GPU Execution Flow**
- Kernel launch pipeline
- CUDA streams and concurrency
- Memory operations (H2D, D2D, D2H)

✅ **Performance Analysis**
- Profiling with Nsight Systems
- Kernel analysis with Nsight Compute
- Bottleneck identification

✅ **Optimization**
- Thread block tuning
- Memory bandwidth optimization
- Mixed precision benefits

### Knowledge Check Questions

**Q1**: What are the main GPU operations in one vLLM inference step?
<details>
<summary>Answer</summary>
1. **Memory prep**: Copy inputs (H2D), setup metadata
2. **Model execution**: Embedding → Layers (Attention, MLP, Norm) → Output
3. **Sampling**: Softmax, top-k/p, sample tokens
4. **Memory updates**: Write KV cache, copy outputs (D2H)
</details>

**Q2**: How do CUDA streams improve performance?
<details>
<summary>Answer</summary>
Enable **concurrent execution** of independent operations:
- Memory copies (H2D) while kernels run
- Multiple kernel launches without CPU synchronization
- Overlap computation and communication

Example: Copy next batch inputs while processing current batch
</details>

**Q3**: What does "memory-bound" mean for a kernel?
<details>
<summary>Answer</summary>
Performance limited by **memory bandwidth**, not compute.

Characteristics:
- Low arithmetic intensity (ops/byte)
- High memory utilization (> 80%)
- Compute units idle waiting for data

Example: Decode attention (read large KV cache, few ops)

Fix: Cache reuse, better access patterns, increase compute/memory ratio
</details>

---

## 🚀 Preview: Day 19

Tomorrow you'll explore:
- **Distributed Inference Basics**: Multi-GPU strategies
- **Tensor Parallelism**: Splitting model across GPUs
- **Communication Primitives**: All-reduce, all-gather
- **NCCL Integration**: Efficient GPU-to-GPU communication

**Preparation**:
- Review distributed computing concepts
- Understand collective operations
- Familiarize with multi-GPU setups

---

## 📚 Additional Resources

**Tools**:
- [ ] Nsight Systems documentation
- [ ] Nsight Compute user guide
- [ ] CUDA profiler API

**Concepts**:
- [ ] CUDA streams and events
- [ ] Memory coalescing
- [ ] Kernel occupancy
- [ ] Roofline model

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
