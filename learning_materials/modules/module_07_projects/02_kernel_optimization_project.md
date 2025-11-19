# Project 7.2: CUDA Kernel Optimization Challenge

## Project Overview

**Duration:** 3-4 hours
**Difficulty:** ⭐⭐⭐⭐☆ (Advanced)
**Prerequisites:** Module 3 (CUDA Kernels & Optimization), strong C/C++ skills

### Objective

Optimize a provided "slow" CUDA kernel by **2x or more** through systematic application of CUDA optimization techniques. You will profile the kernel, identify bottlenecks, apply optimizations, and document your improvements with detailed performance analysis.

### Why This Matters

GPU kernel optimization is the cornerstone of efficient ML inference. A 2x speedup in a critical kernel can:
- **Halve infrastructure costs** (same throughput with half the GPUs)
- **Double user capacity** (serve 2x more requests)
- **Improve user experience** (reduce latency)

This project simulates real optimization work at NVIDIA, where kernel engineers routinely achieve 5-10x speedups through careful optimization.

### Real-World Context

This project mirrors:
- **NVIDIA**: Optimizing cuBLAS, cuDNN, and TensorRT kernels
- **OpenAI**: Custom Triton kernels for GPT models
- **Anthropic**: Optimizing attention mechanisms for Claude

---

## Learning Objectives

By completing this project, you will:

1. ✅ Master systematic kernel profiling with Nsight Compute
2. ✅ Apply memory optimization techniques (coalescing, shared memory, tiling)
3. ✅ Optimize compute throughput (occupancy, warp efficiency, tensor cores)
4. ✅ Conduct roofline analysis to identify limiting factors
5. ✅ Document optimization process professionally

---

## The Challenge: Matrix Softmax Kernel

### Problem Statement

You are given a **naive softmax kernel** that computes softmax across rows of a matrix. The kernel works correctly but is extremely slow due to poor memory access patterns and suboptimal parallelization.

**Input:** Matrix of shape `[batch_size, seq_len]` where `batch_size` can be 32-128 and `seq_len` can be 512-4096

**Operation:** For each row `i`: `output[i, j] = exp(input[i, j]) / sum(exp(input[i, :]))`

**Goal:** Optimize the kernel to achieve **2x or better speedup** while maintaining numerical accuracy (error < 1e-5).

### Baseline (Naive) Implementation

```cuda
// naive_softmax.cu - DO NOT USE THIS IN PRODUCTION!

__global__ void naive_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size) {
        // Find max (for numerical stability)
        float max_val = -INFINITY;
        for (int col = 0; col < seq_len; col++) {
            float val = input[row * seq_len + col];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            float val = input[row * seq_len + col];
            sum += expf(val - max_val);
        }

        // Normalize and write output
        for (int col = 0; col < seq_len; col++) {
            float val = input[row * seq_len + col];
            output[row * seq_len + col] = expf(val - max_val) / sum;
        }
    }
}
```

### Performance Characteristics of Naive Kernel

**Problems:**
1. **Redundant memory reads**: Each element read 3 times (max, sum, normalize)
2. **No shared memory usage**: All reads from slow global memory
3. **Sequential processing**: No parallelism within row
4. **Warp divergence**: Conditional in max-finding loop
5. **Low occupancy**: Uses too many registers

**Expected Performance:**
- ~0.5 TFLOPS (should be 10+ TFLOPS)
- ~50 GB/s memory bandwidth (should be 500+ GB/s)
- ~25% occupancy (should be 50%+)

---

## Requirements Specification

### Functional Requirements

#### FR1: Correctness
- Output must match reference implementation (PyTorch `F.softmax`) within 1e-5 absolute error
- Must handle edge cases: all zeros, very large values, very small values
- No NaN or Inf in output for valid inputs

#### FR2: Performance
- **Minimum:** 2x speedup over naive implementation
- **Target:** 3-5x speedup
- **Stretch:** 5-10x speedup

#### FR3: Generality
- Support arbitrary `seq_len` (powers of 2 from 128 to 8192)
- Support arbitrary `batch_size` (1 to 1024)
- Work on multiple GPU architectures (Ampere, Hopper)

### Non-Functional Requirements

#### NFR1: Profiling
- Complete Nsight Compute profiling report
- Roofline analysis showing kernel characteristics
- Bottleneck identification with evidence

#### NFR2: Documentation
- Document each optimization technique applied
- Before/after metrics for each optimization
- Trade-offs and design decisions explained

#### NFR3: Testing
- Unit tests for correctness
- Performance benchmarks across input sizes
- Numerical accuracy tests

---

## Optimization Strategy

### Phase 1: Profiling and Analysis (30 minutes)

#### Step 1.1: Baseline Profiling

```bash
# Compile with profiling info
nvcc -O3 -lineinfo -arch=sm_80 naive_softmax.cu -o naive_softmax

# Profile with Nsight Compute
ncu --set full --export baseline_profile naive_softmax

# Key metrics to collect:
# - Memory throughput (Global Load/Store Throughput)
# - Compute throughput (FP32 FLOPS)
# - Occupancy (Achieved Occupancy)
# - Warp execution efficiency
# - Memory access patterns (coalescing efficiency)
```

#### Step 1.2: Roofline Analysis

```python
# scripts/roofline_analysis.py

def compute_roofline_metrics(kernel_metrics):
    """
    Compute arithmetic intensity and plot on roofline model.

    Arithmetic Intensity = FLOPS / Bytes Transferred
    """
    flops = kernel_metrics['flops']
    bytes_transferred = kernel_metrics['global_load_bytes'] + \
                        kernel_metrics['global_store_bytes']

    arithmetic_intensity = flops / bytes_transferred

    # Compare to hardware limits
    gpu_peak_flops = 19.5e12  # A100: 19.5 TFLOPS FP32
    gpu_peak_bandwidth = 1555e9  # A100: 1.5 TB/s

    # Determine limiting factor
    compute_bound_limit = gpu_peak_flops
    memory_bound_limit = arithmetic_intensity * gpu_peak_bandwidth

    if compute_bound_limit < memory_bound_limit:
        limiting_factor = "Compute Bound"
    else:
        limiting_factor = "Memory Bound"

    return {
        'arithmetic_intensity': arithmetic_intensity,
        'limiting_factor': limiting_factor,
        'peak_flops_utilization': flops / gpu_peak_flops,
        'peak_bandwidth_utilization': bytes_transferred / gpu_peak_bandwidth,
    }
```

#### Expected Baseline Results

```
Kernel: naive_softmax_kernel
Duration: 450 μs (batch=64, seq_len=2048)
Global Load Throughput: 52 GB/s (3.3% of peak)
Global Store Throughput: 16 GB/s (1.0% of peak)
FP32 Throughput: 0.48 TFLOPS (2.5% of peak)
Achieved Occupancy: 23%
Warp Execution Efficiency: 87%
Arithmetic Intensity: 0.15 FLOPS/byte
Limiting Factor: MEMORY BOUND
```

### Phase 2: Memory Optimization (60 minutes)

#### Optimization 2.1: Reduce Redundant Reads

**Problem:** Each element read 3 times from global memory.

**Solution:** Cache values in registers or shared memory.

```cuda
__global__ void optimized_v1_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size) {
        // Allocate shared memory for one row (if small enough)
        extern __shared__ float shared_row[];

        // Load entire row into shared memory
        for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
            shared_row[col] = input[row * seq_len + col];
        }
        __syncthreads();

        // Find max using shared memory
        float max_val = -INFINITY;
        for (int col = 0; col < seq_len; col++) {
            max_val = fmaxf(max_val, shared_row[col]);
        }

        // Compute sum
        float sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            sum += expf(shared_row[col] - max_val);
        }

        // Normalize and write
        for (int col = 0; col < seq_len; col++) {
            output[row * seq_len + col] = expf(shared_row[col] - max_val) / sum;
        }
    }
}
```

**Expected Improvement:** 1.3-1.5x speedup (fewer global memory accesses)

#### Optimization 2.2: Parallel Reduction

**Problem:** Max and sum computed sequentially by one thread.

**Solution:** Use warp-level primitives for parallel reduction.

```cuda
// Warp-level reduction utilities
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_v2_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    if (row < batch_size) {
        const float* row_input = input + row * seq_len;
        float* row_output = output + row * seq_len;

        // Phase 1: Parallel max reduction
        float thread_max = -INFINITY;
        for (int col = tid; col < seq_len; col += blockDim.x) {
            thread_max = fmaxf(thread_max, row_input[col]);
        }

        // Warp-level reduction
        thread_max = warp_reduce_max(thread_max);

        // Block-level reduction via shared memory
        __shared__ float warp_maxes[32];  // Max 1024 threads = 32 warps
        int warp_id = tid / 32;
        if (lane_id == 0) {
            warp_maxes[warp_id] = thread_max;
        }
        __syncthreads();

        // Final reduction by first warp
        float block_max = -INFINITY;
        if (warp_id == 0) {
            float val = (tid < (blockDim.x / 32)) ? warp_maxes[tid] : -INFINITY;
            val = warp_reduce_max(val);
            if (tid == 0) {
                warp_maxes[0] = val;
            }
        }
        __syncthreads();
        block_max = warp_maxes[0];

        // Phase 2: Parallel sum of exponentials
        float thread_sum = 0.0f;
        for (int col = tid; col < seq_len; col += blockDim.x) {
            thread_sum += expf(row_input[col] - block_max);
        }

        thread_sum = warp_reduce_sum(thread_sum);

        __shared__ float warp_sums[32];
        if (lane_id == 0) {
            warp_sums[warp_id] = thread_sum;
        }
        __syncthreads();

        float block_sum = 0.0f;
        if (warp_id == 0) {
            float val = (tid < (blockDim.x / 32)) ? warp_sums[tid] : 0.0f;
            val = warp_reduce_sum(val);
            if (tid == 0) {
                warp_sums[0] = val;
            }
        }
        __syncthreads();
        block_sum = warp_sums[0];

        // Phase 3: Parallel normalization
        for (int col = tid; col < seq_len; col += blockDim.x) {
            row_output[col] = expf(row_input[col] - block_max) / block_sum;
        }
    }
}
```

**Expected Improvement:** 2-3x speedup over naive (parallel reduction + better memory)

#### Optimization 2.3: Memory Coalescing

**Problem:** Thread 0 accesses index 0, thread 1 accesses index `seq_len`, causing strided access.

**Solution:** Transpose access pattern so consecutive threads access consecutive memory.

```cuda
// Launch with: <<<batch_size, 256>>>
// Each block processes one row with 256 threads

__global__ void optimized_v3_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block handles one row
    int row = blockIdx.x;

    // Coalesced memory access: consecutive threads access consecutive elements
    const float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    // ... (same reduction logic as v2, but with guaranteed coalescing)
}
```

**Expected Improvement:** Additional 1.2-1.5x (memory bandwidth utilization)

### Phase 3: Compute Optimization (45 minutes)

#### Optimization 3.1: Online Softmax (Fused Max + Sum)

**Problem:** Two passes over data (max, then sum).

**Solution:** Compute max and sum in single pass using online algorithm.

```cuda
// Online softmax: compute max and sum in one pass
// Based on: https://arxiv.org/abs/1805.02867

__global__ void online_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row < batch_size) {
        const float* row_input = input + row * seq_len;
        float* row_output = output + row * seq_len;

        // Online softmax: maintain running max and sum
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        // Each thread processes elements with stride
        for (int col = tid; col < seq_len; col += blockDim.x) {
            float val = row_input[col];

            // Update running max and sum
            float old_max = running_max;
            running_max = fmaxf(running_max, val);

            // Adjust sum for new max
            running_sum = running_sum * expf(old_max - running_max) +
                          expf(val - running_max);
        }

        // Reduce across threads (warp + block level)
        // ... (similar to v2 but with special handling for online algorithm)

        // Write normalized output
        for (int col = tid; col < seq_len; col += blockDim.x) {
            row_output[col] = expf(row_input[col] - block_max) / block_sum;
        }
    }
}
```

**Expected Improvement:** 1.3-1.5x (fewer memory passes, better cache utilization)

#### Optimization 3.2: Vectorized Loads

**Problem:** Loading one float at a time.

**Solution:** Use `float4` for vectorized loads (4x floats per instruction).

```cuda
__global__ void vectorized_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Use float4 for vectorized loads
    const float4* row_input_vec = reinterpret_cast<const float4*>(
        input + row * seq_len
    );
    float4* row_output_vec = reinterpret_cast<float4*>(
        output + row * seq_len
    );

    int num_vec_elements = seq_len / 4;

    // Process 4 elements at a time
    float thread_max = -INFINITY;
    for (int idx = tid; idx < num_vec_elements; idx += blockDim.x) {
        float4 vals = row_input_vec[idx];
        thread_max = fmaxf(thread_max, vals.x);
        thread_max = fmaxf(thread_max, vals.y);
        thread_max = fmaxf(thread_max, vals.z);
        thread_max = fmaxf(thread_max, vals.w);
    }

    // ... (rest of kernel similar to v2)
}
```

**Expected Improvement:** 1.2-1.3x (better memory throughput)

### Phase 4: Final Tuning (30 minutes)

#### Optimization 4.1: Occupancy Tuning

```bash
# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak optimized_kernel

# Tune block size
# Try: 128, 256, 512, 1024 threads per block
# Measure impact on occupancy and performance
```

#### Optimization 4.2: Launch Configuration

```cuda
// Optimize grid and block dimensions
dim3 block(256);  // Empirically determined optimal
dim3 grid(batch_size);

// Dynamic shared memory sizing
size_t shared_mem_size = ...; // Calculate based on seq_len

optimized_kernel<<<grid, block, shared_mem_size>>>(input, output, ...);
```

---

## Testing and Validation

### Correctness Tests

```python
# tests/test_correctness.py

import torch
import numpy as np
from cuda_kernels import softmax_cuda  # Your CUDA kernel wrapper

def test_correctness_basic():
    """Test against PyTorch reference."""
    batch_size, seq_len = 32, 1024

    # Random input
    x = torch.randn(batch_size, seq_len, device='cuda')

    # Reference (PyTorch)
    reference = torch.nn.functional.softmax(x, dim=1)

    # Your kernel
    output = softmax_cuda(x)

    # Compare
    max_error = torch.max(torch.abs(reference - output)).item()
    assert max_error < 1e-5, f"Error too large: {max_error}"

def test_numerical_stability():
    """Test with extreme values."""
    x = torch.tensor([[1000.0, 1001.0, 999.0]], device='cuda')

    reference = torch.nn.functional.softmax(x, dim=1)
    output = softmax_cuda(x)

    assert torch.allclose(reference, output, atol=1e-5)

def test_edge_cases():
    """Test edge cases."""
    # All zeros
    x = torch.zeros(4, 128, device='cuda')
    output = softmax_cuda(x)
    assert torch.allclose(output, torch.full_like(output, 1.0/128), atol=1e-5)

    # Single element
    x = torch.randn(1, 1, device='cuda')
    output = softmax_cuda(x)
    assert torch.allclose(output, torch.ones_like(output), atol=1e-5)
```

### Performance Benchmarks

```python
# benchmarks/benchmark_softmax.py

import torch
import time
from cuda_kernels import (
    naive_softmax_cuda,
    optimized_v1_softmax_cuda,
    optimized_v2_softmax_cuda,
    optimized_final_softmax_cuda,
)

def benchmark_kernel(kernel_fn, x, num_iterations=1000):
    """Benchmark kernel performance."""
    # Warmup
    for _ in range(10):
        kernel_fn(x)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        kernel_fn(x)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    return elapsed_ms / num_iterations

def main():
    configs = [
        (32, 512),
        (64, 1024),
        (128, 2048),
        (64, 4096),
    ]

    kernels = {
        'Naive': naive_softmax_cuda,
        'Optimized V1': optimized_v1_softmax_cuda,
        'Optimized V2': optimized_v2_softmax_cuda,
        'Final': optimized_final_softmax_cuda,
        'PyTorch': lambda x: torch.nn.functional.softmax(x, dim=1),
    }

    print("Batch Size | Seq Len | Naive (ms) | V1 (ms) | V2 (ms) | Final (ms) | PyTorch (ms) | Speedup")
    print("-" * 100)

    for batch_size, seq_len in configs:
        x = torch.randn(batch_size, seq_len, device='cuda')

        results = {}
        for name, kernel in kernels.items():
            time_ms = benchmark_kernel(kernel, x)
            results[name] = time_ms

        speedup = results['Naive'] / results['Final']

        print(f"{batch_size:10d} | {seq_len:7d} | "
              f"{results['Naive']:10.4f} | "
              f"{results['Optimized V1']:7.4f} | "
              f"{results['Optimized V2']:7.4f} | "
              f"{results['Final']:10.4f} | "
              f"{results['PyTorch']:12.4f} | "
              f"{speedup:7.2f}x")

if __name__ == "__main__":
    main()
```

---

## Deliverables

### 1. Optimized CUDA Kernel(s)
- Source code with clear comments
- Multiple optimization versions (v1, v2, final)
- Build system (Makefile or CMakeLists.txt)

### 2. Profiling Reports
- Nsight Compute reports for each version
- Roofline analysis diagrams
- Bottleneck identification

### 3. Performance Analysis Document
```markdown
# Softmax Kernel Optimization Report

## Executive Summary
- **Final Speedup:** 3.2x over naive implementation
- **Key Optimizations:** Parallel reduction, vectorized loads, online algorithm
- **Limiting Factor:** Memory bandwidth (achieved 85% of peak)

## Baseline Analysis
[Nsight Compute metrics, bottlenecks identified]

## Optimization Journey

### V1: Shared Memory Caching
- **Technique:** Cache row in shared memory
- **Result:** 1.4x speedup
- **Metrics:** [before/after comparison]

### V2: Parallel Reduction
- **Technique:** Warp primitives for max/sum
- **Result:** 2.1x speedup cumulative
- **Metrics:** [...]

[Continue for all optimizations]

## Final Results
[Comparison table, charts]

## Lessons Learned
[What worked, what didn't, trade-offs]
```

### 4. Test Suite
- Correctness tests (passing)
- Numerical stability tests
- Performance benchmarks

### 5. Presentation Slides (Optional)
- Problem statement
- Optimization approach
- Key results
- Lessons learned

---

## Evaluation Rubric

### Performance (40 points)
- ✅ 2x speedup: 20 points
- ✅ 3x speedup: 30 points
- ✅ 4x+ speedup: 40 points

### Profiling & Analysis (20 points)
- ✅ Complete Nsight Compute reports: 10 points
- ✅ Roofline analysis: 5 points
- ✅ Bottleneck identification: 5 points

### Code Quality (20 points)
- ✅ Clean, well-commented code: 10 points
- ✅ Multiple optimization versions: 5 points
- ✅ Proper error handling: 5 points

### Testing (10 points)
- ✅ Correctness tests pass: 5 points
- ✅ Edge cases covered: 5 points

### Documentation (10 points)
- ✅ Optimization report: 5 points
- ✅ Build instructions: 3 points
- ✅ Usage examples: 2 points

**Total: 100 points**
**Passing: 70 points (requires 2x speedup minimum)**

---

## Common Pitfalls

### Pitfall 1: Premature Optimization
**Problem:** Optimizing before profiling.
**Solution:** Always profile first to identify real bottlenecks.

### Pitfall 2: Breaking Correctness
**Problem:** Optimization introduces numerical errors.
**Solution:** Test after every change; use reference implementation.

### Pitfall 3: Over-Optimizing Non-Bottlenecks
**Problem:** Spending time on parts that don't matter.
**Solution:** Focus on top 3 bottlenecks from profiler.

### Pitfall 4: Ignoring Occupancy
**Problem:** Using too many registers/shared memory.
**Solution:** Check occupancy metrics; adjust resource usage.

### Pitfall 5: Not Testing on Target Hardware
**Problem:** Optimization works on one GPU but not another.
**Solution:** Test on target architecture; use architecture-specific features carefully.

---

## Advanced Optimizations (Stretch Goals)

### 1. Tensor Core Utilization
Explore using tensor cores for certain operations (requires specific data layouts).

### 2. Multi-Pass Optimization
For very long sequences (>8K), implement multi-pass algorithm.

### 3. Triton Implementation
Rewrite kernel in Triton for easier optimization and better portability.

### 4. Adaptive Block Sizing
Dynamically choose block size based on `seq_len`.

### 5. Fused Operations
Fuse softmax with subsequent operations (e.g., softmax + attention).

---

## Resources

### CUDA Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Research Papers
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

### Code Examples
- NVIDIA CUDA Samples: `reduction`, `transpose`
- PyTorch CUDA Extensions: [Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

### Roofline Model
- [Roofline Model Visualization](https://crd.lbl.gov/departments/computer-science/par/research/roofline/)

---

**Ready to optimize? Start with profiling the naive kernel, then apply optimizations incrementally. Test and measure after each change!**

*Project 7.2: CUDA Kernel Optimization | Duration: 3-4 hours*
