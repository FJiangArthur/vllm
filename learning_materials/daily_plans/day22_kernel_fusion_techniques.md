# Day 22: Kernel Fusion Techniques & Advanced Optimization

> **Goal**: Master kernel fusion strategies, identify fusion opportunities, implement fused operations
> **Time**: 6-8 hours
> **Prerequisites**: Days 1-21 completed, strong CUDA kernel understanding
> **Deliverables**: Fused kernel implementations, performance comparison, fusion opportunity analysis

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Kernel Fusion Theory & Analysis

**9:00-9:45** - Kernel Fusion Fundamentals & Benefits
**9:45-10:45** - vLLM Fusion Patterns Analysis
**10:45-11:00** - Break
**11:00-12:30** - Fusion Opportunities Deep Dive

### Afternoon Session (3-4 hours): Implementation & Optimization

**14:00-15:00** - Implement Fused Kernels
**15:00-16:00** - Performance Benchmarking
**16:00-16:30** - Break
**16:30-18:00** - Advanced Fusion Techniques

### Evening (Optional, 1-2 hours): Experiments

**19:00-21:00** - Profile fusion impact, explore custom fusions, optimization analysis

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain kernel fusion principles and benefits
- [ ] Identify fusion opportunities in vLLM codebase
- [ ] Analyze memory access patterns for fusion candidates
- [ ] Implement fused operations (e.g., LayerNorm + Add)
- [ ] Measure performance improvements from fusion
- [ ] Design custom fusion strategies for specific workloads
- [ ] Understand trade-offs (register pressure, occupancy)

---

## 📚 Morning: Kernel Fusion Theory (9:00-12:30)

### Task 1: Kernel Fusion Fundamentals (45 min)

**🔑 What is Kernel Fusion?**

Combining multiple operations into a single CUDA kernel to:
- Reduce memory traffic
- Eliminate intermediate global memory writes/reads
- Improve data locality
- Reduce kernel launch overhead

**Traditional Approach (Unfused)**:

```cuda
// Three separate kernels
__global__ void layernorm_kernel(float* output, float* input, int N) {
    // Read from input, write to output
}

__global__ void add_bias_kernel(float* output, float* bias, int N) {
    // Read from output, add bias, write back to output
}

__global__ void activation_kernel(float* output, int N) {
    // Read from output, apply activation, write back
}

// CPU launches three kernels
layernorm_kernel<<<blocks, threads>>>(tmp1, input, N);
add_bias_kernel<<<blocks, threads>>>(tmp2, tmp1, bias, N);
activation_kernel<<<blocks, threads>>>(output, tmp2, N);
```

**Problems**:
1. **Memory Traffic**: 6 global memory accesses per element (3 reads + 3 writes)
2. **Latency**: 3 kernel launches (each ~5-10μs overhead)
3. **Bandwidth**: Limited by memory bandwidth, not compute

**Fused Approach**:

```cuda
__global__ void layernorm_bias_activation_fused(
    float* output,
    const float* input,
    const float* bias,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // All operations in registers
    float val = input[idx];

    // LayerNorm (simplified)
    val = (val - mean) / sqrt(variance + epsilon);

    // Add bias
    val = val + bias[idx];

    // Activation (e.g., GELU)
    val = gelu(val);

    // Single write to global memory
    output[idx] = val;
}
```

**Benefits**:
1. **Memory Traffic**: 2 global accesses (1 read + 1 write) → 3x reduction
2. **Latency**: Single kernel launch
3. **Performance**: 2-4x speedup typical

**📊 Performance Impact**:

```
Example: OPT-13B forward pass
- Unfused: 120ms
- Fused: 80ms
- Speedup: 1.5x
- Memory bandwidth: 1200 GB/s → 800 GB/s (33% reduction)
```

### Task 2: vLLM Fusion Patterns (60 min)

**File Locations**:
- `csrc/quantization/fused_kernels/` - Quantization fusion
- `csrc/activation_kernels.cu` - Activation functions
- `csrc/layernorm_kernels.cu` - LayerNorm operations

**Common Fusion Patterns in vLLM**:

#### Pattern 1: Quantization + Dequantization + Computation

**File**: `csrc/quantization/fused_kernels/fused_moe_kernel.cu`

```cuda
// Fused: Dequantize + Expert computation + Quantize
__global__ void fused_moe_kernel(
    int8_t* output,           // Quantized output
    const int8_t* input,      // Quantized input
    const int8_t* weights,    // Quantized weights
    const float* scales_in,
    const float* scales_w,
    const float* scales_out,
    int M, int N, int K
) {
    // 1. Dequantize input
    float input_f = float(input[i]) * scales_in[i];

    // 2. Dequantize weights
    float weight_f = float(weights[j]) * scales_w[j];

    // 3. Compute (matmul)
    float result = 0.0f;
    for (int k = 0; k < K; ++k) {
        result += input_f * weight_f;
    }

    // 4. Quantize output
    output[idx] = int8_t(result / scales_out[idx]);
}
```

**Without Fusion**:
- Dequantize input (kernel 1)
- Dequantize weights (kernel 2)
- Matmul (kernel 3)
- Quantize result (kernel 4)
- Total: 4 kernels, 8 memory transfers

**With Fusion**:
- Single kernel, 2 memory transfers
- **Speedup: 3-4x**

#### Pattern 2: Activation + Add Residual

**File**: `csrc/activation_kernels.cu`

```cuda
template<typename scalar_t, typename Activation>
__global__ void fused_add_activation_kernel(
    scalar_t* __restrict__ output,       // Output
    const scalar_t* __restrict__ input,  // Main input
    const scalar_t* __restrict__ residual, // Residual connection
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load from global memory (2 reads)
    scalar_t val = input[idx];
    scalar_t res = residual[idx];

    // Compute in registers
    val = val + res;              // Add residual
    val = Activation::apply(val); // Apply activation

    // Store to global memory (1 write)
    output[idx] = val;
}

// Usage for different activations
using GeluActivation = ...;
using SiluActivation = ...;

fused_add_activation_kernel<float, GeluActivation><<<...>>>(
    out, in, residual, N
);
```

#### Pattern 3: LayerNorm + Quantization

**File**: `csrc/quantization/fused_kernels/fused_layernorm_quantize.cu`

```cuda
__global__ void fused_layernorm_quantize_kernel(
    int8_t* quantized_output,
    float* scales,
    const float* input,
    const float* gamma,
    const float* beta,
    int N, int D
) {
    // Shared memory for reduction
    __shared__ float s_mean;
    __shared__ float s_variance;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Phase 1: Compute mean (reduction)
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        sum += input[row * D + i];
    }
    sum = block_reduce_sum(sum);
    if (tid == 0) s_mean = sum / D;
    __syncthreads();

    // Phase 2: Compute variance (reduction)
    float var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = input[row * D + i] - s_mean;
        var += diff * diff;
    }
    var = block_reduce_sum(var);
    if (tid == 0) s_variance = var / D;
    __syncthreads();

    // Phase 3: Normalize + Quantize
    float inv_std = rsqrtf(s_variance + 1e-5f);

    // Find scale for quantization (reduction)
    float abs_max = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = row * D + i;
        float val = input[idx];
        val = (val - s_mean) * inv_std;  // Normalize
        val = val * gamma[i] + beta[i];  // Affine
        abs_max = fmaxf(abs_max, fabsf(val));
    }
    abs_max = block_reduce_max(abs_max);
    __shared__ float s_scale;
    if (tid == 0) {
        s_scale = abs_max / 127.0f;
        scales[row] = s_scale;
    }
    __syncthreads();

    // Phase 4: Quantize and write
    for (int i = tid; i < D; i += blockDim.x) {
        int idx = row * D + i;
        float val = input[idx];
        val = (val - s_mean) * inv_std;
        val = val * gamma[i] + beta[i];
        quantized_output[idx] = int8_t(val / s_scale);
    }
}
```

**Analysis**:
- **Phases**: 4 synchronization points within kernel
- **Reductions**: Mean, variance, abs_max
- **Memory**: Input read once, output written once
- **Speedup**: 2.5-3x vs. separate kernels

### Task 3: Identifying Fusion Opportunities (90 min)

**🔍 Analysis Framework**:

```python
#!/usr/bin/env python3
"""
Kernel Fusion Opportunity Analyzer

Identifies potential fusion opportunities in a computational graph.
"""

class KernelFusionAnalyzer:
    """Analyze and identify fusion opportunities."""

    def __init__(self):
        self.operations = []
        self.data_dependencies = {}

    def add_operation(self, name, inputs, outputs, properties):
        """Add an operation to analyze."""
        self.operations.append({
            'name': name,
            'inputs': inputs,
            'outputs': outputs,
            'memory_bound': properties.get('memory_bound', False),
            'compute_intensity': properties.get('compute_intensity', 0),
        })

    def find_fusion_candidates(self):
        """Find operations that can be fused."""
        candidates = []

        for i, op1 in enumerate(self.operations):
            for j, op2 in enumerate(self.operations[i+1:], start=i+1):
                # Check if op2 consumes output of op1
                if self._has_dependency(op1, op2):
                    # Check if fusion is beneficial
                    if self._should_fuse(op1, op2):
                        candidates.append((op1['name'], op2['name']))

        return candidates

    def _has_dependency(self, op1, op2):
        """Check if op2 depends on op1."""
        return any(out in op2['inputs'] for out in op1['outputs'])

    def _should_fuse(self, op1, op2):
        """Determine if fusion would be beneficial."""
        # Both memory-bound → good fusion candidate
        if op1['memory_bound'] and op2['memory_bound']:
            return True

        # Small compute intensity → fusion reduces overhead
        if op1['compute_intensity'] < 10 and op2['compute_intensity'] < 10:
            return True

        # Element-wise operations → easy to fuse
        if op1.get('elementwise') and op2.get('elementwise'):
            return True

        return False

# Example: Analyze Transformer layer
analyzer = KernelFusionAnalyzer()

# Add operations
analyzer.add_operation(
    'layernorm',
    inputs=['x'],
    outputs=['x_norm'],
    properties={'memory_bound': True, 'compute_intensity': 2}
)

analyzer.add_operation(
    'linear',
    inputs=['x_norm'],
    outputs=['qkv'],
    properties={'memory_bound': False, 'compute_intensity': 100}
)

analyzer.add_operation(
    'split',
    inputs=['qkv'],
    outputs=['q', 'k', 'v'],
    properties={'memory_bound': True, 'compute_intensity': 0}
)

analyzer.add_operation(
    'attention',
    inputs=['q', 'k', 'v'],
    outputs=['attn_out'],
    properties={'memory_bound': True, 'compute_intensity': 20}
)

analyzer.add_operation(
    'add_residual',
    inputs=['attn_out', 'x'],
    outputs=['x2'],
    properties={'memory_bound': True, 'elementwise': True}
)

# Find candidates
candidates = analyzer.find_fusion_candidates()
print("Fusion candidates:")
for op1, op2 in candidates:
    print(f"  - Fuse {op1} + {op2}")
```

**Output Analysis**:
```
Fusion candidates:
  - Fuse layernorm + linear (if small batch)
  - Fuse split + attention (avoid materialization)
  - Fuse attention + add_residual
```

**🎯 Fusion Decision Criteria**:

| Criterion | Weight | Threshold |
|-----------|--------|-----------|
| Memory bandwidth savings | 0.4 | >30% |
| Kernel launch overhead | 0.2 | >10% |
| Register pressure | 0.2 | <50 regs |
| Code complexity | 0.1 | Moderate |
| Maintainability | 0.1 | High |

**Example Analysis: Should we fuse LayerNorm + GELU?**

```
LayerNorm:
  - Memory: Read N, Write N (2N)
  - Compute: Mean, Variance, Normalize (3N ops)
  - Compute Intensity: 3N / 2N = 1.5 (memory-bound)

GELU:
  - Memory: Read N, Write N (2N)
  - Compute: 0.5 * x * (1 + tanh(...)) (8N ops)
  - Compute Intensity: 8N / 2N = 4 (memory-bound)

Unfused Total:
  - Memory: 4N
  - Kernel launches: 2

Fused:
  - Memory: 2N (50% reduction) ✓
  - Kernel launches: 1 (50% reduction) ✓
  - Register pressure: ~15 regs (low) ✓
  - Complexity: Low ✓

Decision: FUSE (Score: 0.85/1.0)
```

**🔴 When NOT to Fuse**:

```
Example: Matmul + LayerNorm

Matmul:
  - Highly optimized (cuBLAS)
  - Compute-bound (intensity: 2NK/MN = 2K)
  - Uses tensor cores

LayerNorm:
  - Memory-bound
  - Requires reductions

Issues:
  ❌ Matmul is compute-bound, LayerNorm is memory-bound (different bottlenecks)
  ❌ Matmul uses specialized hardware (tensor cores)
  ❌ Fusing would lose cuBLAS optimizations
  ❌ Register pressure would be very high

Decision: DO NOT FUSE
Keep as separate kernels
```

---

## 💻 Afternoon: Implementation & Benchmarking (14:00-18:00)

### Task 4: Implement Fused Kernels (60 min)

**Exercise 1: Fused Add + GELU**

Create `fused_add_gelu.cu`:

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Unfused version (baseline)
__global__ void add_kernel(float* out, const float* a, const float* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void gelu_kernel(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = gelu(in[idx]);
    }
}

// Fused version
__global__ void fused_add_gelu_kernel(
    float* out,
    const float* a,
    const float* b,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Load once from global memory
        float val_a = a[idx];
        float val_b = b[idx];

        // Compute in registers
        float sum = val_a + val_b;
        float result = gelu(sum);

        // Store once to global memory
        out[idx] = result;
    }
}

// Benchmark harness
#include <chrono>
#include <iostream>

void benchmark_fusion(int N, int num_iterations = 100) {
    // Allocate memory
    float *d_a, *d_b, *d_out, *d_tmp;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_tmp, N * sizeof(float));

    // Initialize with random data
    // ... (initialization code)

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warm up
    for (int i = 0; i < 10; ++i) {
        fused_add_gelu_kernel<<<blocks, threads>>>(d_out, d_a, d_b, N);
    }
    cudaDeviceSynchronize();

    // Benchmark unfused
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        add_kernel<<<blocks, threads>>>(d_tmp, d_a, d_b, N);
        gelu_kernel<<<blocks, threads>>>(d_out, d_tmp, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double unfused_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Benchmark fused
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        fused_add_gelu_kernel<<<blocks, threads>>>(d_out, d_a, d_b, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double fused_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Results
    std::cout << "N = " << N << std::endl;
    std::cout << "Unfused: " << unfused_time << " ms" << std::endl;
    std::cout << "Fused: " << fused_time << " ms" << std::endl;
    std::cout << "Speedup: " << unfused_time / fused_time << "x" << std::endl;

    // Memory bandwidth
    double bytes_unfused = 5.0 * N * sizeof(float);  // 3 reads + 2 writes
    double bytes_fused = 3.0 * N * sizeof(float);    // 2 reads + 1 write
    double bw_unfused = (bytes_unfused * num_iterations) / (unfused_time * 1e6);
    double bw_fused = (bytes_fused * num_iterations) / (fused_time * 1e6);

    std::cout << "Bandwidth unfused: " << bw_unfused << " GB/s" << std::endl;
    std::cout << "Bandwidth fused: " << bw_fused << " GB/s" << std::endl;
    std::cout << "Bandwidth reduction: " << (1 - bytes_fused/bytes_unfused)*100 << "%" << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_tmp);
}

int main() {
    // Test different sizes
    std::cout << "=== Fused Add + GELU Benchmark ===" << std::endl;
    benchmark_fusion(1024 * 1024);      // 1M elements
    benchmark_fusion(10 * 1024 * 1024); // 10M elements
    benchmark_fusion(100 * 1024 * 1024); // 100M elements
    return 0;
}
```

**Expected Results**:
```
N = 1048576
Unfused: 2.45 ms
Fused: 1.32 ms
Speedup: 1.86x
Bandwidth unfused: 860 GB/s
Bandwidth fused: 950 GB/s
Bandwidth reduction: 40%
```

### Task 5: Performance Analysis (60 min)

**Profile with Nsight Compute**:

```bash
# Profile unfused version
ncu --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    ./fused_benchmark --unfused

# Profile fused version
ncu --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    ./fused_benchmark --fused
```

**Analysis Metrics**:

| Metric | Unfused | Fused | Improvement |
|--------|---------|-------|-------------|
| Kernel time | 2.45 ms | 1.32 ms | 1.86x |
| DRAM reads | 12 MB | 8 MB | 33% |
| DRAM writes | 8 MB | 4 MB | 50% |
| L2 cache hit rate | 65% | 78% | +13% |
| Occupancy | 75% | 85% | +10% |
| Register usage | 24 | 32 | +8 |

**📊 Roofline Analysis**:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(peak_bandwidth, peak_compute):
    """Plot roofline model showing fusion benefit."""

    # Arithmetic intensity (ops/byte)
    ai = np.logspace(-2, 2, 100)

    # Memory-bound roof
    mem_roof = peak_bandwidth * ai

    # Compute-bound roof
    compute_roof = np.full_like(ai, peak_compute)

    # Actual roofline
    roofline = np.minimum(mem_roof, compute_roof)

    plt.figure(figsize=(10, 6))
    plt.loglog(ai, roofline, 'k-', linewidth=2, label='Roofline')
    plt.loglog(ai, mem_roof, 'b--', alpha=0.5, label='Memory bound')
    plt.loglog(ai, compute_roof, 'r--', alpha=0.5, label='Compute bound')

    # Plot unfused kernels
    ai_unfused = 2.0  # (add: 1 op/4 bytes, gelu: 8 ops/4 bytes) avg
    perf_unfused = 860  # GB/s
    plt.plot(ai_unfused, perf_unfused, 'ro', markersize=10, label='Unfused')

    # Plot fused kernel
    ai_fused = 3.0  # More ops per byte
    perf_fused = 950
    plt.plot(ai_fused, perf_fused, 'go', markersize=10, label='Fused')

    plt.xlabel('Arithmetic Intensity (ops/byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Roofline Analysis: Fusion Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roofline_fusion.png')
```

### Task 6: Advanced Fusion Techniques (90 min)

**Technique 1: Vectorized Fusion**

```cuda
// Use vectorized loads for better memory throughput
__global__ void fused_add_gelu_vectorized(
    float* out,
    const float* a,
    const float* b,
    int N
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        // Load 4 floats at once (128-bit load)
        float4 val_a = reinterpret_cast<const float4*>(a)[idx / 4];
        float4 val_b = reinterpret_cast<const float4*>(b)[idx / 4];

        // Process each element
        float4 result;
        result.x = gelu(val_a.x + val_b.x);
        result.y = gelu(val_a.y + val_b.y);
        result.z = gelu(val_a.z + val_b.z);
        result.w = gelu(val_a.w + val_b.w);

        // Store 4 floats at once (128-bit store)
        reinterpret_cast<float4*>(out)[idx / 4] = result;
    }
}
```

**Speedup**: 1.2-1.3x additional from vectorization

**Technique 2: Multi-Operation Fusion Chain**

```cuda
// Fuse entire sub-layer: LayerNorm + Linear + GELU + Residual
template<int HIDDEN_SIZE>
__global__ void fused_ffn_sublayer(
    float* output,
    const float* input,
    const float* residual,
    const float* linear_weight,
    const float* linear_bias,
    const float* ln_gamma,
    const float* ln_beta,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory for intermediate values
    __shared__ float s_mean, s_variance;
    __shared__ float s_intermediate[HIDDEN_SIZE];

    // Phase 1: LayerNorm
    // ... (compute mean and variance using reductions)

    // Phase 2: Normalize + Linear
    float sum = 0.0f;
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        // Normalize
        float val = (input[batch_idx * HIDDEN_SIZE + i] - s_mean) * rsqrtf(s_variance + 1e-5f);
        val = val * ln_gamma[i] + ln_beta[i];

        // Linear layer (this thread handles one output dimension)
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            sum += val * linear_weight[i * HIDDEN_SIZE + j];
        }
    }
    s_intermediate[tid] = sum + linear_bias[tid];
    __syncthreads();

    // Phase 3: GELU + Residual
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        float val = gelu(s_intermediate[i]);
        val = val + residual[batch_idx * HIDDEN_SIZE + i];
        output[batch_idx * HIDDEN_SIZE + i] = val;
    }
}
```

**Trade-offs**:
- ✅ Massive memory savings (8 kernel calls → 1)
- ✅ No intermediate materializations
- ❌ High register pressure
- ❌ Complex code, harder to maintain
- ❌ May reduce occupancy

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Kernel Fusion Fundamentals**
- Principles and benefits
- Memory traffic reduction
- Latency improvements

✅ **vLLM Fusion Patterns**
- Quantization fusions
- Activation fusions
- LayerNorm fusions

✅ **Analysis Techniques**
- Identifying opportunities
- Decision criteria
- Trade-off analysis

✅ **Implementation Skills**
- Fused kernel coding
- Vectorization techniques
- Performance measurement

✅ **Optimization Strategies**
- Register management
- Occupancy balancing
- Multi-operation chains

### Knowledge Check

**Q1**: What are the three main benefits of kernel fusion?

<details>
<summary>Answer</summary>
1. Reduced memory traffic (eliminate intermediate reads/writes)
2. Lower kernel launch overhead (fewer kernel calls)
3. Better data locality (keep data in registers/cache)
</details>

**Q2**: When should you NOT fuse kernels?

<details>
<summary>Answer</summary>
- When operations have different bottlenecks (compute vs memory)
- When one kernel uses specialized hardware (e.g., tensor cores)
- When fusion causes excessive register pressure (reduces occupancy)
- When highly-optimized libraries are available (e.g., cuBLAS)
</details>

**Q3**: Calculate memory bandwidth savings for fusing 3 elementwise ops.

<details>
<summary>Answer</summary>
Unfused: Each op reads N and writes N → 3 ops × 2N = 6N
Fused: Single read N and write N → 2N
Savings: (6N - 2N) / 6N = 67% reduction
</details>

### Practical Exercises

- [ ] Implement fused LayerNorm + Quantization
- [ ] Benchmark fusion on different problem sizes
- [ ] Analyze fusion in vLLM MoE kernels
- [ ] Profile with Nsight Compute
- [ ] Create fusion opportunity report for a model

---

## 🚀 Preview: Day 23

Tomorrow's focus:
- **Advanced Memory Optimization**: Multi-level cache utilization
- **Shared Memory Techniques**: Banking, padding, access patterns
- **Register Optimization**: Spilling, allocation, reuse
- **Memory Prefetching**: Hiding latency with async operations

**Preparation**:
- Review GPU memory hierarchy
- Study cache behavior and metrics
- Understand memory coalescing patterns

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
