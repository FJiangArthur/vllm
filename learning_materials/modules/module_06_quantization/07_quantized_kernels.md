# Quantized CUDA Kernels for LLM Inference

## Table of Contents
1. [Introduction to Quantized Kernels](#introduction-to-quantized-kernels)
2. [INT8/INT4 GEMM Kernels](#int8int4-gemm-kernels)
3. [Kernel Optimization Techniques](#kernel-optimization-techniques)
4. [vLLM Quantization Kernels](#vllm-quantization-kernels)
5. [Benchmarking Kernels](#benchmarking-kernels)
6. [Custom Kernel Development](#custom-kernel-development)

---

## Introduction to Quantized Kernels

### Why Custom Kernels?

Standard PyTorch operations don't fully exploit quantized data types. Custom CUDA kernels are essential for:

1. **Fused Operations**: Combine dequantization + GEMM
2. **Efficient Memory Access**: Optimal layouts for INT4/INT8
3. **Tensor Core Utilization**: Hardware acceleration
4. **Reduced Overhead**: Minimize dequantization cost

### Kernel Categories

```
vLLM Quantization Kernels:
├── INT8 Kernels (w8a8/)
│   ├── Weight-Activation INT8 GEMM
│   └── Dynamic quantization kernels
├── INT4 Kernels (marlin/, gptq/, awq/)
│   ├── Packed INT4 GEMM
│   └── Group-wise dequantization
├── FP8 Kernels (cutlass_w4a8/)
│   └── FP8 Tensor Core operations
└── Activation Kernels
    ├── Dynamic quantization
    └── Static dequantization
```

---

## INT8/INT4 GEMM Kernels

### INT8 Matrix Multiplication

#### Naive Implementation

```cuda
// Simplified INT8 GEMM (educational purpose)
__global__ void int8_gemm_naive(
    const int8_t* A,      // [M, K]
    const int8_t* B,      // [K, N]
    int32_t* C,           // [M, N] output (INT32 accumulation)
    float scale_a,
    float scale_b,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        // Compute dot product
        for (int k = 0; k < K; k++) {
            int8_t a_val = A[row * K + k];
            int8_t b_val = B[k * N + col];
            sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
        }

        // Store INT32 result
        C[row * N + col] = sum;
    }
}
```

**Problems:**
- No shared memory usage
- Poor memory coalescing
- No Tensor Core utilization
- Sequential computation

#### Tensor Core Implementation

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Tensor Core INT8 GEMM
__global__ void int8_gemm_tensor_core(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int M, int N, int K
) {
    // WMMA dimensions: 16x16x16 for INT8
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Warp and lane ID
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0);

    // Tile over K dimension
    for (int k_step = 0; k_step < K; k_step += WMMA_K) {
        int a_row = warp_row * WMMA_M;
        int a_col = k_step;
        int b_row = k_step;
        int b_col = warp_col * WMMA_N;

        // Load fragments from global memory
        load_matrix_sync(a_frag, A + a_row * K + a_col, K);
        load_matrix_sync(b_frag, B + b_row * N + b_col, N);

        // Perform INT8 matrix multiply-accumulate
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    int c_row = warp_row * WMMA_M;
    int c_col = warp_col * WMMA_N;
    store_matrix_sync(C + c_row * N + c_col, c_frag, N, mem_row_major);
}
```

**Performance:**
- ~10-20x faster than naive
- Full Tensor Core utilization
- Efficient memory access patterns

### INT4 Packed GEMM

INT4 requires special handling due to packing:

```cuda
// INT4 GEMM with packed weights
__global__ void int4_gemm_kernel(
    const half* X,           // [M, K] FP16 activations
    const int32_t* W_packed, // [N, K/8] packed INT4 weights
    const half* scales,      // [N, K/group_size] FP16 scales
    half* Y,                 // [M, N] FP16 output
    int M, int N, int K,
    int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row < M && col < N) {
        float sum = 0.0f;

        // Process K dimension in groups
        for (int k = 0; k < K; k++) {
            // Unpack INT4 weight
            int pack_idx = k / 8;
            int sub_idx = k % 8;

            int32_t packed = W_packed[col * (K/8) + pack_idx];
            int8_t w_int4 = (packed >> (sub_idx * 4)) & 0x0F;

            // Convert to signed: -8 to 7
            if (w_int4 > 7) w_int4 -= 16;

            // Get scale for this group
            int group_idx = k / group_size;
            float scale = __half2float(scales[col * (K/group_size) + group_idx]);

            // Dequantize and multiply
            float w_fp = static_cast<float>(w_int4) * scale;
            float x_fp = __half2float(X[row * K + k]);

            sum += x_fp * w_fp;
        }

        Y[row * N + col] = __float2half(sum);
    }
}
```

---

## Kernel Optimization Techniques

### 1. Shared Memory Tiling

```cuda
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void int8_gemm_shared_memory(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    __shared__ int8_t A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int32_t sum = 0;

    // Tile across K dimension
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        A_tile[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
            A[row * K + a_col] : 0;

        B_tile[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
            B[b_row * N + col] : 0;

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 2. Vectorized Loads

```cuda
// Load 4 INT8 values at once using vectorized load
__global__ void int8_gemm_vectorized(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    using vec_t = int4;  // 128-bit vector (16 bytes = 16 INT8 values)

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;

        // Process K in chunks of 16
        for (int k = 0; k < K; k += 16) {
            // Vectorized load of 16 INT8 values
            vec_t a_vec = reinterpret_cast<const vec_t*>(A + row * K + k)[0];
            vec_t b_vec = reinterpret_cast<const vec_t*>(B + k * N + col)[0];

            // Extract and multiply individual elements
            int8_t* a_vals = reinterpret_cast<int8_t*>(&a_vec);
            int8_t* b_vals = reinterpret_cast<int8_t*>(&b_vec);

            for (int i = 0; i < 16; i++) {
                sum += a_vals[i] * b_vals[i];
            }
        }

        C[row * N + col] = sum;
    }
}
```

### 3. Fused Dequantization

```cuda
// Fused INT8 dequantize + GEMM
__global__ void fused_dequant_gemm(
    const half* X,          // FP16 activations
    const int8_t* W_q,      // INT8 weights
    const half* W_scales,   // Per-channel scales
    half* Y,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        float scale = __half2float(W_scales[col]);

        // Fused dequantize and multiply
        for (int k = 0; k < K; k++) {
            float x = __half2float(X[row * K + k]);
            float w = static_cast<float>(W_q[col * K + k]) * scale;  // Dequant
            sum += x * w;  // Multiply
        }

        Y[row * N + col] = __float2half(sum);
    }
}
```

---

## vLLM Quantization Kernels

### Directory Structure

```bash
csrc/quantization/
├── awq/
│   ├── gemm_kernels.cu      # AWQ INT4 GEMM
│   └── gemm_kernels.h
├── gptq/
│   ├── q_gemm.cu            # GPTQ INT4 GEMM
│   └── matrix.cuh
├── gptq_marlin/
│   ├── gptq_marlin.cu       # Optimized GPTQ/Marlin kernels
│   └── gptq_marlin.cuh
├── marlin/
│   ├── marlin_cuda_kernel.cu  # Marlin INT4 kernels (fastest)
│   ├── marlin.cuh
│   └── marlin_repack.cu
├── w8a8/
│   ├── w8a8_gemm.cu         # INT8 weight-activation GEMM
│   └── cutlass_kernels/
└── activation_kernels.cu     # Dynamic quantization kernels
```

### Example: Marlin Kernel Integration

```python
# File: vllm/model_executor/layers/quantization/marlin.py (simplified)

import torch

def marlin_gemm(
    x: torch.Tensor,              # [batch, in_features] FP16
    weight_packed: torch.Tensor,  # [out_features, in_features/8] INT32
    scales: torch.Tensor,         # [out_features, num_groups] FP16
    workspace: torch.Tensor,      # Temporary workspace
    num_bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Marlin INT4 GEMM kernel call.

    Marlin is one of the fastest INT4 kernels available.
    """
    import vllm._C

    output = vllm._C.marlin_gemm(
        x,
        weight_packed,
        scales,
        workspace,
        num_bits,
        group_size,
    )

    return output
```

### AWQ Kernel

```cpp
// csrc/quantization/awq/gemm_kernels.cu (simplified)

void awq_gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    int num_out_channels = _kernel.size(1);

    auto options = torch::TensorOptions()
        .dtype(_in_feats.dtype())
        .device(_in_feats.device());

    auto _out_feats = torch::empty({num_in_feats, num_out_channels}, options);

    // Launch CUDA kernel
    dim3 num_blocks((num_out_channels + 63) / 64);
    dim3 threads_per_block(64);

    gemm_forward_4bit_cuda_kernel<<<num_blocks, threads_per_block>>>(
        num_in_feats,
        num_in_channels,
        num_out_channels,
        _in_feats.data_ptr<half>(),
        _kernel.data_ptr<int>(),
        _scaling_factors.data_ptr<half>(),
        _zeros.data_ptr<int>(),
        _out_feats.data_ptr<half>()
    );
}
```

---

## Benchmarking Kernels

### Micro-Benchmarking INT8 Kernels

```python
import torch
import time

def benchmark_int8_kernel(
    kernel_fn,
    M, N, K,
    num_iterations=100,
):
    """
    Benchmark INT8 kernel performance.

    Args:
        kernel_fn: CUDA kernel function
        M, N, K: Matrix dimensions
        num_iterations: Number of iterations

    Returns:
        Performance metrics
    """
    # Create test data
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
    C = torch.zeros((M, N), dtype=torch.int32, device='cuda')

    # Warmup
    for _ in range(10):
        kernel_fn(A, B, C)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        kernel_fn(A, B, C)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Compute metrics
    avg_time = elapsed / num_iterations * 1000  # ms
    ops = 2 * M * N * K  # multiply-accumulate operations
    tops = (ops * num_iterations / elapsed) / 1e12  # TOPS

    # Memory bandwidth
    bytes_read = (M * K + K * N) * 1  # INT8 = 1 byte
    bytes_written = M * N * 4  # INT32 = 4 bytes
    total_bytes = bytes_read + bytes_written
    bandwidth_gbs = (total_bytes * num_iterations / elapsed) / 1e9

    return {
        'avg_time_ms': avg_time,
        'tops': tops,
        'bandwidth_gb_s': bandwidth_gbs,
        'size': (M, N, K),
    }


# Example usage
def compare_kernels():
    """Compare different INT8 kernel implementations."""
    sizes = [
        (128, 128, 128),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    kernels = {
        'PyTorch': lambda A, B, C: torch.matmul(A.float(), B.float().T),
        'cuBLAS INT8': lambda A, B, C: torch.ops.aten._int_mm(A, B.T),
        # Add custom kernels here
    }

    print(f"{'Size':<20} {'Kernel':<15} {'Time (ms)':<12} {'TOPS':<8} {'BW (GB/s)':<10}")
    print("-" * 75)

    for size in sizes:
        M, N, K = size
        for name, kernel in kernels.items():
            result = benchmark_int8_kernel(kernel, M, N, K)
            print(f"{str(size):<20} {name:<15} {result['avg_time_ms']:<12.3f} "
                  f"{result['tops']:<8.2f} {result['bandwidth_gb_s']:<10.2f}")
```

### Profiling with Nsight Compute

```bash
# Profile specific kernel
ncu --set full --target-processes all \
    --kernel-name int8_gemm_tensor_core \
    --launch-count 1 \
    --export profile_int8 \
    python benchmark_script.py

# Analyze metrics
ncu --import profile_int8.ncu-rep \
    --page details \
    --print-summary per-kernel
```

**Key Metrics to Check:**
- Achieved occupancy
- Memory throughput (% of peak)
- Tensor Core utilization
- SM efficiency
- Warp execution efficiency

---

## Custom Kernel Development

### Step-by-Step: Custom INT8 Kernel

#### 1. Define Kernel

```cuda
// custom_int8_kernel.cu

__global__ void custom_int8_gemm(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K
) {
    // Your implementation here
}
```

#### 2. Create PyTorch Extension

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_int8_kernels',
    ext_modules=[
        CUDAExtension(
            name='custom_int8_kernels',
            sources=['custom_int8_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_80',  # Ampere
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

#### 3. Python Wrapper

```python
# wrapper.py
import torch
import custom_int8_kernels

def custom_int8_matmul(A, B):
    """
    Wrapper for custom INT8 kernel.

    Args:
        A: [M, K] INT8 tensor
        B: [K, N] INT8 tensor

    Returns:
        C: [M, N] INT32 tensor
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Dimension mismatch"

    C = torch.zeros((M, N), dtype=torch.int32, device=A.device)

    # Launch kernel
    custom_int8_kernels.gemm_forward(A, B, C, M, N, K)

    return C
```

#### 4. Testing and Validation

```python
def test_custom_kernel():
    """Test custom kernel against reference."""
    M, N, K = 128, 128, 128

    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')

    # Reference (PyTorch)
    C_ref = torch.matmul(A.int(), B.T.int())

    # Custom kernel
    C_custom = custom_int8_matmul(A, B)

    # Validate
    assert torch.allclose(C_ref, C_custom), "Results don't match!"

    print("✓ Kernel validation passed!")


def benchmark_custom_kernel():
    """Benchmark custom kernel vs reference."""
    import time

    M, N, K = 4096, 4096, 4096

    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')

    # Warmup
    for _ in range(10):
        _ = custom_int8_matmul(A, B)

    torch.cuda.synchronize()

    # Benchmark custom
    start = time.perf_counter()
    for _ in range(100):
        C_custom = custom_int8_matmul(A, B)
    torch.cuda.synchronize()
    time_custom = (time.perf_counter() - start) / 100

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(100):
        C_ref = torch.matmul(A.int(), B.T.int())
    torch.cuda.synchronize()
    time_ref = (time.perf_counter() - start) / 100

    print(f"Custom kernel: {time_custom*1000:.3f} ms")
    print(f"PyTorch:       {time_ref*1000:.3f} ms")
    print(f"Speedup:       {time_ref/time_custom:.2f}x")
```

---

## Summary

### Key Takeaways

1. **Custom Kernels Essential**: Standard ops don't fully exploit quantization
2. **Tensor Cores**: Critical for INT8/FP8 performance
3. **Fusion**: Dequantize + GEMM fusion reduces overhead
4. **vLLM Kernels**: State-of-the-art implementations available
5. **Benchmarking**: Always profile to verify performance

### Best Practices

1. Use Tensor Cores for large matrices
2. Fuse dequantization with computation
3. Optimize memory access patterns
4. Profile with Nsight Compute
5. Validate against reference implementations

---

## References

1. NVIDIA CUTLASS Library: https://github.com/NVIDIA/cutlass
2. vLLM Quantization Kernels: csrc/quantization/
3. "Efficient Memory Management for Large Language Model Serving with PagedAttention"
4. NVIDIA Tensor Core Programming Guide

---

**Module 6.7 Complete** • Next: [Accuracy Evaluation](08_accuracy_evaluation.md)
