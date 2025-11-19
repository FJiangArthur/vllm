# Day 13: Custom Operators & Advanced Kernel Techniques

> **Goal**: Master creating custom PyTorch operators, kernel fusion, and advanced optimization techniques
> **Time**: 6-8 hours
> **Prerequisites**: Days 8-12 completed, strong CUDA and C++ knowledge
> **Deliverables**: Custom operator implementation, fused kernels, Triton code, performance comparison

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Custom PyTorch Operators

**9:00-10:00** - PyTorch Extension Basics
**10:00-11:00** - Writing Custom CUDA Operators
**11:00-11:15** - Break
**11:15-12:30** - Binding C++/CUDA to Python

### Afternoon Session (3-4 hours): Advanced Techniques

**14:00-15:00** - Kernel Fusion Patterns
**15:00-16:00** - Triton: Python GPU Programming
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Build Fused RMSNorm+Quantize Operator

### Evening (Optional, 1-2 hours): Optimization

**19:00-21:00** - Performance tuning, comparison with existing ops, optimization case studies

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Create custom PyTorch CUDA extensions
- [ ] Implement fused kernels for better performance
- [ ] Use pybind11 for Python bindings
- [ ] Write GPU kernels in Triton
- [ ] Profile and optimize custom operators
- [ ] Integrate custom ops into production code
- [ ] Understand when fusion helps vs. hurts

---

## 📚 Morning: Custom PyTorch Operators (9:00-12:30)

### Task 1: PyTorch Extension Basics (60 min)

**Why Custom Operators?**

```
Reasons to write custom ops:
  1. Performance: Fuse multiple operations
  2. New functionality: Not available in PyTorch
  3. Hardware-specific: Leverage GPU features
  4. Memory efficiency: Reduce intermediate tensors

vLLM examples:
  - PagedAttention (not in PyTorch)
  - Fused quantization + matmul
  - Custom sampling kernels
  - Cache reshape operations
```

**Extension Structure**:

```
my_custom_op/
├── csrc/
│   ├── ops.h              # C++ declarations
│   ├── ops.cpp            # CPU implementations
│   └── ops_cuda.cu        # CUDA implementations
├── my_custom_op/
│   ├── __init__.py        # Python package
│   └── ops.py             # Python wrappers
├── setup.py               # Build script
└── README.md
```

**Setup.py for CUDA Extension**:

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architectures to compile for
def get_cuda_architectures():
    # Auto-detect or specify
    return ["75", "80", "86", "89"]  # V100, A100, RTX 3090, H100

setup(
    name="my_custom_op",
    version="0.1.0",
    author="Your Name",
    description="Custom CUDA operators for vLLM",
    packages=["my_custom_op"],
    ext_modules=[
        CUDAExtension(
            name="my_custom_op._C",  # Python module name
            sources=[
                "csrc/ops.cpp",      # CPU code
                "csrc/ops_cuda.cu",  # CUDA code
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    *[f"-gencode=arch=compute_{arch},code=sm_{arch}"
                      for arch in get_cuda_architectures()],
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    install_requires=["torch>=2.0.0"],
)
```

**Installation**:

```bash
# Development mode (editable)
pip install -e .

# This compiles C++/CUDA code and installs Python package
```

### Task 2: Writing Custom CUDA Operators (60 min)

**Example: Fused ReLU + Bias**

Create: `csrc/ops_cuda.cu`

```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel: Fused ReLU + Bias
// y = max(0, x + bias)

__global__ void relu_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    const int M,  // Batch size
    const int N   // Feature dimension
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;

    if (idx < total) {
        const int col = idx % N;

        // Load input and bias
        float val = x[idx] + bias[col];

        // Apply ReLU
        y[idx] = fmaxf(0.0f, val);
    }
}

// C++ wrapper
torch::Tensor relu_bias_cuda(
    torch::Tensor x,      // [M, N]
    torch::Tensor bias    // [N]
) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(1) == bias.size(0), "Dimension mismatch");

    const int M = x.size(0);
    const int N = x.size(1);

    // Allocate output
    auto y = torch::empty_like(x);

    // Launch kernel
    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;

    relu_bias_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N
    );

    // Check for errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "CUDA error: ", cudaGetErrorString(error));

    return y;
}
```

**CPU Fallback** (optional but recommended):

Create: `csrc/ops.cpp`

```cpp
#include <torch/extension.h>

// CPU version (fallback)
torch::Tensor relu_bias_cpu(
    torch::Tensor x,
    torch::Tensor bias
) {
    // Simple CPU implementation
    auto y = x.clone();
    y.add_(bias.unsqueeze(0));  // Broadcasting
    y.relu_();
    return y;
}

// Dispatcher: Choose CUDA or CPU
torch::Tensor relu_bias(
    torch::Tensor x,
    torch::Tensor bias
) {
    if (x.is_cuda()) {
        return relu_bias_cuda(x, bias);
    } else {
        return relu_bias_cpu(x, bias);
    }
}
```

### Task 3: Python Bindings (75 min)

**Pybind11 Bindings**:

Create: `csrc/ops.cpp` (continued)

```cpp
#include <torch/extension.h>

// Forward declarations
torch::Tensor relu_bias_cuda(torch::Tensor x, torch::Tensor bias);
torch::Tensor relu_bias_cpu(torch::Tensor x, torch::Tensor bias);

// Dispatcher
torch::Tensor relu_bias(torch::Tensor x, torch::Tensor bias) {
    if (x.is_cuda()) {
        return relu_bias_cuda(x, bias);
    } else {
        return relu_bias_cpu(x, bias);
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_bias", &relu_bias,
          "Fused ReLU + Bias operation",
          py::arg("x"),
          py::arg("bias"));

    // Can add more functions here
    // m.def("another_op", &another_op, ...);
}
```

**Python Wrapper**:

Create: `my_custom_op/ops.py`

```python
import torch
import my_custom_op._C as _C

def relu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + Bias operation.

    Args:
        x: Input tensor [M, N]
        bias: Bias vector [N]

    Returns:
        Output tensor [M, N] = max(0, x + bias)
    """
    # Input validation
    assert x.dim() == 2, "x must be 2D"
    assert bias.dim() == 1, "bias must be 1D"
    assert x.size(1) == bias.size(0), "Dimension mismatch"

    # Call C++ implementation
    return _C.relu_bias(x, bias)

# Make it a PyTorch autograd function (for backward pass)
class ReluBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        y = _C.relu_bias(x, bias)
        ctx.save_for_backward(x, bias, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, bias, y = ctx.saved_tensors

        # Gradient for ReLU: pass through if y > 0
        grad_x = grad_output * (y > 0).float()

        # Gradient for bias: sum over batch dimension
        grad_bias = grad_x.sum(dim=0)

        return grad_x, grad_bias

# Functional API
def relu_bias_autograd(x, bias):
    return ReluBiasFunction.apply(x, bias)
```

**Usage**:

```python
import torch
import my_custom_op

# Create inputs
x = torch.randn(128, 512, device='cuda')
bias = torch.randn(512, device='cuda')

# Use custom op
y = my_custom_op.relu_bias(x, bias)

# Compare with PyTorch
y_pytorch = torch.relu(x + bias)

print(f"Max diff: {(y - y_pytorch).abs().max().item()}")  # Should be ~0

# Benchmark
import time
iters = 1000

# Warmup
for _ in range(10):
    _ = my_custom_op.relu_bias(x, bias)
    _ = torch.relu(x + bias)

torch.cuda.synchronize()

# Custom op
start = time.time()
for _ in range(iters):
    _ = my_custom_op.relu_bias(x, bias)
torch.cuda.synchronize()
custom_time = time.time() - start

# PyTorch
start = time.time()
for _ in range(iters):
    _ = torch.relu(x + bias)
torch.cuda.synchronize()
pytorch_time = time.time() - start

print(f"Custom op: {custom_time*1000/iters:.3f} ms")
print(f"PyTorch:   {pytorch_time*1000/iters:.3f} ms")
print(f"Speedup:   {pytorch_time/custom_time:.2f}x")

# Expected: ~1.5-2x speedup (one kernel launch vs two)
```

---

## 🔬 Afternoon: Advanced Techniques (14:00-18:00)

### Task 4: Kernel Fusion Patterns (60 min)

Study: `learning_materials/modules/module_03_cuda_kernels/05_kernel_fusion_techniques.md`

**Common Fusion Patterns in vLLM**:

```
1. Normalization + Linear
   LayerNorm(x) @ W  →  Fused: Norm and matmul in one kernel

2. Linear + Activation
   x @ W + b, then ReLU/GELU  →  Fused GEMM with activation

3. Attention + Projection
   Attention(Q,K,V) @ W_out  →  Can fuse in some cases

4. Quantize + Matmul
   Quantize(W), then W @ x  →  Dequant during GEMM (we saw this!)

5. Residual + Normalization
   x + residual, then LayerNorm  →  Single pass through memory
```

**Example: Fused RMSNorm + Quantize**

```cuda
// RMSNorm: y = x / sqrt(mean(x^2)) * scale
// Then quantize to INT8

__global__ void fused_rmsnorm_quantize_kernel(
    const half* __restrict__ input,      // [M, N] FP16
    const half* __restrict__ weight,     // [N] FP16 (scale)
    int8_t* __restrict__ output,         // [M, N] INT8
    half* __restrict__ scale_out,        // [M] quantization scales
    const int M,
    const int N,
    const float eps = 1e-6f
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) return;

    const half* x = input + row * N;

    // Step 1: Compute RMSNorm
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Reduce across threads
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Compute RMS
    float rms = sqrtf(shared_sum[0] / N + eps);
    float inv_rms = 1.0f / rms;

    // Step 2: Normalize and find max (for quantization)
    float max_val = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = __half2float(x[i]) * inv_rms;
        float scaled = val * __half2float(weight[i]);
        max_val = fmaxf(max_val, fabsf(scaled));

        // Store in shared memory for quantization
        shared_sum[i % blockDim.x] = scaled;  // Reuse shared mem
    }

    // Reduce max
    __shared__ float shared_max[256];
    shared_max[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    float quant_scale = shared_max[0] / 127.0f;

    // Store scale
    if (tid == 0) {
        scale_out[row] = __float2half(quant_scale);
    }

    // Step 3: Quantize
    int8_t* out = output + row * N;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = __half2float(x[i]) * inv_rms * __half2float(weight[i]);
        int8_t qval = (int8_t)roundf(val / quant_scale);
        qval = max(-127, min(127, qval));
        out[i] = qval;
    }
}

// Without fusion: 3 kernel launches (RMSNorm, Scale, Quantize)
// With fusion: 1 kernel launch
// Speedup: ~2-3x (fewer memory accesses, less overhead)
```

### Task 5: Triton - Python GPU Programming (60 min)

**What is Triton?**

```
Triton: Python-based GPU programming language
  - Write GPU kernels in Python-like syntax
  - Automatic optimization (tiling, coalescing, etc.)
  - Easier than CUDA for many use cases
  - Used in vLLM for some operations

Advantages:
  ✓ Simpler syntax than CUDA
  ✓ Automatic memory optimization
  ✓ Fast iteration (no C++ compilation)
  ✗ Less control than raw CUDA
  ✗ Newer, less mature ecosystem
```

**Triton Example: Vector Add**:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    output_ptr,  # Pointer to output
    n_elements,  # Size
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # Get program ID
    pid = tl.program_id(axis=0)

    # Compute offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary checking
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Allocate output
    output = torch.empty_like(x)

    # Get size
    n_elements = x.numel()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

# Usage
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
z = vector_add(x, y)

print((z - (x + y)).abs().max())  # Should be ~0
```

**Triton: Fused Softmax**:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)

    # Compute row offset
    row_start = row_idx * n_cols

    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row_ptr = input_ptr + row_start + col_offsets
    row = tl.load(row_ptr, mask=mask, other=-float('inf'))

    # Find max (for numerical stability)
    row_max = tl.max(row, axis=0)

    # Subtract max and compute exp
    numerator = tl.exp(row - row_max)

    # Compute sum
    denominator = tl.sum(numerator, axis=0)

    # Compute softmax
    softmax_output = numerator / denominator

    # Store result
    output_row_ptr = output_ptr + row_start + col_offsets
    tl.store(output_row_ptr, softmax_output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Fused softmax in Triton.
    x: [M, N] input tensor
    """
    M, N = x.shape
    output = torch.empty_like(x)

    # Next power of 2 for BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Launch one program per row
    grid = (M,)

    softmax_kernel[grid](
        x, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

# Test
x = torch.randn(128, 512, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, dim=1)

print(f"Max diff: {(y_triton - y_torch).abs().max().item()}")

# Benchmark
import time
iters = 1000

for _ in range(10):  # Warmup
    _ = softmax(x)
    _ = torch.softmax(x, dim=1)

torch.cuda.synchronize()

start = time.time()
for _ in range(iters):
    _ = softmax(x)
torch.cuda.synchronize()
triton_time = time.time() - start

start = time.time()
for _ in range(iters):
    _ = torch.softmax(x, dim=1)
torch.cuda.synchronize()
torch_time = time.time() - start

print(f"Triton: {triton_time*1000/iters:.3f} ms")
print(f"PyTorch: {torch_time*1000/iters:.3f} ms")
print(f"Speedup: {torch_time/triton_time:.2f}x")
```

### Task 6: Build Fused RMSNorm+Quantize (90 min)

**Exercise**: Complete implementation with all optimizations

```python
# File: my_custom_op/fused_ops.py

import torch
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_quant_kernel(
    input_ptr,          # Input [M, N]
    weight_ptr,         # Weight [N]
    output_ptr,         # Output INT8 [M, N]
    scale_ptr,          # Quantization scales [M]
    M, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Compute row start
    row_start = row_idx * N

    # Load weight
    col_offsets = tl.arange(0, BLOCK_SIZE)
    weight_mask = col_offsets < N
    weight = tl.load(weight_ptr + col_offsets, mask=weight_mask)

    # Load input row
    input_ptrs = input_ptr + row_start + col_offsets
    x = tl.load(input_ptrs, mask=weight_mask, other=0.0)

    # Compute RMSNorm
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms = tl.sqrt(mean_sq + eps)
    x_norm = x / rms

    # Apply weight
    x_scaled = x_norm * weight

    # Find max for quantization
    x_abs = tl.abs(x_scaled)
    max_val = tl.max(x_abs, axis=0)

    # Compute scale
    quant_scale = max_val / 127.0

    # Store scale
    if tl.program_id(1) == 0:  # First thread
        tl.store(scale_ptr + row_idx, quant_scale)

    # Quantize
    x_quant_float = x_scaled / quant_scale
    x_quant = x_quant_float.to(tl.int8)

    # Clamp to INT8 range
    x_quant = tl.where(x_quant > 127, 127, x_quant)
    x_quant = tl.where(x_quant < -127, -127, x_quant)

    # Store output
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, x_quant, mask=weight_mask)

def fused_rmsnorm_quantize(
    x: torch.Tensor,      # [M, N] FP16/FP32
    weight: torch.Tensor,  # [N] FP16/FP32
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + INT8 Quantization

    Returns:
        output: INT8 tensor [M, N]
        scales: FP16/FP32 tensor [M]
    """
    M, N = x.shape
    assert weight.shape[0] == N

    # Allocate output
    output = torch.empty((M, N), dtype=torch.int8, device=x.device)
    scales = torch.empty(M, dtype=x.dtype, device=x.device)

    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    fused_rmsnorm_quant_kernel[grid](
        x, weight, output, scales,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output, scales

# Test and benchmark
if __name__ == "__main__":
    M, N = 1024, 4096
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, device='cuda', dtype=torch.float16)

    # Run fused version
    output, scales = fused_rmsnorm_quantize(x, weight)

    # Verify against unfused
    # 1. RMSNorm
    x_norm = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-6)
    x_scaled = x_norm * weight

    # 2. Quantize
    max_vals = torch.max(torch.abs(x_scaled), dim=1)[0]
    scales_ref = max_vals / 127.0
    output_ref = torch.clamp(
        torch.round(x_scaled / scales_ref.unsqueeze(1)),
        -127, 127
    ).to(torch.int8)

    print(f"Output match: {torch.allclose(output.float(), output_ref.float(), atol=1)}")
    print(f"Scales match: {torch.allclose(scales, scales_ref, rtol=1e-2)}")

    # Benchmark
    # TODO: Add timing code
```

---

## 📊 Performance Analysis (Evening, 19:00-21:00)

### Task 7: Comprehensive Op Comparison

```python
#!/usr/bin/env python3
"""
Compare custom ops vs PyTorch native ops
"""

import torch
import time
import numpy as np

def benchmark_op(name, func, *args, iters=1000):
    """Benchmark an operation"""

    # Warmup
    for _ in range(10):
        _ = func(*args)
    torch.cuda.synchronize()

    # Measure
    start = time.time()
    for _ in range(iters):
        _ = func(*args)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    avg_time_ms = elapsed * 1000 / iters
    return avg_time_ms

# Test configurations
configs = [
    {"M": 128, "N": 512},
    {"M": 1024, "N": 4096},
    {"M": 4096, "N": 4096},
]

for config in configs:
    M, N = config["M"], config["N"]
    print(f"\n{'='*60}")
    print(f"Configuration: M={M}, N={N}")
    print(f"{'='*60}")

    x = torch.randn(M, N, device='cuda')
    bias = torch.randn(N, device='cuda')

    # Compare operations
    ops = {
        "PyTorch (unfused)": lambda: torch.relu(x + bias),
        "Custom (fused)": lambda: my_custom_op.relu_bias(x, bias),
        # Add more comparisons
    }

    for name, op in ops.items():
        time_ms = benchmark_op(name, op)
        print(f"{name:25s}: {time_ms:.3f} ms")
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Custom Operators**
- PyTorch extension structure
- CUDA kernel integration
- Pybind11 bindings
- Autograd integration

✅ **Kernel Fusion**
- Common fusion patterns
- Implementation techniques
- Performance benefits

✅ **Triton Programming**
- Python-based GPU kernels
- Automatic optimization
- Use cases and trade-offs

✅ **Advanced Optimization**
- When to fuse operations
- Performance analysis
- Production integration

### Knowledge Check

**Question 1**: When should you fuse operations?
<details>
<summary>Answer</summary>
Fuse when: (1) Operations are memory-bound, (2) Intermediate results are large, (3) Kernel launch overhead is significant. Don't fuse when operations are compute-bound or when fusion increases register pressure/reduces occupancy.
</details>

**Question 2**: What are advantages of Triton over CUDA?
<details>
<summary>Answer</summary>
Triton advantages: (1) Simpler Python-like syntax, (2) Automatic tiling and memory optimization, (3) Faster development iteration. CUDA advantages: (1) More control over optimization, (2) Mature ecosystem, (3) Better for complex kernels.
</details>

**Question 3**: How do you create a PyTorch autograd function?
<details>
<summary>Answer</summary>
Subclass torch.autograd.Function, implement forward() (with ctx.save_for_backward()) and backward() methods. The backward method receives output gradients and returns input gradients.
</details>

### Daily Reflection

**What went well?**
- [ ] I can create custom CUDA operators
- [ ] I understand kernel fusion benefits
- [ ] I wrote Triton kernels successfully
- [ ] I integrated custom ops with PyTorch

**What was challenging?**
- [ ] Pybind11 syntax and compilation
- [ ] Triton memory model
- [ ] Debugging custom ops
- [ ] Performance tuning

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 14

Tomorrow is Week 2 review day:
- **Comprehensive Review**: All Week 2 concepts
- **Integration Project**: Build end-to-end optimized pipeline
- **Performance Analysis**: Compare optimizations
- **Week 3 Preview**: System components and integration

**Preparation**:
- Review all Week 2 notes
- Organize code examples
- Prepare questions

---

## 📚 Additional Resources

**Module Materials**:
- [ ] `modules/module_03_cuda_kernels/05_kernel_fusion_techniques.md`

**vLLM Files to Study**:
- [ ] `csrc/ops.h` - Operator declarations
- [ ] `setup.py` - Build configuration
- [ ] `vllm/model_executor/layers/` - Custom layer implementations

**External Resources**:
- [ ] PyTorch Extension Tutorial
- [ ] Triton Documentation and Tutorials
- [ ] "Kernel Fusion in Deep Learning" papers

---

**Day 13 Complete! Custom operator mastery achieved! 🛠️**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
