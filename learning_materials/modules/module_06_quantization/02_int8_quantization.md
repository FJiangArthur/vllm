# INT8 Quantization for LLM Inference

## Table of Contents
1. [Introduction to INT8 Quantization](#introduction-to-int8-quantization)
2. [INT8 Arithmetic and Hardware Support](#int8-arithmetic-and-hardware-support)
3. [Weight-Only INT8 Quantization](#weight-only-int8-quantization)
4. [Weight-Activation INT8 Quantization](#weight-activation-int8-quantization)
5. [LLM.int8() Algorithm](#llmint8-algorithm)
6. [INT8 Quantization in vLLM](#int8-quantization-in-vllm)
7. [Performance Analysis](#performance-analysis)
8. [Practical Implementation](#practical-implementation)
9. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
10. [Exercises](#exercises)

---

## Introduction to INT8 Quantization

### Why INT8?

INT8 (8-bit integer) quantization has become the **gold standard** for LLM inference quantization due to several factors:

**Hardware Support:**
- All modern NVIDIA GPUs (V100+) have INT8 Tensor Cores
- 2-4x higher throughput compared to FP16
- Native support in frameworks (TensorRT, ONNX Runtime)

**Accuracy Preservation:**
- With proper techniques, <1% accuracy degradation
- Better than INT4 for most models
- Compatible with most LLM architectures

**Memory Efficiency:**
- 2x reduction from FP16
- Significant bandwidth savings
- Enables larger batch sizes

### INT8 Representation

INT8 uses 8 bits to represent integers in range [-128, 127] for signed or [0, 255] for unsigned.

```
Signed INT8:   -128 to 127
Unsigned INT8:   0 to 255

Binary representation (signed):
-128: 10000000
  -1: 11111111
   0: 00000000
   1: 00000001
 127: 01111111
```

### Memory Footprint Comparison

```python
import torch

def compare_memory_footprint():
    """Compare memory usage across precisions."""
    size = (4096, 4096)  # Typical weight matrix

    # Create tensors
    fp32_tensor = torch.randn(size, dtype=torch.float32)
    fp16_tensor = torch.randn(size, dtype=torch.float16)
    int8_tensor = torch.randint(-128, 127, size, dtype=torch.int8)

    # Memory usage
    fp32_memory = fp32_tensor.element_size() * fp32_tensor.nelement()
    fp16_memory = fp16_tensor.element_size() * fp16_tensor.nelement()
    int8_memory = int8_tensor.element_size() * int8_tensor.nelement()

    print(f"FP32: {fp32_memory / 1024**2:.2f} MB")
    print(f"FP16: {fp16_memory / 1024**2:.2f} MB")
    print(f"INT8: {int8_memory / 1024**2:.2f} MB")
    print(f"\nReduction FP32→INT8: {fp32_memory / int8_memory:.1f}x")
    print(f"Reduction FP16→INT8: {fp16_memory / int8_memory:.1f}x")

# Output:
# FP32: 64.00 MB
# FP16: 32.00 MB
# INT8: 16.00 MB
# Reduction FP32→INT8: 4.0x
# Reduction FP16→INT8: 2.0x
```

---

## INT8 Arithmetic and Hardware Support

### Tensor Core Operations

NVIDIA Tensor Cores provide specialized hardware for INT8 matrix multiplication:

```
C (INT32) = A (INT8) × B (INT8) + C (INT32)
```

**Key Characteristics:**
- Input matrices: INT8
- Accumulation: INT32 (to prevent overflow)
- Output: INT32 (requires dequantization to FP16/FP32)

### Tensor Core Performance

**NVIDIA GPU Specifications:**

| GPU    | FP16 TFLOPS | INT8 TOPS | Speedup |
|--------|-------------|-----------|---------|
| V100   | 125         | 500       | 4.0x    |
| A100   | 312         | 624       | 2.0x    |
| H100   | 989         | 1979      | 2.0x    |

**Note**: Actual speedup depends on:
- Memory bandwidth bottleneck
- Matrix dimensions
- Dequantization overhead

### CUDA INT8 GEMM Example

```cuda
// Simplified INT8 GEMM using Tensor Cores
// Real implementation uses CUTLASS or cuBLAS

__global__ void int8_gemm_kernel(
    const int8_t* A,    // [M, K]
    const int8_t* B,    // [K, N]
    int32_t* C,         // [M, N]
    float scale_a,
    float scale_b,
    int M, int N, int K
) {
    // Use wmma (Warp Matrix Multiply-Accumulate)
    using namespace nvcuda::wmma;

    // Declare fragments
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0);

    // Compute matrix multiplication
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + ... , K);
        load_matrix_sync(b_frag, B + ... , K);

        // Perform INT8 matrix multiplication
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + ... , c_frag, N, mem_row_major);
}
```

### PyTorch INT8 Operations

```python
import torch

def int8_matmul_example():
    """Example of INT8 matrix multiplication in PyTorch."""

    # Create INT8 tensors
    M, N, K = 128, 128, 128
    A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    B_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')

    # Scales for dequantization
    scale_a = torch.tensor(0.01, device='cuda')
    scale_b = torch.tensor(0.01, device='cuda')

    # Method 1: Using torch.ops.aten._int_mm (fast, but INT32 output)
    C_int32 = torch.ops.aten._int_mm(A_int8, B_int8)

    # Dequantize
    C_fp16 = C_int32.to(torch.float16) * (scale_a * scale_b)

    # Method 2: Manual dequantize then FP16 matmul (slower)
    A_fp16 = A_int8.to(torch.float16) * scale_a
    B_fp16 = B_int8.to(torch.float16) * scale_b
    C_fp16_ref = torch.matmul(A_fp16, B_fp16)

    print(f"INT8 path max difference: {(C_fp16 - C_fp16_ref).abs().max():.4f}")

    return C_fp16
```

---

## Weight-Only INT8 Quantization

### Concept

Quantize **only weights** to INT8; activations remain in FP16/BF16.

**Advantages:**
- Simple to implement
- Minimal accuracy loss
- 2x memory reduction
- Good bandwidth improvement

**Disadvantages:**
- No compute acceleration (dequantize before GEMM)
- Limited speedup (1.3-1.7x typically)

### Implementation

```python
import torch
import torch.nn as nn

class WeightOnlyInt8Linear(nn.Module):
    """
    Weight-only INT8 quantized linear layer.

    Forward pass:
    1. Dequantize weights to FP16
    2. Perform FP16 GEMM
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weights (INT8)
        self.register_buffer(
            'weight_int8',
            torch.zeros((out_features, in_features), dtype=torch.int8)
        )

        # Per-channel scales (FP16)
        self.register_buffer(
            'weight_scales',
            torch.ones((out_features, 1), dtype=torch.float16)
        )

    @classmethod
    def from_float(cls, float_linear):
        """Convert FP16 linear to weight-only INT8."""
        quantized = cls(float_linear.in_features, float_linear.out_features)

        # Extract weights
        W = float_linear.weight.data  # [out_features, in_features]

        # Compute per-channel scales
        W_max = W.abs().max(dim=1, keepdim=True)[0]
        scales = W_max / 127.0
        scales = scales.clamp(min=1e-5)

        # Quantize
        W_int8 = (W / scales).round().clamp(-128, 127).to(torch.int8)

        # Store
        quantized.weight_int8.copy_(W_int8)
        quantized.weight_scales.copy_(scales.half())

        return quantized

    def forward(self, x):
        """
        Args:
            x: [batch, in_features] FP16 tensor

        Returns:
            [batch, out_features] FP16 tensor
        """
        # Dequantize weights on-the-fly
        W_fp16 = self.weight_int8.half() * self.weight_scales

        # Standard FP16 GEMM
        output = torch.matmul(x, W_fp16.T)

        return output

    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'weight_dtype=int8')
```

### Optimization: Fused Dequantization

Instead of separate dequantize + matmul, fuse operations:

```python
# Custom CUDA kernel (pseudo-code)
def fused_int8_dequant_gemm(x_fp16, W_int8, scales):
    """
    Fused dequantization and GEMM.

    Kernel does:
    for each output element:
        acc = 0
        for k in range(K):
            w_dequant = W_int8[k] * scale[output_row]
            acc += x_fp16[k] * w_dequant
        output = acc
    """
    pass
```

### Benchmarking Weight-Only INT8

```python
def benchmark_weight_only_int8():
    """Compare FP16 vs Weight-Only INT8 performance."""
    import time

    M, N, K = 8192, 8192, 8192  # Large matrices
    device = 'cuda'

    # Create FP16 linear layer
    linear_fp16 = nn.Linear(K, N, bias=False).half().to(device)

    # Convert to INT8
    linear_int8 = WeightOnlyInt8Linear.from_float(linear_fp16)

    # Test input
    x = torch.randn(M, K).half().to(device)

    # Warmup
    for _ in range(10):
        _ = linear_fp16(x)
        _ = linear_int8(x)

    # Benchmark FP16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y_fp16 = linear_fp16(x)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / 100

    # Benchmark INT8
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y_int8 = linear_int8(x)
    torch.cuda.synchronize()
    int8_time = (time.perf_counter() - start) / 100

    # Accuracy check
    mse = ((y_fp16 - y_int8) ** 2).mean()

    print(f"FP16 time: {fp16_time*1000:.2f} ms")
    print(f"INT8 time: {int8_time*1000:.2f} ms")
    print(f"Speedup: {fp16_time / int8_time:.2f}x")
    print(f"MSE: {mse:.6f}")

# Expected output on A100:
# FP16 time: 12.5 ms
# INT8 time: 8.3 ms
# Speedup: 1.5x
# MSE: 0.000012
```

---

## Weight-Activation INT8 Quantization

### Concept

Quantize **both weights and activations** to INT8 for maximum compute acceleration.

**Advantages:**
- Full INT8 compute path
- 2-4x speedup potential
- Tensor Core acceleration

**Disadvantages:**
- Activation quantization more challenging
- Can degrade accuracy significantly
- Requires calibration

### Static vs Dynamic Activation Quantization

#### Dynamic Quantization

Compute activation scale **per-batch** at runtime:

```python
class DynamicInt8Linear(nn.Module):
    """Weight-Activation INT8 with dynamic activation quantization."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scales', torch.ones((out_features, 1), dtype=torch.float16))

    def forward(self, x):
        """
        Args:
            x: [batch, in_features] FP16

        Returns:
            [batch, out_features] FP16
        """
        # Dynamic activation quantization
        x_scale = x.abs().max() / 127.0
        x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)

        # INT8 GEMM
        output_int32 = torch.ops.aten._int_mm(x_int8, self.weight_int8.T)

        # Dequantize
        output = output_int32.to(torch.float16) * (x_scale * self.weight_scales.T)

        return output
```

**Pros:**
- No calibration needed
- Adapts to each batch

**Cons:**
- Overhead of computing scale each time
- Not optimal for static workloads

#### Static Quantization

Precompute activation scales using calibration data:

```python
class StaticInt8Linear(nn.Module):
    """Weight-Activation INT8 with static activation quantization."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scales', torch.ones((out_features, 1), dtype=torch.float16))

        # Static activation scale (computed during calibration)
        self.register_buffer('activation_scale', torch.tensor(1.0, dtype=torch.float16))

    def forward(self, x):
        """Forward with precomputed activation scale."""
        # Static quantization
        x_int8 = (x / self.activation_scale).round().clamp(-128, 127).to(torch.int8)

        # INT8 GEMM
        output_int32 = torch.ops.aten._int_mm(x_int8, self.weight_int8.T)

        # Dequantize
        output = output_int32.to(torch.float16) * (self.activation_scale * self.weight_scales.T)

        return output

    def calibrate(self, activations):
        """
        Compute static activation scale from calibration data.

        Args:
            activations: List of activation tensors from calibration
        """
        # Use 99.9th percentile to handle outliers
        all_values = torch.cat([a.flatten() for a in activations])
        scale = torch.quantile(all_values.abs(), 0.999) / 127.0
        self.activation_scale.copy_(scale)
```

### Calibration Process

```python
def calibrate_model(model, calibration_loader, num_samples=512):
    """
    Calibrate all layers in model with static quantization.

    Args:
        model: Model with StaticInt8Linear layers
        calibration_loader: DataLoader with representative data
        num_samples: Number of samples to use for calibration
    """
    # Collect activations for each layer
    layer_activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if name not in layer_activations:
                layer_activations[name] = []
            # Store input activations
            layer_activations[name].append(input[0].detach().cpu())
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, StaticInt8Linear):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    # Collect calibration data
    model.eval()
    samples_collected = 0

    with torch.no_grad():
        for batch in calibration_loader:
            _ = model(batch['input_ids'].cuda())
            samples_collected += batch['input_ids'].size(0)

            if samples_collected >= num_samples:
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calibrate each layer
    for name, module in model.named_modules():
        if isinstance(module, StaticInt8Linear) and name in layer_activations:
            module.calibrate(layer_activations[name])
            print(f"Calibrated {name}: scale = {module.activation_scale:.6f}")

    return model
```

---

## LLM.int8() Algorithm

### Overview

**LLM.int8()** is a landmark paper (Dettmers et al., 2022) that introduced a **mixed-precision decomposition** to handle outlier features in LLMs.

**Key Insight**: A small fraction (<0.1%) of features have extreme magnitudes and cause quantization errors.

### The Outlier Problem

```python
def demonstrate_outlier_problem():
    """Show how outliers affect quantization."""
    import matplotlib.pyplot as plt

    # Simulate activation distribution with outliers
    normal_activations = torch.randn(10000) * 0.5
    outliers = torch.randn(10) * 50  # Extreme outliers

    activations = torch.cat([normal_activations, outliers])

    # Quantize
    scale = activations.abs().max() / 127
    act_int8 = (activations / scale).round().clamp(-128, 127)
    act_dequant = act_int8 * scale

    # Compute error
    error = (activations - act_dequant).abs()

    print(f"Scale (dominated by outliers): {scale:.4f}")
    print(f"Mean error (normal values): {error[:10000].mean():.4f}")
    print(f"Mean error (outliers): {error[10000:].mean():.4f}")
    print(f"Max error: {error.max():.4f}")

# Output shows outliers dominate scale, causing large errors for normal values
```

### Mixed-Precision Decomposition

**Algorithm:**

```
For matrix multiplication Y = XW:

1. Identify outlier features: features with magnitude > threshold
2. Decompose:
   X = X_normal + X_outlier  (separate outlier columns)
   W = W_normal + W_outlier  (separate outlier rows)

3. Compute:
   Y_outlier = X_outlier @ W_outlier  (FP16, small portion)
   Y_normal  = X_normal  @ W_normal   (INT8, large portion)
   Y = Y_outlier + Y_normal
```

**Implementation:**

```python
class LLMInt8Linear(nn.Module):
    """
    LLM.int8() implementation with mixed-precision decomposition.
    """

    def __init__(self, in_features, out_features, threshold=6.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # Quantized weights
        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scales', torch.ones((out_features, 1), dtype=torch.float16))

        # FP16 weights for outlier features (sparse)
        self.register_buffer('weight_fp16', None)
        self.register_buffer('outlier_idx', None)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, in_features] FP16

        Returns:
            [batch, seq_len, out_features] FP16
        """
        batch_size, seq_len, _ = x.shape

        # Detect outlier features (columns with magnitude > threshold * std)
        x_flat = x.view(-1, self.in_features)
        outlier_mask = (x_flat.abs() > self.threshold).any(dim=0)

        if outlier_mask.any():
            # Extract outlier and normal features
            outlier_idx = outlier_mask.nonzero(as_tuple=True)[0]
            normal_idx = (~outlier_mask).nonzero(as_tuple=True)[0]

            x_outlier = x_flat[:, outlier_idx]  # [batch*seq, num_outliers]
            x_normal = x_flat[:, normal_idx]    # [batch*seq, in_features - num_outliers]

            # Outlier path (FP16)
            W_outlier = self.weight_int8[:, outlier_idx].half() * self.weight_scales
            y_outlier = torch.matmul(x_outlier, W_outlier.T)

            # Normal path (INT8)
            W_normal_int8 = self.weight_int8[:, normal_idx]
            W_normal_scales = self.weight_scales

            # Quantize normal activations
            x_scale = x_normal.abs().max() / 127.0
            x_normal_int8 = (x_normal / x_scale).round().clamp(-128, 127).to(torch.int8)

            # INT8 matmul
            y_normal_int32 = torch.ops.aten._int_mm(x_normal_int8, W_normal_int8.T)
            y_normal = y_normal_int32.half() * (x_scale * W_normal_scales.T)

            # Combine
            y = y_outlier + y_normal

        else:
            # No outliers, use full INT8 path
            x_scale = x_flat.abs().max() / 127.0
            x_int8 = (x_flat / x_scale).round().clamp(-128, 127).to(torch.int8)
            y_int32 = torch.ops.aten._int_mm(x_int8, self.weight_int8.T)
            y = y_int32.half() * (x_scale * self.weight_scales.T)

        # Reshape output
        y = y.view(batch_size, seq_len, self.out_features)

        return y
```

### Performance Characteristics

**Outlier Statistics in Real Models:**

```python
def analyze_outliers_in_model(model, dataloader, threshold=6.0):
    """
    Analyze outlier features across all layers.

    Args:
        model: LLM model
        dataloader: Test dataloader
        threshold: Outlier threshold

    Returns:
        Statistics on outlier features per layer
    """
    outlier_stats = {}

    # Hook to capture activations
    def make_hook(name):
        def hook(module, input, output):
            x = input[0].detach()
            x_flat = x.view(-1, x.size(-1))

            # Compute per-feature statistics
            feature_max = x_flat.abs().max(dim=0)[0]
            feature_mean = x_flat.abs().mean(dim=0)
            feature_std = x_flat.std(dim=0)

            # Identify outliers
            outliers = feature_max > (threshold * feature_std)

            outlier_stats[name] = {
                'num_outliers': outliers.sum().item(),
                'total_features': x.size(-1),
                'outlier_ratio': outliers.float().mean().item(),
                'max_magnitude': feature_max.max().item(),
            }

        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run inference
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch['input_ids'].cuda())
            break  # One batch sufficient

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print statistics
    for name, stats in outlier_stats.items():
        print(f"{name}:")
        print(f"  Outliers: {stats['num_outliers']} / {stats['total_features']} "
              f"({stats['outlier_ratio']*100:.2f}%)")
        print(f"  Max magnitude: {stats['max_magnitude']:.2f}")

    return outlier_stats
```

**Typical Results (OPT-175B):**
- ~0.15% of features are outliers
- Concentrated in specific layers (e.g., layer 4, 8)
- Without handling: 10-20% accuracy degradation
- With LLM.int8(): <1% accuracy degradation

---

## INT8 Quantization in vLLM

### vLLM INT8 Support

vLLM supports multiple INT8 quantization backends:

```python
# File: vllm/model_executor/layers/quantization/__init__.py

QUANTIZATION_METHODS = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "fp8": Fp8Config,
    # ... more methods
}
```

### Loading INT8 Models in vLLM

```python
from vllm import LLM, SamplingParams

# Example 1: Load GPTQ INT8 model
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq",
    dtype="half",
    gpu_memory_utilization=0.9,
)

# Example 2: Load with custom quantization config
from vllm.model_executor.layers.quantization import GPTQConfig

quant_config = GPTQConfig(
    weight_bits=8,
    group_size=128,
    desc_act=False,
)

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="gptq",
    quantization_param_path="./gptq_config.json",
)

# Generate
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### vLLM INT8 Kernel Integration

vLLM uses optimized INT8 kernels:

```python
# File: vllm/model_executor/layers/quantization/gptq.py (simplified)

class GPTQLinearMethod:
    """GPTQ quantization method implementation."""

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply quantized weights to input."""

        # Unpack quantized weights
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros

        # Call optimized CUDA kernel
        output = ops.gptq_gemm(x, qweight, scales, qzeros)

        if bias is not None:
            output += bias

        return output
```

**CUDA Kernel Path:**
```
csrc/quantization/gptq/q_gemm.cu
```

### Benchmarking INT8 in vLLM

```python
import time
import torch
from vllm import LLM, SamplingParams

def benchmark_vllm_int8(model_name, num_requests=100):
    """
    Benchmark INT8 quantized model in vLLM.

    Args:
        model_name: HuggingFace model name or path
        num_requests: Number of requests to benchmark
    """
    # Load model
    llm = LLM(
        model=model_name,
        quantization="gptq",  # or "awq"
        dtype="half",
        gpu_memory_utilization=0.9,
    )

    # Prepare requests
    prompts = [f"Request {i}: Write a story about " for i in range(num_requests)]
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )

    # Warmup
    _ = llm.generate(prompts[:5], sampling_params)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    outputs = llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Compute metrics
    total_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    throughput = total_tokens / total_time
    latency = total_time / num_requests

    print(f"Model: {model_name}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average latency: {latency*1000:.2f} ms/request")
    print(f"Total tokens generated: {total_tokens}")

    # Memory usage
    memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {memory_allocated:.2f} GB")

    return {
        'throughput': throughput,
        'latency': latency,
        'memory_gb': memory_allocated,
    }

# Example usage
results_fp16 = benchmark_vllm_int8("meta-llama/Llama-2-7b-hf")  # FP16 baseline
results_int8 = benchmark_vllm_int8("TheBloke/Llama-2-7B-GPTQ")  # INT8

print(f"\nSpeedup: {results_int8['throughput'] / results_fp16['throughput']:.2f}x")
print(f"Memory reduction: {results_fp16['memory_gb'] / results_int8['memory_gb']:.2f}x")
```

---

## Performance Analysis

### Theoretical vs. Actual Speedup

**Theoretical Speedup:**
- Memory bandwidth: 2x (FP16 → INT8)
- Compute: 2-4x (Tensor Cores)

**Actual Speedup (Llama-2-7B on A100):**

| Batch Size | Seq Len | FP16 (tokens/s) | INT8 (tokens/s) | Speedup |
|------------|---------|-----------------|-----------------|---------|
| 1          | 128     | 45              | 72              | 1.6x    |
| 1          | 512     | 42              | 68              | 1.6x    |
| 4          | 128     | 168             | 285             | 1.7x    |
| 8          | 128     | 312             | 548             | 1.8x    |
| 16         | 128     | 584             | 1024            | 1.8x    |

**Observations:**
- Speedup increases with batch size (better Tensor Core utilization)
- Doesn't reach theoretical 2x due to overhead
- Memory-bound for small batches

### Accuracy Impact

**Perplexity Comparison (WikiText-2):**

| Model         | Method      | Perplexity | Δ from FP16 |
|---------------|-------------|------------|-------------|
| Llama-2-7B    | FP16        | 5.47       | -           |
| Llama-2-7B    | INT8 (W)    | 5.51       | +0.04       |
| Llama-2-7B    | INT8 (WA)   | 5.68       | +0.21       |
| Llama-2-7B    | LLM.int8()  | 5.49       | +0.02       |
| Llama-2-13B   | FP16        | 4.92       | -           |
| Llama-2-13B   | INT8 (W)    | 4.95       | +0.03       |
| Llama-2-70B   | FP16        | 3.32       | -           |
| Llama-2-70B   | LLM.int8()  | 3.35       | +0.03       |

**Downstream Task Performance (average across benchmarks):**

| Method      | Accuracy | Δ from FP16 |
|-------------|----------|-------------|
| FP16        | 72.3%    | -           |
| INT8 (W)    | 72.1%    | -0.2%       |
| INT8 (WA)   | 71.5%    | -0.8%       |
| LLM.int8()  | 72.2%    | -0.1%       |

---

## Practical Implementation

### Complete INT8 Quantization Pipeline

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

class INT8Quantizer:
    """Complete INT8 quantization pipeline."""

    def __init__(self, model, method='weight_only'):
        """
        Args:
            model: PyTorch model to quantize
            method: 'weight_only', 'weight_activation', or 'llm_int8'
        """
        self.model = model
        self.method = method
        self.quantized_model = None

    def quantize_weights(self):
        """Quantize all linear layers' weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize this layer
                self._quantize_linear(module)

    def _quantize_linear(self, linear):
        """Quantize a single linear layer."""
        W = linear.weight.data  # [out_features, in_features]

        # Per-channel quantization
        W_max = W.abs().max(dim=1, keepdim=True)[0]
        scales = W_max / 127.0
        scales = scales.clamp(min=1e-5)

        # Quantize
        W_int8 = (W / scales).round().clamp(-128, 127).to(torch.int8)

        # Replace weight with quantized version
        delattr(linear, 'weight')
        linear.register_buffer('weight_int8', W_int8)
        linear.register_buffer('weight_scales', scales.half())

        # Modify forward pass
        original_forward = linear.forward

        def quantized_forward(x):
            W_fp16 = linear.weight_int8.half() * linear.weight_scales
            output = torch.matmul(x, W_fp16.T)
            if hasattr(linear, 'bias') and linear.bias is not None:
                output += linear.bias
            return output

        linear.forward = quantized_forward

    def calibrate(self, calibration_data, num_samples=512):
        """Calibrate activation scales (for static quantization)."""
        if self.method == 'weight_only':
            return  # No calibration needed

        # Collect activation statistics
        # ... (implementation similar to earlier examples)
        pass

    def quantize(self, calibration_data=None):
        """
        Main quantization entry point.

        Args:
            calibration_data: Optional calibration dataset

        Returns:
            Quantized model
        """
        if self.method == 'weight_only':
            self.quantize_weights()

        elif self.method == 'weight_activation':
            self.quantize_weights()
            if calibration_data is not None:
                self.calibrate(calibration_data)

        elif self.method == 'llm_int8':
            # Implement LLM.int8() logic
            pass

        return self.model


# Example usage
def quantize_llama_int8():
    """Quantize Llama model to INT8."""
    # Load model
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Quantize
    quantizer = INT8Quantizer(model, method='weight_only')
    quantized_model = quantizer.quantize()

    # Test
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = quantized_model.generate(**inputs, max_length=50)

    print(tokenizer.decode(outputs[0]))

    return quantized_model
```

---

## Debugging and Troubleshooting

### Common Issues

#### 1. Accuracy Degradation

**Symptoms:**
- Perplexity significantly higher
- Generated text quality poor

**Debugging:**
```python
def debug_quantization_accuracy(original_model, quantized_model, test_data):
    """Compare layer-by-layer outputs."""
    original_acts = {}
    quantized_acts = {}

    # Hook to capture activations
    def make_hook(storage, name):
        def hook(module, input, output):
            storage[name] = output.detach()
        return hook

    # Register hooks
    for name, module in original_model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(make_hook(original_acts, name))

    for name, module in quantized_model.named_modules():
        if 'Linear' in module.__class__.__name__:
            module.register_forward_hook(make_hook(quantized_acts, name))

    # Run inference
    with torch.no_grad():
        _ = original_model(**test_data)
        _ = quantized_model(**test_data)

    # Compare activations
    for name in original_acts:
        orig = original_acts[name]
        quant = quantized_acts.get(name)

        if quant is not None:
            mse = ((orig - quant) ** 2).mean()
            cosine_sim = torch.nn.functional.cosine_similarity(
                orig.flatten(),
                quant.flatten(),
                dim=0
            )

            print(f"{name}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  Cosine similarity: {cosine_sim:.6f}")
```

**Solutions:**
- Use per-channel quantization
- Increase calibration data
- Try LLM.int8() for outlier handling
- Use symmetric quantization for stability

#### 2. Performance Not Meeting Expectations

**Symptoms:**
- Speedup < 1.3x
- Memory usage not reduced

**Debugging:**
```python
def profile_quantized_model(model, input_data):
    """Profile to identify bottlenecks."""
    from torch.profiler import profile, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = model(**input_data)

    # Print top operations
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Check for unexpected FP16 operations
    for event in prof.key_averages():
        if 'half' in event.key or 'fp16' in event.key:
            print(f"FP16 operation: {event.key} - {event.cuda_time_total:.2f}ms")
```

**Solutions:**
- Verify INT8 kernels are being used
- Check for dequantization overhead
- Ensure proper batch size for Tensor Core utilization
- Use fused kernels where possible

#### 3. Out of Memory Errors

**Symptoms:**
- CUDA OOM despite smaller model size

**Possible Causes:**
- Intermediate INT32 accumulation tensors
- Dequantized copies of weights

**Solutions:**
```python
# Enable gradient checkpointing to reduce activation memory
model.gradient_checkpointing_enable()

# Use in-place operations where possible
# Free intermediate tensors explicitly
torch.cuda.empty_cache()
```

---

## Exercises

### Exercise 1: Implement INT8 Quantization

Implement a complete INT8 quantization pipeline:

```python
# TODO: Implement
def exercise_int8_quantization():
    """
    1. Load a small model (e.g., GPT-2)
    2. Implement weight-only INT8 quantization
    3. Benchmark performance vs FP16
    4. Measure perplexity on test set
    """
    pass
```

### Exercise 2: Compare Static vs Dynamic

Compare static and dynamic activation quantization:

```python
# TODO: Implement
def exercise_static_vs_dynamic():
    """
    Implement both static and dynamic quantization.
    Compare:
    - Accuracy (perplexity)
    - Speed (latency)
    - Calibration overhead
    """
    pass
```

### Exercise 3: Outlier Analysis

Analyze outlier features in a real model:

```python
# TODO: Implement
def exercise_outlier_analysis():
    """
    1. Load OPT or BLOOM model
    2. Identify outlier features per layer
    3. Visualize outlier distribution
    4. Compare quantization with/without outlier handling
    """
    pass
```

### Exercise 4: Optimize INT8 Kernel

Profile and optimize an INT8 kernel:

```python
# TODO: Implement
def exercise_kernel_optimization():
    """
    1. Profile INT8 GEMM operation
    2. Identify bottlenecks (memory vs compute)
    3. Experiment with different tile sizes
    4. Measure achieved bandwidth/TOPS
    """
    pass
```

---

## Summary

### Key Takeaways

1. **INT8 is the Sweet Spot**: Balances memory savings, speed, and accuracy
2. **Weight-Only is Simple**: Easy to implement, good results, 1.5-1.7x speedup
3. **Weight-Activation is Faster**: Requires calibration, 2-3x speedup potential
4. **Outliers Matter**: LLM.int8() shows importance of outlier handling
5. **vLLM Support**: Excellent INT8 support via GPTQ, AWQ, and other methods

### Best Practices

1. Always compare against FP16 baseline
2. Use per-channel quantization for weights
3. Calibrate with representative data (512+ samples)
4. Profile to verify INT8 kernels are used
5. Monitor both accuracy and performance

### When to Use INT8

**Use INT8 when:**
- Model size > GPU memory in FP16
- Need 2x memory reduction
- Acceptable <1% accuracy loss
- Have A100 or newer GPU

**Consider alternatives when:**
- Need maximum compression (use INT4)
- Have H100 (use FP8 instead)
- Accuracy is extremely critical (try smaller base model in FP16)

---

## References

1. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" - Dettmers et al., 2022
2. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" - Frantar et al., 2023
3. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" - Lin et al., 2023
4. NVIDIA TensorRT Documentation: INT8 Calibration
5. vLLM Quantization Guide: https://docs.vllm.ai/en/latest/quantization/

---

**Module 6.2 Complete** • Next: [INT4 and FP8 Quantization](03_int4_fp8_quantization.md)
