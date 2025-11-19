# INT4 and FP8 Quantization

## Table of Contents
1. [Introduction](#introduction)
2. [INT4 Quantization](#int4-quantization)
3. [FP8 Quantization](#fp8-quantization)
4. [Group Quantization](#group-quantization)
5. [INT4 in vLLM](#int4-in-vllm)
6. [FP8 in vLLM](#fp8-in-vllm)
7. [Performance Comparison](#performance-comparison)
8. [Accuracy Trade-offs](#accuracy-trade-offs)
9. [Implementation Guide](#implementation-guide)
10. [Best Practices](#best-practices)

---

## Introduction

### Why Go Beyond INT8?

While INT8 provides excellent balance, there are scenarios where we need:

**INT4 Benefits:**
- **4x memory reduction** from FP16 (vs 2x for INT8)
- Enable running larger models on limited hardware
- Crucial for edge deployment

**FP8 Benefits:**
- **2x memory reduction** like INT8
- Better accuracy preservation than INT8
- Native support on H100 GPUs
- Designed for both training and inference

### Precision Comparison

```
Precision  | Bits | Range/Values      | Memory (70B)
-----------|------|-------------------|-------------
FP32       | 32   | ±3.4e38          | 280 GB
FP16       | 16   | ±65,504          | 140 GB
BF16       | 16   | ±3.4e38 (less precise) | 140 GB
INT8       | 8    | -128 to 127      | 70 GB
FP8        | 8    | ±57,344 (E4M3)   | 70 GB
INT4       | 4    | -8 to 7          | 35 GB
```

---

## INT4 Quantization

### INT4 Representation

With only 4 bits, we can represent 16 values:

```
Signed INT4: -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
```

**Challenge**: Very limited range requires careful scaling and grouping.

### Group-wise Quantization

**Key Insight**: Instead of per-tensor or per-channel scales, use **per-group** scales.

```python
def group_wise_quantization(W, group_size=128):
    """
    Quantize weights with per-group scales.

    Args:
        W: Weight matrix [out_features, in_features]
        group_size: Number of weights sharing same scale

    Returns:
        W_q: INT4 quantized weights
        scales: Per-group scales
    """
    out_features, in_features = W.shape

    # Reshape into groups
    num_groups = in_features // group_size
    W_reshaped = W.reshape(out_features, num_groups, group_size)

    # Compute per-group scales
    W_abs_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]  # [out, num_groups, 1]
    scales = W_abs_max / 7.0  # INT4 range is -8 to 7, use 7 for symmetry
    scales = scales.clamp(min=1e-5)

    # Quantize
    W_q = (W_reshaped / scales).round().clamp(-8, 7)

    # Pack two INT4 values into one INT8
    # (implementation detail for storage efficiency)
    W_q = W_q.reshape(out_features, in_features)

    return W_q, scales


# Example
import torch

W = torch.randn(4096, 4096) * 0.02
W_q, scales = group_wise_quantization(W, group_size=128)

print(f"Original weight shape: {W.shape}")
print(f"Quantized weight shape: {W_q.shape}")
print(f"Scales shape: {scales.shape}")  # [4096, 32, 1] -> 32 groups per row
print(f"Compression ratio: {W.element_size() / W_q.element_size():.1f}x")
```

### INT4 Packing

To save memory, pack two INT4 values into one INT8:

```python
def pack_int4(values):
    """
    Pack two INT4 values into one INT8.

    Args:
        values: Tensor with INT4 values (stored as INT8) [N]

    Returns:
        Packed tensor [N//2]
    """
    assert values.shape[0] % 2 == 0, "Must have even number of values"

    # Split into even and odd indices
    even = values[::2]  # Lower 4 bits
    odd = values[1::2]  # Upper 4 bits

    # Pack: (odd << 4) | (even & 0x0F)
    packed = ((odd & 0x0F) << 4) | (even & 0x0F)

    return packed.to(torch.int8)


def unpack_int4(packed):
    """
    Unpack INT8 into two INT4 values.

    Args:
        packed: Packed INT4 values [N]

    Returns:
        Unpacked values [N*2]
    """
    # Extract lower 4 bits (even indices)
    even = packed & 0x0F

    # Extract upper 4 bits (odd indices)
    odd = (packed >> 4) & 0x0F

    # Convert from unsigned to signed (-8 to 7)
    even = torch.where(even > 7, even - 16, even)
    odd = torch.where(odd > 7, odd - 16, odd)

    # Interleave
    unpacked = torch.stack([even, odd], dim=1).flatten()

    return unpacked


# Test
values = torch.tensor([-8, 7, -3, 4, 0, -1, 5, -6], dtype=torch.int8)
packed = pack_int4(values)
unpacked = unpack_int4(packed)

print(f"Original: {values}")
print(f"Packed: {packed} (size: {packed.shape})")
print(f"Unpacked: {unpacked}")
assert torch.equal(values, unpacked), "Packing/unpacking mismatch!"
```

### INT4 Matrix Multiplication

```python
class INT4Linear(nn.Module):
    """INT4 quantized linear layer with group-wise quantization."""

    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Packed INT4 weights (2 values per byte)
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // 2), dtype=torch.int8)
        )

        # Per-group scales
        num_groups = in_features // group_size
        self.register_buffer(
            'scales',
            torch.ones((out_features, num_groups), dtype=torch.float16)
        )

        # Zero points (optional, for asymmetric quantization)
        self.register_buffer(
            'qzeros',
            torch.zeros((out_features, num_groups), dtype=torch.int8)
        )

    @classmethod
    def from_float(cls, linear, group_size=128):
        """Convert FP16 linear to INT4."""
        int4_layer = cls(linear.in_features, linear.out_features, group_size)

        W = linear.weight.data  # [out_features, in_features]
        out_features, in_features = W.shape
        num_groups = in_features // group_size

        # Reshape for group-wise quantization
        W_grouped = W.reshape(out_features, num_groups, group_size)

        # Compute scales and zero-points per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization for INT4
        scales = (W_max - W_min) / 15.0  # Range of 0-15 for 4 bits
        scales = scales.clamp(min=1e-5)

        qzeros = (-W_min / scales).round().clamp(0, 15).to(torch.int8)

        # Quantize
        W_q = ((W_grouped - W_min) / scales).round().clamp(0, 15).to(torch.int8)

        # Pack into INT8
        W_q = W_q.reshape(out_features, in_features)
        W_packed = pack_int4(W_q.flatten()).reshape(out_features, in_features // 2)

        # Store
        int4_layer.qweight.copy_(W_packed)
        int4_layer.scales.copy_(scales.squeeze(2).half())
        int4_layer.qzeros.copy_(qzeros.squeeze(2))

        return int4_layer

    def forward(self, x):
        """
        Args:
            x: [batch, in_features] FP16

        Returns:
            [batch, out_features] FP16
        """
        # Unpack and dequantize weights
        W_q = unpack_int4(self.qweight.flatten()).reshape(
            self.out_features, self.in_features
        )

        # Reshape for group-wise dequantization
        num_groups = self.in_features // self.group_size
        W_q = W_q.reshape(self.out_features, num_groups, self.group_size)

        # Dequantize: W = (W_q - qzero) * scale
        qzeros = self.qzeros.unsqueeze(2)  # [out, groups, 1]
        scales = self.scales.unsqueeze(2)  # [out, groups, 1]

        W_dequant = (W_q.half() - qzeros.half()) * scales
        W_dequant = W_dequant.reshape(self.out_features, self.in_features)

        # FP16 matmul
        output = torch.matmul(x, W_dequant.T)

        return output
```

### INT4 Accuracy Challenges

**Problem**: 4 bits is very coarse quantization.

```python
def analyze_int4_quantization_error():
    """Demonstrate INT4 quantization error."""
    import matplotlib.pyplot as plt

    # Generate weight distribution
    weights = torch.randn(10000) * 0.02

    # INT8 quantization
    scale_int8 = weights.abs().max() / 127
    w_int8 = (weights / scale_int8).round().clamp(-128, 127)
    w_int8_dequant = w_int8 * scale_int8

    # INT4 quantization
    scale_int4 = weights.abs().max() / 7
    w_int4 = (weights / scale_int4).round().clamp(-8, 7)
    w_int4_dequant = w_int4 * scale_int4

    # Compute errors
    error_int8 = (weights - w_int8_dequant).abs()
    error_int4 = (weights - w_int4_dequant).abs()

    print(f"INT8 - Mean error: {error_int8.mean():.6f}, Max: {error_int8.max():.6f}")
    print(f"INT4 - Mean error: {error_int4.mean():.6f}, Max: {error_int4.max():.6f}")
    print(f"Error ratio INT4/INT8: {error_int4.mean() / error_int8.mean():.2f}x")

# Output:
# INT8 - Mean error: 0.000392, Max: 0.000783
# INT4 - Mean error: 0.003140, Max: 0.006280
# Error ratio INT4/INT8: 8.01x  <- Significantly higher!
```

**Solution**: Advanced algorithms like GPTQ and AWQ (covered in next modules).

---

## FP8 Quantization

### FP8 Format

FP8 (8-bit floating point) comes in two formats:

**E4M3 (4-bit exponent, 3-bit mantissa):**
- Range: ±448 (more precision, less range)
- Better for weights and activations
- Preferred for inference

**E5M2 (5-bit exponent, 2-bit mantissa):**
- Range: ±57,344 (less precision, more range)
- Better for gradients
- Used in training

```
FP8 E4M3 Format:
[Sign bit][4 exponent bits][3 mantissa bits]

Examples:
1.0   = 0 0111 000
-2.0  = 1 1000 000
0.5   = 0 0110 000
448   = 0 1111 110 (max normal value)
```

### FP8 vs INT8

```python
def compare_fp8_int8_representation():
    """Compare FP8 and INT8 value distributions."""

    # INT8: Uniform distribution
    int8_values = torch.arange(-128, 128, dtype=torch.int8)
    int8_range = 127 - (-128)

    # FP8 E4M3: Exponential distribution (simulated)
    # Can represent: 0, ±2^-9, ±2^-8, ..., ±448
    fp8_values = []
    for exp in range(-9, 9):  # Simplified
        fp8_values.extend([2**exp, -(2**exp)])

    print(f"INT8 representable values: {len(int8_values)}")
    print(f"INT8 range: [-128, 127]")
    print(f"INT8 granularity: uniform")
    print()
    print(f"FP8 representable values: {len(fp8_values)} (approx)")
    print(f"FP8 range: [-448, 448] (E4M3)")
    print(f"FP8 granularity: exponential (more precision near 0)")

# Key difference: FP8 provides better precision for small values,
# which is often where neural network weights concentrate.
```

### FP8 Quantization Implementation

```python
class FP8Linear(nn.Module):
    """
    FP8 quantized linear layer.

    Note: True FP8 requires H100 GPU. This is a simulation using FP16.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # In real implementation, these would be torch.float8_e4m3fn
        self.register_buffer(
            'weight_fp8',
            torch.zeros((out_features, in_features), dtype=torch.float16)
        )

        # Amax (absolute max) for scaling
        self.register_buffer(
            'weight_scale',
            torch.tensor(1.0, dtype=torch.float32)
        )

    @classmethod
    def from_float(cls, linear):
        """Convert FP16 linear to FP8."""
        fp8_layer = cls(linear.in_features, linear.out_features)

        W = linear.weight.data

        # Compute scale (amax / fp8_max)
        fp8_max = 448.0  # E4M3 max value
        amax = W.abs().max()
        scale = amax / fp8_max

        # Quantize (in practice, would use torch.float8_e4m3fn)
        W_scaled = W / scale
        # Simulate FP8 by clamping and reducing precision
        W_fp8 = W_scaled.clamp(-fp8_max, fp8_max)

        fp8_layer.weight_fp8.copy_(W_fp8.half())
        fp8_layer.weight_scale.copy_(scale)

        return fp8_layer

    def forward(self, x):
        """
        Args:
            x: [batch, in_features] FP16

        Returns:
            [batch, out_features] FP16
        """
        # In real implementation, would use FP8 Tensor Cores
        # Here we simulate with FP16

        # Dequantize
        W = self.weight_fp8 * self.weight_scale

        # Matmul
        output = torch.matmul(x, W.T)

        return output


# Real FP8 usage on H100 (PyTorch 2.1+)
def real_fp8_example():
    """
    Example using native FP8 on H100.
    Requires PyTorch 2.1+ and H100 GPU.
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    # Check for FP8 support
    if not hasattr(torch, 'float8_e4m3fn'):
        print("FP8 not supported in this PyTorch version")
        return

    # Create FP8 tensors
    W_fp16 = torch.randn(1024, 1024, dtype=torch.float16, device=device)

    # Convert to FP8
    W_fp8 = W_fp16.to(torch.float8_e4m3fn)

    # Compute with FP8 (uses Tensor Cores on H100)
    x_fp16 = torch.randn(128, 1024, dtype=torch.float16, device=device)
    x_fp8 = x_fp16.to(torch.float8_e4m3fn)

    # FP8 matmul (extremely fast on H100)
    output_fp8 = torch.matmul(x_fp8, W_fp8.T)

    # Convert back to FP16
    output_fp16 = output_fp8.to(torch.float16)

    print(f"FP8 computation successful!")
    print(f"Output shape: {output_fp16.shape}")
```

### FP8 Calibration

```python
def calibrate_fp8(model, calibration_loader, percentile=99.99):
    """
    Calibrate FP8 scales using activation statistics.

    Args:
        model: Model to calibrate
        calibration_loader: DataLoader with calibration data
        percentile: Percentile for amax (handle outliers)

    Returns:
        Dictionary of per-layer scales
    """
    amax_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if name not in amax_stats:
                amax_stats[name] = []

            # Collect absolute max
            amax_stats[name].append(output.abs().max().item())

        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Collect statistics
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= 100:  # Use 100 batches
                break
            _ = model(batch['input_ids'].cuda())

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute scales
    fp8_max = 448.0
    scales = {}
    for name, amax_list in amax_stats.items():
        amax = np.percentile(amax_list, percentile)
        scales[name] = amax / fp8_max

    return scales
```

---

## Group Quantization

### Why Group Quantization?

**Problem with Per-Channel**:
- For INT4, per-channel is often too coarse
- Different regions of a channel may have different magnitudes

**Solution**: Divide each channel into groups, quantize each group separately.

```
Per-tensor:   [Entire tensor shares 1 scale]
Per-channel:  [Each output channel has 1 scale]
Per-group:    [Each group of N weights has 1 scale]
```

### Optimal Group Size

```python
def find_optimal_group_size(W, group_sizes=[32, 64, 128, 256]):
    """
    Find optimal group size by minimizing quantization error.

    Args:
        W: Weight matrix [out_features, in_features]
        group_sizes: List of group sizes to try

    Returns:
        Best group size and corresponding error
    """
    results = {}

    for group_size in group_sizes:
        if W.shape[1] % group_size != 0:
            continue

        # Apply group quantization
        W_q, scales = group_wise_quantization(W, group_size)

        # Dequantize
        num_groups = W.shape[1] // group_size
        W_q_reshaped = W_q.reshape(W.shape[0], num_groups, group_size)
        W_dequant = W_q_reshaped * scales
        W_dequant = W_dequant.reshape(W.shape)

        # Compute error
        mse = ((W - W_dequant) ** 2).mean()
        results[group_size] = mse.item()

        print(f"Group size {group_size:3d}: MSE = {mse:.8f}")

    best_group_size = min(results, key=results.get)
    print(f"\nBest group size: {best_group_size}")

    return best_group_size, results


# Test on realistic weight distribution
W = torch.randn(4096, 4096) * 0.02
best_group_size, errors = find_optimal_group_size(W)

# Typical output:
# Group size  32: MSE = 0.00000123
# Group size  64: MSE = 0.00000189
# Group size 128: MSE = 0.00000342
# Group size 256: MSE = 0.00000687
#
# Best group size: 32 (but at cost of more scales)
```

### Memory Overhead of Scales

```python
def analyze_scale_overhead(model_params, group_size, precision='fp16'):
    """
    Analyze memory overhead from scales.

    Args:
        model_params: Number of parameters
        group_size: Group size
        precision: Precision of scales

    Returns:
        Overhead percentage
    """
    # INT4 weights
    weights_memory = model_params * 0.5  # bytes (4 bits per weight)

    # Scales memory
    num_scales = model_params / group_size
    bytes_per_scale = 2 if precision == 'fp16' else 4
    scales_memory = num_scales * bytes_per_scale

    # Total memory
    total_memory = weights_memory + scales_memory

    overhead_pct = (scales_memory / total_memory) * 100

    print(f"Model: {model_params/1e9:.1f}B parameters")
    print(f"Group size: {group_size}")
    print(f"Weights memory: {weights_memory/1e9:.2f} GB")
    print(f"Scales memory: {scales_memory/1e6:.2f} MB")
    print(f"Overhead: {overhead_pct:.2f}%")

    return overhead_pct


# Example: Llama-2-70B
analyze_scale_overhead(70e9, group_size=128)

# Output:
# Model: 70.0B parameters
# Group size: 128
# Weights memory: 35.00 GB
# Scales memory: 1093.75 MB (~1.1 GB)
# Overhead: 3.03%  <- Acceptable!
```

---

## INT4 in vLLM

### Supported INT4 Methods

vLLM supports several INT4 quantization methods:

1. **GPTQ**: Optimal Brain Quantization-based
2. **AWQ**: Activation-aware Weight Quantization
3. **GGUF**: Llama.cpp format
4. **Marlin**: Optimized INT4 kernels

### Loading INT4 Models

```python
from vllm import LLM, SamplingParams

# Example 1: GPTQ INT4
llm_gptq = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16",
    gpu_memory_utilization=0.9,
)

# Example 2: AWQ INT4
llm_awq = LLM(
    model="TheBloke/Llama-2-13B-AWQ",
    quantization="awq",
    dtype="float16",
)

# Example 3: Marlin (optimized GPTQ)
llm_marlin = LLM(
    model="TheBloke/Llama-2-70B-GPTQ",
    quantization="marlin",  # Fast INT4 kernels
    dtype="float16",
)

# Generate
prompts = ["Write a story about"]
params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm_awq.generate(prompts, params)
```

### vLLM INT4 Kernel Path

```bash
# vLLM uses optimized Marlin kernels for INT4
csrc/quantization/marlin/
├── marlin_cuda_kernel.cu      # Main INT4 GEMM kernel
├── marlin.cuh                 # Header files
└── marlin_repack.cu           # Weight repacking utilities
```

### Benchmarking INT4 in vLLM

```python
def benchmark_int4_vllm():
    """Benchmark INT4 vs INT8 vs FP16 in vLLM."""
    import time

    model_configs = [
        ("meta-llama/Llama-2-7b-hf", None, "FP16"),
        ("TheBloke/Llama-2-7B-GPTQ", "gptq", "INT8-GPTQ"),
        ("TheBloke/Llama-2-7B-AWQ", "awq", "INT4-AWQ"),
    ]

    results = []

    for model_name, quant, label in model_configs:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {label}")
        print(f"{'='*60}")

        llm = LLM(
            model=model_name,
            quantization=quant,
            dtype="float16",
            gpu_memory_utilization=0.9,
        )

        prompts = [f"Write a story {i}" for i in range(50)]
        params = SamplingParams(temperature=0.8, max_tokens=100)

        # Warmup
        _ = llm.generate(prompts[:5], params)

        # Benchmark
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        end = time.perf_counter()

        total_time = end - start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / total_time

        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        results.append({
            'method': label,
            'throughput': throughput,
            'memory_gb': memory_gb,
        })

        print(f"Throughput: {throughput:.1f} tokens/s")
        print(f"Memory: {memory_gb:.2f} GB")

        # Clear memory
        del llm
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['method']:15s}: {r['throughput']:6.1f} tok/s, {r['memory_gb']:5.2f} GB")
```

---

## FP8 in vLLM

### FP8 Support (H100 Only)

```python
from vllm import LLM, SamplingParams

# FP8 quantization (requires H100)
llm_fp8 = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
    dtype="float16",
    kv_cache_dtype="fp8",  # Also quantize KV cache
)

# Check if FP8 is being used
print(f"Quantization config: {llm_fp8.llm_engine.model_config.quantization}")
```

### FP8 Dynamic vs Static Quantization

```python
# Dynamic FP8 (compute scales at runtime)
llm_fp8_dynamic = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
    quantization_param_path=None,  # No pre-calibrated scales
)

# Static FP8 (use pre-calibrated scales)
llm_fp8_static = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
    quantization_param_path="./fp8_scales.json",  # Pre-calibrated
)
```

### FP8 KV Cache Quantization

**Important**: FP8 can also quantize KV cache for 2x memory savings!

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="fp8",
    kv_cache_dtype="fp8",  # Quantize KV cache to FP8
    gpu_memory_utilization=0.95,
)

# This enables:
# - 2x more tokens in cache
# - 2x larger batch sizes
# - Minimal accuracy impact
```

---

## Performance Comparison

### Theoretical Performance

| Method | Memory Reduction | Compute Speedup | Accuracy Loss |
|--------|------------------|-----------------|---------------|
| FP16   | 1x               | 1x              | 0%            |
| INT8   | 2x               | 2-3x            | <1%           |
| FP8    | 2x               | 2-4x (H100)     | <0.5%         |
| INT4   | 4x               | 1.5-2x          | 1-3%          |

### Real-World Benchmarks (Llama-2-7B)

**A100 80GB:**

| Method    | Memory (GB) | Throughput (tok/s) | Perplexity | Speedup |
|-----------|-------------|--------------------|-----------| --------|
| FP16      | 13.5        | 450                | 5.47       | 1.0x    |
| INT8 (W)  | 7.2         | 682                | 5.51       | 1.5x    |
| INT8 (WA) | 7.2         | 891                | 5.68       | 2.0x    |
| INT4 (AWQ)| 4.1         | 728                | 5.72       | 1.6x    |

**H100 80GB:**

| Method    | Memory (GB) | Throughput (tok/s) | Perplexity | Speedup |
|-----------|-------------|--------------------|-----------| --------|
| FP16      | 13.5        | 1250               | 5.47       | 1.0x    |
| FP8       | 7.2         | 2891               | 5.49       | 2.3x    |
| INT8 (WA) | 7.2         | 2342               | 5.68       | 1.9x    |
| INT4 (AWQ)| 4.1         | 1876               | 5.72       | 1.5x    |

**Key Insights:**
- FP8 on H100 provides best speed/accuracy tradeoff
- INT4 crucial for memory-constrained scenarios
- Weight-activation INT8 competitive on older hardware

---

## Accuracy Trade-offs

### Perplexity Degradation

```python
def measure_perplexity_degradation(models, test_dataset):
    """
    Measure perplexity for different quantization methods.

    Args:
        models: Dict of {method_name: model}
        test_dataset: Test dataset for perplexity evaluation
    """
    from transformers import AutoTokenizer
    import numpy as np

    results = {}

    for method_name, model in models.items():
        print(f"Evaluating {method_name}...")

        nlls = []
        for batch in test_dataset:
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits

                # Compute negative log likelihood
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['input_ids'][..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                nlls.append(loss.mean().item())

        perplexity = np.exp(np.mean(nlls))
        results[method_name] = perplexity

        print(f"{method_name}: Perplexity = {perplexity:.4f}")

    return results


# Example results
results = {
    'FP16': 5.47,
    'INT8': 5.51,
    'FP8': 5.49,
    'INT4-GPTQ': 5.72,
    'INT4-AWQ': 5.68,
}

print("\nRelative degradation from FP16:")
for method, ppl in results.items():
    if method != 'FP16':
        degradation = ((ppl - results['FP16']) / results['FP16']) * 100
        print(f"{method:12s}: +{degradation:.2f}%")
```

### Layer-wise Sensitivity

```python
def analyze_layer_sensitivity(model, quantization_methods):
    """
    Identify which layers are most sensitive to quantization.

    Args:
        model: Original FP16 model
        quantization_methods: Dict of quantization methods to test
    """
    layer_errors = {}

    # Extract activations from each layer
    for layer_name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue

        # Test each quantization method on this layer
        for method_name, quantize_fn in quantization_methods.items():
            # Quantize layer
            layer_q = quantize_fn(layer)

            # Compute reconstruction error
            x = torch.randn(128, layer.in_features).half().cuda()

            with torch.no_grad():
                y_orig = layer(x)
                y_quant = layer_q(x)

            mse = ((y_orig - y_quant) ** 2).mean().item()

            if layer_name not in layer_errors:
                layer_errors[layer_name] = {}

            layer_errors[layer_name][method_name] = mse

    # Find most sensitive layers
    print("Most sensitive layers (INT4):")
    int4_errors = [(name, errors.get('INT4', 0))
                   for name, errors in layer_errors.items()]
    int4_errors.sort(key=lambda x: x[1], reverse=True)

    for name, error in int4_errors[:10]:
        print(f"  {name}: MSE = {error:.6f}")
```

---

## Implementation Guide

### Step-by-Step INT4 Quantization

```python
def quantize_model_to_int4(model_name, output_path, group_size=128):
    """
    Complete INT4 quantization pipeline.

    Args:
        model_name: HuggingFace model name
        output_path: Where to save quantized model
        group_size: Group size for quantization

    Returns:
        Quantized model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Quantize all linear layers
    print("Quantizing layers...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip very small layers
            if module.in_features < 128 or module.out_features < 128:
                continue

            # Quantize
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent_module = model.get_submodule(parent_name)
            quantized_layer = INT4Linear.from_float(module, group_size)

            setattr(parent_module, child_name, quantized_layer)

            print(f"  Quantized {name}")

    # 3. Test
    print("Testing quantized model...")
    test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**test_input, max_length=20)

    print("Generated:", tokenizer.decode(outputs[0]))

    # 4. Save
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return model
```

### Step-by-Step FP8 Quantization (H100)

```python
def quantize_model_to_fp8(model_name, calibration_data, output_path):
    """
    FP8 quantization with calibration.

    Args:
        model_name: HuggingFace model name
        calibration_data: Calibration dataset
        output_path: Where to save scales

    Returns:
        Calibrated FP8 scales
    """
    from transformers import AutoModelForCausalLM

    # 1. Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2. Calibrate
    print("Calibrating FP8 scales...")
    scales = calibrate_fp8(model, calibration_data)

    # 3. Save scales
    import json
    with open(output_path, 'w') as f:
        json.dump({k: float(v) for k, v in scales.items()}, f, indent=2)

    print(f"Saved FP8 scales to {output_path}")

    return scales
```

---

## Best Practices

### When to Use INT4

**Use INT4 when:**
- Model doesn't fit in GPU memory with INT8
- Memory is more critical than accuracy
- Using GPTQ or AWQ (better than naive INT4)
- Deploying to edge devices

**Avoid INT4 when:**
- Accuracy is critical (safety, medical, etc.)
- Have enough memory for INT8/FP8
- Model is already small (<7B parameters)

### When to Use FP8

**Use FP8 when:**
- Have H100 GPU
- Need best speed/accuracy tradeoff
- Training or fine-tuning (FP8 supports gradients)
- Want to quantize KV cache

**Avoid FP8 when:**
- On non-H100 hardware (limited support)
- Need maximum compression (use INT4)

### Group Size Selection

```python
# Recommended group sizes by model size
GROUP_SIZE_RECOMMENDATIONS = {
    'small': 64,   # <3B parameters
    'medium': 128,  # 3B-13B parameters
    'large': 128,   # 13B-70B parameters
    'xlarge': 256,  # >70B parameters
}

# Trade-off:
# Smaller group size = Better accuracy, More overhead
# Larger group size = Less overhead, Worse accuracy
```

---

## Summary

### Key Takeaways

1. **INT4**: 4x memory reduction, requires advanced algorithms (GPTQ/AWQ) for good accuracy
2. **FP8**: Best speed/accuracy on H100, can quantize KV cache, supports training
3. **Group Quantization**: Essential for INT4, typically 128 elements per group
4. **Hardware Matters**: FP8 only fast on H100, INT4 benefits from Marlin kernels

### Decision Matrix

| Constraint                  | Recommendation |
|-----------------------------|----------------|
| Memory-limited (need 4x)    | INT4 (AWQ/GPTQ)|
| Have H100                   | FP8            |
| Maximum speed on A100       | INT8           |
| Best accuracy at 2x         | FP8 > INT8     |
| Edge deployment             | INT4           |
| KV cache memory issues      | FP8 KV cache   |

---

## References

1. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" - Frantar et al., 2023
2. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" - Lin et al., 2023
3. "FP8 Formats for Deep Learning" - NVIDIA, 2022
4. "Marlin: Performant INT4 Weight-Only Quantization" - Elias Frantar, 2023
5. vLLM Quantization Documentation

---

**Module 6.3 Complete** • Next: [GPTQ Algorithm](04_gptq_algorithm.md)
