# Quantization Fundamentals for LLM Inference

## Table of Contents
1. [Introduction to Quantization](#introduction-to-quantization)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Quantization Types and Schemes](#quantization-types-and-schemes)
4. [Memory and Performance Benefits](#memory-and-performance-benefits)
5. [Quantization in vLLM Architecture](#quantization-in-vllm-architecture)
6. [Basic Quantization Implementation](#basic-quantization-implementation)
7. [Trade-offs and Considerations](#trade-offs-and-considerations)
8. [Practical Exercises](#practical-exercises)

---

## Introduction to Quantization

### What is Quantization?

Quantization is the process of mapping continuous or high-precision values to a discrete, lower-precision representation. In the context of Large Language Models (LLMs), quantization primarily involves reducing the precision of model weights and/or activations from high-precision formats (like FP32 or FP16) to lower-precision formats (like INT8, INT4, or FP8).

**Key Motivation:**
- **Memory Reduction**: A 70B parameter model in FP16 requires ~140GB of memory. With INT4 quantization, this drops to ~35GB.
- **Bandwidth Optimization**: Lower precision reduces memory bandwidth requirements, a critical bottleneck in LLM inference.
- **Compute Acceleration**: Modern GPUs have specialized hardware for low-precision operations (e.g., Tensor Cores for INT8/FP8).
- **Cost Efficiency**: Smaller models can run on cheaper hardware or enable higher batch sizes.

### Why Quantization Matters for LLM Inference

LLM inference is primarily **memory-bandwidth bound** rather than compute-bound. This means:

```
Time per token = (Model size in bytes) / (Memory bandwidth)
```

**Example Calculation:**
- Model: Llama-2-70B in FP16 (140GB)
- GPU: A100 80GB (2TB/s memory bandwidth)
- Time to load weights per token: 140GB / 2TB/s = 70ms

With INT4 quantization (35GB):
- Time to load weights per token: 35GB / 2TB/s = 17.5ms
- **Theoretical speedup: 4x**

### Historical Context

Quantization has evolved significantly:

1. **Early Deep Learning (2015-2018)**: Post-training quantization for CNNs
2. **BERT Era (2018-2020)**: Quantization-aware training, INT8 inference
3. **LLM Explosion (2020-2023)**: Weight-only quantization, GPTQ, AWQ
4. **Modern Era (2023+)**: FP8, INT4, hybrid quantization schemes

---

## Mathematical Foundations

### Basic Quantization Formula

The core quantization operation maps a floating-point value to an integer:

```
Q(x) = round(x / scale) + zero_point
```

**Dequantization** (converting back):
```
x_approx = (Q(x) - zero_point) * scale
```

Where:
- `x`: Original floating-point value
- `Q(x)`: Quantized integer value
- `scale`: Scaling factor (floating-point)
- `zero_point`: Offset for asymmetric quantization (integer)

### Symmetric vs. Asymmetric Quantization

#### Symmetric Quantization

```
scale = max(|x_min|, |x_max|) / (2^(bits-1) - 1)
zero_point = 0
Q(x) = round(x / scale)
```

**Advantages:**
- Simpler computation (no zero-point)
- Faster inference
- Better hardware support

**Disadvantages:**
- Inefficient for asymmetric distributions
- Wastes representation range

**Example:**
```python
import numpy as np

def symmetric_quantize(x, bits=8):
    """Symmetric quantization to signed integer."""
    qmax = 2**(bits-1) - 1  # 127 for INT8
    qmin = -qmax             # -127 for INT8

    # Calculate scale
    scale = np.max(np.abs(x)) / qmax

    # Quantize
    x_q = np.round(x / scale)
    x_q = np.clip(x_q, qmin, qmax).astype(np.int8)

    return x_q, scale

# Example usage
weights = np.random.randn(100) * 0.5  # Mean 0, std 0.5
w_q, scale = symmetric_quantize(weights)
print(f"Original range: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"Quantized range: [{w_q.min()}, {w_q.max()}]")
print(f"Scale: {scale:.6f}")
```

#### Asymmetric Quantization

```
scale = (x_max - x_min) / (2^bits - 1)
zero_point = round(-x_min / scale)
Q(x) = round(x / scale) + zero_point
```

**Advantages:**
- Better for asymmetric distributions (e.g., activations after ReLU)
- More efficient use of quantization range

**Disadvantages:**
- Additional computation for zero-point
- Slightly slower

**Example:**
```python
def asymmetric_quantize(x, bits=8):
    """Asymmetric quantization to unsigned integer."""
    qmax = 2**bits - 1  # 255 for UINT8
    qmin = 0

    # Calculate scale and zero-point
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / qmax
    zero_point = int(np.round(-x_min / scale))

    # Quantize
    x_q = np.round(x / scale) + zero_point
    x_q = np.clip(x_q, qmin, qmax).astype(np.uint8)

    return x_q, scale, zero_point

# Example with ReLU activations (non-negative)
activations = np.abs(np.random.randn(100) * 0.5)
a_q, scale, zp = asymmetric_quantize(activations)
print(f"Original range: [{activations.min():.3f}, {activations.max():.3f}]")
print(f"Zero-point: {zp}")
```

### Per-Tensor vs. Per-Channel Quantization

#### Per-Tensor Quantization

**Single** scale (and zero-point) for the entire tensor.

```python
# Shape: (out_features, in_features)
weights = np.random.randn(4096, 4096) * 0.02
w_q, scale = symmetric_quantize(weights)  # Single scale value
```

**Advantages:**
- Simple and fast
- Minimal overhead

**Disadvantages:**
- Poor accuracy for tensors with varying ranges across channels
- Outliers affect entire tensor

#### Per-Channel Quantization

**Separate** scale for each output channel (row in weight matrix).

```python
def per_channel_quantize(W, bits=8, axis=0):
    """Per-channel symmetric quantization."""
    qmax = 2**(bits-1) - 1

    # Calculate per-channel scales
    W_abs_max = np.abs(W).max(axis=1-axis, keepdims=True)
    scales = W_abs_max / qmax

    # Quantize
    W_q = np.round(W / scales)
    W_q = np.clip(W_q, -qmax, qmax).astype(np.int8)

    return W_q, scales

# Example
weights = np.random.randn(4096, 4096) * np.linspace(0.01, 0.05, 4096).reshape(-1, 1)
w_q, scales = per_channel_quantize(weights, axis=0)
print(f"Scales shape: {scales.shape}")  # (4096, 1)
print(f"Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
```

**Advantages:**
- Much better accuracy preservation
- Handles outlier channels

**Disadvantages:**
- More complex computation
- Additional memory for scales (usually negligible)

### Quantization Error Analysis

The quantization error is:
```
error = x - x_approx = x - (round(x/scale) * scale)
```

**Mean Squared Error (MSE):**
```
MSE = E[(x - x_approx)^2]
```

For uniform quantization, the error is bounded:
```
|error| ≤ scale / 2
```

**Example:**
```python
def compute_quantization_error(x, x_q, scale):
    """Compute quantization error metrics."""
    x_dequant = x_q * scale

    mse = np.mean((x - x_dequant)**2)
    mae = np.mean(np.abs(x - x_dequant))
    max_error = np.max(np.abs(x - x_dequant))
    snr = 10 * np.log10(np.var(x) / mse)

    return {
        'MSE': mse,
        'MAE': mae,
        'Max Error': max_error,
        'SNR (dB)': snr
    }

# Compare per-tensor vs per-channel
weights = np.random.randn(1024, 1024) * 0.02
w_q_tensor, scale_tensor = symmetric_quantize(weights)
w_q_channel, scales_channel = per_channel_quantize(weights)

error_tensor = compute_quantization_error(weights, w_q_tensor, scale_tensor)
error_channel = compute_quantization_error(weights, w_q_channel, scales_channel)

print("Per-Tensor Error:", error_tensor)
print("Per-Channel Error:", error_channel)
```

---

## Quantization Types and Schemes

### Weight-Only Quantization

Quantizes **only** the model weights; activations remain in higher precision (FP16/BF16).

**Characteristics:**
- Most common for LLM inference
- Memory bandwidth reduction without accuracy loss
- Used by GPTQ, AWQ, RTN

**Pseudo-code:**
```python
# During inference
def forward_weight_only_quant(x, W_q, scales):
    """
    x: FP16 activations [batch, in_features]
    W_q: INT8 weights [out_features, in_features]
    scales: FP16 per-channel scales [out_features, 1]
    """
    # Dequantize on-the-fly
    W_fp16 = W_q.to(torch.float16) * scales

    # Standard FP16 GEMM
    output = torch.matmul(x, W_fp16.T)
    return output
```

**Memory Savings:**
- FP16 → INT8: 2x reduction
- FP16 → INT4: 4x reduction

### Weight-Activation Quantization

Quantizes **both** weights and activations.

**Characteristics:**
- Maximum compute acceleration
- Requires careful calibration
- Used by SmoothQuant, INT8 PTQ

**Pseudo-code:**
```python
def forward_weight_activation_quant(x, W_q, w_scales, a_scale):
    """
    x: FP16 activations
    W_q: INT8 weights
    w_scales: weight scales
    a_scale: activation scale
    """
    # Quantize activations
    x_q = (x / a_scale).round().clamp(-128, 127).to(torch.int8)

    # INT8 GEMM (fast!)
    output_q = torch.ops.aten._int_mm(x_q, W_q.T)

    # Dequantize output
    output = output_q.to(torch.float16) * (a_scale * w_scales.T)
    return output
```

**Speedup Potential:**
- A100: ~2-3x for INT8 vs FP16
- H100: ~4-6x for FP8 vs FP16

### Dynamic vs. Static Quantization

#### Dynamic Quantization
- Activation scales computed **at runtime** for each batch
- Better accuracy, more overhead

```python
def dynamic_quantize_activation(x):
    """Compute scale dynamically per batch."""
    scale = x.abs().max() / 127.0
    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale
```

#### Static Quantization
- Activation scales **precomputed** using calibration data
- Faster, but requires representative calibration set

```python
# Calibration phase
def calibrate(model, calibration_dataloader):
    activation_scales = {}

    with torch.no_grad():
        for batch in calibration_dataloader:
            # Collect activation statistics
            for name, activation in model.activations.items():
                if name not in activation_scales:
                    activation_scales[name] = []
                activation_scales[name].append(activation.abs().max())

    # Compute final scales (e.g., 99.9th percentile)
    for name in activation_scales:
        activation_scales[name] = np.percentile(activation_scales[name], 99.9)

    return activation_scales
```

---

## Memory and Performance Benefits

### Memory Footprint Analysis

**Model Size Formula:**
```
Size = num_parameters * bytes_per_parameter
```

**Example: Llama-2-70B**

| Precision | Bytes/Param | Total Size | Reduction |
|-----------|-------------|------------|-----------|
| FP32      | 4           | 280 GB     | 1x        |
| FP16/BF16 | 2           | 140 GB     | 2x        |
| INT8      | 1           | 70 GB      | 4x        |
| INT4      | 0.5         | 35 GB      | 8x        |
| FP8       | 1           | 70 GB      | 4x        |

**KV Cache Savings:**

For a single token position:
```
KV_cache_size = 2 * num_layers * hidden_dim * (num_heads * head_dim)
```

Example (Llama-2-7B):
- FP16: 2 * 32 layers * 4096 dim * 2 bytes = 524 KB per token
- INT8: 262 KB per token (2x reduction)
- **For 2048 context**: FP16 = 1 GB, INT8 = 512 MB

### Bandwidth Analysis

**Memory Bandwidth Bottleneck:**

```python
def estimate_inference_time(model_size_gb, batch_size, seq_len, bandwidth_tbs):
    """
    Estimate token generation time based on memory bandwidth.

    Args:
        model_size_gb: Model size in GB
        batch_size: Batch size
        seq_len: Sequence length
        bandwidth_tbs: Memory bandwidth in TB/s
    """
    # Weight loading dominates for small batches
    weight_time_ms = (model_size_gb / bandwidth_tbs) * 1000

    # KV cache loading (depends on context length)
    kv_cache_gb = (seq_len * batch_size * 0.0005)  # Approximate
    kv_time_ms = (kv_cache_gb / bandwidth_tbs) * 1000

    total_time_ms = weight_time_ms + kv_time_ms
    return total_time_ms

# Compare FP16 vs INT4
model_params = 70e9  # 70B parameters
bandwidth = 2.0  # A100 80GB: 2 TB/s

fp16_time = estimate_inference_time(140, 1, 2048, bandwidth)
int4_time = estimate_inference_time(35, 1, 2048, bandwidth)

print(f"FP16 time per token: {fp16_time:.2f} ms")
print(f"INT4 time per token: {int4_time:.2f} ms")
print(f"Speedup: {fp16_time / int4_time:.2f}x")
```

### Compute Acceleration

**Tensor Core Utilization:**

Modern NVIDIA GPUs have specialized Tensor Cores:

| GPU | FP16 TFLOPS | INT8 TOPS | FP8 TFLOPS |
|-----|-------------|-----------|------------|
| A100 | 312 | 624 | - |
| H100 | 989 | 1979 | 1979 |

**Effective Speedup Factors:**

```python
def compute_speedup_potential(precision_from, precision_to, hardware):
    """
    Estimate speedup from quantization.

    Factors:
    1. Memory bandwidth reduction
    2. Compute acceleration
    3. Overhead
    """
    bandwidth_ratio = {
        ('fp16', 'int8'): 2.0,
        ('fp16', 'int4'): 4.0,
        ('fp16', 'fp8'): 2.0,
    }

    compute_ratio = {
        ('fp16', 'int8'): 2.0 if hardware == 'A100' else 2.5,
        ('fp16', 'fp8'): 2.0 if hardware == 'H100' else 1.0,
    }

    # Actual speedup is limited by the minimum
    bw_speedup = bandwidth_ratio.get((precision_from, precision_to), 1.0)
    comp_speedup = compute_ratio.get((precision_from, precision_to), 1.0)

    # Account for overhead (dequantization, etc.)
    overhead_factor = 0.85 if precision_to == 'int4' else 0.95

    effective_speedup = min(bw_speedup, comp_speedup) * overhead_factor
    return effective_speedup
```

---

## Quantization in vLLM Architecture

### vLLM Quantization Pipeline

vLLM supports multiple quantization schemes through a modular architecture:

```
vllm/
├── model_executor/
│   └── layers/
│       └── quantization/
│           ├── base_config.py          # Base quantization interface
│           ├── awq.py                  # AWQ implementation
│           ├── gptq.py                 # GPTQ implementation
│           ├── fp8.py                  # FP8 quantization
│           ├── bitsandbytes.py         # 4-bit/8-bit quantization
│           └── ...
```

### Key Components

#### 1. Quantization Config

```python
# Example from vllm/model_executor/layers/quantization/base_config.py
class QuantizationConfig:
    """Base class for quantization configurations."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the quantization method."""
        pass

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Return supported activation dtypes."""
        pass

    @abstractmethod
    def get_min_capability(self) -> int:
        """Return minimum CUDA compute capability."""
        pass

    @abstractmethod
    def get_quant_method(self):
        """Return the quantization method class."""
        pass
```

#### 2. Quantized Linear Layer

```python
# Simplified version from vLLM
class QuantizedLinear(nn.Module):
    """Base quantized linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_config: QuantizationConfig,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_config = quant_config

        # Quantized weight storage
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // 8), dtype=torch.int32)
        )

        # Scales and zero-points
        self.register_buffer(
            'scales',
            torch.zeros(out_features, dtype=torch.float16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implemented by specific quantization method
        raise NotImplementedError
```

### Loading Quantized Models in vLLM

```python
from vllm import LLM, SamplingParams

# Example 1: Loading GPTQ model
llm_gptq = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16",
    gpu_memory_utilization=0.9,
)

# Example 2: Loading AWQ model
llm_awq = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",
)

# Example 3: FP8 quantization (H100)
llm_fp8 = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
    dtype="float16",
)

# Generate
prompts = ["Once upon a time"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm_gptq.generate(prompts, sampling_params)
```

### vLLM Quantization Kernel Integration

vLLM uses custom CUDA kernels for efficient quantized operations:

```
csrc/quantization/
├── awq/
│   ├── gemm_kernels.cu          # AWQ GEMM kernels
│   └── dequantize.cuh
├── gptq/
│   ├── q_gemm.cu                # GPTQ matrix multiplication
│   └── matrix.cuh
├── marlin/
│   ├── marlin_cuda_kernel.cu   # Marlin 4-bit kernels
│   └── marlin.cuh
└── w8a8/
    └── w8a8_gemm.cu             # INT8 weight-activation kernels
```

**Example Kernel Call:**
```python
# From vllm/model_executor/layers/quantization/awq.py
def apply_awq(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
):
    """Apply AWQ quantized linear layer."""
    import awq_inference_engine

    output = awq_inference_engine.gemm_forward_cuda(
        x,
        qweight,
        scales,
        qzeros,
    )
    return output
```

---

## Basic Quantization Implementation

### Implementing Naive Post-Training Quantization

```python
import torch
import torch.nn as nn

class NaiveQuantizedLinear(nn.Module):
    """
    Simple per-channel symmetric INT8 quantization.
    For educational purposes only - not production-ready!
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Storage for quantized weights (INT8)
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features), dtype=torch.int8)
        )

        # Per-channel scales
        self.register_buffer(
            'scales',
            torch.ones(out_features, 1, dtype=torch.float16)
        )

    @classmethod
    def from_float(cls, float_linear: nn.Linear):
        """Convert a FP16/FP32 linear layer to quantized."""
        quantized = cls(float_linear.in_features, float_linear.out_features)

        # Get float weights
        W = float_linear.weight.data  # [out_features, in_features]

        # Compute per-channel scales
        W_abs_max = W.abs().max(dim=1, keepdim=True)[0]  # [out_features, 1]
        scales = W_abs_max / 127.0
        scales = scales.clamp(min=1e-5)  # Avoid division by zero

        # Quantize
        qweight = (W / scales).round().clamp(-128, 127).to(torch.int8)

        # Store
        quantized.qweight.copy_(qweight)
        quantized.scales.copy_(scales.half())

        # Copy bias if exists
        if float_linear.bias is not None:
            quantized.register_buffer('bias', float_linear.bias.data.half())
        else:
            quantized.register_buffer('bias', None)

        return quantized

    def forward(self, x):
        """
        Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor [batch, in_features] in FP16

        Returns:
            Output tensor [batch, out_features] in FP16
        """
        # Dequantize weights on-the-fly
        W_dequant = self.qweight.half() * self.scales

        # Standard FP16 matmul
        output = torch.matmul(x, W_dequant.T)

        if self.bias is not None:
            output += self.bias

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, quantized=INT8'


# Example usage
def quantize_model(model):
    """Recursively quantize all Linear layers in a model."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with quantized version
            quantized_linear = NaiveQuantizedLinear.from_float(module)
            setattr(model, name, quantized_linear)
        else:
            # Recursively quantize child modules
            quantize_model(module)
    return model


# Test
if __name__ == '__main__':
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 512)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model_fp16 = SimpleModel().half().cuda()

    # Quantize
    model_int8 = quantize_model(model_fp16)

    # Test inference
    x = torch.randn(4, 512).half().cuda()

    with torch.no_grad():
        y_fp16 = model_fp16(x)
        y_int8 = model_int8(x)

    # Check error
    mse = ((y_fp16 - y_int8) ** 2).mean()
    print(f"MSE between FP16 and INT8: {mse.item():.6f}")

    # Check memory
    def get_model_size(model):
        total = 0
        for param in model.buffers():
            total += param.nelement() * param.element_size()
        return total / (1024**2)  # MB

    print(f"FP16 model size: {get_model_size(model_fp16):.2f} MB")
    print(f"INT8 model size: {get_model_size(model_int8):.2f} MB")
```

### Calibration for Static Quantization

```python
def calibrate_activations(model, calibration_loader, num_batches=10):
    """
    Collect activation statistics for static quantization.

    Args:
        model: The model to calibrate
        calibration_loader: DataLoader with calibration data
        num_batches: Number of batches to use

    Returns:
        Dictionary of activation ranges per layer
    """
    activation_stats = {}
    hooks = []

    def register_hook(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []

            # Collect max absolute value
            activation_stats[name].append(output.abs().max().item())

        return hook

    # Register hooks for all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(register_hook(name))
            hooks.append(hook)

    # Run calibration
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break

            # Forward pass
            _ = model(batch['input_ids'].cuda())

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute final scales (use 99.9th percentile to handle outliers)
    activation_scales = {}
    for name, values in activation_stats.items():
        activation_scales[name] = np.percentile(values, 99.9) / 127.0

    return activation_scales
```

---

## Trade-offs and Considerations

### Accuracy vs. Speed Trade-off

**General Trend:**
```
Higher Precision → Better Accuracy, Slower Inference
Lower Precision  → Worse Accuracy, Faster Inference
```

**Empirical Results (Llama-2-7B on WikiText-2):**

| Method    | Perplexity | Latency (ms) | Memory (GB) | Speedup |
|-----------|------------|--------------|-------------|---------|
| FP16      | 5.47       | 25           | 13.5        | 1.0x    |
| INT8 (W)  | 5.51       | 15           | 7.0         | 1.7x    |
| INT8 (WA) | 5.68       | 12           | 7.0         | 2.1x    |
| INT4 (W)  | 5.72       | 10           | 3.5         | 2.5x    |
| FP8       | 5.49       | 11           | 7.0         | 2.3x    |

### When to Use Each Quantization Scheme

#### Weight-Only INT8/INT4
**Use when:**
- Memory is the bottleneck
- Need minimal accuracy loss
- Don't have weight-activation quantized kernels

**Avoid when:**
- Compute-bound workloads
- Need maximum speed

#### Weight-Activation INT8
**Use when:**
- Have modern GPU (A100, H100)
- Need balanced speed/accuracy
- Batch size > 1

**Avoid when:**
- Model has extreme outliers
- FP8 is available (H100)

#### FP8
**Use when:**
- Using H100 GPU
- Need best accuracy/speed trade-off
- Training or fine-tuning

**Avoid when:**
- On older GPUs (A100, V100)
- Need maximum compression

### Quantization-Aware Considerations

**Outlier Handling:**
- Some models (e.g., OPT, BLOOM) have extreme outliers
- SmoothQuant migrates difficulty to weights
- AWQ protects important weights

**Layer-wise Sensitivity:**
- First/last layers more sensitive
- Attention layers vs. FFN layers
- Consider mixed-precision

**Calibration Data:**
- Should be representative of target distribution
- 128-512 samples usually sufficient
- Quality > Quantity

---

## Practical Exercises

### Exercise 1: Implement Quantization

Implement a complete quantization pipeline:

```python
# TODO: Implement this
def quantize_and_benchmark(model_name, quantization_method):
    """
    1. Load a model
    2. Apply quantization
    3. Benchmark latency and memory
    4. Evaluate perplexity
    """
    pass
```

### Exercise 2: Compare Quantization Schemes

Create a comparison table for different quantization methods on the same model.

**Requirements:**
- Test on Llama-2-7B or similar
- Measure: perplexity, latency, memory
- Methods: FP16, INT8, INT4, FP8

### Exercise 3: Analyze Quantization Error

Implement error analysis:

```python
def analyze_layer_wise_error(original_model, quantized_model, test_data):
    """
    Analyze quantization error per layer:
    1. Hook into each layer
    2. Compute activation differences
    3. Identify most affected layers
    """
    pass
```

### Exercise 4: Calibration Study

Study the effect of calibration data size and quality:

```python
def calibration_sensitivity_analysis(model, dataset):
    """
    Test quantization accuracy with:
    - Different calibration sizes: 16, 64, 256, 1024 samples
    - Different data sources
    - Random data vs. real data
    """
    pass
```

---

## Summary

### Key Takeaways

1. **Quantization is Essential**: For large models, quantization is not optional—it's required for efficient inference.

2. **Multiple Schemes**: No one-size-fits-all solution. Choose based on:
   - Hardware capabilities
   - Accuracy requirements
   - Memory vs. speed priority

3. **vLLM Integration**: vLLM provides excellent quantization support through modular architecture.

4. **Calibration Matters**: For static and advanced quantization, quality calibration data is crucial.

5. **Trade-offs**: Always measure the accuracy/speed/memory triangle for your specific use case.

### Next Steps

In the following modules, we'll dive deep into:
- **Module 2**: INT8 Quantization in detail
- **Module 3**: INT4 and FP8 quantization
- **Module 4**: GPTQ algorithm
- **Module 5**: AWQ algorithm
- **Module 6**: SmoothQuant
- **Module 7**: Quantized kernel implementations

---

## References

1. "A Survey on Model Compression for Large Language Models" - arXiv:2308.07633
2. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" - arXiv:2208.07339
3. vLLM Documentation: https://docs.vllm.ai/en/latest/quantization/
4. NVIDIA TensorRT Quantization Toolkit
5. "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"

---

**Module 6.1 Complete** • Next: [INT8 Quantization](02_int8_quantization.md)
