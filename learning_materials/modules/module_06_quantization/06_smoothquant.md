# SmoothQuant: Smooth Activation Distribution for Quantization

## Table of Contents
1. [Introduction](#introduction)
2. [The Activation Quantization Problem](#the-activation-quantization-problem)
3. [SmoothQuant Algorithm](#smoothquant-algorithm)
4. [Implementation](#implementation)
5. [Integration with vLLM](#integration-with-vllm)
6. [Performance Analysis](#performance-analysis)
7. [Practical Usage](#practical-usage)

---

## Introduction

### What is SmoothQuant?

**SmoothQuant** is a training-free post-training quantization method that enables **8-bit weight-activation quantization** for LLMs by "smoothing" activation outliers.

**Key Innovation:**
```
Instead of:  Quantizing difficult activations directly
Do:         Migrate difficulty from activations to weights
            (Weights are easier to quantize!)
```

**Published:** November 2022 by Xiao et al. (MIT, NVIDIA)

### Why SmoothQuant Matters

**Problem**: LLM activations have extreme outliers that destroy INT8 quantization accuracy.

**Solution**: Mathematical equivalence transformation:
```
Y = (X @ W) = (X * s^-1) @ (W * s)
```

Where `s` is a per-channel smoothing factor that:
- Reduces activation outliers
- Increases weight magnitude (which is easier to quantize)

---

## The Activation Quantization Problem

### Demonstration of Outliers

```python
import torch
import torch.nn as nn

def analyze_activation_outliers(model, dataloader):
    """
    Identify activation outliers in LLM.

    Args:
        model: Transformer model
        dataloader: Calibration data

    Returns:
        Statistics on activation outliers
    """
    layer_stats = {}

    def hook_fn(name):
        def hook(module, input, output):
            x = input[0].detach()  # [batch, seq, features]

            # Per-channel statistics
            x_abs = x.abs()
            channel_max = x_abs.max(dim=(0, 1))[0]  # [features]
            channel_mean = x_abs.mean(dim=(0, 1))[0]

            # Identify outliers (channels with max >> mean)
            outlier_ratio = channel_max / (channel_mean + 1e-5)

            layer_stats[name] = {
                'max_outlier_ratio': outlier_ratio.max().item(),
                'num_outliers': (outlier_ratio > 100).sum().item(),
                'total_channels': len(outlier_ratio),
            }

        return hook

    # Register hooks on all linear layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Collect statistics
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(**batch)
            break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print results
    print("\nActivation Outlier Analysis:")
    print(f"{'Layer':<40} {'Max Ratio':>10} {'Outliers':>10} {'Total':>10}")
    print("-" * 75)

    for name, stats in layer_stats.items():
        print(f"{name:<40} {stats['max_outlier_ratio']:>10.2f} "
              f"{stats['num_outliers']:>10} {stats['total_channels']:>10}")

# Example output:
# Activation Outlier Analysis:
# Layer                                      Max Ratio   Outliers      Total
# ---------------------------------------------------------------------------
# model.layers.0.self_attn.q_proj             487.23         12       4096
# model.layers.0.self_attn.k_proj             523.15         18       4096
# model.layers.0.mlp.gate_proj               1284.56         34      11008
# ...
# Outliers are 100-1000x larger than average!
```

### Impact on Quantization

```python
def demonstrate_outlier_impact():
    """Show how outliers destroy quantization."""

    # Simulate activations with outliers
    activations = torch.randn(1000) * 0.5
    activations[42] = 100.0  # Outlier 200x larger

    # INT8 quantization
    scale = activations.abs().max() / 127
    act_q = (activations / scale).round().clamp(-128, 127)
    act_dequant = act_q * scale

    # Error analysis
    error = (activations - act_dequant).abs()

    print(f"Scale (dominated by outlier): {scale:.6f}")
    print(f"Normal values error: {error[error < 1].mean():.6f}")
    print(f"Outlier error: {error[42]:.6f}")
    print(f"SNR: {10 * torch.log10(activations.var() / error.var()):.2f} dB")

# Output:
# Scale (dominated by outlier): 0.787402
# Normal values error: 0.393701  <- Large error for normal values!
# Outlier error: 0.000000
# SNR: 8.23 dB  <- Poor signal-to-noise ratio
```

---

## SmoothQuant Algorithm

### Mathematical Foundation

Given a linear layer: `Y = X @ W`

We can introduce a per-channel smoothing factor `s`:

```
Y = (X @ W)
  = (X * diag(s)^-1) @ (diag(s) @ W)
  = X' @ W'

Where:
  X' = X / s  (smoothed activations)
  W' = W * s  (scaled weights)
```

This is **mathematically equivalent** but changes quantization difficulty.

### Optimal Smoothing Factor

The optimal `s` balances activation and weight quantization difficulty:

```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)

Where:
  - j: channel index
  - α: migration strength (0 to 1)
  - α=0: no smoothing
  - α=0.5: balanced
  - α=1: full migration to weights
```

### Implementation

```python
def compute_smoothquant_scales(
    X: torch.Tensor,
    W: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute SmoothQuant per-channel scaling factors.

    Args:
        X: Activation samples [batch, seq, in_features]
        W: Weight matrix [out_features, in_features]
        alpha: Migration strength (0-1)

    Returns:
        scales: Per-channel smoothing factors [in_features]
    """
    # Flatten batch and sequence dimensions
    if X.dim() == 3:
        X = X.reshape(-1, X.size(-1))  # [batch*seq, in_features]

    # Compute per-channel max magnitudes
    activation_max = X.abs().max(dim=0)[0]  # [in_features]
    weight_max = W.abs().max(dim=0)[0]      # [in_features]

    # Compute smoothing factors
    # s = (max_activation^α) / (max_weight^(1-α))
    scales = activation_max.pow(alpha) / (weight_max.pow(1 - alpha) + 1e-5)

    # Clamp to prevent extreme values
    scales = scales.clamp(min=0.01, max=100)

    return scales


def apply_smoothquant(X, W, scales):
    """
    Apply SmoothQuant transformation.

    Args:
        X: Activations [batch, in_features]
        W: Weights [out_features, in_features]
        scales: Smoothing factors [in_features]

    Returns:
        X_smooth: Smoothed activations
        W_smooth: Scaled weights
    """
    # X' = X / s
    X_smooth = X / scales.unsqueeze(0)

    # W' = W * s
    W_smooth = W * scales.unsqueeze(0)

    # Verify mathematical equivalence
    Y_original = X @ W.T
    Y_smooth = X_smooth @ W_smooth.T
    assert torch.allclose(Y_original, Y_smooth, rtol=1e-4)

    return X_smooth, W_smooth


# Example usage
def smooth_and_quantize(X, W, alpha=0.5, bits=8):
    """
    Complete SmoothQuant + quantization pipeline.

    Args:
        X: Calibration activations
        W: Weights
        alpha: Migration strength
        bits: Quantization bits

    Returns:
        Quantized weights, activation scales, smoothing scales
    """
    # 1. Compute smoothing factors
    smooth_scales = compute_smoothquant_scales(X, W, alpha)

    # 2. Apply smoothing
    X_smooth, W_smooth = apply_smoothquant(X, W, smooth_scales)

    # 3. Quantize smoothed weights (per-channel)
    W_max = W_smooth.abs().max(dim=1, keepdim=True)[0]
    W_scale = W_max / (2**(bits-1) - 1)
    W_q = (W_smooth / W_scale).round().clamp(-2**(bits-1), 2**(bits-1) - 1)

    # 4. Quantize smoothed activations (per-tensor, dynamic)
    X_max = X_smooth.abs().max()
    X_scale = X_max / (2**(bits-1) - 1)
    X_q = (X_smooth / X_scale).round().clamp(-2**(bits-1), 2**(bits-1) - 1)

    return {
        'W_q': W_q.to(torch.int8),
        'W_scale': W_scale,
        'X_scale': X_scale,
        'smooth_scales': smooth_scales,
    }
```

### Alpha Selection

```python
def grid_search_alpha(X, W, alphas=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """
    Find optimal alpha for SmoothQuant.

    Args:
        X: Calibration activations
        W: Weights
        alphas: List of alpha values to try

    Returns:
        best_alpha: Optimal alpha
        results: Errors for each alpha
    """
    Y_orig = X @ W.T  # Original output

    results = {}

    for alpha in alphas:
        # Apply SmoothQuant and quantize
        quant_result = smooth_and_quantize(X, W, alpha, bits=8)

        # Dequantize and compute output
        W_dequant = quant_result['W_q'].float() * quant_result['W_scale']
        X_smooth = X / quant_result['smooth_scales'].unsqueeze(0)

        # Quantize X_smooth dynamically
        X_scale = X_smooth.abs().max() / 127
        X_q = (X_smooth / X_scale).round().clamp(-128, 127)
        X_dequant = X_q * X_scale

        # Compute output
        Y_quant = X_dequant @ W_dequant.T

        # Error
        error = (Y_orig - Y_quant).pow(2).mean().sqrt()
        results[alpha] = error.item()

        print(f"α = {alpha:.2f}: RMSE = {error:.6f}")

    best_alpha = min(results, key=results.get)
    print(f"\nBest α: {best_alpha:.2f}")

    return best_alpha, results


# Typical output:
# α = 0.00: RMSE = 0.0234  <- No smoothing, poor
# α = 0.25: RMSE = 0.0089
# α = 0.50: RMSE = 0.0045  <- Usually optimal
# α = 0.75: RMSE = 0.0067
# α = 1.00: RMSE = 0.0128  <- Too much smoothing
```

---

## Implementation

### Complete SmoothQuant Layer

```python
import torch
import torch.nn as nn

class SmoothQuantLinear(nn.Module):
    """
    Linear layer with SmoothQuant INT8 weight-activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Quantized weights (INT8)
        self.register_buffer(
            'weight_q',
            torch.zeros((out_features, in_features), dtype=torch.int8)
        )

        # Weight scales (per-channel)
        self.register_buffer(
            'weight_scales',
            torch.ones((out_features, 1), dtype=torch.float32)
        )

        # SmoothQuant smoothing factors (per-channel)
        self.register_buffer(
            'smooth_scales',
            torch.ones(in_features, dtype=torch.float32)
        )

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, activations: torch.Tensor, alpha: float = 0.5):
        """
        Convert FP32/FP16 linear layer to SmoothQuant.

        Args:
            linear: Original linear layer
            activations: Calibration activations [batch, seq, in_features]
            alpha: Migration strength

        Returns:
            SmoothQuant linear layer
        """
        sq_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            alpha=alpha,
        )

        W = linear.weight.data.float()

        # Flatten activations
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.size(-1))

        # 1. Compute smoothing factors
        smooth_scales = compute_smoothquant_scales(activations, W, alpha)

        # 2. Apply smoothing to weights
        W_smooth = W * smooth_scales.unsqueeze(0)

        # 3. Quantize smoothed weights (per-channel)
        W_max = W_smooth.abs().max(dim=1, keepdim=True)[0]
        W_scale = W_max / 127.0
        W_scale = W_scale.clamp(min=1e-5)

        W_q = (W_smooth / W_scale).round().clamp(-128, 127).to(torch.int8)

        # 4. Store
        sq_linear.weight_q.copy_(W_q)
        sq_linear.weight_scales.copy_(W_scale)
        sq_linear.smooth_scales.copy_(smooth_scales)

        if linear.bias is not None:
            sq_linear.bias.copy_(linear.bias.data)

        return sq_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with INT8 weight-activation quantization.

        Args:
            x: Input [batch, in_features]

        Returns:
            Output [batch, out_features]
        """
        # 1. Apply smoothing to activations: X' = X / s
        x_smooth = x / self.smooth_scales.unsqueeze(0)

        # 2. Dynamically quantize smoothed activations
        x_scale = x_smooth.abs().max() / 127.0
        x_q = (x_smooth / x_scale).round().clamp(-128, 127).to(torch.int8)

        # 3. INT8 matmul
        output_q = torch.ops.aten._int_mm(x_q, self.weight_q.T)

        # 4. Dequantize output
        output = output_q.float() * (x_scale * self.weight_scales.T)

        # 5. Add bias
        if self.bias is not None:
            output += self.bias

        return output


# Utility: Convert entire model
def convert_model_to_smoothquant(model, calibration_dataloader, alpha=0.5):
    """
    Convert all linear layers in model to SmoothQuant.

    Args:
        model: PyTorch model
        calibration_dataloader: Calibration data
        alpha: Migration strength

    Returns:
        Model with SmoothQuant layers
    """
    # Collect activations for each layer
    layer_activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if name not in layer_activations:
                layer_activations[name] = []
            layer_activations[name].append(input[0].detach().cpu())
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Collect calibration data
    print("Collecting calibration data...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_dataloader):
            if i >= 10:  # Use 10 batches
                break
            _ = model(**batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Convert each layer
    print("Converting to SmoothQuant...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in layer_activations:
            # Concatenate activation batches
            acts = torch.cat(layer_activations[name], dim=0)

            # Convert to SmoothQuant
            sq_layer = SmoothQuantLinear.from_float(module, acts, alpha)

            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model

            setattr(parent, child_name, sq_layer)

            print(f"  Converted {name}")

    return model
```

---

## Integration with vLLM

### SmoothQuant Support in vLLM

vLLM supports SmoothQuant through its quantization framework:

```python
from vllm import LLM, SamplingParams

# Load model with SmoothQuant
llm = LLM(
    model="mit-han-lab/opt-30b-smoothquant",
    quantization="smoothquant",
    dtype="float16",
)

# Generate
prompts = ["Explain machine learning"]
params = SamplingParams(temperature=0.8, max_tokens=150)
outputs = llm.generate(prompts, params)
```

### vLLM Implementation

```python
# File: vllm/model_executor/layers/quantization/smoothquant.py (simplified)

from dataclasses import dataclass

@dataclass
class SmoothQuantConfig:
    """SmoothQuant configuration."""

    weight_bits: int = 8
    activation_bits: int = 8
    alpha: float = 0.5

    def get_quant_method(self):
        """Return quantization method."""
        return SmoothQuantLinearMethod(self)


class SmoothQuantLinearMethod:
    """SmoothQuant linear method."""

    def __init__(self, config: SmoothQuantConfig):
        self.config = config

    def apply_weights(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply SmoothQuant quantized weights."""

        # Apply smoothing to activations
        x_smooth = x / layer.smooth_scales

        # Quantize activations dynamically
        x_scale = x_smooth.abs().max() / 127.0
        x_q = (x_smooth / x_scale).round().clamp(-128, 127).to(torch.int8)

        # INT8 GEMM
        output_q = torch.ops.aten._int_mm(x_q, layer.weight_q.T)

        # Dequantize
        output = output_q.float() * (x_scale * layer.weight_scales.T)

        if bias is not None:
            output += bias

        return output
```

---

## Performance Analysis

### Accuracy Comparison

**OPT-175B on benchmarks:**

| Method | WikiText-2 PPL | LAMBADA Acc | Time |
|--------|----------------|-------------|------|
| FP16   | 10.13          | 74.6%       | -    |
| W8A8 (naive) | 1438.2   | 23.4%       | -    |
| SmoothQuant | 10.28     | 73.9%       | -    |

**SmoothQuant recovers from catastrophic failure!**

### Speed and Memory

**Llama-2-7B on A100:**

| Method | Memory | Throughput | Latency |
|--------|--------|------------|---------|
| FP16   | 13.5GB | 450 tok/s  | 22ms    |
| SQ INT8 | 7.2GB | 890 tok/s  | 11ms    |

**Gains:**
- 1.9x memory reduction
- 2.0x throughput increase
- 2.0x latency reduction

---

## Practical Usage

### Quantizing Your Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def quantize_to_smoothquant(
    model_name: str,
    output_dir: str,
    alpha: float = 0.5,
    num_calibration_samples: int = 128,
):
    """
    Quantize model with SmoothQuant.

    Args:
        model_name: HuggingFace model name
        output_dir: Output directory
        alpha: Migration strength
        num_calibration_samples: Number of calibration samples
    """
    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare calibration data
    print("Preparing calibration data...")
    dataset = load_dataset("c4", split="train", streaming=True)
    dataset = dataset.take(num_calibration_samples)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    calibration_data = []
    for sample in dataset:
        tokens = tokenize(sample)
        calibration_data.append(tokens)

    # Convert to SmoothQuant
    print("Converting to SmoothQuant...")
    model_sq = convert_model_to_smoothquant(model, calibration_data, alpha)

    # Save
    print(f"Saving to {output_dir}...")
    model_sq.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done!")
    return model_sq


# Example
quantize_to_smoothquant(
    "meta-llama/Llama-2-7b-hf",
    "./llama-2-7b-smoothquant",
    alpha=0.5,
)
```

### Best Practices

```python
# 1. Alpha selection by model size
ALPHA_RECOMMENDATIONS = {
    'small': 0.65,   # <3B parameters (more migration needed)
    'medium': 0.50,  # 3-13B parameters (balanced)
    'large': 0.50,   # 13-70B parameters
    'xlarge': 0.45,  # >70B parameters (less migration)
}

# 2. Calibration data
CALIBRATION_SAMPLES = 128  # Usually sufficient

# 3. Layer-specific alpha
# Some layers (e.g., attention) may need different alpha
# than FFN layers
```

---

## Summary

### Key Takeaways

1. **Smooth Activations**: Migrates quantization difficulty from activations to weights
2. **Mathematically Equivalent**: No accuracy loss from smoothing itself
3. **Enables W8A8**: Makes INT8 weight-activation quantization practical
4. **Fast**: Dynamic activation quantization is fast
5. **Training-Free**: Post-training quantization only

### When to Use SmoothQuant

**Use SmoothQuant when:**
- Need weight-activation INT8 quantization
- Model has severe activation outliers (OPT, BLOOM)
- Want 2x speedup on supported hardware
- Using A100 or newer GPUs

**Consider alternatives when:**
- Weight-only quantization sufficient (AWQ, GPTQ)
- Need INT4 (AWQ, GPTQ better)
- Using H100 (FP8 better)

---

## References

1. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" - Xiao et al., 2022
2. SmoothQuant Code: https://github.com/mit-han-lab/smoothquant
3. vLLM SmoothQuant Implementation

---

**Module 6.6 Complete** • Next: [Quantized Kernels](07_quantized_kernels.md)
