# Day 12: Quantization Kernels - INT8, INT4, FP8 & Mixed Precision

> **Goal**: Master quantization techniques in vLLM: algorithms, kernels, and performance trade-offs
> **Time**: 6-8 hours
> **Prerequisites**: Days 8-11 completed, understanding of number representations
> **Deliverables**: Quantization kernel implementations, accuracy analysis, performance benchmarks

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Quantization Fundamentals

**9:00-10:00** - Quantization Theory & Number Formats
**10:00-11:00** - AWQ, GPTQ, SmoothQuant Methods
**11:00-11:15** - Break
**11:15-12:30** - Weight Quantization Kernels

### Afternoon Session (3-4 hours): Advanced Quantization

**14:00-15:00** - Activation Quantization & KV Cache
**15:00-16:00** - FP8 and Mixed Precision
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Implement INT8 GEMM with Dequant

### Evening (Optional, 1-2 hours): Analysis

**19:00-21:00** - Accuracy analysis, performance benchmarks, trade-off exploration

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain different quantization methods (symmetric, asymmetric, per-channel)
- [ ] Understand AWQ, GPTQ, and SmoothQuant algorithms
- [ ] Implement quantization/dequantization kernels
- [ ] Optimize INT8/INT4 GEMM operations
- [ ] Handle FP8 precision on modern GPUs
- [ ] Analyze accuracy vs performance trade-offs
- [ ] Integrate quantization into inference pipeline

---

## 📚 Morning: Quantization Fundamentals (9:00-12:30)

### Task 1: Quantization Theory (60 min)

**What is Quantization?**

```
Goal: Represent FP16/FP32 weights/activations with fewer bits

FP16: 16 bits per value
  - Sign: 1 bit
  - Exponent: 5 bits
  - Mantissa: 10 bits
  - Range: ~±65,504, precision: ~0.001

INT8: 8 bits per value
  - Range: -128 to 127 (signed)
  - Storage: 2x less memory
  - Compute: 2-4x faster (on INT8 tensor cores)

INT4: 4 bits per value
  - Range: -8 to 7 (signed)
  - Storage: 4x less memory
  - Compute: 4-8x faster (on specialized hardware)
```

**Quantization Methods**:

```python
# 1. Symmetric Quantization (zero-point = 0)
def quantize_symmetric(x_fp32, num_bits=8):
    """
    Q = round(x / scale)
    where scale = max(abs(x)) / (2^(num_bits-1) - 1)
    """
    max_val = np.max(np.abs(x_fp32))
    scale = max_val / (2 ** (num_bits - 1) - 1)

    x_int = np.round(x_fp32 / scale)
    x_int = np.clip(x_int, -(2**(num_bits-1)), 2**(num_bits-1) - 1)

    return x_int.astype(np.int8), scale

def dequantize_symmetric(x_int, scale):
    """
    x_fp32 = x_int * scale
    """
    return x_int.astype(np.float32) * scale

# Example:
x = np.array([0.1, -0.5, 1.2, -1.8, 0.3])
x_int, scale = quantize_symmetric(x, num_bits=8)
print(f"Original: {x}")
print(f"Quantized: {x_int}, scale: {scale}")
print(f"Dequantized: {dequantize_symmetric(x_int, scale)}")

# Output:
# Original: [ 0.1 -0.5  1.2 -1.8  0.3]
# Quantized: [7 -35 85 -127 21], scale: 0.014173228
# Dequantized: [ 0.099  -0.496   1.205  -1.800   0.298]
```

```python
# 2. Asymmetric Quantization (with zero-point)
def quantize_asymmetric(x_fp32, num_bits=8):
    """
    Q = round(x / scale + zero_point)
    scale = (max(x) - min(x)) / (2^num_bits - 1)
    zero_point = round(-min(x) / scale)
    """
    min_val = np.min(x_fp32)
    max_val = np.max(x_fp32)

    scale = (max_val - min_val) / (2 ** num_bits - 1)
    zero_point = int(np.round(-min_val / scale))

    x_int = np.round(x_fp32 / scale + zero_point)
    x_int = np.clip(x_int, 0, 2**num_bits - 1)

    return x_int.astype(np.uint8), scale, zero_point

def dequantize_asymmetric(x_int, scale, zero_point):
    """
    x_fp32 = (x_int - zero_point) * scale
    """
    return (x_int.astype(np.float32) - zero_point) * scale

# Asymmetric is better for activations (often non-negative)
```

```python
# 3. Per-Channel vs Per-Tensor Quantization

# Per-Tensor: One scale for entire tensor
def quantize_per_tensor(weight, num_bits=8):
    scale = np.max(np.abs(weight)) / (2**(num_bits-1) - 1)
    return np.round(weight / scale).astype(np.int8), scale

# Per-Channel: Different scale for each output channel
def quantize_per_channel(weight, num_bits=8):
    """
    Weight shape: [out_channels, in_channels]
    Each output channel gets its own scale
    """
    num_channels = weight.shape[0]
    scales = np.zeros(num_channels)
    quantized = np.zeros_like(weight, dtype=np.int8)

    for c in range(num_channels):
        channel_weight = weight[c, :]
        max_val = np.max(np.abs(channel_weight))
        scale = max_val / (2**(num_bits-1) - 1)
        scales[c] = scale
        quantized[c, :] = np.round(channel_weight / scale)

    return quantized, scales

# Per-channel: Better accuracy (different ranges per channel)
# Per-tensor: Simpler, faster
```

**Quantization Error Analysis**:

```python
# Quantization introduces error
def analyze_quantization_error(x_fp32, num_bits):
    x_int, scale = quantize_symmetric(x_fp32, num_bits)
    x_dequant = dequantize_symmetric(x_int, scale)

    # Mean Squared Error
    mse = np.mean((x_fp32 - x_dequant) ** 2)

    # Signal-to-Noise Ratio
    signal_power = np.mean(x_fp32 ** 2)
    noise_power = np.mean((x_fp32 - x_dequant) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)

    print(f"{num_bits}-bit quantization:")
    print(f"  MSE: {mse:.6f}")
    print(f"  SNR: {snr_db:.2f} dB")

# Typical results:
# 8-bit: SNR ~40-50 dB (good)
# 4-bit: SNR ~20-30 dB (acceptable with calibration)
# 2-bit: SNR ~10-15 dB (significant degradation)
```

### Task 2: AWQ, GPTQ, SmoothQuant Methods (60 min)

**1. GPTQ (GPT Quantization)**

```
Algorithm: Optimal Brain Quantization (OBQ) variant
  - Layer-by-layer quantization
  - Minimize reconstruction error using Hessian information
  - Fast calibration (128 samples)

Process:
  1. Compute Hessian H for layer weights W
  2. For each weight group:
     a. Quantize weights
     b. Compute optimal adjustment to minimize error
     c. Update remaining weights to compensate

Advantages:
  ✓ 4-bit quantization with minimal accuracy loss
  ✓ Fast calibration
  ✓ Works well for large models

Implementation: Group-wise quantization
  - Group size: 32-128 weights
  - Each group has own scale/zero-point
```

**GPTQ Kernel Pattern**:

```cuda
// GPTQ stores quantized weights + group scales

// Weight layout (INT4):
// [num_groups, group_size_packed]
// where group_size_packed = group_size / 2 (2 INT4 values per byte)

// Scales layout:
// [num_groups] in FP16

__global__ void gptq_dequantize_kernel(
    const uint8_t* __restrict__ qweight,  // Packed INT4 weights
    const half* __restrict__ scales,      // Per-group scales
    const half* __restrict__ zeros,       // Per-group zero-points
    half* __restrict__ weight,            // Output FP16 weights
    const int num_groups,
    const int group_size
) {
    const int group_idx = blockIdx.x;
    const int idx_in_group = threadIdx.x;

    if (idx_in_group >= group_size) return;

    // Load scale and zero-point for this group
    const half scale = scales[group_idx];
    const half zero = zeros[group_idx];

    // Each byte contains 2 INT4 values
    const int packed_idx = group_idx * (group_size / 2) + idx_in_group / 2;
    const uint8_t packed = qweight[packed_idx];

    // Extract INT4 value (low or high 4 bits)
    int8_t qval;
    if (idx_in_group % 2 == 0) {
        qval = (packed & 0x0F) - 8;  // Low 4 bits, offset to signed
    } else {
        qval = ((packed >> 4) & 0x0F) - 8;  // High 4 bits
    }

    // Dequantize: w = (q - zero) * scale
    const int out_idx = group_idx * group_size + idx_in_group;
    weight[out_idx] = __float2half(
        (__half2float(scale) * (qval - __half2float(zero)))
    );
}
```

**2. AWQ (Activation-aware Weight Quantization)**

```
Key Insight: Not all weights are equally important!
  - Some weights (associated with salient activations) are more critical
  - Protect important weights from quantization error

Algorithm:
  1. Collect activation statistics on calibration data
  2. Identify salient channels (high activation magnitude)
  3. Scale down salient channels before quantization
  4. Scale up during inference

Process:
  For each layer:
    1. Compute per-channel activation scales: s = mean(|X|)
    2. Apply to weights: W' = W * diag(s)
    3. Quantize scaled weights: Q(W')
    4. At inference: Y = (Q(W') @ X) / s
       (division absorbed into scaling factors)

Benefits:
  ✓ Better accuracy than naive quantization
  ✓ 4-bit with minimal degradation
  ✓ No extra computation at inference
```

**AWQ Kernel Pattern**:

```cuda
// AWQ: Quantized weights + per-channel scales

__global__ void awq_gemm_kernel(
    const int8_t* __restrict__ qweight,   // INT4/INT8 weights
    const half* __restrict__ input,       // FP16 input
    const half* __restrict__ scales,      // Per-channel scales
    half* __restrict__ output,
    const int M, const int N, const int K
) {
    // Fused: Dequantize + GEMM
    // This is more efficient than separate dequant + GEMM

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Compute dot product
    for (int k = 0; k < K; k++) {
        // Load quantized weight
        int8_t qw = qweight[col * K + k];

        // Load input
        float in_val = __half2float(input[row * K + k]);

        // Accumulate (dequantization happens inline)
        sum += qw * in_val;
    }

    // Apply per-channel scale
    sum *= __half2float(scales[col]);

    // Write output
    output[row * N + col] = __float2half(sum);
}
```

**3. SmoothQuant**

```
Problem: Outliers in activations make quantization hard

Solution: Transfer difficulty from activations to weights
  - Weights are easier to quantize (static)
  - Activations have outliers (dynamic)

Algorithm:
  For each layer:
    1. Compute per-channel activation range: s = max(|X|)
    2. Compute per-channel weight range: t = max(|W|)
    3. Choose smoothing factor: α (0 to 1)
    4. Scale: W' = W * s^α, X' = X / s^α

    Intuition: Transfer variance from X to W

Benefits:
  ✓ Enables INT8 activation quantization
  ✓ Better than quantizing activations directly
  ✓ Particularly good for LLMs with outliers

vLLM support: Partial (mainly focuses on weight quant)
```

### Task 3: Weight Quantization Kernels (75 min)

**File**: `csrc/quantization/gptq/gptq_gemm.cu`

**INT4 Weight Packing**:

```cuda
// Packing: Store 2 INT4 values in 1 byte

__global__ void pack_int4_weights(
    const int8_t* __restrict__ unpacked,  // [N, K] INT8 (-8 to 7)
    uint8_t* __restrict__ packed,         // [N, K/2] packed
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * K;

    if (idx * 2 >= total_elements) return;

    // Get two INT4 values
    int8_t val0 = unpacked[idx * 2];
    int8_t val1 = unpacked[idx * 2 + 1];

    // Clamp to 4-bit range [-8, 7]
    val0 = max(-8, min(7, val0));
    val1 = max(-8, min(7, val1));

    // Convert to unsigned [0, 15]
    uint8_t uval0 = val0 + 8;
    uint8_t uval1 = val1 + 8;

    // Pack: low 4 bits = val0, high 4 bits = val1
    packed[idx] = (uval1 << 4) | uval0;
}
```

**GEMM with On-the-fly Dequantization**:

```cuda
// Fused dequantization + matrix multiplication
// More efficient than separate kernels!

template<int GROUP_SIZE>
__global__ void int4_gemm_dequant_kernel(
    const uint8_t* __restrict__ qweight,  // [N, K/2] packed INT4
    const half* __restrict__ scales,      // [N / GROUP_SIZE]
    const half* __restrict__ input,       // [M, K] FP16
    half* __restrict__ output,            // [M, N] FP16
    const int M, const int N, const int K
) {
    // Shared memory for input tile
    __shared__ half input_tile[TILE_M][TILE_K];

    // Shared memory for dequantized weight tile
    __shared__ half weight_tile[TILE_K][TILE_N];

    const int bx = blockIdx.x;  // Block in N dimension
    const int by = blockIdx.y;  // Block in M dimension
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    float acc = 0.0f;

    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load input tile
        const int input_row = by * TILE_M + ty;
        const int input_col = k_tile + tx;
        if (input_row < M && input_col < K) {
            input_tile[ty][tx] = input[input_row * K + input_col];
        }

        // Load and dequantize weight tile
        const int weight_row = k_tile + ty;  // K dimension
        const int weight_col = bx * TILE_N + tx;  // N dimension

        if (weight_row < K && weight_col < N) {
            // Determine group
            const int group_idx = weight_col / GROUP_SIZE;
            const half scale = scales[group_idx];

            // Load packed weight
            const int packed_idx = weight_col * (K / 2) + weight_row / 2;
            const uint8_t packed = qweight[packed_idx];

            // Extract INT4
            int8_t qval;
            if (weight_row % 2 == 0) {
                qval = (packed & 0x0F) - 8;
            } else {
                qval = ((packed >> 4) & 0x0F) - 8;
            }

            // Dequantize
            weight_tile[ty][tx] = __float2half(
                __half2float(scale) * qval
            );
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += __half2float(input_tile[ty][k]) *
                   __half2float(weight_tile[k][tx]);
        }

        __syncthreads();
    }

    // Write output
    const int out_row = by * TILE_M + ty;
    const int out_col = bx * TILE_N + tx;
    if (out_row < M && out_col < N) {
        output[out_row * N + out_col] = __float2half(acc);
    }
}
```

---

## 🔬 Afternoon: Advanced Quantization (14:00-18:00)

### Task 4: Activation Quantization & KV Cache (60 min)

**Quantizing KV Cache**:

```
Motivation: KV cache is memory bottleneck
  - OPT-13B, 2048 context: ~1 GB KV cache per sequence
  - INT8 KV cache: 0.5 GB (2x reduction)
  - FP8 KV cache: 0.5 GB with better accuracy

Challenges:
  - Activations have outliers
  - Per-token quantization needed (dynamic)
  - Must maintain accuracy
```

**INT8 KV Cache Kernel**:

```cuda
// Quantize and cache K, V in INT8

__global__ void quantize_kv_cache_int8_kernel(
    const half* __restrict__ key,           // [num_tokens, num_heads, head_size]
    const half* __restrict__ value,
    int8_t* __restrict__ key_cache_int8,    // Quantized cache
    int8_t* __restrict__ value_cache_int8,
    half* __restrict__ key_scales,          // Per-token scales
    half* __restrict__ value_scales,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int num_heads,
    const int head_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (token_idx >= num_tokens) return;

    const int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;

    // Load K vector for this head
    const int src_offset = token_idx * num_heads * head_size + head_idx * head_size;
    const half* k_src = key + src_offset;

    // Find max for quantization
    float max_val = 0.0f;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(__half2float(k_src[i])));
    }

    // Reduce max across threads
    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(
                shared_max[threadIdx.x],
                shared_max[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    max_val = shared_max[0];
    const float scale = max_val / 127.0f;

    // Save scale (per-token)
    if (threadIdx.x == 0) {
        key_scales[token_idx * num_heads + head_idx] = __float2half(scale);
    }

    // Quantize and write to cache
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = __half2float(k_src[i]);
        int8_t qval = (int8_t)roundf(val / scale);
        qval = max(-127, min(127, qval));

        // Write to cache (with proper layout)
        const int cache_offset = slot_idx * num_heads * head_size
                               + head_idx * head_size + i;
        key_cache_int8[cache_offset] = qval;
    }

    // Same for value (omitted for brevity)
}
```

**Dequantization in Attention**:

```cuda
// Modified attention kernel to handle INT8 KV cache

template<int HEAD_SIZE>
__global__ void paged_attention_int8_kv_kernel(
    half* __restrict__ output,
    const half* __restrict__ query,
    const int8_t* __restrict__ key_cache_int8,
    const int8_t* __restrict__ value_cache_int8,
    const half* __restrict__ key_scales,     // ← New: scales
    const half* __restrict__ value_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    // Load query
    __shared__ half q_shared[HEAD_SIZE];
    // ... (same as FP16 version)

    // Compute attention scores
    const int context_len = context_lens[seq_idx];

    for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += blockDim.x) {
        // Get block and offset
        const int block_idx = token_idx / BLOCK_SIZE;
        const int block_offset = token_idx % BLOCK_SIZE;
        const int physical_block = block_tables[seq_idx * MAX_BLOCKS + block_idx];

        // Load quantized K
        int8_t k_int8[HEAD_SIZE];
        const int k_cache_offset = physical_block * BLOCK_SIZE * num_heads * HEAD_SIZE
                                  + block_offset * num_heads * HEAD_SIZE
                                  + head_idx * HEAD_SIZE;

        for (int d = 0; d < HEAD_SIZE; d++) {
            k_int8[d] = key_cache_int8[k_cache_offset + d];
        }

        // Load scale for this token
        const half k_scale = key_scales[physical_block * BLOCK_SIZE * num_heads
                                       + block_offset * num_heads + head_idx];

        // Compute Q · K with dequantization
        float score = 0.0f;
        for (int d = 0; d < HEAD_SIZE; d++) {
            // Dequantize K on-the-fly
            float k_val = __half2float(k_scale) * k_int8[d];
            score += __half2float(q_shared[d]) * k_val;
        }

        // ... (rest of attention computation)
    }
}
```

### Task 5: FP8 and Mixed Precision (60 min)

**FP8 Format (H100, Ada GPUs)**:

```
E4M3 (4-bit exponent, 3-bit mantissa):
  - Range: ~±240
  - Precision: Lower than FP16
  - Best for: Weights, forward pass activations

E5M2 (5-bit exponent, 2-bit mantissa):
  - Range: ~±57,344
  - Precision: Very low
  - Best for: Gradients (need range)

Benefits:
  - 2x memory reduction vs FP16
  - 2x compute throughput (on supported hardware)
  - Better than INT8 for some workloads
```

**FP8 Kernel Example**:

```cuda
#include <cuda_fp8.h>

__global__ void fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ weight,  // FP8 weights
    const half* __restrict__ input,             // FP16 input
    half* __restrict__ output,
    const half* __restrict__ weight_scale,      // Scaling factor
    const int M, const int N, const int K
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Dot product
    for (int k = 0; k < K; k++) {
        // Load FP8 weight and convert to FP32
        __nv_fp8_e4m3 w_fp8 = weight[col * K + k];
        float w_fp32 = __half2float(__nv_cvt_fp8_to_halfraw(w_fp8));

        // Scale weight
        w_fp32 *= __half2float(weight_scale[col]);

        // Load input
        float in_val = __half2float(input[row * K + k]);

        // Accumulate
        sum += w_fp32 * in_val;
    }

    output[row * N + col] = __float2half(sum);
}
```

**Mixed Precision Strategy**:

```
Typical setup in vLLM:
  - Weights: INT4/INT8 (most memory usage)
  - KV Cache: FP8 or INT8 (large for long context)
  - Activations: FP16 (dynamic, harder to quantize)
  - Accumulation: FP32 (maintain precision)

Example: Llama-2-70B on A100 (80GB)
  - Weights: INT4 (~35 GB)
  - KV Cache: FP8 (~20 GB per batch of 100 seqs, 2K context)
  - Remaining: Activations, overhead (~25 GB)
  - Total: Can fit with good batch size!

Without quantization:
  - Weights: FP16 (~140 GB) ← Doesn't fit!
```

### Task 6: Implement INT8 GEMM with Dequant (90 min)

**Exercise**: Create optimized INT8 GEMM

Create: `~/projects/vllm/my_kernels/int8_gemm.cu`

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // For tensor cores

using namespace nvcuda;

// INT8 GEMM using tensor cores (Ampere+)
// C (FP16) = A (INT8) @ B (INT8) with per-channel dequantization

__global__ void int8_gemm_tensor_core_kernel(
    const int8_t* __restrict__ A,          // [M, K] INT8
    const int8_t* __restrict__ B,          // [K, N] INT8
    half* __restrict__ C,                  // [M, N] FP16 output
    const half* __restrict__ A_scales,     // [M] per-row scales
    const half* __restrict__ B_scales,     // [N] per-column scales
    const int M, const int N, const int K
) {
    // Tensor core WMMA (Warp Matrix Multiply-Accumulate) API
    // Fragment sizes: 16x16x16 for INT8

    const int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0);

    // Loop over K dimension
    for (int k = 0; k < K; k += 16) {
        const int m = warp_m * 16;
        const int n = warp_n * 16;

        if (m < M && n < N && k < K) {
            // Load A and B fragments
            wmma::load_matrix_sync(a_frag, A + m * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + n, N);

            // Multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Convert to FP16 and apply scales
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag_fp16;

    const int m = warp_m * 16;
    const int n = warp_n * 16;

    // Convert INT32 accumulator to FP16
    for (int i = 0; i < c_frag.num_elements; i++) {
        // Determine which row and column this element belongs to
        const int row_offset = i / 16;
        const int col_offset = i % 16;

        // Get scales
        const float scale_a = __half2float(A_scales[m + row_offset]);
        const float scale_b = __half2float(B_scales[n + col_offset]);

        // Dequantize: C_fp16 = C_int32 * scale_a * scale_b
        c_frag_fp16.x[i] = __float2half(
            c_frag.x[i] * scale_a * scale_b
        );
    }

    // Store result
    if (m < M && n < N) {
        wmma::store_matrix_sync(C + m * N + n, c_frag_fp16, N, wmma::mem_row_major);
    }
}

// Host launcher
void int8_gemm_tensor_core(
    const int8_t* A,
    const int8_t* B,
    half* C,
    const half* A_scales,
    const half* B_scales,
    int M, int N, int K
) {
    // Each warp handles 16x16 output tile
    dim3 block(128);  // 4 warps
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    int8_gemm_tensor_core_kernel<<<grid, block>>>(
        A, B, C, A_scales, B_scales, M, N, K
    );

    cudaDeviceSynchronize();
}

// Benchmark
void benchmark_int8_gemm() {
    const int M = 4096, N = 4096, K = 4096;

    // Allocate
    int8_t *d_A, *d_B;
    half *d_C, *d_A_scales, *d_B_scales;

    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMalloc(&d_A_scales, M * sizeof(half));
    cudaMalloc(&d_B_scales, N * sizeof(half));

    // Initialize (random data)
    // ... (initialization code)

    // Warmup
    int8_gemm_tensor_core(d_A, d_B, d_C, d_A_scales, d_B_scales, M, N, K);

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        int8_gemm_tensor_core(d_A, d_B, d_C, d_A_scales, d_B_scales, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Calculate TFLOPS
    // INT8 GEMM: 2 * M * N * K operations
    float tflops = (2.0f * M * N * K * 100) / (elapsed_ms / 1000.0f) / 1e12;

    printf("INT8 GEMM Performance:\n");
    printf("  Matrix size: %dx%d @ %dx%d\n", M, N, K, N);
    printf("  Time per GEMM: %.3f ms\n", elapsed_ms / 100.0f);
    printf("  Throughput: %.2f TFLOPS\n", tflops);

    // Expected on A100: ~300-400 TFLOPS (INT8 tensor cores)
}
```

---

## 📊 Analysis (Evening, 19:00-21:00)

### Task 7: Accuracy vs Performance Analysis

**Accuracy Benchmark**:

```python
#!/usr/bin/env python3
"""Compare accuracy of different quantization methods"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

def measure_perplexity(model, tokenizer, text):
    """Compute perplexity on text"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

# Load model in different precisions
model_fp16 = LLM("meta-llama/Llama-2-7b-hf", dtype="float16")
model_int8 = LLM("meta-llama/Llama-2-7b-hf", quantization="int8")
model_int4_awq = LLM("meta-llama/Llama-2-7b-awq", quantization="awq")
model_int4_gptq = LLM("meta-llama/Llama-2-7b-gptq", quantization="gptq")

# Test text
test_texts = [
    "The quick brown fox jumps over the lazy dog",
    # ... more test cases
]

# Compare
results = {
    "FP16": [],
    "INT8": [],
    "INT4-AWQ": [],
    "INT4-GPTQ": []
}

for text in test_texts:
    # Generate and compare
    # ... (implementation)

print("Accuracy Results:")
print(f"FP16 (baseline): perplexity = {np.mean(results['FP16']):.2f}")
print(f"INT8: perplexity = {np.mean(results['INT8']):.2f} (delta: +{...:.2f}%)")
print(f"INT4-AWQ: perplexity = {np.mean(results['INT4-AWQ']):.2f}")
print(f"INT4-GPTQ: perplexity = {np.mean(results['INT4-GPTQ']):.2f}")
```

**Performance Benchmark**:

```python
# Measure throughput for different quantizations

configs = [
    {"name": "FP16", "quant": None},
    {"name": "INT8", "quant": "int8"},
    {"name": "INT4-AWQ", "quant": "awq"},
    {"name": "INT4-GPTQ", "quant": "gptq"},
]

for config in configs:
    llm = LLM(
        "meta-llama/Llama-2-7b-hf",
        quantization=config["quant"]
    )

    # Benchmark
    tokens_generated, time_taken = benchmark_throughput(llm)

    throughput = tokens_generated / time_taken

    print(f"{config['name']:15s}: {throughput:.1f} tokens/s")

# Expected results (A100, batch=32):
# FP16          : 2500 tokens/s (baseline)
# INT8          : 3200 tokens/s (1.3x)
# INT4-AWQ      : 4500 tokens/s (1.8x)
# INT4-GPTQ     : 4300 tokens/s (1.7x)
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Quantization Theory**
- Symmetric vs asymmetric quantization
- Per-tensor vs per-channel methods
- Error analysis and SNR

✅ **Advanced Methods**
- GPTQ algorithm and implementation
- AWQ activation-aware quantization
- SmoothQuant for outliers

✅ **Kernel Implementation**
- INT4 weight packing
- Fused dequantization + GEMM
- INT8 KV cache quantization
- FP8 support

✅ **Performance Analysis**
- Accuracy vs speed trade-offs
- Memory savings analysis
- Real-world benchmarks

### Knowledge Check

**Question 1**: What is the difference between GPTQ and AWQ?
<details>
<summary>Answer</summary>
GPTQ uses optimal brain quantization with Hessian information to minimize reconstruction error, quantizing layer-by-layer. AWQ focuses on protecting weights associated with salient (high-magnitude) activations by scaling channels before quantization. AWQ generally provides better accuracy, especially at 4-bit.
</details>

**Question 2**: Why fuse dequantization with GEMM?
<details>
<summary>Answer</summary>
Fusing dequantization with GEMM avoids materializing full FP16 weight matrix in memory. Instead, weights are dequantized on-the-fly in registers/shared memory during computation, saving memory bandwidth and improving performance.
</details>

**Question 3**: What are the trade-offs of INT8 KV cache?
<details>
<summary>Answer</summary>
Benefits: 2x memory reduction, enabling larger batch sizes or longer contexts. Trade-offs: Small accuracy degradation (~1-2% perplexity increase), additional dequantization compute (usually minor), per-token scale storage overhead.
</details>

### Daily Reflection

**What went well?**
- [ ] I understand quantization fundamentals
- [ ] I can implement quantization kernels
- [ ] I know trade-offs of different methods
- [ ] I can analyze accuracy vs performance

**What was challenging?**
- [ ] INT4 packing/unpacking logic
- [ ] Tensor core programming
- [ ] FP8 format details
- [ ] Accuracy analysis methodology

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 13

Tomorrow you'll explore custom operators and kernel fusion:
- **Custom Ops**: Building PyTorch extensions
- **Kernel Fusion**: Combining operations
- **Triton**: Python-based GPU programming
- **Performance Tuning**: Advanced optimization

**Preparation**:
- Review PyTorch extension documentation
- Read about Triton language
- Review kernel fusion concepts

---

## 📚 Additional Resources

**Module Materials**:
- [ ] `modules/module_03_cuda_kernels/05_kernel_fusion_techniques.md`

**vLLM Files to Study**:
- [ ] `csrc/quantization/gptq/gptq_gemm.cu`
- [ ] `csrc/quantization/awq/awq_gemm.cu`
- [ ] `vllm/model_executor/layers/quantization/`

**Papers**:
- [ ] "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- [ ] "AWQ: Activation-aware Weight Quantization for LLM Compression"
- [ ] "SmoothQuant: Accurate and Efficient Post-Training Quantization"
- [ ] "FP8 Formats for Deep Learning"

---

**Day 12 Complete! Quantization expertise achieved! ⚡**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
