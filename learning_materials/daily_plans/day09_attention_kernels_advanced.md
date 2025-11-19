# Day 9: Advanced Attention Kernels - Optimizations & Variants

> **Goal**: Master advanced attention kernel techniques: PagedAttention V2, Flash Attention, GQA optimizations
> **Time**: 6-8 hours
> **Prerequisites**: Day 8 completed, understanding of basic attention kernels
> **Deliverables**: Comparative analysis document, optimization annotations, Flash Attention implementation

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Advanced Kernel Variants

**9:00-10:00** - PagedAttention V2 Deep Dive
**10:00-11:00** - Flash Attention Algorithm & Implementation
**11:00-11:15** - Break
**11:15-12:30** - Multi-Query & Grouped-Query Attention

### Afternoon Session (3-4 hours): Optimization Techniques

**14:00-15:00** - Kernel Fusion Patterns in vLLM
**15:00-16:00** - Shared Memory Optimization Techniques
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Implement Flash Attention Core

### Evening (Optional, 1-2 hours): Performance Analysis

**19:00-21:00** - Benchmark different attention variants, comparative profiling

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain differences between PagedAttention V1 and V2
- [ ] Understand Flash Attention algorithm and memory benefits
- [ ] Implement online softmax computation
- [ ] Optimize kernels using shared memory and tiling
- [ ] Handle Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
- [ ] Apply kernel fusion techniques
- [ ] Compare performance characteristics of different attention variants

---

## 📚 Morning: Advanced Kernel Variants (9:00-12:30)

### Task 1: PagedAttention V2 Analysis (60 min)

**What's Different in V2?**

Study file: `csrc/attention/attention_kernels.cu` - find `paged_attention_v2_kernel`

**Key Improvements**:

```
V1 vs V2 Comparison:

PagedAttention V1:
  ✓ Single-pass over KV blocks
  ✓ Each thread block processes one (seq, head)
  ✗ All threads process same tokens (redundant work)
  ✗ Limited by shared memory for long sequences

PagedAttention V2:
  ✓ Two-stage computation for long sequences
  ✓ Parallel processing of token chunks
  ✓ Better load balancing across threads
  ✓ Reduced redundant computation
  ✗ Slightly more complex implementation
```

**V2 Algorithm Overview**:

```cuda
// V2 uses partitioning for long contexts
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v2_kernel(
    scalar_t* __restrict__ out,
    float* __restrict__ exp_sums,        // ← New: partial sums
    float* __restrict__ max_logits,      // ← New: partial maxes
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int max_num_blocks_per_seq,
    const float scale,
    const int partition_size              // ← New: partition size
) {
    // Step 1: Partition the context into chunks
    const int seq_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    const int context_len = context_lens[seq_idx];
    const int partition_start = partition_idx * partition_size;
    const int partition_end = min(partition_start + partition_size, context_len);

    if (partition_start >= context_len) return;

    // Step 2: Process this partition (similar to V1)
    // Compute partial results: max, sum, weighted output

    float max_logit = -FLT_MAX;
    float sum_exp = 0.0f;
    scalar_t partial_out[HEAD_SIZE] = {0};

    // ... compute attention for tokens in [partition_start, partition_end)

    // Step 3: Store partial results
    max_logits[...] = max_logit;
    exp_sums[...] = sum_exp;
    // partial outputs stored in global memory

    __syncthreads();

    // Step 4: Final reduction (in separate kernel launch or at end)
    // Combine partitions using stable softmax formula
}

// Reduction kernel to combine partitions
__global__ void paged_attention_v2_reduce_kernel(...) {
    // Combine partial results from all partitions
    // Use numerically stable softmax combination
}
```

**Why Partitioning Helps**:

```
Long sequence (2048 tokens):

V1 approach:
  - One block processes all 2048 tokens
  - Limited parallelism within block
  - Memory constraints for storing logits

V2 approach:
  - Split into 4 partitions of 512 tokens
  - 4 blocks process in parallel
  - Each needs less shared memory
  - Combine results at end

Speedup: ~1.5-2x for long contexts (>1024 tokens)
```

**📝 Exercise: Trace V2 Execution**

```
Scenario:
  - Context length: 1024 tokens
  - Partition size: 256 tokens
  - Number of partitions: 4

Execution flow:

Launch 1: paged_attention_v2_kernel
  Grid: (num_seqs, 4, num_heads)  ← 4 partitions
  Block (0, 0, 0): Processes tokens 0-255
  Block (0, 1, 0): Processes tokens 256-511
  Block (0, 2, 0): Processes tokens 512-767
  Block (0, 3, 0): Processes tokens 768-1023

  Each block computes:
    - max_logit for its partition
    - sum_exp for its partition
    - partial_output weighted by its attention

Launch 2: paged_attention_v2_reduce_kernel
  Combines 4 partial results using:
    global_max = max(max_0, max_1, max_2, max_3)
    adjusted_sums = sum_i * exp(max_i - global_max)
    final_sum = sum(adjusted_sums)
    final_output = sum(partial_out_i * adjusted_sum_i) / final_sum
```

### Task 2: Flash Attention Deep Dive (60 min)

Study: `learning_materials/modules/module_03_cuda_kernels/04_flash_attention_explained.md`

**Flash Attention Core Innovation**:

```
Standard Attention Problem:
  1. Compute S = Q @ K^T          [N, N] matrix
  2. Compute P = softmax(S)       [N, N] matrix  ← MEMORY BOTTLENECK
  3. Compute O = P @ V            [N, d] output

  Memory: O(N²) for attention matrix
  HBM accesses: Multiple passes over data

Flash Attention Solution:
  1. Tile the computation (block by block)
  2. Never materialize full attention matrix
  3. Compute softmax online (incrementally)
  4. Single pass through data

  Memory: O(N) - only store output and statistics
  HBM accesses: Minimized through tiling
```

**Online Softmax Algorithm**:

```python
# Traditional softmax (requires full vector)
def softmax(x):
    max_x = max(x)
    exp_x = [exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

# Online softmax (can process incrementally)
def online_softmax_update(m_old, d_old, m_new, d_new):
    """
    Update running max and denominator when seeing new data

    Args:
        m_old: previous max
        d_old: previous sum(exp(x - m_old))
        m_new: max of new data
        d_new: sum(exp(x_new - m_new))

    Returns:
        m_global: global max
        d_global: adjusted global denominator
    """
    m_global = max(m_old, m_new)
    d_global = d_old * exp(m_old - m_global) + d_new * exp(m_new - m_global)
    return m_global, d_global

# Apply when combining tiles
for tile in tiles:
    m_tile, d_tile, o_tile = process_tile(tile)
    m, d = online_softmax_update(m, d, m_tile, d_tile)
    # Adjust output contribution
    o = o * exp(m_old - m) + o_tile * exp(m_tile - m)
```

**Flash Attention Kernel Pseudocode**:

```cuda
__global__ void flash_attention_kernel(
    float* O,                // Output [N, d]
    const float* Q,          // Query [N, d]
    const float* K,          // Key [N, d]
    const float* V,          // Value [N, d]
    const int N,             // Sequence length
    const int d,             // Head dimension
    const int Bc,            // Block size for K, V
    const int Br             // Block size for Q
) {
    // Shared memory for tiles
    __shared__ float Q_tile[Br][d];
    __shared__ float K_tile[Bc][d];
    __shared__ float V_tile[Bc][d];
    __shared__ float S_tile[Br][Bc];  // Attention scores tile

    // Each block processes Br query rows
    const int block_start = blockIdx.x * Br;

    // Initialize output and statistics
    float O_local[Br][d] = {0};
    float m[Br] = {-INFINITY};  // Running max
    float l[Br] = {0};           // Running sum

    // Load Q tile (this block's queries)
    load_tile_Q(Q_tile, Q, block_start, Br, d);

    // Loop over KV tiles
    const int num_tiles = (N + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Load K, V tiles
        load_tile_KV(K_tile, V_tile, K, V, tile_idx * Bc, Bc, d);
        __syncthreads();

        // Compute attention scores: S = Q @ K^T
        compute_tile_scores(S_tile, Q_tile, K_tile, Br, Bc, d);

        // For each query row in this tile
        for (int i = threadIdx.x; i < Br; i += blockDim.x) {
            // Find max in this row of S_tile
            float m_new = -INFINITY;
            for (int j = 0; j < Bc; ++j) {
                m_new = fmax(m_new, S_tile[i][j]);
            }

            // Update global max
            float m_old = m[i];
            m[i] = fmax(m_old, m_new);

            // Compute exp(S - m) and sum
            float l_new = 0;
            for (int j = 0; j < Bc; ++j) {
                S_tile[i][j] = exp(S_tile[i][j] - m[i]);
                l_new += S_tile[i][j];
            }

            // Update denominator with correction
            float correction = exp(m_old - m[i]);
            l[i] = l[i] * correction + l_new;

            // Update output: O = correction * O_old + S_tile @ V_tile
            for (int d_idx = 0; d_idx < d; ++d_idx) {
                O_local[i][d_idx] *= correction;
                for (int j = 0; j < Bc; ++j) {
                    O_local[i][d_idx] += S_tile[i][j] * V_tile[j][d_idx];
                }
            }
        }
        __syncthreads();
    }

    // Final normalization: O = O / l
    for (int i = threadIdx.x; i < Br; i += blockDim.x) {
        for (int d_idx = 0; d_idx < d; ++d_idx) {
            O_local[i][d_idx] /= l[i];
        }
    }

    // Write output
    write_output(O, O_local, block_start, Br, d);
}
```

**Key Advantages**:

```
Memory Usage:
  Standard: O(N²) - full attention matrix
  Flash: O(N) - only tiles and output

HBM Accesses (A100):
  Standard: ~3-5 passes over Q, K, V
  Flash: 1 pass over Q, K, V

Speedup:
  Short sequences (N < 1024): 1-2x
  Long sequences (N > 2048): 2-4x
  Very long (N > 8192): 5-10x
```

### Task 3: Multi-Query & Grouped-Query Attention (75 min)

**Multi-Query Attention (MQA)**:

```
Standard Multi-Head Attention:
  Q: [batch, num_heads, seq_len, head_dim]
  K: [batch, num_heads, seq_len, head_dim]
  V: [batch, num_heads, seq_len, head_dim]

  Each head has its own K, V

Multi-Query Attention (MQA):
  Q: [batch, num_heads, seq_len, head_dim]
  K: [batch, 1, seq_len, head_dim]           ← SINGLE K/V
  V: [batch, 1, seq_len, head_dim]

  All heads share K, V

Benefits:
  - Reduced KV cache size (num_heads × smaller)
  - Faster inference (less memory bandwidth)
  - Slight accuracy degradation (~1-2%)

Used in: Falcon, PaLM, StarCoder
```

**Grouped-Query Attention (GQA)**:

```
Grouped-Query Attention (GQA):
  Q: [batch, num_heads, seq_len, head_dim]
  K: [batch, num_kv_heads, seq_len, head_dim]
  V: [batch, num_kv_heads, seq_len, head_dim]

  Where num_kv_heads < num_heads (e.g., 8 vs 32)
  Each group of Q heads shares one KV head

Example: 32 Q heads, 8 KV heads
  - Group size = 32 / 8 = 4
  - Q heads 0-3 use KV head 0
  - Q heads 4-7 use KV head 1
  - ...

Benefits:
  - 4x smaller KV cache than MHA
  - Better quality than MQA
  - Good throughput/quality trade-off

Used in: Llama-2, Mistral, Gemma
```

**vLLM Implementation**:

```cuda
// In vLLM attention kernels, GQA is handled via head mapping

template<int NUM_KV_HEADS>
__global__ void paged_attention_gqa_kernel(...) {
    const int q_head_idx = blockIdx.y;
    const int num_q_heads = gridDim.y;

    // Map Q head to KV head
    const int kv_head_idx = q_head_idx * NUM_KV_HEADS / num_q_heads;

    // Load Q for this head
    const float* q_ptr = q + q_head_idx * HEAD_SIZE;

    // Load K, V from mapped KV head
    const float* k_ptr = k_cache + kv_head_idx * ...;
    const float* v_ptr = v_cache + kv_head_idx * ...;

    // Rest of attention computation...
}
```

**📊 Memory Savings Comparison**:

```python
# Model: Llama-2-7B
num_layers = 32
num_q_heads = 32
num_kv_heads = 8  # GQA
head_dim = 128
seq_len = 2048
dtype = 2  # FP16 bytes

# Multi-Head Attention (MHA)
kv_cache_mha = (
    2 * num_layers * num_q_heads * seq_len * head_dim * dtype
)
# = 2 * 32 * 32 * 2048 * 128 * 2 = 1,073,741,824 bytes = 1 GB per sequence

# Grouped-Query Attention (GQA)
kv_cache_gqa = (
    2 * num_layers * num_kv_heads * seq_len * head_dim * dtype
)
# = 2 * 32 * 8 * 2048 * 128 * 2 = 268,435,456 bytes = 256 MB per sequence

# Savings: 4x reduction (32/8 = 4)
```

---

## 🔬 Afternoon: Optimization Techniques (14:00-18:00)

### Task 4: Kernel Fusion Patterns (60 min)

Study: `learning_materials/modules/module_03_cuda_kernels/05_kernel_fusion_techniques.md`

**What is Kernel Fusion?**

```
Without fusion (3 kernel launches):
  Kernel 1: LayerNorm          (read X, write Y)
  Kernel 2: Attention          (read Y, write Z)
  Kernel 3: Add residual       (read X, Z, write W)

  Memory traffic: 3 reads of X, 2 reads of Y, ...
  Kernel launch overhead: 3× (~5-10 μs each)

With fusion (1 kernel launch):
  Fused kernel: LayerNorm + Attention + Residual

  Memory traffic: 1 read of X, intermediate in registers
  Kernel launch overhead: 1×

Speedup: 1.5-3x depending on operations
```

**Examples in vLLM**:

```cuda
// Example 1: RoPE (Rotary Position Embedding) + Attention
// Fused in some backends

__global__ void rope_fused_attention_kernel(
    float* out,
    const float* q,
    const float* k_cache,
    const float* cos_sin_cache,  // RoPE rotation matrix
    ...
) {
    // 1. Apply RoPE to Q (in registers, not written to global mem)
    float q_rotated[HEAD_SIZE];
    apply_rope_inline(q_rotated, q, cos_sin_cache);

    // 2. Compute attention using rotated Q
    // (no intermediate write of q_rotated!)
    compute_attention(out, q_rotated, k_cache, ...);
}

// Benefit: Saved one global memory write + read of Q
```

**Fusion Opportunities**:

```
vLLM fusion targets:

1. Quantization + MatMul
   - Dequantize weights on-the-fly during GEMM
   - File: csrc/quantization/gptq/gptq_gemm.cu

2. Activation + Bias
   - Apply activation (GELU, SiLU) immediately after adding bias
   - File: csrc/activation_kernels.cu

3. Softmax + Sampling
   - Compute softmax and sample token in one kernel
   - Reduces sampling latency

4. Attention + Output Projection
   - Compute attention and immediately apply output projection
   - Experimental in some backends
```

### Task 5: Shared Memory Optimization (60 min)

Study: `learning_materials/modules/module_03_cuda_kernels/07_shared_memory_optimization.md`

**Shared Memory Bank Conflicts**:

```cuda
// A100 shared memory: 32 banks, 4 bytes per bank

// BAD: Bank conflicts
__shared__ float data[32][32];

// Thread 0 accesses data[0][0]   → Bank 0
// Thread 1 accesses data[1][0]   → Bank 0  ← CONFLICT!
// Thread 2 accesses data[2][0]   → Bank 0  ← CONFLICT!
// All threads access same bank!

// GOOD: Pad to avoid conflicts
__shared__ float data[32][33];  // +1 padding

// Thread 0 accesses data[0][0]   → Bank 0
// Thread 1 accesses data[1][0]   → Bank 1  ✓
// Thread 2 accesses data[2][0]   → Bank 2  ✓
// No conflicts!

// Speedup: 2-5x for memory-intensive kernels
```

**Optimizing Tile Size**:

```cuda
// Trade-off: Larger tiles vs. occupancy

// Option 1: Large tiles (better reuse)
constexpr int TILE_SIZE = 64;
__shared__ float tile[TILE_SIZE][TILE_SIZE];  // 64*64*4 = 16 KB

// A100: 164 KB shared memory per SM
// Max blocks per SM: 164 / 16 = 10 blocks
// With 1024 max threads per SM: 10 blocks * 256 threads = 2560 threads
// Occupancy: 2560 / 2048 (max) = 125% ✓ Good!

// Option 2: Small tiles (better occupancy)
constexpr int TILE_SIZE = 32;
__shared__ float tile[TILE_SIZE][TILE_SIZE];  // 32*32*4 = 4 KB

// Max blocks per SM: 164 / 4 = 41 blocks (limited by other factors)
// Likely higher occupancy but less data reuse

// vLLM choice: Medium tiles (16-32) for balance
```

**vLLM Shared Memory Patterns**:

```cuda
// Pattern 1: Cooperative loading
__global__ void attention_kernel(...) {
    __shared__ float q_shared[HEAD_SIZE];

    // All threads cooperatively load Q
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
        q_shared[i] = q_global[i];
    }
    __syncthreads();

    // All threads use Q from shared memory
}

// Pattern 2: Reduction in shared memory
__global__ void reduction_kernel(...) {
    __shared__ float partial_sums[NUM_THREADS];

    // Each thread computes partial result
    partial_sums[threadIdx.x] = compute_partial();
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 has final result
}

// Pattern 3: Double buffering (advanced)
__global__ void pipelined_kernel(...) {
    __shared__ float buffer_a[TILE_SIZE];
    __shared__ float buffer_b[TILE_SIZE];

    load_async(buffer_a, ...);  // Load first tile

    for (int i = 0; i < num_tiles - 1; ++i) {
        load_async(buffer_b, ...);  // Load next while processing current
        __syncthreads();
        process(buffer_a);          // Process current
        __syncthreads();
        swap(buffer_a, buffer_b);   // Swap buffers
    }

    process(buffer_a);  // Process last tile
}
```

### Task 6: Implement Flash Attention Core (90 min)

**Exercise**: Implement simplified Flash Attention

Create: `~/projects/vllm/my_kernels/flash_attention_simple.cu`

```cuda
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

// Simplified Flash Attention (single tile for learning)
template<int HEAD_SIZE, int TILE_SIZE>
__global__ void flash_attention_simple_kernel(
    float* __restrict__ out,              // [batch, heads, seq_len, head_size]
    const float* __restrict__ q,          // [batch, heads, seq_len, head_size]
    const float* __restrict__ k,          // [batch, heads, seq_len, head_size]
    const float* __restrict__ v,          // [batch, heads, seq_len, head_size]
    const int seq_len,
    const float scale
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (query_idx >= seq_len) return;

    // Shared memory for K, V tiles
    __shared__ float k_tile[TILE_SIZE][HEAD_SIZE];
    __shared__ float v_tile[TILE_SIZE][HEAD_SIZE];

    // Load query into registers
    float q_vec[HEAD_SIZE];
    const int q_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + query_idx) * HEAD_SIZE;
    for (int d = 0; d < HEAD_SIZE; d++) {
        q_vec[d] = q[q_offset + d];
    }

    // Initialize running statistics
    float m = -FLT_MAX;  // Running max
    float l = 0.0f;      // Running sum
    float output[HEAD_SIZE] = {0.0f};

    const int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    // Process K, V in tiles
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const int tile_start = tile_idx * TILE_SIZE;
        const int tile_size = min(TILE_SIZE, seq_len - tile_start);

        // Cooperatively load K, V tile into shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            const int kv_idx = tile_start + i;
            const int kv_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + kv_idx) * HEAD_SIZE;

            for (int d = 0; d < HEAD_SIZE; d++) {
                k_tile[i][d] = k[kv_offset + d];
                v_tile[i][d] = v[kv_offset + d];
            }
        }
        __syncthreads();

        // Compute attention for this tile
        float scores[TILE_SIZE];
        float m_new = m;

        for (int i = 0; i < tile_size; i++) {
            // Compute Q · K
            float score = 0.0f;
            for (int d = 0; d < HEAD_SIZE; d++) {
                score += q_vec[d] * k_tile[i][d];
            }
            score *= scale;
            scores[i] = score;
            m_new = fmaxf(m_new, score);
        }

        // Update running max and denominator
        float m_old = m;
        m = m_new;

        // Correction factor for previous tiles
        float correction = expf(m_old - m);
        l = l * correction;

        // Update output with correction
        for (int d = 0; d < HEAD_SIZE; d++) {
            output[d] *= correction;
        }

        // Accumulate current tile contribution
        for (int i = 0; i < tile_size; i++) {
            float attn_weight = expf(scores[i] - m);
            l += attn_weight;

            for (int d = 0; d < HEAD_SIZE; d++) {
                output[d] += attn_weight * v_tile[i][d];
            }
        }

        __syncthreads();
    }

    // Normalize output
    for (int d = 0; d < HEAD_SIZE; d++) {
        output[d] /= l;
    }

    // Write output
    const int out_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + query_idx) * HEAD_SIZE;
    for (int d = 0; d < HEAD_SIZE; d++) {
        out[out_offset + d] = output[d];
    }
}

// TODO: Add launcher function, test harness
```

**Testing**:

```python
import torch
import numpy as np

def test_flash_attention():
    batch = 2
    heads = 4
    seq_len = 128
    head_size = 64

    # Create random inputs
    q = torch.randn(batch, heads, seq_len, head_size, device='cuda')
    k = torch.randn(batch, heads, seq_len, head_size, device='cuda')
    v = torch.randn(batch, heads, seq_len, head_size, device='cuda')

    # Reference: PyTorch scaled dot-product attention
    scale = 1.0 / np.sqrt(head_size)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    reference = torch.matmul(attn, v)

    # Your implementation
    output = flash_attention_simple(q, k, v)

    # Compare
    diff = torch.abs(reference - output).max().item()
    print(f"Max difference: {diff}")
    assert diff < 1e-3, "Implementation incorrect!"
    print("✓ Test passed!")

test_flash_attention()
```

---

## 📊 Performance Analysis (Evening, 19:00-21:00)

### Task 7: Benchmark Attention Variants

**Create Benchmark Script**:

```python
#!/usr/bin/env python3
"""
Benchmark different attention implementations
"""

import torch
import time
from vllm.attention.ops import paged_attn

def benchmark_attention_variant(
    variant: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_size: int,
    num_iterations: int = 100
):
    # Setup inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_size,
                    device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_size,
                    device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_size,
                    device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        if variant == "pytorch":
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        elif variant == "paged_v1":
            # Use vLLM paged attention V1
            pass
        # Add other variants

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iterations):
        if variant == "pytorch":
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # ... other variants

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_iterations * 1000  # ms
    return avg_time

# Run benchmarks
configs = [
    {"batch": 32, "heads": 32, "seq": 128, "head_size": 128},
    {"batch": 32, "heads": 32, "seq": 512, "head_size": 128},
    {"batch": 32, "heads": 32, "seq": 2048, "head_size": 128},
]

for config in configs:
    print(f"\n{'='*60}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    for variant in ["pytorch", "paged_v1", "paged_v2", "flash"]:
        try:
            time_ms = benchmark_attention_variant(variant, **config)
            print(f"{variant:15s}: {time_ms:.3f} ms")
        except Exception as e:
            print(f"{variant:15s}: Error - {e}")
```

**Expected Results** (on A100):

```
Config: batch=32, heads=32, seq=128, head_size=128
pytorch        : 0.523 ms
paged_v1       : 0.612 ms  (overhead from paging)
paged_v2       : 0.598 ms
flash          : 0.445 ms  ✓ fastest

Config: batch=32, heads=32, seq=2048, head_size=128
pytorch        : 12.3 ms
paged_v1       : 8.7 ms   ✓ benefits from not materializing matrix
paged_v2       : 7.2 ms   ✓ partitioning helps
flash          : 5.1 ms   ✓ best for long sequences
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **PagedAttention V2**
- Partitioning strategy for long contexts
- Parallel processing improvements
- Reduction and combination of partial results

✅ **Flash Attention**
- Online softmax algorithm
- Tiling and memory optimization
- HBM access minimization

✅ **GQA/MQA**
- Reduced KV cache memory
- Head mapping implementation
- Performance/quality trade-offs

✅ **Optimization Techniques**
- Kernel fusion patterns
- Shared memory optimization
- Bank conflict avoidance

### Knowledge Check

**Question 1**: How does PagedAttention V2 handle long sequences differently than V1?
<details>
<summary>Answer</summary>
V2 partitions long sequences into chunks and processes them in parallel across multiple thread blocks, then combines results. This provides better parallelism and reduces per-block memory requirements compared to V1's single-block approach.
</details>

**Question 2**: What is the key innovation of Flash Attention?
<details>
<summary>Answer</summary>
Flash Attention never materializes the full N×N attention matrix. Instead, it uses tiling and online softmax to compute attention in blocks, dramatically reducing memory usage from O(N²) to O(N) and minimizing HBM accesses.
</details>

**Question 3**: How much KV cache memory does GQA save compared to MHA?
<details>
<summary>Answer</summary>
GQA saves memory proportional to the ratio num_q_heads / num_kv_heads. For example, if you have 32 query heads and 8 KV heads, you save 4x KV cache memory (32/8 = 4).
</details>

### Daily Reflection

**What went well?**
- [ ] I understand advanced attention variants
- [ ] I can implement online softmax
- [ ] I understand kernel fusion benefits
- [ ] I can optimize shared memory usage

**What was challenging?**
- [ ] Flash Attention algorithm complexity
- [ ] Numerical stability in online softmax
- [ ] Shared memory bank conflicts
- [ ] Performance analysis and profiling

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 10

Tomorrow you'll explore memory management kernels:
- **Cache Kernels**: KV cache copy, reshape, swap operations
- **Memory Pools**: GPU/CPU memory management
- **Copy Operations**: Optimized host-device transfers
- **Kernel Optimization**: Memory-bound kernel tuning

**Preparation**:
- Review KV cache concepts from Week 1
- Read `modules/module_04_system_components/06_kv_cache_management.md`
- Understand CUDA memory copy patterns

---

## 📚 Additional Resources

**Module Materials**:
- [ ] `modules/module_03_cuda_kernels/04_flash_attention_explained.md`
- [ ] `modules/module_03_cuda_kernels/05_kernel_fusion_techniques.md`
- [ ] `modules/module_03_cuda_kernels/07_shared_memory_optimization.md`

**vLLM Files to Study**:
- [ ] `csrc/attention/attention_kernels.cu` (V2 implementation)
- [ ] `vllm/attention/backends/flash_attn.py`
- [ ] `csrc/quantization/fused_kernels/` (fusion examples)

**Papers**:
- [ ] FlashAttention: Fast and Memory-Efficient Exact Attention
- [ ] FlashAttention-2: Faster Attention with Better Parallelism
- [ ] GQA: Training Generalized Multi-Query Transformer Models

---

**Day 9 Complete! Advanced attention mastery achieved! 🔥**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
