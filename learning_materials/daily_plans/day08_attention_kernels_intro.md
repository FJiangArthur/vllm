# Day 8: Attention Kernels Introduction - CUDA Fundamentals

> **Goal**: Understand vLLM's attention kernel architecture, CUDA memory hierarchy in context, and basic kernel patterns
> **Time**: 6-8 hours
> **Prerequisites**: Week 1 completed, basic CUDA knowledge, understanding of PagedAttention algorithm
> **Deliverables**: Annotated kernel code, memory access pattern diagrams, simple CUDA kernel implementation

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): CUDA Fundamentals Review

**9:00-9:45** - CUDA Memory Hierarchy in vLLM Context
**9:45-10:45** - Attention Kernel Architecture Overview
**10:45-11:00** - Break
**11:00-12:30** - PagedAttention Kernel V1 Walkthrough

### Afternoon Session (3-4 hours): Deep Dive & Practice

**14:00-15:00** - Kernel Launch Configuration Analysis
**15:00-16:00** - Memory Access Pattern Tracing
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Implement Simple Attention Kernel

### Evening (Optional, 1-2 hours): Profiling

**19:00-21:00** - Profile attention kernels with Nsight Compute, analyze metrics

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain CUDA memory hierarchy and how vLLM leverages it
- [ ] Navigate vLLM's attention kernel codebase (csrc/attention/)
- [ ] Understand PagedAttention kernel V1 implementation
- [ ] Analyze kernel launch configurations and thread mapping
- [ ] Trace memory access patterns in attention kernels
- [ ] Profile attention kernels and interpret basic metrics

---

## 📚 Morning: CUDA Fundamentals & Architecture (9:00-12:30)

### Task 1: CUDA Memory Hierarchy Review (45 min)

**Memory Types in CUDA**:

```cuda
// 1. Global Memory (Device Memory)
// - Largest, slowest (~400 cycles latency)
// - Accessible by all threads across all blocks
// - Persists across kernel launches
__global__ void kernel() {
    extern __shared__ float data[];  // Declared at kernel level
}

// 2. Shared Memory
// - Per-block, fast (~20 cycles)
// - Shared among threads in same block
// - Limited size (48KB - 96KB per block)
__shared__ float shared_data[256];

// 3. Registers
// - Per-thread, fastest (~1 cycle)
// - Limited (64K registers per SM)
float thread_local_var;  // Likely in register

// 4. Constant Memory
// - Read-only, cached
// - Broadcast efficiency for same address
__constant__ float const_data[1024];

// 5. Texture/Surface Memory
// - Read-only, cached
// - Optimized for 2D spatial locality
```

**Memory Hierarchy Performance**:

```
Type          | Latency    | Bandwidth      | Size
--------------|------------|----------------|-------------
Registers     | 1 cycle    | ~20 TB/s       | 256 KB/SM
Shared Memory | ~20 cycles | ~15 TB/s       | 48-164 KB/SM
L1 Cache      | ~28 cycles | ~12 TB/s       | 128 KB/SM
L2 Cache      | ~200 cycles| ~3 TB/s        | 40-80 MB
Global Memory | ~400 cycles| 1-2 TB/s       | 40-80 GB

Key insight: Shared memory is ~20x faster than global memory!
```

**🔍 vLLM Memory Usage Patterns**:

Study: `learning_materials/modules/module_03_cuda_kernels/01_cuda_memory_hierarchy.md`

```cuda
// PagedAttention typical pattern:
// 1. Load Q from global → shared (coalesced)
// 2. Loop over KV blocks:
//    a. Load K block from global → shared
//    b. Compute Q @ K^T using shared memory
//    c. Accumulate attention scores
// 3. Load V blocks and compute output
// 4. Write output to global memory

// This minimizes global memory accesses!
```

### Task 2: Attention Kernel Architecture (60 min)

**Directory Structure**:

```bash
cd ~/projects/vllm/csrc/attention/

# List files
ls -la

# Expected structure:
# attention_kernels.cu           ← Main kernel implementations
# attention_generic.cuh          ← Template-based generic implementation
# attention_dtypes.h             ← Data type definitions
# dtype_float16.cuh              ← FP16 specializations
# dtype_bfloat16.cuh             ← BF16 specializations
# dtype_fp8.cuh                  ← FP8 specializations (newer)
# attention_utils.cuh            ← Helper functions
```

**Files to Study**:

1. **`csrc/attention/attention_kernels.cu`** - Main entry points
2. **`csrc/attention/attention_generic.cuh`** - Core implementation
3. **`csrc/attention/dtype_float16.cuh`** - Data type handling
4. **`csrc/attention/attention_utils.cuh`** - Utilities

**📝 Exercise: Create Kernel Inventory**

```bash
# Find all kernel entry points
grep -n "__global__" csrc/attention/attention_kernels.cu

# Find all template instantiations
grep -n "template<" csrc/attention/attention_generic.cuh
```

**Expected Kernels**:

```
1. paged_attention_v1_kernel
   - Basic PagedAttention implementation
   - Used for most models

2. paged_attention_v2_kernel
   - Optimized version with better memory patterns
   - Uses different block processing strategy

3. Single-query attention kernels
   - Optimized for batch_size=1 decode
   - Different thread mapping

4. Flash attention kernels (if available)
   - Fused online softmax
   - Reduced memory footprint
```

### Task 3: PagedAttention Kernel V1 Walkthrough (90 min)

**File**: `csrc/attention/attention_kernels.cu`

**Kernel Signature**:

```cuda
template<
  typename scalar_t,      // float16, bfloat16, float32
  int HEAD_SIZE,          // 64, 80, 96, 128, 256
  int BLOCK_SIZE,         // KV cache block size (16)
  int NUM_THREADS,        // Threads per block
  int PARTITION_SIZE      // For large context handling
>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,              // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,    // [num_seqs]
    const int max_num_blocks_per_seq,
    const float scale,                       // 1/sqrt(head_size)
    const float alibi_slope                  // For ALiBi positional encoding
) {
    // Kernel implementation
}
```

**🔍 Understanding Thread Organization**:

```cuda
// Thread mapping:
// - One thread block per (sequence, head) pair
// - blockIdx.x = sequence index
// - blockIdx.y = head index
// - Each thread block processes one attention output

const int seq_idx = blockIdx.x;
const int head_idx = blockIdx.y;
const int num_heads = gridDim.y;

// Thread ID within block
const int thread_id = threadIdx.x;

// Load query vector for this head (shared by all threads in block)
const scalar_t* q_ptr = q + seq_idx * num_heads * HEAD_SIZE
                          + head_idx * HEAD_SIZE;
```

**📊 Memory Layout Visualization**:

```
Query (Q): [num_seqs, num_heads, head_size]
  Seq 0, Head 0: [q0, q1, q2, ..., q127]  ← One thread block loads this
  Seq 0, Head 1: [q0, q1, q2, ..., q127]
  Seq 1, Head 0: [q0, q1, q2, ..., q127]
  ...

K Cache (paged): [num_blocks, num_kv_heads, head_size/x, block_size, x]
  Block 0: [16 tokens worth of keys]
  Block 1: [16 tokens worth of keys]
  ...

Block Table: [num_seqs, max_num_blocks_per_seq]
  Seq 0: [5, 12, 7, 3, ...]  ← Physical block IDs
  Seq 1: [2, 8, 15, ...]
  ...

The block table provides indirection:
  Logical token 0-15   → Physical block 5
  Logical token 16-31  → Physical block 12
  Logical token 32-47  → Physical block 7
```

**🧩 Kernel Algorithm - High Level**:

```cuda
__global__ void paged_attention_v1_kernel(...) {
    // 1. SETUP: Identify work unit
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int thread_id = threadIdx.x;

    // 2. LOAD QUERY: Cooperatively load Q vector into shared memory
    __shared__ scalar_t q_smem[HEAD_SIZE];
    if (thread_id < HEAD_SIZE) {
        q_smem[thread_id] = q[seq_idx][head_idx][thread_id];
    }
    __syncthreads();

    // 3. ATTENTION LOOP: Process KV cache blocks
    float max_logit = -FLT_MAX;
    float sum_exp = 0.0f;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 3a. First pass: Compute attention scores, find max, compute softmax
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Get physical block ID from block table
        const int physical_block_id = block_tables[seq_idx][block_idx];

        // Each thread processes subset of tokens in block
        for (int i = thread_id; i < BLOCK_SIZE; i += NUM_THREADS) {
            const int token_idx = block_idx * BLOCK_SIZE + i;
            if (token_idx >= context_len) break;

            // Load key vector from cache
            scalar_t k_vec[HEAD_SIZE];
            load_key(k_vec, k_cache, physical_block_id, head_idx, i);

            // Compute dot product: Q · K
            float logit = 0.0f;
            for (int d = 0; d < HEAD_SIZE; ++d) {
                logit += q_smem[d] * k_vec[d];
            }
            logit *= scale;  // Scale by 1/sqrt(head_size)

            // Update max (for numerical stability)
            max_logit = max(max_logit, logit);

            // Store logit for later
            logits[token_idx] = logit;
        }
    }

    // 3b. Reduction: Find global max across threads
    __shared__ float shared_max[NUM_THREADS];
    shared_max[thread_id] = max_logit;
    __syncthreads();

    // Tree reduction to find max
    for (int stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared_max[thread_id] = max(shared_max[thread_id],
                                       shared_max[thread_id + stride]);
        }
        __syncthreads();
    }
    max_logit = shared_max[0];

    // 3c. Compute exp and sum
    sum_exp = 0.0f;
    for (int i = thread_id; i < context_len; i += NUM_THREADS) {
        float exp_val = expf(logits[i] - max_logit);
        logits[i] = exp_val;
        sum_exp += exp_val;
    }

    // Reduction: Sum across threads
    __shared__ float shared_sum[NUM_THREADS];
    shared_sum[thread_id] = sum_exp;
    __syncthreads();

    for (int stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_sum[0];

    // 4. OUTPUT COMPUTATION: Weighted sum of values
    scalar_t output[HEAD_SIZE] = {0};

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_id = block_tables[seq_idx][block_idx];

        for (int i = thread_id; i < BLOCK_SIZE; i += NUM_THREADS) {
            const int token_idx = block_idx * BLOCK_SIZE + i;
            if (token_idx >= context_len) break;

            // Load value vector
            scalar_t v_vec[HEAD_SIZE];
            load_value(v_vec, v_cache, physical_block_id, head_idx, i);

            // Attention weight (normalized)
            float attn_weight = logits[token_idx] / sum_exp;

            // Accumulate: output += attn_weight * V
            for (int d = 0; d < HEAD_SIZE; ++d) {
                output[d] += attn_weight * v_vec[d];
            }
        }
    }

    // 5. REDUCTION: Sum outputs across threads
    // (implementation detail varies)

    // 6. WRITE OUTPUT
    if (thread_id == 0) {
        for (int d = 0; d < HEAD_SIZE; ++d) {
            out[seq_idx][head_idx][d] = output[d];
        }
    }
}
```

**🎯 Key Takeaways**:

1. **One thread block per (sequence, head)**: Natural parallelism mapping
2. **Shared memory for Q**: Avoid redundant global memory loads
3. **Online algorithm**: Process KV cache in blocks, don't materialize full attention matrix
4. **Two-pass softmax**: First pass for max/sum, second for output (numerically stable)
5. **Block table indirection**: Enables paged memory access

---

## 🔬 Afternoon: Analysis & Practice (14:00-18:00)

### Task 4: Kernel Launch Configuration (60 min)

**File**: `vllm/attention/ops/paged_attn.py` (Python side)

Find kernel launch code:

```bash
grep -n "paged_attention_v1" vllm/attention/ops/paged_attn.py -A 20
```

**Launch Configuration Example**:

```python
# In vllm/attention/ops/paged_attn.py

def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> None:
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]

    # Grid configuration:
    # - One block per (sequence, head) pair
    grid = (num_seqs, num_heads)

    # Block configuration:
    # - Typically 128 or 256 threads per block
    block = (128,)  # NUM_THREADS

    # Launch kernel
    if head_size == 64:
        paged_attention_v1_kernel_64[grid, block](...)
    elif head_size == 128:
        paged_attention_v1_kernel_128[grid, block](...)
    # ... other head sizes
```

**📊 Resource Calculation**:

```python
# Example scenario:
num_seqs = 32        # Batch size
num_heads = 32       # Attention heads
head_size = 128      # Head dimension
num_threads = 128    # Threads per block

# Total thread blocks = num_seqs × num_heads
total_blocks = 32 * 32 = 1024

# GPU: A100 (108 SMs)
# Concurrent blocks per SM: Limited by shared memory usage

# Shared memory per block (estimate):
# - Q vector: 128 * 2 bytes (fp16) = 256 bytes
# - Reduction buffers: ~1 KB
# - Logits storage: depends on context length
# Total: ~10-20 KB per block

# A100 shared memory: 164 KB per SM
# Max blocks per SM: 164 / 20 = ~8 blocks

# Total concurrent blocks: 108 * 8 = 864 blocks
# Our 1024 blocks: Run in 2 waves

# This is good! Nearly full GPU utilization
```

### Task 5: Memory Access Pattern Analysis (60 min)

**🔍 Trace a Single Thread's Memory Accesses**:

```
Scenario:
  - Thread 0 in block (seq=0, head=0)
  - Context length: 48 tokens (3 blocks)
  - Block table: [5, 12, 7]

Memory accesses in first K loop iteration:

1. Load Q (shared by all threads):
   Address: q[0][0][0:128]
   Pattern: Coalesced (consecutive threads load consecutive elements)
   Location: Global → Shared

2. Load K for token 0 (thread 0):
   Physical block: block_tables[0][0] = 5
   Address: k_cache[5][0][0:128][0]
   Pattern: Strided (head_size elements)
   Location: Global → Register

3. Compute Q·K: Register operations (fast!)

4. Load K for token 16 (thread 0 in next block):
   Physical block: block_tables[0][1] = 12
   Address: k_cache[12][0][0:128][0]
   Pattern: Non-contiguous (different physical block)
   Location: Global → Register

Key insight: Block table causes irregular memory access!
But this is necessary for paged memory flexibility.
```

**Memory Coalescing Analysis**:

Study: `learning_materials/modules/module_03_cuda_kernels/06_memory_coalescing.md`

```cuda
// Good: Coalesced access (consecutive threads access consecutive elements)
// Loading Q into shared memory
if (thread_id < HEAD_SIZE) {
    q_smem[thread_id] = q[seq_idx * num_heads * HEAD_SIZE +
                          head_idx * HEAD_SIZE +
                          thread_id];  // ← thread_id consecutive
}

// Potentially uncoalesced: Loading K from paged cache
// Threads may access different physical blocks
// But within a block, access pattern is regular
```

### Task 6: Hands-On Implementation (90 min)

**Exercise 1**: Implement Simplified Attention Kernel

Create: `~/projects/vllm/my_kernels/simple_attention.cu`

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Simplified non-paged attention kernel for learning
template<int HEAD_SIZE, int BLOCK_SIZE>
__global__ void simple_attention_kernel(
    float* __restrict__ out,           // [num_seqs, num_heads, HEAD_SIZE]
    const float* __restrict__ q,       // [num_seqs, num_heads, HEAD_SIZE]
    const float* __restrict__ k,       // [num_seqs, num_heads, seq_len, HEAD_SIZE]
    const float* __restrict__ v,       // [num_seqs, num_heads, seq_len, HEAD_SIZE]
    const int seq_len,
    const float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Shared memory for query
    __shared__ float q_smem[HEAD_SIZE];

    // Load query
    if (tid < HEAD_SIZE) {
        const int q_offset = (seq_idx * gridDim.y + head_idx) * HEAD_SIZE;
        q_smem[tid] = q[q_offset + tid];
    }
    __syncthreads();

    // Compute attention scores
    extern __shared__ float logits[];  // Dynamic shared memory for logits

    float max_logit = -INFINITY;

    // Each thread processes multiple tokens
    for (int token = tid; token < seq_len; token += blockDim.x) {
        float score = 0.0f;

        const int k_offset = ((seq_idx * gridDim.y + head_idx) * seq_len + token) * HEAD_SIZE;

        // Compute Q·K for this token
        for (int d = 0; d < HEAD_SIZE; d++) {
            score += q_smem[d] * k[k_offset + d];
        }
        score *= scale;

        logits[token] = score;
        max_logit = fmaxf(max_logit, score);
    }

    // TODO: Add reduction for max, softmax computation, and output
    // This is Exercise 2!
}

// Host function to launch kernel
void simple_attention(
    float* out,
    const float* q,
    const float* k,
    const float* v,
    int num_seqs,
    int num_heads,
    int seq_len,
    int head_size
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(128);

    float scale = 1.0f / sqrtf(head_size);

    // Calculate shared memory size
    size_t smem_size = seq_len * sizeof(float);  // For logits

    if (head_size == 128) {
        simple_attention_kernel<128, 128><<<grid, block, smem_size>>>(
            out, q, k, v, seq_len, scale
        );
    }
    // Add other head sizes...

    cudaDeviceSynchronize();
}
```

**Exercise 2**: Complete the kernel implementation

Add:
1. Reduction to find max across threads
2. Softmax computation (exp and sum)
3. Output computation (weighted sum of V)
4. Reduction to combine thread outputs

**Exercise 3**: Build and test

```bash
# Compile
nvcc -o simple_attention simple_attention.cu -lcuda

# Create test harness in Python
# Use PyTorch to verify correctness
```

---

## 📊 Profiling Session (Evening, 19:00-21:00)

### Task 7: Profile Real vLLM Kernels

**Setup Profiling**:

```bash
# Profile a simple inference
nsys profile -o attention_profile \
    --trace=cuda,nvtx \
    python examples/offline_inference.py

# View in Nsight Systems GUI
nsys-ui attention_profile.qdrep
```

**Focus Areas**:

1. **Find attention kernels**: Look for `paged_attention_v1_kernel` in timeline
2. **Measure duration**: How long does each kernel take?
3. **Check occupancy**: Are SMs fully utilized?
4. **Memory throughput**: Compare to theoretical max

**Detailed Kernel Analysis**:

```bash
# Profile with Nsight Compute (detailed metrics)
ncu --set full -o attention_detailed \
    python examples/offline_inference.py

# View report
ncu-ui attention_detailed.ncu-rep
```

**Metrics to Check**:

```
1. SM Efficiency: Should be > 60%
2. Memory Throughput: Compare to peak bandwidth
3. Warp Execution Efficiency: Check for divergence
4. Shared Memory Usage: How much is used?
5. Register Usage: Affecting occupancy?
6. Cache Hit Rates: L1, L2 performance
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **CUDA Memory Hierarchy**
- Global, shared, register memory characteristics
- Performance implications of memory choices
- vLLM's memory usage patterns

✅ **Attention Kernel Architecture**
- Directory structure and file organization
- PagedAttention V1 kernel walkthrough
- Thread organization and mapping

✅ **Memory Access Patterns**
- Coalescing requirements
- Impact of paged memory layout
- Optimization strategies

✅ **Profiling Basics**
- Using Nsight Systems and Compute
- Key metrics to track
- Identifying bottlenecks

### Knowledge Check

**Question 1**: Why does vLLM load the query vector into shared memory?
<details>
<summary>Answer</summary>
All threads in the block need to access the same query vector multiple times (once per token in the context). Loading into shared memory (~20x faster than global) and sharing among threads is much more efficient than each thread loading from global memory.
</details>

**Question 2**: What is the grid configuration for PagedAttention kernels and why?
<details>
<summary>Answer</summary>
Grid = (num_seqs, num_heads), meaning one thread block per (sequence, head) pair. This naturally parallelizes the computation since each attention head output is independent and can be computed in parallel.
</details>

**Question 3**: How does the block table affect memory access patterns?
<details>
<summary>Answer</summary>
The block table causes potentially non-contiguous memory accesses since consecutive logical tokens may be stored in non-consecutive physical blocks. This reduces memory access efficiency but is necessary for the flexibility of paged memory management.
</details>

**Question 4**: What are the two main phases of the PagedAttention kernel?
<details>
<summary>Answer</summary>
1. Compute attention scores (Q·K), apply softmax (with numerical stability via max subtraction)
2. Compute output as weighted sum of values (attention_weights · V)
</details>

### Daily Reflection

**What went well?**
- [ ] I understand CUDA memory hierarchy
- [ ] I can navigate attention kernel code
- [ ] I understand the kernel algorithm
- [ ] I can profile kernels

**What was challenging?**
- [ ] Complex template code
- [ ] Memory layout understanding
- [ ] Profiling tool usage
- [ ] CUDA programming concepts

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 9

Tomorrow you'll dive into advanced attention kernel optimizations:
- **PagedAttention V2**: Improved memory access patterns
- **Flash Attention**: Fused online softmax, reduced memory
- **Multi-Query/Grouped-Query Attention**: Specialized kernels
- **Performance Optimization**: Techniques and trade-offs

**Preparation**:
- Review today's kernel walkthrough notes
- Read `learning_materials/modules/module_03_cuda_kernels/04_flash_attention_explained.md`
- Ensure profiling tools are working well

---

## 📚 Additional Resources

**Module Materials**:
- [ ] `modules/module_03_cuda_kernels/01_cuda_memory_hierarchy.md`
- [ ] `modules/module_03_cuda_kernels/02_kernel_optimization_basics.md`
- [ ] `modules/module_03_cuda_kernels/03_attention_kernel_walkthrough.md`

**vLLM Files to Study**:
- [ ] `csrc/attention/attention_kernels.cu`
- [ ] `csrc/attention/attention_generic.cuh`
- [ ] `vllm/attention/ops/paged_attn.py`

**External Resources**:
- [ ] NVIDIA CUDA C Programming Guide - Memory Hierarchy
- [ ] "Efficient Attention" blog posts
- [ ] FlashAttention paper (for tomorrow)

---

**Day 8 Complete! You're now ready for advanced kernel optimizations! 🚀**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
