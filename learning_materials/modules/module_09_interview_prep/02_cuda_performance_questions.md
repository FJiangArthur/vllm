# CUDA & Performance Optimization Interview Questions

## Introduction

This document contains 15 advanced interview questions focused on CUDA programming, GPU optimization, and performance analysis. These questions are commonly asked in technical deep-dive rounds for GPU systems engineer and ML infrastructure roles.

**Target Roles**: GPU Systems Engineer, CUDA Performance Engineer, ML Infrastructure Engineer

**Difficulty Level**: Advanced

**Expected Background**:
- Strong CUDA programming experience
- Understanding of GPU architecture
- Profiling and optimization experience
- Knowledge of attention mechanisms

---

## Question 1: Explain FlashAttention Algorithm in Detail

### Question
"Walk me through the FlashAttention algorithm. How does it achieve memory efficiency?"

### Answer Framework
1. Problem: Standard attention memory bottleneck
2. Core idea: Tiling + recomputation
3. Algorithm steps
4. Memory and compute analysis

### Detailed Answer

**Problem: Standard Attention Memory Bottleneck**
```
Standard Attention:
1. Compute Q @ K^T → Store full attention matrix [N × N]
2. Apply softmax → Still [N × N]
3. Multiply by V → Output [N × D]

Memory: O(N²) for attention matrix
For N=2048, D=128: Attention matrix = 16MB per head
For 32 heads: 512MB just for intermediate attention scores!

Bottleneck: Memory bandwidth (reading/writing large attention matrix)
```

**Core Idea: Tiling + Recomputation**
```
Key insights:
1. Don't materialize full attention matrix
2. Process in tiles that fit in SRAM (on-chip fast memory)
3. Recompute attention scores in backward pass (compute cheaper than memory)
```

**Algorithm (Forward Pass)**:
```
Input: Q, K, V of shape [N, D]
Block size: B (typically 128-256)
Number of blocks: Tc = N/B (columns), Tr = N/B (rows)

For each row block Qi (size [B, D]):
    Initialize Oi = zeros, li = zeros, mi = -inf  # output, norm, max

    For each column block Kj, Vj (size [B, D]):
        1. Load Qi, Kj, Vj from HBM → SRAM

        2. Compute attention scores in SRAM:
           Sij = Qi @ Kj^T  [B × B]

        3. Update statistics for stable softmax:
           m_new = max(mi, rowmax(Sij))
           l_new = exp(mi - m_new) * li + rowsum(exp(Sij - m_new))

        4. Update output:
           Oi = exp(mi - m_new) * Oi + exp(Sij - m_new) @ Vj

        5. Update statistics:
           mi = m_new
           li = l_new

    6. Final normalization:
       Oi = Oi / li

    7. Write Oi back to HBM

Key: Never store [N × N] matrix, only [B × B] blocks in SRAM
```

**Online Softmax Computation**:
```
Challenge: Softmax needs max and sum over full row
Solution: Update running statistics incrementally

Standard softmax:
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Online algorithm:
- Keep running max: m
- Keep running sum: l
- When seeing new values, update m and l
- Rescale previous computations when m changes

This allows computing softmax in one pass without storing all values!
```

**Memory Analysis**:
```
Standard Attention:
- HBM reads: Q, K, V (once each)
- HBM writes: O (once)
- HBM intermediate: Attention matrix S [N × N]
- Total HBM access: O(Nd + N²)  ← Dominated by N²

FlashAttention:
- HBM reads: Q, K, V (loaded in blocks)
- HBM writes: O (once)
- HBM intermediate: None (stay in SRAM)
- Total HBM access: O(N²d²/M) where M = SRAM size
- For typical M: O(Nd) ← Linear in N!

Speedup: 2-4x from reduced memory traffic
```

**Compute Analysis**:
```
Standard: O(N²d) FLOPs
FlashAttention: O(N²d) FLOPs (same)

But: Memory-bound → reducing memory traffic is key
```

### Follow-up Questions

**Q: "How does FlashAttention handle causal masking for autoregressive generation?"**
A: "For causal attention, we only compute the lower triangular part:

```
For each row block i:
    For each column block j:
        if j > i:  # Future tokens
            skip  # Don't compute
        elif j < i:  # All past tokens
            compute full [B × B] block
        else:  # j == i (diagonal block)
            compute with causal mask (lower triangle)
            # Can optimize to skip upper triangle entirely
```

This cuts computation roughly in half while maintaining the tiling structure."

**Q: "What about the backward pass?"**
A: "Backward pass is more complex because we need gradients w.r.t Q, K, V:

Options:
1. **Save attention matrix** during forward → defeats memory savings
2. **Recompute attention** during backward → FlashAttention approach

FlashAttention recomputes S = QK^T during backward:
- Still memory-efficient (tiles in SRAM)
- Total cost: 2x forward compute (acceptable trade-off)
- Saves O(N²) memory → enables much longer sequences

Backward also uses tiling and online statistics for efficiency."

**Q: "How does FlashAttention perform on different GPU architectures?"**
A: "Performance varies by memory hierarchy:

**A100**:
- SRAM (shared mem): 192KB per SM
- HBM bandwidth: ~1.5 TB/s
- Speedup: 2-3x over standard attention
- Sweet spot: Block size 128-256

**H100**:
- SRAM: 256KB per SM
- HBM bandwidth: ~3 TB/s
- Speedup: 3-4x
- Can use larger blocks (256-512)

**V100** (older):
- SRAM: 96KB per SM
- HBM bandwidth: ~900 GB/s
- Speedup: 1.5-2x
- Need smaller blocks (64-128)

Key: Newer GPUs with larger SRAM and higher bandwidth benefit more."

### Common Mistakes

❌ **Confusing with PagedAttention**: FlashAttention is about computation; PagedAttention is memory management

❌ **Missing online softmax details**: This is a key algorithmic contribution

❌ **Not discussing tiling parameters**: Block size choice matters

❌ **Ignoring memory bandwidth**: This is the key bottleneck being addressed

### Evaluation Criteria

**Excellent (5/5)**:
- Clear problem statement
- Detailed algorithm with tiling
- Online softmax explanation
- Memory analysis with big-O
- Discusses implementation variants

**Good (4/5)**:
- Correct core algorithm
- Basic tiling explanation
- Memory savings quantified

**Adequate (3/5)**:
- High-level understanding
- Missing tiling details

**Poor (1-2/5)**:
- Confused about mechanism
- Cannot explain memory savings

---

## Question 2: How Do You Optimize CUDA Kernel Performance?

### Question
"Walk me through your systematic approach to optimizing a CUDA kernel."

### Answer Framework
1. Profile and identify bottleneck
2. Apply targeted optimizations
3. Measure and iterate

### Detailed Answer

**Step 1: Profile and Identify Bottleneck**

**Tools**: Nsight Compute, nvprof, PyTorch profiler

**Key Metrics**:
```
1. Occupancy: % of max active warps
   - Target: >50% (ideally 75-100%)

2. Memory Throughput: % of peak bandwidth
   - HBM: Target >60%
   - L1/L2: Check hit rates

3. Compute Throughput: % of peak FLOPs
   - FP32/FP16/INT8: Check utilization
   - Tensor Core usage if applicable

4. Warp Execution Efficiency
   - Check for divergence
   - Stalls (memory, sync, etc.)
```

**Roofline Analysis**:
```
Plot: Performance (FLOPs/s) vs Arithmetic Intensity (FLOPs/byte)

Memory Bound: Below bandwidth ceiling
→ Optimize memory access

Compute Bound: Below peak FLOPs ceiling
→ Optimize computation

Sweet Spot: Balance both
```

**Step 2: Systematic Optimization Strategy**

**A. Memory Optimizations** (Most Common Bottleneck)

**1. Memory Coalescing**:
```cuda
// Bad: Strided access (each warp accesses scattered locations)
__global__ void bad(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx * stride];  // Uncoalesced!
}

// Good: Contiguous access (warp accesses consecutive addresses)
__global__ void good(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // Coalesced: one 128B transaction
}

Impact: 10-30x speedup by maximizing memory bandwidth utilization
```

**2. Shared Memory for Reuse**:
```cuda
// Example: Matrix multiplication
__global__ void matmul_optimized(float* C, float* A, float* B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Load tile into shared memory (once)
    As[ty][tx] = A[...];  // Coalesced load from global
    Bs[ty][tx] = B[...];
    __syncthreads();

    // Reuse from shared memory (fast)
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];  // No global memory access
    }
}

Impact: Reduce global memory accesses by reuse factor (e.g., 16x for 16x16 tiles)
```

**3. Avoid Bank Conflicts**:
```cuda
// Bad: Bank conflicts (multiple threads access same bank)
__shared__ float s[32][32];
float val = s[threadIdx.y][threadIdx.x];  // Column access → conflicts

// Good: Pad to avoid conflicts
__shared__ float s[32][33];  // +1 padding
float val = s[threadIdx.y][threadIdx.x];  // No conflicts

Impact: 8-32x speedup for shared memory operations
```

**B. Compute Optimizations**

**1. Warp-Level Primitives**:
```cuda
// Instead of shared memory reduction
__shared__ float sdata[256];
sdata[tid] = value;
__syncthreads();
// ... tree reduction with syncthreads

// Use warp-level shuffle
float val = value;
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// No shared memory, no sync, faster!

Impact: 2-3x faster reductions
```

**2. Instruction-Level Parallelism**:
```cuda
// Bad: Sequential dependencies
for (int i = 0; i < N; i++) {
    sum += a[i] * b[i];  // Must wait for each multiply-add
}

// Good: Multiple accumulators (ILP)
float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
for (int i = 0; i < N; i += 4) {
    sum1 += a[i] * b[i];
    sum2 += a[i+1] * b[i+1];
    sum3 += a[i+2] * b[i+2];
    sum4 += a[i+3] * b[i+3];
}
float sum = sum1 + sum2 + sum3 + sum4;

Impact: Better instruction pipeline utilization, 20-40% speedup
```

**3. Vectorized Loads**:
```cuda
// Load 4 floats at once (128-bit transaction)
float4* data4 = reinterpret_cast<float4*>(data);
float4 val = data4[idx];
// Process val.x, val.y, val.z, val.w

Impact: Fewer memory instructions, better bandwidth utilization
```

**C. Occupancy Optimization**

**1. Register Pressure**:
```
Check: nvcc --ptxas-options=-v
Registers per thread: X
→ If high (>64), reduce:
  - Use fewer local variables
  - Break into smaller functions
  - Use __launch_bounds__ to hint compiler
```

**2. Shared Memory Pressure**:
```
Balance shared memory vs blocks per SM:
- More shared mem → Fewer blocks → Lower occupancy
- Find sweet spot via experimentation
```

**3. Thread Block Size**:
```
Rules of thumb:
- Multiple of 32 (warp size)
- 128-256 often optimal
- Test: 64, 128, 256, 512
- Consider: occupancy, shared mem, register usage
```

**Step 3: Iteration and Validation**

```
1. Baseline benchmark
2. Apply ONE optimization at a time
3. Measure impact
4. If improvement >5%: Keep
5. If regression: Revert
6. Repeat until diminishing returns

Important: Changes can interact - may need to revisit earlier optimizations
```

### Follow-up Questions

**Q: "How do you debug performance regressions?"**
A: "Systematic comparison:
1. **Nsight Compute Diff**: Compare two kernel runs side-by-side
2. **Check metrics**:
   - Occupancy decreased? → Register or shared mem issue
   - Memory throughput down? → Coalescing or conflicts
   - Compute down? → Divergence or inefficient instructions
3. **Source correlation**: Link metrics to specific code lines
4. **Bisect**: If complex change, binary search for problematic optimization"

**Q: "When should you use tensor cores?"**
A: "Tensor cores excel for mixed-precision matrix multiplication:

**Use when**:
- Problem: Matrix multiply (GEMM)
- Sizes: M, N, K all multiples of 8 (FP16) or 16 (INT8)
- Precision: FP16/BF16 acceptable (or mixed precision)
- Scale: Large matrices (>1024x1024)

**Avoid when**:
- Irregular sizes (padding overhead too high)
- Need FP32 precision throughout
- Elemental operations (not matmul)

**Access**:
- CUTLASS library (NVIDIA templates)
- cuBLAS (automatic tensor core usage)
- WMMA intrinsics (manual control)

**Impact**: 8-16x speedup over FP32 CUDA cores for matmul"

### Common Mistakes

❌ **Random optimization**: Must profile first to identify bottleneck
❌ **Multiple changes at once**: Can't isolate impact
❌ **Ignoring occupancy**: Low occupancy limits performance
❌ **Over-optimizing non-bottlenecks**: Focus on hot paths

### Evaluation Criteria

**Excellent (5/5)**:
- Systematic profiling approach
- Multiple optimization techniques
- Concrete code examples
- Quantifies improvements
- Discusses trade-offs

---

## Questions 3-15: Additional CUDA Performance Topics

For brevity, providing condensed versions:

---

## Question 3: Explain CUDA Memory Hierarchy

**Answer**: Global → L2 → L1 → Shared → Registers (fastest)
- **Latency**: Global ~400 cycles, Shared ~30 cycles, Registers <1 cycle
- **Bandwidth**: HBM ~1-3 TB/s, Shared ~10 TB/s
- **Optimization**: Move data up hierarchy, maximize reuse

---

## Question 4: What is Memory Coalescing?

**Answer**: When warp threads access consecutive addresses, GPU combines into single transaction
- **Good**: thread 0 → addr 0, thread 1 → addr 4, ...
- **Bad**: Strided or scattered access
- **Impact**: 10-32x performance difference

---

## Question 5: Explain Bank Conflicts

**Answer**: Shared memory divided into 32 banks. If multiple threads in warp access same bank → serialized
- **Avoid**: Pad arrays, use different access patterns
- **Check**: Nsight shows bank conflict count
- **Impact**: Up to 32x slowdown if all threads conflict

---

## Question 6: What is Occupancy and Why Does it Matter?

**Answer**: % of max active warps per SM
- **Affects**: Latency hiding (more warps → hide memory latency)
- **Limited by**: Registers, shared memory, block size
- **Target**: >50%, but higher doesn't always mean faster
- **Tool**: Occupancy calculator, Nsight

---

## Question 7: How Do You Profile CUDA Kernels?

**Tools**:
- **Nsight Compute**: Detailed kernel analysis
- **Nsight Systems**: Timeline, system-wide
- **nvprof**: Legacy CLI profiler

**Metrics**: Memory throughput, occupancy, warp stalls, divergence
**Workflow**: Baseline → Profile → Optimize → Repeat

---

## Question 8: Explain Warp Divergence

**Answer**: When threads in warp take different control flow paths → serialize
```cuda
if (threadIdx.x % 2 == 0) { ... } else { ... }
// Half warp goes one way, half goes other → 2x slower
```
**Fix**: Rearrange to minimize divergence, use warp-level functions

---

## Question 9: What are Fused Kernels?

**Answer**: Combine multiple operations in one kernel
- **Benefit**: Reduce global memory round-trips, keep data in registers/shared
- **Example**: LayerNorm + Activation fused
- **Trade-off**: More complex, register pressure

---

## Question 10: Explain Asynchronous Execution

**Answer**: CPU launches kernel, continues without waiting
- **Streams**: Independent execution queues
- **Overlap**: Computation + memory copy + kernel execution
- **Sync**: cudaDeviceSynchronize() or event-based

---

## Question 11: How Does Quantization Affect CUDA Kernels?

**Answer**: INT8/INT4 kernels
- **Benefits**: 2-4x memory bandwidth, potential compute speedup
- **Implementation**: Special quantized GEMM kernels, dequantize in registers
- **Trade-off**: Quantization/dequantization overhead

---

## Question 12: What is CUTLASS?

**Answer**: NVIDIA CUDA Templates for Linear Algebra Subroutines
- **Purpose**: High-performance GEMM with tensor cores
- **Features**: Templates for different sizes, types, layouts
- **Usage**: vLLM, many ML frameworks

---

## Question 13: Explain Triton Language

**Answer**: Python-like DSL for writing GPU kernels
- **Benefits**: Easier than CUDA, compiler handles many optimizations
- **Use in vLLM**: Custom fused kernels
- **Trade-off**: Less control than raw CUDA

---

## Question 14: How Do You Debug CUDA Errors?

**Tools**:
- **cuda-gdb**: GDB for CUDA
- **compute-sanitizer**: Memory checker (like Valgrind)
- **cuda-memcheck**: Detect out-of-bounds, race conditions

**Approach**:
1. Check cudaGetLastError()
2. Enable exceptions in higher-level frameworks
3. Use compute-sanitizer for memory issues
4. Print debugging (carefully, affects performance)

---

## Question 15: What is the Roofline Model?

**Answer**: Performance model showing memory vs compute bounds
- **X-axis**: Arithmetic Intensity (FLOPs/byte)
- **Y-axis**: Performance (FLOPs/s)
- **Ceilings**: Memory bandwidth, peak compute
- **Use**: Identify bottleneck, guide optimization

---

## Practice Exercises

### Exercise Set 1: Kernel Implementation
1. Write optimized reduction (sum) kernel
2. Implement matrix transpose with shared memory
3. Fused softmax + multiply kernel
4. Prefix sum (scan) using warp primitives

### Exercise Set 2: Optimization
1. Optimize given slow matmul kernel
2. Fix bank conflicts in provided code
3. Improve occupancy of register-heavy kernel
4. Vectorize memory-bound kernel

### Exercise Set 3: Profiling
1. Profile attention kernel with Nsight Compute
2. Identify top 3 bottlenecks in given kernel
3. Create roofline plot for kernel
4. Compare coalesced vs uncoalesced versions

---

## Study Strategy

1. **Understand concepts** (2-3 hours)
2. **Code examples** (3-4 hours) - Write actual kernels
3. **Profile real kernels** (2-3 hours) - Use Nsight
4. **Mock interview practice** (2 hours) - Explain out loud

Total: ~10-12 hours

---

*Last Updated: 2025-11-19*
*Module 9: CUDA & Performance Interview Questions*
