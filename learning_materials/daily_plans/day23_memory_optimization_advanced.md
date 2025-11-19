# Day 23: Advanced Memory Optimization Techniques

> **Goal**: Master multi-level cache optimization, shared memory patterns, and memory prefetching
> **Time**: 6-8 hours
> **Prerequisites**: Days 1-22 completed, solid understanding of GPU memory hierarchy
> **Deliverables**: Optimized kernels, cache analysis reports, memory access pattern visualizations

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Cache Optimization & Analysis

**9:00-9:45** - GPU Memory Hierarchy Deep Dive
**9:45-10:45** - Cache-Aware Algorithm Design
**10:45-11:00** - Break
**11:00-12:30** - Shared Memory Optimization Patterns

### Afternoon Session (3-4 hours): Implementation & Profiling

**14:00-15:00** - Register Optimization Techniques
**15:00-16:00** - Memory Prefetching & Async Operations
**16:00-16:30** - Break
**16:30-18:00** - Comprehensive Memory Profiling

### Evening (Optional, 1-2 hours): Advanced Topics

**19:00-21:00** - Texture memory, constant memory, unified memory analysis

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Optimize for each level of GPU memory hierarchy
- [ ] Design cache-friendly algorithms and data layouts
- [ ] Eliminate shared memory bank conflicts
- [ ] Minimize register spilling and optimize allocation
- [ ] Implement memory prefetching for latency hiding
- [ ] Profile and analyze memory bottlenecks
- [ ] Apply memory optimization to vLLM kernels

---

## 📚 Morning: Cache Optimization (9:00-12:30)

### Task 1: GPU Memory Hierarchy (45 min)

**🏗️ Complete Memory Hierarchy**:

```
┌─────────────────────────────────────────────────────┐
│ Registers (Per-thread)                              │
│ - Size: ~64 KB per SM                               │
│ - Latency: 1 cycle                                  │
│ - Bandwidth: ~20 TB/s                               │
│ - Scope: Thread-private                             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ L1 Cache / Shared Memory (Per-SM)                  │
│ - Size: 128-256 KB (configurable split)            │
│ - Latency: ~30 cycles                               │
│ - Bandwidth: ~10 TB/s                               │
│ - Scope: Thread block                               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ L2 Cache (Global, all SMs)                          │
│ - Size: 40-60 MB (A100)                             │
│ - Latency: ~200 cycles                              │
│ - Bandwidth: ~5 TB/s                                │
│ - Scope: GPU-wide                                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ Global Memory (HBM)                                 │
│ - Size: 40-80 GB                                    │
│ - Latency: 400-800 cycles                           │
│ - Bandwidth: 1.5-2.0 TB/s                           │
│ - Scope: GPU-wide                                   │
└─────────────────────────────────────────────────────┘
```

**Performance Impact**:

```
Memory Access Latency (A100):
  Registers:      1 cycle     (0.5 ns @ 2 GHz)
  Shared/L1:     30 cycles    (15 ns)
  L2 cache:     200 cycles    (100 ns)
  HBM (DRAM):   600 cycles    (300 ns)

Latency Ratio:
  HBM is 600x slower than registers!
  Cache hit vs miss: 3-20x performance difference
```

**🎯 Optimization Strategy**:

1. **Maximize Register Usage** (fastest)
2. **Use Shared Memory** for data reuse within block
3. **Optimize for L2 Cache** hits
4. **Minimize Global Memory** accesses
5. **Coalesce** remaining global accesses

### Task 2: Cache-Aware Algorithm Design (60 min)

**Example: Matrix Transpose Optimization**

**Naive Version (Poor Cache Behavior)**:

```cuda
// BAD: Uncoalesced writes
__global__ void transpose_naive(float* out, const float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        // Coalesced reads (good)
        // But uncoalesced writes (bad)
        out[x * N + y] = in[y * N + x];
    }
}
```

**Problem**: Writes to `out` are strided by N (not coalesced)

**Cache-Optimized Version (Using Shared Memory)**:

```cuda
#define TILE_SIZE 32

__global__ void transpose_optimized(float* out, const float* in, int N) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Coalesced read from global memory
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Coalesced write to global memory
    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

**Key Insights**:
1. Use shared memory as staging area
2. Both reads and writes are now coalesced
3. `+ 1` padding avoids bank conflicts
4. **Speedup**: 5-10x over naive version

**Performance Analysis**:

```
Matrix size: 4096x4096 (64 MB)

Naive version:
  - Time: 8.5 ms
  - Bandwidth: 7.5 GB/s (terrible)
  - L2 hit rate: 45%

Optimized version:
  - Time: 0.9 ms
  - Bandwidth: 71 GB/s (9.5x better)
  - L2 hit rate: 92%
  - Speedup: 9.4x
```

**vLLM Application: Attention Score Transpose**

```cuda
// In PagedAttention, we often need to transpose attention scores
// From: [num_heads, seq_len, head_dim]
// To:   [num_heads, head_dim, seq_len]

template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_attention_scores(
    float* out,          // [num_heads, head_dim, seq_len]
    const float* in,     // [num_heads, seq_len, head_dim]
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Shared memory with bank conflict avoidance
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Process one head at a time
    int head_idx = blockIdx.z;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Input offset for this head
    const float* in_head = in + head_idx * seq_len * head_dim;
    float* out_head = out + head_idx * head_dim * seq_len;

    // Load tile (coalesced)
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < head_dim && (y + i) < seq_len) {
            tile[threadIdx.y + i][threadIdx.x] =
                in_head[(y + i) * head_dim + x];
        }
    }
    __syncthreads();

    // Write transposed tile (coalesced)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < seq_len && (y + i) < head_dim) {
            out_head[(y + i) * seq_len + x] =
                tile[threadIdx.x][threadIdx.y + i];
        }
    }
}
```

### Task 3: Shared Memory Bank Conflicts (90 min)

**🏦 Understanding Bank Conflicts**:

Shared memory is organized into 32 banks (for 32 threads in a warp):

```
Banks:  [ 0 ][ 1 ][ 2 ]...[ 31 ][ 0 ][ 1 ]...
Index:    0    1    2  ...  31   32   33  ...

Address mapping: bank_id = (address / 4) % 32
```

**Conflict Types**:

```cuda
__shared__ float shared[32][32];

// NO CONFLICT: Each thread accesses different bank
shared[threadIdx.y][threadIdx.x];  // threadIdx.x spans 0-31
// bank_id = threadIdx.x (all different)

// 2-WAY CONFLICT: Two threads access same bank
shared[threadIdx.y][threadIdx.x * 2];  // Even indices only
// bank_id = (threadIdx.x * 2) % 32 (two threads → same bank)

// 32-WAY CONFLICT: All threads access same bank (WORST)
shared[threadIdx.y][0];  // All threads access column 0
// bank_id = 0 (all threads → bank 0)
```

**Impact on Performance**:

```
No conflict:     1 cycle (parallel access)
2-way conflict:  2 cycles (serialized)
4-way conflict:  4 cycles
32-way conflict: 32 cycles (completely serialized!)
```

**🔧 Avoiding Bank Conflicts**:

**Problem: Strided Access**

```cuda
__shared__ float data[1024];

// 2-way bank conflict
int idx = threadIdx.x * 2;  // Stride of 2
float val = data[idx];
```

**Solution 1: Padding**

```cuda
// Add padding to change stride
__shared__ float data[32][33];  // Extra column

// Now no conflict
float val = data[threadIdx.y][threadIdx.x];
```

**Solution 2: Shuffle Access Pattern**

```cuda
__shared__ float data[32][32];

// XOR to spread accesses across banks
int idx = threadIdx.x ^ threadIdx.y;
float val = data[threadIdx.y][idx];
```

**vLLM Example: Reduction with Shared Memory**

```cuda
// Efficient warp reduction without bank conflicts
template<typename T, int BLOCK_SIZE>
__device__ T warp_reduce_sum(T val) {
    // Use warp shuffle instead of shared memory
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val) {
    __shared__ T shared[32];  // One per warp

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Reduce within warp (no shared memory needed)
    val = warp_reduce_sum<T, 32>(val);

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        val = (lane < (BLOCK_SIZE / 32)) ? shared[lane] : T(0);
        val = warp_reduce_sum<T, 32>(val);
    }

    return val;
}

// Usage in LayerNorm kernel
__global__ void layernorm_optimized(
    float* out,
    const float* in,
    const float* gamma,
    const float* beta,
    int N, int D
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Compute mean using optimized reduction
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        sum += in[row * D + i];
    }
    sum = block_reduce_sum<float, 256>(sum);  // No bank conflicts!

    __shared__ float s_mean;
    if (tid == 0) s_mean = sum / D;
    __syncthreads();

    // ... rest of layernorm
}
```

**📊 Bank Conflict Analysis**:

```bash
# Profile with Nsight Compute
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_shared.pct \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./kernel

# Ideal values:
# - bytes per sector: 100% (full utilization)
# - bank conflicts: 0 (no conflicts)
```

---

## 💻 Afternoon: Register & Prefetching (14:00-18:00)

### Task 4: Register Optimization (60 min)

**📊 Register Pressure Analysis**:

```
A100 Registers per SM: 65,536 registers
Max threads per SM: 2,048 threads

Registers per thread vs Occupancy:
  32 regs → 100% occupancy (2048 threads)
  64 regs → 100% occupancy (1024 threads)
  128 regs → 50% occupancy (512 threads)
  256 regs → 25% occupancy (256 threads)

Low occupancy → Cannot hide memory latency!
```

**🔍 Checking Register Usage**:

```bash
# Compile with register usage info
nvcc -Xptxas=-v kernel.cu

# Output:
# ptxas info    : Used 48 registers, 4096 bytes smem
# ptxas info    : Function properties for kernel:
#                 48 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

**Register Spilling** (BAD):

```
When kernel uses too many registers:
  - Excess values spilled to local memory
  - Local memory backed by global DRAM (slow!)
  - 100-200x latency penalty
  - Massive performance degradation
```

**🎯 Reducing Register Usage**:

**Technique 1: Kernel Splitting**

```cuda
// BAD: One large kernel (many registers)
__global__ void large_computation(float* out, const float* in, int N) {
    // Compute A
    float a1 = ...;
    float a2 = ...;
    // ... 20 intermediate values

    // Compute B
    float b1 = ...;
    float b2 = ...;
    // ... 20 more intermediate values

    // Total: 40+ registers → spilling!
}

// GOOD: Split into two kernels
__global__ void compute_A(float* temp, const float* in, int N) {
    // Only need ~20 registers
}

__global__ void compute_B(float* out, const float* temp, int N) {
    // Only need ~20 registers
}
```

**Technique 2: Reduce Variable Lifetime**

```cuda
// BAD: Variables live too long
__global__ void bad_lifetime(float* out, const float* in, int N) {
    float a = load_a();  // Allocated early
    float b = load_b();
    float c = load_c();

    // ... lots of code ...

    // Only used at end
    float result = compute(a, b, c);
}

// GOOD: Minimize variable lifetime
__global__ void good_lifetime(float* out, const float* in, int N) {
    // ... lots of code ...

    // Load right before use
    float a = load_a();
    float b = load_b();
    float c = load_c();
    float result = compute(a, b, c);
}
```

**Technique 3: Compiler Hints**

```cuda
// Limit register usage (may reduce performance)
__global__ __launch_bounds__(256, 4)  // 256 threads, 4 blocks per SM
void kernel(...) {
    // Compiler forced to use fewer registers
}

// Or use maxrregcount
nvcc -maxrregcount=32 kernel.cu
```

**vLLM Example: Attention Kernel Register Optimization**

```cuda
// Optimized register allocation in PagedAttention
template<typename scalar_t, int HEAD_SIZE>
__global__ void paged_attention_register_optimized(
    scalar_t* out,
    const scalar_t* q,
    const scalar_t* k_cache,
    const scalar_t* v_cache,
    // ... other parameters
) {
    // Load query into shared memory (not registers)
    __shared__ scalar_t s_query[HEAD_SIZE];

    if (threadIdx.x < HEAD_SIZE) {
        s_query[threadIdx.x] = q[...];
    }
    __syncthreads();

    // Process in chunks to reuse registers
    constexpr int CHUNK_SIZE = 16;
    float acc = 0.0f;  // Accumulator in register

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        // Local variables only live within loop iteration
        float score = 0.0f;

        // Compute attention score
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            float k_val = k_cache[...];  // Loaded and discarded
            score += s_query[i] * k_val;
        }

        // ... process score, update acc
    }  // score and k_val registers freed here

    out[...] = acc;
}
```

### Task 5: Memory Prefetching (60 min)

**🚀 Async Memory Operations (CUDA 11+)**:

```cuda
#include <cuda_pipeline.h>

// Overlap compute and memory access
__global__ void async_prefetch_kernel(
    float* out,
    const float* in,
    int N
) {
    __shared__ float buffer[2][256];  // Double buffering

    int current = 0;
    int next = 1;

    // Create pipeline
    __pipeline_wait_prior(0);

    // Prefetch first chunk
    __pipeline_memcpy_async(&buffer[current][threadIdx.x],
                           &in[threadIdx.x],
                           sizeof(float));
    __pipeline_commit();

    // Main loop with prefetching
    for (int i = 0; i < N / 256; ++i) {
        // Prefetch next chunk while processing current
        if (i < N / 256 - 1) {
            __pipeline_memcpy_async(&buffer[next][threadIdx.x],
                                   &in[(i + 1) * 256 + threadIdx.x],
                                   sizeof(float));
            __pipeline_commit();
        }

        // Wait for current chunk
        __pipeline_wait_prior(1);

        // Process current chunk (compute overlaps with next prefetch)
        float val = buffer[current][threadIdx.x];
        float result = expensive_compute(val);
        out[i * 256 + threadIdx.x] = result;

        // Swap buffers
        current = 1 - current;
        next = 1 - next;
    }
}
```

**Performance Gain**:
- Without prefetch: Compute waits for memory (serial)
- With prefetch: Compute overlaps with memory (parallel)
- **Speedup**: 1.5-2x for memory-bound kernels

**Manual Prefetching Pattern**:

```cuda
// Software pipelining for older CUDA
__global__ void manual_prefetch(float* out, const float* in, int N) {
    const int UNROLL = 4;

    // Prefetch first values
    float vals[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        vals[i] = in[threadIdx.x + i * blockDim.x];
    }

    // Main loop
    for (int base = 0; base < N; base += UNROLL * blockDim.x) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            int idx = base + threadIdx.x + i * blockDim.x;

            // Process current value
            float result = compute(vals[i]);
            out[idx] = result;

            // Prefetch next value (for next iteration)
            if (idx + UNROLL * blockDim.x < N) {
                vals[i] = in[idx + UNROLL * blockDim.x];
            }
        }
    }
}
```

### Task 6: Comprehensive Memory Profiling (90 min)

**🔬 Profiling Workflow**:

**Step 1: Identify Memory Bottlenecks**

```bash
# High-level profile
nsys profile --stats=true ./program

# Look for:
# - High DRAM utilization
# - Low compute utilization
# - Memory-bound kernels
```

**Step 2: Detailed Memory Analysis**

```bash
# Deep dive into memory behavior
ncu --set full \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --section Occupancy \
    ./program

# Key metrics:
# - DRAM throughput (target: >60% of peak)
# - L2 cache hit rate (target: >50%)
# - Shared memory bank conflicts (target: 0)
# - Memory efficiency (target: >80%)
```

**Step 3: Access Pattern Analysis**

```bash
# Check coalescing
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./program

# Coalescing efficiency = requests / (sectors / 4)
# Target: >90%
```

**📊 Memory Optimization Checklist**:

```python
#!/usr/bin/env python3
"""Memory Optimization Analysis Tool"""

class MemoryOptimizationAnalyzer:
    def __init__(self, ncu_output):
        self.metrics = self.parse_ncu_output(ncu_output)

    def analyze(self):
        issues = []

        # Check 1: Coalescing
        coalescing_eff = self.compute_coalescing_efficiency()
        if coalescing_eff < 0.9:
            issues.append({
                'type': 'COALESCING',
                'severity': 'HIGH',
                'value': coalescing_eff,
                'recommendation': 'Reorganize memory accesses for coalescing'
            })

        # Check 2: Bank conflicts
        bank_conflicts = self.metrics.get('bank_conflicts', 0)
        if bank_conflicts > 100:
            issues.append({
                'type': 'BANK_CONFLICTS',
                'severity': 'MEDIUM',
                'value': bank_conflicts,
                'recommendation': 'Add padding or change access pattern'
            })

        # Check 3: L2 cache hit rate
        l2_hit_rate = self.metrics.get('l2_hit_rate', 0)
        if l2_hit_rate < 0.5:
            issues.append({
                'type': 'CACHE_MISSES',
                'severity': 'MEDIUM',
                'value': l2_hit_rate,
                'recommendation': 'Improve data locality and reuse'
            })

        # Check 4: Register spilling
        spill_stores = self.metrics.get('spill_stores', 0)
        if spill_stores > 0:
            issues.append({
                'type': 'REGISTER_SPILLING',
                'severity': 'HIGH',
                'value': spill_stores,
                'recommendation': 'Reduce register usage or split kernel'
            })

        # Check 5: Occupancy
        occupancy = self.metrics.get('occupancy', 0)
        if occupancy < 0.5:
            issues.append({
                'type': 'LOW_OCCUPANCY',
                'severity': 'MEDIUM',
                'value': occupancy,
                'recommendation': 'Reduce shared mem/register usage'
            })

        return issues

    def generate_report(self):
        print("="*60)
        print("MEMORY OPTIMIZATION ANALYSIS REPORT")
        print("="*60)

        issues = self.analyze()

        if not issues:
            print("✅ No major issues found!")
            return

        # Sort by severity
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        issues.sort(key=lambda x: severity_order[x['severity']])

        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['type']} [{issue['severity']}]")
            print(f"   Value: {issue['value']:.2f}")
            print(f"   → {issue['recommendation']}")

        print("\n" + "="*60)
        print(f"Total issues: {len(issues)}")
        print("="*60)
```

**Example Report**:

```
============================================================
MEMORY OPTIMIZATION ANALYSIS REPORT
============================================================

1. REGISTER_SPILLING [HIGH]
   Value: 1024.00 bytes
   → Reduce register usage or split kernel

2. COALESCING [HIGH]
   Value: 0.45
   → Reorganize memory accesses for coalescing

3. CACHE_MISSES [MEDIUM]
   Value: 0.35
   → Improve data locality and reuse

4. LOW_OCCUPANCY [MEDIUM]
   Value: 0.42
   → Reduce shared mem/register usage

============================================================
Total issues: 4
============================================================
```

**🎯 Optimization Priority**:

1. **Fix register spilling** (biggest impact)
2. **Improve coalescing** (2-5x speedup)
3. **Increase cache hit rate** (1.5-2x speedup)
4. **Resolve bank conflicts** (up to 32x for worst cases)
5. **Improve occupancy** (better latency hiding)

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Memory Hierarchy Mastery**
- Understanding each level
- Latency and bandwidth trade-offs
- Optimization strategies per level

✅ **Cache Optimization**
- Cache-aware algorithm design
- Data layout for locality
- L2 cache utilization

✅ **Shared Memory Techniques**
- Bank conflict identification and resolution
- Padding strategies
- Optimal access patterns

✅ **Register Optimization**
- Minimizing register pressure
- Avoiding spilling
- Variable lifetime management

✅ **Advanced Techniques**
- Memory prefetching
- Async operations
- Pipeline overlapping

### Knowledge Check

**Q1**: Why is coalesced memory access important?

<details>
<summary>Answer</summary>
Coalesced access allows a warp (32 threads) to load/store data in a single memory transaction. Uncoalesced accesses require multiple transactions (up to 32), reducing bandwidth utilization by up to 32x.
</details>

**Q2**: How does padding prevent bank conflicts?

<details>
<summary>Answer</summary>
Padding changes the memory stride to avoid multiple threads accessing the same bank. Example: array[32][32] → two threads access same bank for column access. array[32][33] → padding breaks the pattern, distributing accesses across banks.
</details>

**Q3**: What's the impact of register spilling?

<details>
<summary>Answer</summary>
Spilled registers go to local memory (backed by slow global DRAM), causing 100-200x latency penalty. Can reduce kernel performance by 10-50x in extreme cases.
</details>

### Practical Exercises

- [ ] Optimize matrix transpose with shared memory
- [ ] Implement reduction without bank conflicts
- [ ] Profile and fix memory bottlenecks in sample kernel
- [ ] Implement prefetching for memory-bound operation
- [ ] Analyze vLLM attention kernel memory patterns

---

## 🚀 Preview: Day 24

Tomorrow's focus:
- **Framework Comparison**: vLLM vs TensorRT-LLM vs HuggingFace TGI
- **Architecture Differences**: Design decisions and trade-offs
- **Performance Analysis**: Benchmarking across frameworks
- **Use Case Mapping**: When to use which framework

**Preparation**:
- Install TensorRT-LLM and HF TGI (if possible)
- Review architectural differences
- Prepare benchmark scenarios

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
