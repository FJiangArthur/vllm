# PagedAttention Algorithm: The Core Innovation

## Module 2, Lesson 2: Deep Dive into vLLM's Memory Management

**Estimated Time:** 3-4 hours
**Difficulty:** Advanced
**Prerequisites:** Lesson 1 (Traditional Attention Problems), Virtual memory concepts

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Virtual Memory Analogy](#the-virtual-memory-analogy)
3. [PagedAttention Overview](#pagedattention-overview)
4. [Block-Based Memory Organization](#block-based-memory-organization)
5. [The Attention Computation with Paging](#the-attention-computation-with-paging)
6. [Mathematical Formulation](#mathematical-formulation)
7. [Implementation Architecture](#implementation-architecture)
8. [Memory Allocation and Deallocation](#memory-allocation-and-deallocation)
9. [Performance Characteristics](#performance-characteristics)
10. [Comparison with Traditional Attention](#comparison-with-traditional-attention)
11. [Hands-On Exercises](#hands-on-exercises)
12. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction

PagedAttention is vLLM's revolutionary approach to managing KV cache memory during LLM inference. Inspired by operating system virtual memory, PagedAttention eliminates the need for contiguous memory allocation while maintaining the efficiency of attention computation.

### The Core Innovation

Instead of storing KV cache as one contiguous block per sequence, PagedAttention:

1. **Divides KV cache into fixed-size blocks (pages)**
2. **Maps logical KV positions to physical memory blocks**
3. **Allows non-contiguous physical storage**
4. **Enables dynamic allocation and deallocation**

This simple yet powerful idea unlocks massive improvements in memory utilization and serving throughput.

### Learning Objectives

By the end of this lesson, you will:

1. Understand the block-based memory organization of PagedAttention
2. Explain how attention computation works with paged memory
3. Analyze the mathematical formulation of paged attention
4. Implement basic block allocation algorithms
5. Calculate memory savings and performance improvements
6. Navigate vLLM's PagedAttention implementation

---

## The Virtual Memory Analogy

### How Operating Systems Manage Memory

Modern operating systems use **virtual memory** to provide processes with the illusion of contiguous memory:

```
Process View (Virtual Memory):
┌──────────────────────────────────┐
│  Contiguous address space        │
│  0x0000 → 0xFFFF                 │
└──────────────────────────────────┘

Physical Memory:
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│Page │  │Page │  │Page │  │Page │
│  0  │  │  2  │  │  1  │  │  3  │
└─────┘  └─────┘  └─────┘  └─────┘
 Addr     Addr     Addr     Addr
 0x100    0x500    0x300    0x700
```

**Page Table** maps virtual addresses to physical addresses:

```
Virtual Page → Physical Frame
    0       →     0x100
    1       →     0x300
    2       →     0x500
    3       →     0x700
```

### Applying to KV Cache

PagedAttention applies the same concept to KV cache:

```
Logical KV Cache (Sequence View):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│Block0│Block1│Block2│Block3│Block4│Block5│
└──────┴──────┴──────┴──────┴──────┴──────┘
Tokens: 0-15   16-31  32-47  48-63  64-79  80-95

Physical GPU Memory:
┌──────┐         ┌──────┐         ┌──────┐
│Block0│         │Block2│         │Block1│
└──────┘         └──────┘         └──────┘
GPU addr         GPU addr         GPU addr
0x1000           0x3000           0x5000

Block Table:
Logical Block → Physical Block
     0        →     0x1000
     1        →     0x5000
     2        →     0x3000
     ...
```

### Key Benefits

1. **No contiguous allocation required**
2. **Fill small memory gaps efficiently**
3. **Allocate/free blocks independently**
4. **Near-zero internal fragmentation**

---

## PagedAttention Overview

### Core Concepts

#### 1. Blocks (Pages)

A **block** is a fixed-size unit of KV cache storage.

```python
# Block configuration
BLOCK_SIZE = 16  # tokens per block

# Each block stores K and V for BLOCK_SIZE tokens
block_shape = (num_layers, 2, BLOCK_SIZE, num_heads, head_dim)
#                           ↑
#                       K and V
```

**Example:** For Llama-2-7B with block_size=16:
- Layers: 32
- Heads: 32
- Head dim: 128
- Block memory: 32 × 2 × 16 × 32 × 128 × 2 bytes = 8,388,608 bytes ≈ 8 MB

#### 2. Block Tables

A **block table** maps logical blocks to physical blocks for each sequence.

```python
# Block table for a sequence
class BlockTable:
    def __init__(self):
        self.blocks: List[PhysicalBlock] = []

    def append_block(self, physical_block: PhysicalBlock):
        """Add a new block to this sequence"""
        self.blocks.append(physical_block)

    def get_physical_block(self, logical_idx: int) -> PhysicalBlock:
        """Map logical block index to physical block"""
        return self.blocks[logical_idx]
```

#### 3. Physical Block Pool

All physical blocks are managed in a pool:

```python
class BlockPool:
    def __init__(self, num_blocks: int):
        self.free_blocks: List[PhysicalBlock] = [
            PhysicalBlock(i) for i in range(num_blocks)
        ]
        self.allocated_blocks: Set[PhysicalBlock] = set()

    def allocate(self) -> PhysicalBlock:
        """Allocate a block from the pool"""
        if not self.free_blocks:
            raise OutOfMemoryError("No free blocks available")
        block = self.free_blocks.pop()
        self.allocated_blocks.add(block)
        return block

    def free(self, block: PhysicalBlock):
        """Return a block to the pool"""
        self.allocated_blocks.remove(block)
        self.free_blocks.append(block)
```

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM Engine                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐         ┌──────────────┐                 │
│  │Sequences │───────▶ │Block Manager │                 │
│  └──────────┘         └──────────────┘                 │
│                              │                          │
│                              ▼                          │
│                       ┌────────────┐                    │
│                       │ Block Pool │                    │
│                       └────────────┘                    │
│                              │                          │
│                              ▼                          │
│                    ┌──────────────────┐                 │
│                    │  GPU Memory      │                 │
│                    │  [Block Storage] │                 │
│                    └──────────────────┘                 │
│                              │                          │
│                              ▼                          │
│                    ┌──────────────────┐                 │
│                    │ Attention Kernel │                 │
│                    │ (PagedAttention) │                 │
│                    └──────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

---

## Block-Based Memory Organization

### Block Size Selection

The block size is a critical parameter balancing:

1. **Large blocks:** Lower metadata overhead, but more internal fragmentation
2. **Small blocks:** Less fragmentation, but higher metadata overhead

**vLLM default: 16 tokens per block**

This provides a good balance for most workloads.

### Memory Layout

Physical memory is organized as a large array of blocks:

```python
# Simplified memory layout
class KVCacheMemory:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim):
        # Shape: [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
        self.data = torch.zeros(
            num_blocks,
            num_layers,
            2,  # K and V
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device='cuda'
        )

    def write_block(self, block_id, layer_id, kv_id, position, data):
        """Write K or V data to a specific position in a block"""
        self.data[block_id, layer_id, kv_id, position] = data

    def read_block(self, block_id, layer_id, kv_id):
        """Read all K or V data from a block"""
        return self.data[block_id, layer_id, kv_id]
```

### Block Allocation Example

Let's trace how blocks are allocated for a growing sequence:

```
Step 1: New sequence arrives, needs 10 tokens
Required blocks: ⌈10 / 16⌉ = 1 block

Allocated:
┌──────────────────┐
│ Block 42         │
│ Tokens: 0-9      │ (10 used, 6 free)
└──────────────────┘

Block Table: [42]

Step 2: Generate 8 more tokens (total: 18)
Required blocks: ⌈18 / 16⌉ = 2 blocks

Allocated:
┌──────────────────┐  ┌──────────────────┐
│ Block 42         │  │ Block 73         │
│ Tokens: 0-15     │  │ Tokens: 16-17    │ (2 used, 14 free)
└──────────────────┘  └──────────────────┘

Block Table: [42, 73]

Step 3: Generate 30 more tokens (total: 48)
Required blocks: ⌈48 / 16⌉ = 3 blocks

Allocated:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Block 42         │  │ Block 73         │  │ Block 15         │
│ Tokens: 0-15     │  │ Tokens: 16-31    │  │ Tokens: 32-47    │
└──────────────────┘  └──────────────────┘  └──────────────────┘

Block Table: [42, 73, 15]
```

### Internal Fragmentation Calculation

For a sequence of length `L` with block size `B`:

```
Blocks needed: ⌈L / B⌉
Last block utilization: (L mod B) / B  if (L mod B) != 0, else 1

Internal fragmentation: B - (L mod B)  if (L mod B) != 0, else 0
```

**Example:** L=50, B=16
```
Blocks needed: ⌈50 / 16⌉ = 4
Last block utilization: (50 mod 16) / 16 = 2 / 16 = 12.5%
Wasted tokens in last block: 16 - 2 = 14
Total allocated: 64, Used: 50
Overall utilization: 50/64 = 78.1%
```

**Average case analysis:**

Assuming uniform distribution of sequence lengths ending:
```
Expected waste per sequence: B / 2 = 8 tokens (for B=16)
Expected utilization: 1 - (B/2) / L  (for large L)

For L=1000: 1 - 8/1000 = 99.2% utilization!
```

Much better than traditional ~60% utilization!

---

## The Attention Computation with Paging

### Traditional vs. Paged Attention

**Traditional attention** assumes contiguous KV cache:

```python
def traditional_attention(q, k_cache, v_cache):
    """
    q: [batch, num_heads, 1, head_dim]  # Current query
    k_cache: [batch, num_heads, seq_len, head_dim]  # Contiguous K
    v_cache: [batch, num_heads, seq_len, head_dim]  # Contiguous V
    """
    # Compute attention scores
    scores = torch.matmul(q, k_cache.transpose(-2, -1))  # [batch, heads, 1, seq_len]
    scores = scores / math.sqrt(q.shape[-1])

    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, v_cache)  # [batch, heads, 1, head_dim]

    return output
```

**Paged attention** gathers K and V from multiple blocks:

```python
def paged_attention(q, kv_cache, block_tables):
    """
    q: [num_seqs, num_heads, head_dim]  # Current queries
    kv_cache: [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
    block_tables: [num_seqs, max_num_blocks]  # Block mappings
    """
    # Gather K and V from blocks
    k_gathered = gather_from_blocks(kv_cache, block_tables, kv_idx=0)
    v_gathered = gather_from_blocks(kv_cache, block_tables, kv_idx=1)

    # Compute attention (same as traditional)
    scores = torch.matmul(q.unsqueeze(2), k_gathered.transpose(-2, -1))
    scores = scores / math.sqrt(q.shape[-1])
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_gathered)

    return output

def gather_from_blocks(kv_cache, block_tables, kv_idx):
    """Gather K or V from non-contiguous blocks"""
    batch_size, max_blocks = block_tables.shape
    block_size = kv_cache.shape[3]

    # For each sequence, gather its blocks
    gathered = []
    for seq_idx in range(batch_size):
        seq_blocks = block_tables[seq_idx]
        seq_data = []

        for block_id in seq_blocks:
            if block_id >= 0:  # Valid block
                # Extract data from this block
                block_data = kv_cache[block_id, :, kv_idx]  # [layer, block_size, heads, dim]
                seq_data.append(block_data)

        # Concatenate blocks for this sequence
        gathered.append(torch.cat(seq_data, dim=1))  # Concat along token dimension

    return torch.stack(gathered)
```

### Efficient Gather Implementation

The key challenge is efficiently gathering from non-contiguous blocks. vLLM uses optimized CUDA kernels:

```cuda
// Simplified CUDA kernel for gathering K cache
__global__ void gather_k_cache(
    float* output,           // [num_seqs, seq_len, num_heads, head_dim]
    const float* kv_cache,   // [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
    const int* block_tables, // [num_seqs, max_blocks]
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int layer_idx
) {
    int seq_idx = blockIdx.x;
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int head_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int dim_idx = threadIdx.x;

    if (seq_idx >= num_seqs || head_idx >= num_heads || dim_idx >= head_dim)
        return;

    // Calculate which block this token belongs to
    int block_local_idx = token_idx / block_size;
    int position_in_block = token_idx % block_size;

    // Look up physical block ID
    int physical_block = block_tables[seq_idx * max_blocks + block_local_idx];

    if (physical_block < 0) return;  // Invalid block

    // Calculate source address in KV cache
    int kv_idx = 0;  // K cache
    int src_offset = physical_block * num_layers * 2 * block_size * num_heads * head_dim
                   + layer_idx * 2 * block_size * num_heads * head_dim
                   + kv_idx * block_size * num_heads * head_dim
                   + position_in_block * num_heads * head_dim
                   + head_idx * head_dim
                   + dim_idx;

    // Calculate destination address
    int dst_offset = seq_idx * seq_len * num_heads * head_dim
                   + token_idx * num_heads * head_dim
                   + head_idx * head_dim
                   + dim_idx;

    // Copy data
    output[dst_offset] = kv_cache[src_offset];
}
```

### Optimization: Fused Kernel

vLLM fuses the gather and attention computation into a single kernel for efficiency:

```
Traditional: Gather K → Gather V → Compute Attention (3 kernels)

PagedAttention: Fused Gather + Attention (1 kernel)
```

This eliminates intermediate memory transfers and significantly improves performance.

---

## Mathematical Formulation

### Standard Attention Formulation

Given query Q, keys K, and values V:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Paged Attention Formulation

With block-based storage, we reformulate attention:

Let:
- `B` = block size
- `n` = sequence length
- `m` = ⌈n / B⌉ = number of blocks
- `K_i` = keys in block i (shape: [B, d_k])
- `V_i` = values in block i (shape: [B, d_v])

The attention output for query q is:

```
Output(q) = Σ_{i=0}^{m-1} α_i V_i

where α_i = softmax_i( q K_i^T / √d_k )
```

### Block-wise Softmax Computation

Computing softmax across blocks requires care:

```
scores = [q K_0^T, q K_1^T, ..., q K_{m-1}^T]

softmax(scores) requires:
1. max_score = max(scores)
2. exp_scores = exp(scores - max_score)
3. sum_exp = Σ exp_scores
4. normalized = exp_scores / sum_exp
```

**Challenge:** Blocks are in different physical locations!

**Solution:** Online softmax with running statistics:

```python
def online_softmax(q, k_blocks, v_blocks):
    """Compute softmax incrementally across blocks"""
    max_score = float('-inf')
    sum_exp = 0.0
    weighted_sum = torch.zeros_like(q)

    # First pass: find max
    for k_block in k_blocks:
        scores = torch.matmul(q, k_block.T) / math.sqrt(q.shape[-1])
        max_score = max(max_score, scores.max())

    # Second pass: compute attention
    for k_block, v_block in zip(k_blocks, v_blocks):
        scores = torch.matmul(q, k_block.T) / math.sqrt(q.shape[-1])
        exp_scores = torch.exp(scores - max_score)
        sum_exp += exp_scores.sum()
        weighted_sum += torch.matmul(exp_scores, v_block)

    # Normalize
    output = weighted_sum / sum_exp
    return output
```

This is simplified; vLLM uses a more efficient single-pass algorithm.

### Flash-Paged Attention

vLLM combines FlashAttention's online computation with paged memory:

```
For each block i:
    1. Load K_i, V_i from block into shared memory
    2. Compute block scores: s_i = q K_i^T
    3. Update running max: m_new = max(m_old, max(s_i))
    4. Update running sum: l_new = l_old * exp(m_old - m_new) + Σ exp(s_i - m_new)
    5. Update output: o_new = o_old * exp(m_old - m_new) + exp(s_i - m_new) V_i
```

This allows single-pass computation without materializing full attention matrix!

---

## Implementation Architecture

### Key Components in vLLM

Let's explore the actual vLLM implementation:

#### 1. Block Pool (`/vllm/v1/core/block_pool.py`)

```python
# Simplified from actual vLLM code
class BlockPool:
    """Manages allocation of physical blocks"""

    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks))
        self.ref_counts: Dict[int, int] = {}

    def allocate(self) -> int:
        """Allocate a block and return its ID"""
        if not self.free_blocks:
            raise ValueError("Out of memory: no free blocks")

        block_id = self.free_blocks.pop(0)
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int) -> None:
        """Free a block"""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)

    def get_num_free_blocks(self) -> int:
        """Return number of free blocks"""
        return len(self.free_blocks)
```

#### 2. KV Cache Manager (`/vllm/v1/core/kv_cache_manager.py`)

```python
class KVCacheManager:
    """Manages KV cache allocation for sequences"""

    def __init__(self, block_size: int, num_gpu_blocks: int):
        self.block_size = block_size
        self.block_pool = BlockPool(block_size, num_gpu_blocks)
        self.block_tables: Dict[int, List[int]] = {}

    def allocate_slots(self, seq_id: int, num_tokens: int) -> None:
        """Allocate blocks for num_tokens"""
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []

        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        current_blocks = len(self.block_tables[seq_id])

        # Allocate additional blocks if needed
        for _ in range(current_blocks, blocks_needed):
            block_id = self.block_pool.allocate()
            self.block_tables[seq_id].append(block_id)

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence"""
        if seq_id in self.block_tables:
            for block_id in self.block_tables[seq_id]:
                self.block_pool.free(block_id)
            del self.block_tables[seq_id]

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get block table for a sequence"""
        return self.block_tables.get(seq_id, [])
```

#### 3. Attention Backend (`/vllm/attention/`)

The attention backend handles the actual paged attention computation:

```python
class PagedAttentionBackend:
    """Backend for computing paged attention"""

    @staticmethod
    def forward(
        query: torch.Tensor,                    # [num_seqs, num_heads, head_dim]
        key_cache: torch.Tensor,                # [num_blocks, ...]
        value_cache: torch.Tensor,              # [num_blocks, ...]
        block_tables: torch.Tensor,             # [num_seqs, max_blocks]
        context_lens: torch.Tensor,             # [num_seqs]
    ) -> torch.Tensor:
        """Compute paged attention"""

        # Call optimized CUDA kernel
        output = paged_attention_cuda.forward(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
        )

        return output
```

### Data Flow

Here's how data flows through the system:

```
1. Request arrives
   ↓
2. Scheduler assigns sequence ID
   ↓
3. KVCacheManager allocates blocks
   ├─→ BlockPool.allocate() → Returns block_id
   └─→ Updates block_table[seq_id]
   ↓
4. Model forward pass
   ├─→ Computes Q, K, V for current token
   └─→ Writes K, V to allocated blocks
   ↓
5. Attention computation
   ├─→ Gathers K, V from blocks using block_table
   └─→ Computes attention
   ↓
6. Token generation
   ↓
7. Repeat steps 3-6 until completion
   ↓
8. Sequence completes
   └─→ KVCacheManager.free_sequence()
       └─→ BlockPool.free() for each block
```

---

## Memory Allocation and Deallocation

### Allocation Strategy

vLLM uses **lazy allocation**: blocks are allocated only when needed.

```python
def append_token_ids(self, seq_id: int, token_ids: List[int]) -> None:
    """Append tokens to a sequence, allocating blocks as needed"""
    if seq_id not in self.sequences:
        self.sequences[seq_id] = Sequence(seq_id)

    seq = self.sequences[seq_id]
    new_length = len(seq.tokens) + len(token_ids)

    # Calculate blocks needed
    blocks_needed = (new_length + self.block_size - 1) // self.block_size

    # Allocate additional blocks if necessary
    while len(seq.block_table) < blocks_needed:
        if self.block_pool.get_num_free_blocks() == 0:
            raise OutOfMemoryError()

        block_id = self.block_pool.allocate()
        seq.block_table.append(block_id)

    # Add tokens
    seq.tokens.extend(token_ids)
```

### Deallocation Timing

Blocks are freed when:

1. **Sequence completes:** All blocks returned to pool
2. **Sequence is preempted:** Blocks may be swapped or freed
3. **Sequence is aborted:** Immediate deallocation

```python
def finish_sequence(self, seq_id: int) -> None:
    """Mark sequence as complete and free resources"""
    if seq_id not in self.sequences:
        return

    seq = self.sequences[seq_id]

    # Free all blocks
    for block_id in seq.block_table:
        self.block_pool.free(block_id)

    # Remove sequence
    del self.sequences[seq_id]

    print(f"Freed {len(seq.block_table)} blocks from sequence {seq_id}")
```

### Reference Counting for Sharing

vLLM uses reference counting to enable block sharing (covered in copy-on-write lesson):

```python
class BlockPool:
    def __init__(self, ...):
        self.ref_counts: Dict[int, int] = {}

    def allocate(self) -> int:
        block_id = self.free_blocks.pop(0)
        self.ref_counts[block_id] = 1  # Initial reference
        return block_id

    def add_ref(self, block_id: int) -> None:
        """Increment reference count for shared block"""
        self.ref_counts[block_id] += 1

    def free(self, block_id: int) -> None:
        """Decrement reference count, free if zero"""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)
```

---

## Performance Characteristics

### Memory Efficiency

**Theoretical Analysis:**

Internal fragmentation per sequence:
```
Expected waste = block_size / 2 = 16 / 2 = 8 tokens
```

For a batch of 100 sequences with average length 1000:
```
Traditional approach:
  Memory allocated: 100 × 2048 × memory_per_token
  Memory used: 100 × 1000 × memory_per_token
  Utilization: 48.8%

PagedAttention approach:
  Memory allocated: 100 × (⌈1000/16⌉ × 16) × memory_per_token
                  = 100 × 1008 × memory_per_token
  Memory used: 100 × 1000 × memory_per_token
  Utilization: 99.2%

Memory savings: (2048 - 1008) / 2048 = 50.8%
```

### Throughput Improvement

With better memory utilization, we can fit more sequences:

```
Traditional: 50 sequences max
PagedAttention: 100 sequences max (2× increase)

Assuming same compute capacity:
Throughput improvement: 2× (100%)
```

Real-world measurements from vLLM paper:
- **2-4× throughput improvement** depending on workload
- **Near-zero memory waste** (>95% utilization)

### Latency Overhead

PagedAttention adds minimal latency overhead:

```
Traditional attention: ~1.0× baseline
PagedAttention (with gather): ~1.05× baseline
PagedAttention (fused kernel): ~1.01× baseline
```

The overhead is negligible compared to throughput gains!

### Compute Efficiency

Block-based computation maintains high GPU utilization:

```
Memory access pattern:
- Traditional: Linear scan (good cache locality)
- PagedAttention: Block-wise access (still good with prefetching)

GPU utilization:
- Traditional: 85-90%
- PagedAttention: 83-88% (minimal difference)
```

---

## Comparison with Traditional Attention

### Memory Utilization

| Metric | Traditional | PagedAttention | Improvement |
|--------|-------------|----------------|-------------|
| Avg utilization | 50-60% | 95-98% | 1.6-2× |
| Internal fragmentation | High | Minimal | Block_size dependent |
| External fragmentation | High | Low | Managed by pool |
| Allocation overhead | High | Low | Amortized |

### Serving Capacity

For an A100 80GB serving Llama-2-7B:

| Approach | Max Batch Size | Throughput (seq/s) |
|----------|---------------|-------------------|
| Traditional (static) | 32 | 3.2 |
| Traditional (optimized) | 45 | 4.5 |
| PagedAttention | 90+ | 9.0+ |

**2-3× improvement in real workloads!**

### Flexibility

| Feature | Traditional | PagedAttention |
|---------|-------------|----------------|
| Dynamic batching | No | Yes |
| Prefix sharing | No | Yes |
| Preemption | Difficult | Easy |
| Variable lengths | Inefficient | Efficient |

---

## Hands-On Exercises

### Exercise 1: Block Allocation Simulator

Implement a simple block allocator:

```python
class SimpleBlockAllocator:
    def __init__(self, total_blocks: int, block_size: int):
        """
        TODO: Initialize allocator
        - Create free block list
        - Initialize block tables for sequences
        - Set up any needed data structures
        """
        pass

    def allocate_sequence(self, seq_id: int, length: int) -> List[int]:
        """
        TODO: Allocate blocks for a sequence of given length
        - Calculate blocks needed
        - Allocate from free list
        - Return block table

        Returns: List of allocated block IDs
        """
        pass

    def append_tokens(self, seq_id: int, num_tokens: int) -> Optional[List[int]]:
        """
        TODO: Append tokens to existing sequence
        - Calculate if new blocks needed
        - Allocate additional blocks
        - Return newly allocated block IDs or None

        Returns: New block IDs or None if no new blocks needed
        """
        pass

    def free_sequence(self, seq_id: int) -> int:
        """
        TODO: Free all blocks for a sequence
        - Get block table for sequence
        - Return blocks to free list
        - Clean up sequence metadata

        Returns: Number of blocks freed
        """
        pass

    def get_utilization(self) -> float:
        """
        TODO: Calculate memory utilization
        - Count allocated blocks
        - Calculate as fraction of total

        Returns: Utilization percentage (0.0 to 1.0)
        """
        pass


# Test your implementation
allocator = SimpleBlockAllocator(total_blocks=100, block_size=16)

# Test 1: Allocate sequences
seq1_blocks = allocator.allocate_sequence(seq_id=1, length=50)
print(f"Sequence 1 allocated blocks: {seq1_blocks}")

# Test 2: Append tokens
new_blocks = allocator.append_tokens(seq_id=1, num_tokens=20)
print(f"New blocks after append: {new_blocks}")

# Test 3: Free sequence
freed = allocator.free_sequence(seq_id=1)
print(f"Freed {freed} blocks")

# Test 4: Check utilization
util = allocator.get_utilization()
print(f"Memory utilization: {util:.2%}")
```

### Exercise 2: Memory Savings Calculator

Calculate memory savings for different scenarios:

```python
def calculate_memory_savings(
    num_sequences: int,
    avg_length: int,
    max_length: int,
    block_size: int,
    memory_per_token: int  # in bytes
) -> dict:
    """
    TODO: Calculate memory usage for traditional vs. PagedAttention

    Returns dictionary with:
    - traditional_memory: Total memory with traditional approach
    - paged_memory: Total memory with PagedAttention
    - savings_bytes: Absolute savings
    - savings_percent: Percentage savings
    - utilization_traditional: Memory utilization (traditional)
    - utilization_paged: Memory utilization (PagedAttention)
    """
    pass


# Test with different scenarios
scenarios = [
    {"name": "Short sequences", "num_seqs": 100, "avg_len": 256, "max_len": 2048},
    {"name": "Medium sequences", "num_seqs": 100, "avg_len": 1024, "max_len": 2048},
    {"name": "Long sequences", "num_seqs": 100, "avg_len": 1800, "max_len": 2048},
    {"name": "Mixed workload", "num_seqs": 200, "avg_len": 800, "max_len": 4096},
]

for scenario in scenarios:
    result = calculate_memory_savings(
        num_sequences=scenario["num_seqs"],
        avg_length=scenario["avg_len"],
        max_length=scenario["max_len"],
        block_size=16,
        memory_per_token=1024  # 1 KB per token
    )
    print(f"\n{scenario['name']}:")
    print(f"  Traditional: {result['traditional_memory'] / 1e9:.2f} GB")
    print(f"  PagedAttention: {result['paged_memory'] / 1e9:.2f} GB")
    print(f"  Savings: {result['savings_percent']:.1f}%")
```

### Exercise 3: Block Gather Implementation

Implement the block gathering logic:

```python
import torch

def gather_kv_from_blocks(
    kv_cache: torch.Tensor,      # [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
    block_table: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
    seq_lengths: torch.Tensor,    # [num_seqs]
    layer_idx: int,
    kv_idx: int                   # 0 for K, 1 for V
) -> torch.Tensor:
    """
    TODO: Gather K or V cache from blocks for all sequences

    Args:
        kv_cache: Physical block storage
        block_table: Mapping from logical to physical blocks
        seq_lengths: Actual length of each sequence
        layer_idx: Which transformer layer
        kv_idx: 0 for keys, 1 for values

    Returns:
        Gathered tensor of shape [num_seqs, max_seq_len, num_heads, head_dim]
    """
    pass


# Test implementation
num_blocks = 100
num_layers = 4
block_size = 16
num_heads = 8
head_dim = 64

# Create dummy KV cache
kv_cache = torch.randn(num_blocks, num_layers, 2, block_size, num_heads, head_dim)

# Create block tables for 3 sequences
block_table = torch.tensor([
    [0, 1, 2, -1],    # Sequence 0: 3 blocks
    [3, 4, -1, -1],   # Sequence 1: 2 blocks
    [5, 6, 7, 8],     # Sequence 2: 4 blocks
])

seq_lengths = torch.tensor([40, 25, 60])

# Gather keys for layer 0
keys = gather_kv_from_blocks(kv_cache, block_table, seq_lengths, layer_idx=0, kv_idx=0)

print(f"Gathered keys shape: {keys.shape}")
# Expected: [3, max_length, num_heads, head_dim]
```

### Exercise 4: PagedAttention Performance Analysis

Analyze performance characteristics:

```python
def analyze_paged_attention_performance(
    sequence_lengths: List[int],
    block_size: int,
    max_sequence_length: int,
    compute_time_per_token: float,  # ms
    gather_overhead_per_block: float  # ms
) -> dict:
    """
    TODO: Analyze performance metrics for PagedAttention

    Calculate:
    1. Number of blocks per sequence
    2. Total gather overhead
    3. Compute time
    4. Total latency
    5. Memory utilization
    6. Compare with traditional approach

    Returns dictionary with performance metrics
    """
    pass


# Test with realistic sequence lengths
sequences = [
    512, 1024, 256, 2000, 800, 1500, 300, 1800,
    600, 1200, 450, 1600, 900, 1300, 700, 1700
]

metrics = analyze_paged_attention_performance(
    sequence_lengths=sequences,
    block_size=16,
    max_sequence_length=2048,
    compute_time_per_token=0.1,  # ms
    gather_overhead_per_block=0.01  # ms
)

print("Performance Analysis:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

### Exercise 5: Explore vLLM Implementation

Explore the actual vLLM codebase:

**Tasks:**

1. Read `/vllm/v1/core/block_pool.py`:
   - Identify the main data structures
   - Find the allocation and deallocation methods
   - Understand reference counting mechanism

2. Examine `/vllm/v1/core/kv_cache_manager.py`:
   - How are block tables managed?
   - When are blocks allocated?
   - How does it handle OOM conditions?

3. Check configuration in `/vllm/config/`:
   - Find where block_size is configured
   - What's the default value?
   - Can it be changed by users?

**Deliverable:** Write a 300-word summary of how vLLM implements PagedAttention, including:
- Key data structures
- Allocation strategy
- Performance optimizations
- Configuration options

### Exercise 6: Implement Online Softmax

Implement the online softmax algorithm used in paged attention:

```python
import torch
import math

def online_softmax_attention(
    query: torch.Tensor,         # [num_heads, head_dim]
    key_blocks: List[torch.Tensor],  # List of [block_size, num_heads, head_dim]
    value_blocks: List[torch.Tensor],  # List of [block_size, num_heads, head_dim]
) -> torch.Tensor:
    """
    TODO: Implement online softmax attention computation

    Process blocks one at a time, maintaining running statistics:
    - Running max score
    - Running sum of exponentials
    - Running weighted sum of values

    Returns:
        Attention output [num_heads, head_dim]
    """
    pass


# Test implementation
num_heads = 8
head_dim = 64
block_size = 16

query = torch.randn(num_heads, head_dim)
key_blocks = [torch.randn(block_size, num_heads, head_dim) for _ in range(5)]
value_blocks = [torch.randn(block_size, num_heads, head_dim) for _ in range(5)]

output = online_softmax_attention(query, key_blocks, value_blocks)
print(f"Attention output shape: {output.shape}")

# Verify correctness by comparing with standard attention
def standard_attention(query, keys, values):
    """Standard attention for comparison"""
    scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1))  # [heads, 1, seq_len]
    scores = scores / math.sqrt(head_dim)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, values).squeeze(1)  # [heads, head_dim]
    return output

# Concatenate blocks for comparison
all_keys = torch.cat(key_blocks, dim=0)  # [total_len, heads, dim]
all_values = torch.cat(value_blocks, dim=0)

expected = standard_attention(query, all_keys, all_values)
difference = torch.abs(output - expected).max()
print(f"Max difference from standard attention: {difference:.6f}")
```

---

## Summary and Key Takeaways

### Core Innovation of PagedAttention

1. **Block-based memory organization**
   - Fixed-size blocks (typically 16 tokens)
   - Non-contiguous physical storage
   - Block tables for address translation

2. **Inspired by virtual memory**
   - Logical vs. physical addressing
   - Page tables → Block tables
   - Memory pool management

3. **Efficient attention computation**
   - Gather K, V from non-contiguous blocks
   - Fused kernels for performance
   - Online softmax for single-pass computation

### Memory Efficiency Gains

| Metric | Improvement |
|--------|-------------|
| Memory utilization | 50-60% → 95-98% |
| Internal fragmentation | ~40% → <2% |
| Wasted memory | ~50% → ~1% |
| Batch size capacity | 1× → 2-3× |

### Performance Characteristics

- **Throughput:** 2-4× improvement in real workloads
- **Latency:** Minimal overhead (<5%)
- **GPU utilization:** Maintained at 85%+
- **Scalability:** Linear with number of blocks

### Implementation Highlights

1. **Block Pool:** Manages free/allocated blocks
2. **KV Cache Manager:** Allocates blocks per sequence
3. **Block Tables:** Map logical to physical blocks
4. **Attention Kernels:** Fused gather + compute

### Enabling Capabilities

PagedAttention enables:

1. **Continuous batching** - Dynamic request scheduling
2. **Prefix sharing** - Copy-on-write for common prefixes
3. **Preemption** - Easy swapping of sequences
4. **Mixed workloads** - Efficient handling of variable lengths

### Key Formulas

Block allocation:
```
blocks_needed = ⌈sequence_length / block_size⌉
```

Internal fragmentation:
```
waste_per_sequence = block_size - (sequence_length mod block_size)
average_waste = block_size / 2
```

Memory savings:
```
savings = (max_length - ⌈avg_length/B⌉×B) / max_length
```

### Design Principles

1. **Separation of concerns:** Logical vs. physical memory
2. **Fixed-size allocation:** Eliminates external fragmentation
3. **Lazy allocation:** Allocate only when needed
4. **Reference counting:** Enable sharing without duplication

### Limitations

1. **Block size tradeoff:** Too large → fragmentation, too small → overhead
2. **Gather overhead:** Small cost for non-contiguous access
3. **Metadata:** Block tables add some memory overhead
4. **Complexity:** More complex than traditional approach

### Practical Recommendations

1. **Default block size (16)** works well for most workloads
2. **Monitor memory utilization** to verify benefits
3. **Consider workload characteristics** when tuning
4. **Use with continuous batching** for maximum benefit

### Preparation for Next Lesson

Next, we'll dive deep into **block tables and address translation**:

- Detailed block table structure
- Address translation mechanisms
- Optimization techniques
- Integration with attention computation

Make sure you understand:
- How blocks eliminate fragmentation
- The gather operation for attention
- Block allocation and deallocation
- Performance characteristics

---

## Additional Resources

### Papers
1. **vLLM Paper:** "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. **FlashAttention:** Memory-efficient attention mechanisms
3. **Virtual Memory:** OS concepts applicable to GPU memory

### Code References
- `/vllm/v1/core/block_pool.py` - Block allocation
- `/vllm/v1/core/kv_cache_manager.py` - KV cache management
- `/vllm/attention/` - Attention backends
- `/vllm/v1/worker/block_table.py` - Block table implementation

### Videos & Tutorials
- vLLM technical talks on PagedAttention
- CUDA memory management tutorials
- Operating systems virtual memory lectures

### Related Techniques
- FlashAttention for memory-efficient computation
- CUDA unified memory
- Sparse attention patterns

---

**Next Lesson:** [03_block_tables_address_translation.md](./03_block_tables_address_translation.md)

---

*Last Updated: 2025-11-19*
*Estimated Completion Time: 3-4 hours*
*Difficulty: Advanced*
