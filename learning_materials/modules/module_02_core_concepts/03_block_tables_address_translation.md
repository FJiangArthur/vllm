# Block Tables and Address Translation

## Module 2, Lesson 3: The Mapping Layer of PagedAttention

**Estimated Time:** 2-3 hours
**Difficulty:** Advanced
**Prerequisites:** Lessons 1-2, Understanding of hash tables and indexing

---

## Table of Contents

1. [Introduction](#introduction)
2. [Block Table Fundamentals](#block-table-fundamentals)
3. [Address Translation Mechanism](#address-translation-mechanism)
4. [Block Table Data Structures](#block-table-data-structures)
5. [Multi-Dimensional Indexing](#multi-dimensional-indexing)
6. [Efficient Address Computation](#efficient-address-computation)
7. [Block Table Operations](#block-table-operations)
8. [GPU Implementation](#gpu-implementation)
9. [Memory Layout Optimization](#memory-layout-optimization)
10. [Performance Analysis](#performance-analysis)
11. [Hands-On Exercises](#hands-on-exercises)
12. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction

Block tables are the critical mapping layer that enables PagedAttention to work efficiently. They translate logical KV cache positions (as seen by the model) into physical memory blocks (as stored in GPU memory).

Think of block tables as the "page tables" of virtual memory, but optimized for the unique access patterns of transformer attention.

### Learning Objectives

By the end of this lesson, you will:

1. Understand the structure and purpose of block tables
2. Master address translation from logical to physical
3. Implement efficient block table operations
4. Optimize memory layout for GPU access
5. Analyze performance characteristics
6. Navigate vLLM's block table implementation

---

## Block Table Fundamentals

### What is a Block Table?

A **block table** is a mapping from logical block indices to physical block identifiers for a single sequence.

```
Logical View (What the model sees):
┌────────┬────────┬────────┬────────┬────────┐
│ Block 0│ Block 1│ Block 2│ Block 3│ Block 4│
└────────┴────────┴────────┴────────┴────────┘
Tokens:  0-15     16-31    32-47    48-63    64-79

Block Table (The mapping):
Logical → Physical
  0     →    42
  1     →    17
  2     →    93
  3     →    8
  4     →    56

Physical Memory (How it's actually stored):
Block 8 ────┐
Block 17 ───┼───┐
Block 42 ───┼───┼───┐
Block 56 ───┼───┼───┼───┐
Block 93 ───┼───┼───┼───┼───┐
            │   │   │   │   │
    [.......|...|...|...|...]  GPU Memory
```

### Key Properties

1. **Per-sequence mapping:** Each sequence has its own block table
2. **Variable length:** Table grows as sequence generates more tokens
3. **Index-based:** Fast O(1) lookup by logical block index
4. **Compact:** Small memory footprint (just integers)

### Block Table vs. Page Table

| Feature | OS Page Table | vLLM Block Table |
|---------|--------------|------------------|
| Purpose | Virtual→Physical address | Logical→Physical block |
| Scope | Per process | Per sequence |
| Size | Large (many entries) | Small (≤max_seq_len/block_size) |
| Access pattern | Random | Sequential during attention |
| Hardware support | MMU/TLB | GPU kernel logic |

---

## Address Translation Mechanism

### Three-Level Translation

To access a KV value, we translate through three levels:

```
Level 1: Token Position → Logical Block + Offset
Level 2: Logical Block → Physical Block (via block table)
Level 3: Physical Block + Offset → Physical Memory Address
```

### Level 1: Token to Logical Block

Given token position `t` and block size `B`:

```python
def token_to_logical_block(token_position: int, block_size: int) -> tuple:
    """Convert token position to logical block and offset"""
    logical_block_idx = token_position // block_size
    block_offset = token_position % block_size
    return logical_block_idx, block_offset
```

**Example:**
```
Token position: 45
Block size: 16

Logical block: 45 // 16 = 2
Block offset: 45 % 16 = 13

Token 45 is at position 13 in logical block 2
```

### Level 2: Logical to Physical Block

Use the block table to map logical to physical:

```python
def logical_to_physical_block(
    logical_block_idx: int,
    block_table: List[int]
) -> int:
    """Map logical block index to physical block ID"""
    if logical_block_idx >= len(block_table):
        raise IndexError(f"Logical block {logical_block_idx} not allocated")

    physical_block_id = block_table[logical_block_idx]
    return physical_block_id
```

**Example:**
```
Logical block: 2
Block table: [42, 17, 93, 8, 56]

Physical block: block_table[2] = 93
```

### Level 3: Physical Block to Memory Address

Calculate the actual memory address in the KV cache tensor:

```python
def physical_block_to_address(
    physical_block_id: int,
    block_offset: int,
    layer_idx: int,
    kv_idx: int,  # 0 for K, 1 for V
    head_idx: int,
    kv_cache_shape: tuple
) -> tuple:
    """
    Calculate memory address in KV cache tensor

    kv_cache shape: [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
    """
    num_blocks, num_layers, _, block_size, num_heads, head_dim = kv_cache_shape

    # Indices into the KV cache tensor
    indices = (
        physical_block_id,  # Which block
        layer_idx,          # Which layer
        kv_idx,            # K or V
        block_offset,      # Position within block
        head_idx,          # Which attention head
        slice(None)        # All dimensions of head
    )

    return indices
```

### Complete Translation Example

```python
def access_kv_value(
    token_position: int,
    layer_idx: int,
    kv_idx: int,
    head_idx: int,
    block_table: List[int],
    kv_cache: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    """Complete address translation and access"""

    # Level 1: Token → Logical block + offset
    logical_block_idx = token_position // block_size
    block_offset = token_position % block_size

    # Level 2: Logical → Physical block
    physical_block_id = block_table[logical_block_idx]

    # Level 3: Physical block → Memory address
    kv_value = kv_cache[
        physical_block_id,
        layer_idx,
        kv_idx,
        block_offset,
        head_idx,
        :  # All head dimensions
    ]

    return kv_value
```

**Complete Example:**
```
Given:
- Token position: 45
- Layer: 2
- KV: Keys (0)
- Head: 5
- Block size: 16
- Block table: [42, 17, 93, 8, 56]

Translation:
1. Logical block: 45 // 16 = 2, Offset: 45 % 16 = 13
2. Physical block: block_table[2] = 93
3. Memory address: kv_cache[93, 2, 0, 13, 5, :]

Result: Key vector for token 45, layer 2, head 5
```

---

## Block Table Data Structures

### Basic Block Table

```python
class BlockTable:
    """Simple block table for a single sequence"""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: List[int] = []  # Physical block IDs

    def append_block(self, physical_block_id: int) -> None:
        """Add a new block to this sequence"""
        self.blocks.append(physical_block_id)

    def get_physical_block(self, logical_idx: int) -> int:
        """Get physical block ID for logical index"""
        return self.blocks[logical_idx]

    def num_blocks(self) -> int:
        """Get number of allocated blocks"""
        return len(self.blocks)

    def capacity(self) -> int:
        """Get total token capacity"""
        return len(self.blocks) * self.block_size

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for GPU usage"""
        return torch.tensor(self.blocks, dtype=torch.int32)
```

### Batch Block Tables

For efficient GPU processing, we organize all block tables into a single tensor:

```python
class BatchBlockTables:
    """Block tables for a batch of sequences"""

    def __init__(self, max_num_sequences: int, max_num_blocks_per_seq: int):
        self.max_num_sequences = max_num_sequences
        self.max_num_blocks_per_seq = max_num_blocks_per_seq

        # Shape: [max_num_sequences, max_num_blocks_per_seq]
        # Use -1 to indicate invalid/empty slots
        self.block_tables = torch.full(
            (max_num_sequences, max_num_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device='cuda'
        )

        self.sequence_lengths = torch.zeros(
            max_num_sequences,
            dtype=torch.int32,
            device='cuda'
        )

    def set_block_table(self, seq_idx: int, blocks: List[int]) -> None:
        """Set block table for a sequence"""
        num_blocks = len(blocks)
        assert num_blocks <= self.max_num_blocks_per_seq

        self.block_tables[seq_idx, :num_blocks] = torch.tensor(
            blocks, dtype=torch.int32, device='cuda'
        )
        # Mark remaining slots as invalid
        if num_blocks < self.max_num_blocks_per_seq:
            self.block_tables[seq_idx, num_blocks:] = -1

    def get_block_table(self, seq_idx: int) -> torch.Tensor:
        """Get block table for a sequence"""
        return self.block_tables[seq_idx]

    def update_block(self, seq_idx: int, logical_idx: int, physical_block: int) -> None:
        """Update a single block mapping"""
        self.block_tables[seq_idx, logical_idx] = physical_block
```

### Optimized Structure from vLLM

```python
# From vllm/v1/worker/block_table.py (simplified)
class BlockTable:
    def __init__(self, max_num_blocks: int, block_size: int):
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.blocks = torch.empty(max_num_blocks, dtype=torch.int32)
        self.num_blocks = 0

    def allocate(self, physical_block_id: int) -> int:
        """Allocate a new block, return logical index"""
        if self.num_blocks >= self.max_num_blocks:
            raise ValueError("Block table full")

        logical_idx = self.num_blocks
        self.blocks[logical_idx] = physical_block_id
        self.num_blocks += 1
        return logical_idx

    def reset(self) -> None:
        """Reset block table for reuse"""
        self.num_blocks = 0
```

---

## Multi-Dimensional Indexing

### KV Cache Tensor Dimensions

The KV cache has 6 dimensions:

```
Shape: [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
        │           │           │   │           │          │
        │           │           │   │           │          └─ Head dimension (64-128)
        │           │           │   │           └──────────── Attention heads (32-64)
        │           │           │   └──────────────────────── Tokens per block (16)
        │           │           └──────────────────────────── K and V (2)
        │           └──────────────────────────────────────── Transformer layers (32-80)
        └──────────────────────────────────────────────────── Physical blocks (variable)
```

### Index Calculation

To access element `[b, l, k, t, h, d]`:

```python
def calculate_linear_index(
    block_id: int,
    layer_id: int,
    kv_id: int,
    token_in_block: int,
    head_id: int,
    dim_id: int,
    shape: tuple
) -> int:
    """
    Calculate linear index in flattened memory

    Shape: [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
    """
    num_blocks, num_layers, _, block_size, num_heads, head_dim = shape

    # Stride for each dimension
    stride_block = num_layers * 2 * block_size * num_heads * head_dim
    stride_layer = 2 * block_size * num_heads * head_dim
    stride_kv = block_size * num_heads * head_dim
    stride_token = num_heads * head_dim
    stride_head = head_dim
    stride_dim = 1

    linear_idx = (
        block_id * stride_block +
        layer_id * stride_layer +
        kv_id * stride_kv +
        token_in_block * stride_token +
        head_id * stride_head +
        dim_id * stride_dim
    )

    return linear_idx
```

### Vectorized Access

For efficiency, we often access entire vectors or matrices:

```python
def access_full_key_vector(
    block_id: int,
    layer_id: int,
    token_in_block: int,
    head_id: int,
    kv_cache: torch.Tensor
) -> torch.Tensor:
    """
    Access complete key vector for a token
    Returns: [head_dim]
    """
    return kv_cache[block_id, layer_id, 0, token_in_block, head_id, :]


def access_all_heads_for_token(
    block_id: int,
    layer_id: int,
    token_in_block: int,
    kv_cache: torch.Tensor
) -> torch.Tensor:
    """
    Access all heads for a token
    Returns: [2, num_heads, head_dim]  (K and V for all heads)
    """
    return kv_cache[block_id, layer_id, :, token_in_block, :, :]


def access_full_block(
    block_id: int,
    layer_id: int,
    kv_cache: torch.Tensor
) -> torch.Tensor:
    """
    Access entire block
    Returns: [2, block_size, num_heads, head_dim]
    """
    return kv_cache[block_id, layer_id, :, :, :, :]
```

---

## Efficient Address Computation

### Precomputed Strides

To avoid repeated calculations, we precompute strides:

```python
class KVCacheIndexer:
    """Efficient indexing into KV cache with precomputed strides"""

    def __init__(self, kv_cache_shape: tuple):
        self.shape = kv_cache_shape
        num_blocks, num_layers, _, block_size, num_heads, head_dim = kv_cache_shape

        # Precompute strides
        self.stride_block = num_layers * 2 * block_size * num_heads * head_dim
        self.stride_layer = 2 * block_size * num_heads * head_dim
        self.stride_kv = block_size * num_heads * head_dim
        self.stride_token = num_heads * head_dim
        self.stride_head = head_dim

    def compute_offset(
        self,
        block_id: int,
        layer_id: int,
        kv_id: int,
        token_in_block: int,
        head_id: int
    ) -> int:
        """Compute offset for base of head_dim vector"""
        return (
            block_id * self.stride_block +
            layer_id * self.stride_layer +
            kv_id * self.stride_kv +
            token_in_block * self.stride_token +
            head_id * self.stride_head
        )
```

### Batch Translation

Translate token positions for entire batches:

```python
def batch_translate_addresses(
    token_positions: torch.Tensor,  # [batch_size]
    block_tables: torch.Tensor,     # [batch_size, max_blocks]
    block_size: int
) -> tuple:
    """
    Translate token positions to physical blocks for entire batch

    Returns:
        physical_blocks: [batch_size]
        offsets: [batch_size]
    """
    # Compute logical blocks and offsets (vectorized)
    logical_blocks = token_positions // block_size
    offsets = token_positions % block_size

    # Gather physical blocks (batch_size lookups)
    batch_indices = torch.arange(token_positions.size(0), device=token_positions.device)
    physical_blocks = block_tables[batch_indices, logical_blocks]

    return physical_blocks, offsets
```

### CUDA Kernel for Translation

```cuda
// Efficient CUDA kernel for address translation
__global__ void translate_and_gather_kernel(
    const float* kv_cache,        // [num_blocks, layers, 2, block_size, heads, dim]
    const int* block_tables,      // [num_seqs, max_blocks]
    const int* token_positions,   // [num_seqs]
    float* output,                // [num_seqs, heads, dim]
    int block_size,
    int num_heads,
    int head_dim,
    int layer_idx,
    int kv_idx
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (dim_idx >= head_dim) return;

    // Level 1: Token → Logical block + offset
    int token_pos = token_positions[seq_idx];
    int logical_block = token_pos / block_size;
    int block_offset = token_pos % block_size;

    // Level 2: Logical → Physical block
    int physical_block = block_tables[seq_idx * max_blocks + logical_block];

    if (physical_block < 0) return;  // Invalid block

    // Level 3: Compute linear index
    int idx = physical_block * stride_block +
              layer_idx * stride_layer +
              kv_idx * stride_kv +
              block_offset * stride_token +
              head_idx * stride_head +
              dim_idx;

    // Gather value
    output[seq_idx * num_heads * head_dim + head_idx * head_dim + dim_idx] = kv_cache[idx];
}
```

---

## Block Table Operations

### Appending Blocks

When a sequence grows and needs more blocks:

```python
def append_block_to_sequence(
    seq_id: int,
    physical_block_id: int,
    block_tables: Dict[int, List[int]]
) -> int:
    """
    Append a new block to sequence's block table

    Returns:
        logical_block_idx: The logical index of the new block
    """
    if seq_id not in block_tables:
        block_tables[seq_id] = []

    logical_idx = len(block_tables[seq_id])
    block_tables[seq_id].append(physical_block_id)

    return logical_idx
```

### Removing Blocks

When a sequence completes or is preempted:

```python
def remove_sequence_blocks(
    seq_id: int,
    block_tables: Dict[int, List[int]],
    block_pool: BlockPool
) -> List[int]:
    """
    Remove all blocks for a sequence

    Returns:
        freed_blocks: List of physical block IDs that were freed
    """
    if seq_id not in block_tables:
        return []

    freed_blocks = block_tables[seq_id].copy()

    # Return blocks to pool
    for block_id in freed_blocks:
        block_pool.free(block_id)

    # Remove block table
    del block_tables[seq_id]

    return freed_blocks
```

### Copying Block Tables

For prefix sharing (copy-on-write):

```python
def copy_block_table(
    src_seq_id: int,
    dst_seq_id: int,
    num_blocks_to_copy: int,
    block_tables: Dict[int, List[int]],
    block_pool: BlockPool
) -> None:
    """
    Copy block table from source to destination sequence
    Used for prefix sharing

    Args:
        src_seq_id: Source sequence
        dst_seq_id: Destination sequence
        num_blocks_to_copy: Number of blocks to share
    """
    if src_seq_id not in block_tables:
        raise ValueError(f"Source sequence {src_seq_id} not found")

    src_blocks = block_tables[src_seq_id]

    if num_blocks_to_copy > len(src_blocks):
        raise ValueError("Requested more blocks than available")

    # Share blocks (increment reference counts)
    shared_blocks = src_blocks[:num_blocks_to_copy]
    for block_id in shared_blocks:
        block_pool.add_ref(block_id)

    # Create new block table
    block_tables[dst_seq_id] = shared_blocks.copy()
```

### Updating Block Tables GPU-Side

```python
def update_block_tables_gpu(
    block_tables_tensor: torch.Tensor,  # [num_seqs, max_blocks]
    seq_indices: torch.Tensor,          # [num_updates]
    logical_indices: torch.Tensor,      # [num_updates]
    physical_blocks: torch.Tensor       # [num_updates]
) -> None:
    """
    Update multiple block table entries on GPU

    Args:
        block_tables_tensor: Block tables for all sequences
        seq_indices: Which sequences to update
        logical_indices: Which logical block in each sequence
        physical_blocks: New physical block IDs
    """
    block_tables_tensor[seq_indices, logical_indices] = physical_blocks
```

---

## GPU Implementation

### Memory Layout for Coalesced Access

To maximize GPU memory bandwidth, we layout block tables for coalesced access:

```
Bad layout (strided access):
Seq 0: [b0, b1, b2, b3, ...]
Seq 1: [b0, b1, b2, b3, ...]
Seq 2: [b0, b1, b2, b3, ...]
       ↑   ↑   ↑   ↑
       Thread 0 accesses these (strided!)

Good layout (coalesced access):
Block 0: [seq0_b0, seq1_b0, seq2_b0, seq3_b0, ...]
Block 1: [seq0_b1, seq1_b1, seq2_b1, seq3_b1, ...]
         ↑        ↑        ↑        ↑
         Consecutive memory accesses (coalesced!)
```

However, vLLM uses the "bad" layout because:
1. Logical block lookups are infrequent (once per block, not per token)
2. Simpler indexing logic
3. More flexible for variable-length sequences

### Shared Memory Optimization

Cache block table in shared memory for repeated access:

```cuda
__global__ void attention_with_cached_block_table(
    const float* kv_cache,
    const int* block_tables,
    float* output,
    int num_seqs,
    int max_blocks,
    int block_size
) {
    // Shared memory for block table
    __shared__ int shared_block_table[MAX_BLOCKS_PER_SEQ];

    int seq_idx = blockIdx.x;

    // Load block table into shared memory (cooperative)
    for (int i = threadIdx.x; i < max_blocks; i += blockDim.x) {
        shared_block_table[i] = block_tables[seq_idx * max_blocks + i];
    }
    __syncthreads();

    // Now all threads can access block table from fast shared memory
    int token_pos = /* ... */;
    int logical_block = token_pos / block_size;
    int physical_block = shared_block_table[logical_block];  // Fast!

    // ... rest of attention computation
}
```

### Register Optimization

For small block tables, store in registers:

```cuda
__device__ inline int translate_address_cached(
    int token_pos,
    int block_size,
    int block0,  // Cached in register
    int block1,
    int block2,
    int block3
) {
    int logical_block = token_pos / block_size;

    // Branchless selection (if small number of blocks)
    int physical_block =
        (logical_block == 0) * block0 +
        (logical_block == 1) * block1 +
        (logical_block == 2) * block2 +
        (logical_block == 3) * block3;

    return physical_block;
}
```

---

## Memory Layout Optimization

### Row-Major vs. Column-Major

The order of dimensions matters for access patterns:

```python
# Option 1: Block-first layout
kv_cache_1 = torch.zeros(
    num_blocks, num_layers, 2, block_size, num_heads, head_dim
)
# Pro: Easy to allocate/deallocate blocks
# Con: Layer access is strided

# Option 2: Layer-first layout
kv_cache_2 = torch.zeros(
    num_layers, num_blocks, 2, block_size, num_heads, head_dim
)
# Pro: Layer access is contiguous
# Con: Block management is complex

# vLLM uses Option 1 for simplicity
```

### Cache Line Alignment

Align blocks to cache line boundaries (typically 128 bytes):

```python
def calculate_aligned_block_size(
    block_size: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int,
    cache_line_size: int = 128
) -> int:
    """Calculate block size aligned to cache lines"""

    # Size of one token's KV data
    token_size = 2 * num_heads * head_dim * dtype_size  # K and V

    # Size of block
    block_bytes = block_size * token_size

    # Align to cache line
    aligned_bytes = ((block_bytes + cache_line_size - 1) // cache_line_size) * cache_line_size

    return aligned_bytes
```

### Memory Pool Organization

Organize physical blocks for locality:

```
Memory Pool Layout:

┌─────────────────────────────────────┐
│ Block 0                              │
├─────────────────────────────────────┤
│ Block 1                              │
├─────────────────────────────────────┤
│ Block 2                              │
├─────────────────────────────────────┤
│ ...                                  │
├─────────────────────────────────────┤
│ Block N-1                            │
└─────────────────────────────────────┘

Each block is aligned and same size
Easy to calculate address: base + block_id * block_size
```

---

## Performance Analysis

### Translation Overhead

**Analysis:** How much overhead does address translation add?

```
Traditional attention:
- Direct memory access: O(1) arithmetic
- Cache access: ~10-100 cycles

Paged attention:
- Address translation: O(1) table lookup
- Block table access: ~10-100 cycles (if in cache)
- Cache access: ~10-100 cycles

Total overhead: ~2× memory access time
```

**Mitigation strategies:**

1. **Cache block tables:** Keep frequently accessed tables in L1/L2 cache
2. **Fused kernels:** Amortize translation across many operations
3. **Prefetching:** Load block table entries before needed

### Memory Bandwidth

Block table access adds minimal bandwidth:

```
Per-token attention computation:
- KV cache access: block_size × head_dim × sizeof(dtype) ≈ 16 × 128 × 2 = 4 KB
- Block table access: 1 × sizeof(int32) = 4 bytes

Block table overhead: 4 / 4096 = 0.1%
```

**Negligible impact on bandwidth!**

### Latency Measurements

Benchmark of address translation:

```python
def benchmark_address_translation():
    num_seqs = 1000
    block_size = 16
    max_blocks = 128

    # Generate random block tables
    block_tables = torch.randint(0, 10000, (num_seqs, max_blocks), device='cuda')
    token_positions = torch.randint(0, 2048, (num_seqs,), device='cuda')

    # Warmup
    for _ in range(100):
        logical_blocks = token_positions // block_size
        batch_idx = torch.arange(num_seqs, device='cuda')
        physical_blocks = block_tables[batch_idx, logical_blocks]

    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        logical_blocks = token_positions // block_size
        batch_idx = torch.arange(num_seqs, device='cuda')
        physical_blocks = block_tables[batch_idx, logical_blocks]
    end.record()

    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)

    print(f"Average translation time: {elapsed / 1000:.3f} ms")
    print(f"Per-sequence: {elapsed / 1000 / num_seqs * 1000:.1f} μs")

# Typical results on A100:
# Average translation time: 0.050 ms
# Per-sequence: 0.05 μs
```

**Conclusion:** Address translation overhead is ~50μs for 1000 sequences - negligible!

---

## Hands-On Exercises

### Exercise 1: Implement Block Table

```python
class MyBlockTable:
    """Implement a complete block table"""

    def __init__(self, block_size: int):
        """
        TODO: Initialize block table
        - Store block size
        - Initialize empty list of blocks
        - Track number of allocated blocks
        """
        pass

    def append_block(self, physical_block_id: int) -> int:
        """
        TODO: Append a new physical block

        Returns:
            logical_block_idx: The logical index assigned
        """
        pass

    def translate(self, token_position: int) -> tuple:
        """
        TODO: Translate token position to physical block and offset

        Returns:
            (physical_block_id, offset_in_block)

        Raises:
            ValueError: If token position is beyond allocated capacity
        """
        pass

    def capacity(self) -> int:
        """
        TODO: Return total token capacity
        """
        pass

    def to_tensor(self) -> torch.Tensor:
        """
        TODO: Convert to PyTorch tensor for GPU use
        """
        pass


# Test your implementation
bt = MyBlockTable(block_size=16)
bt.append_block(42)
bt.append_block(17)
bt.append_block(93)

# Test translation
physical, offset = bt.translate(token_position=25)
print(f"Token 25: Block {physical}, Offset {offset}")  # Should be (17, 9)

# Test capacity
print(f"Capacity: {bt.capacity()}")  # Should be 48
```

### Exercise 2: Batch Translation

```python
def batch_translate(
    token_positions: torch.Tensor,  # [batch_size]
    block_tables: torch.Tensor,     # [batch_size, max_blocks]
    block_size: int
) -> tuple:
    """
    TODO: Implement batch address translation

    Returns:
        physical_blocks: [batch_size]
        offsets: [batch_size]

    Hint: Use vectorized operations (no loops!)
    """
    pass


# Test
block_size = 16
batch_size = 5
max_blocks = 10

block_tables = torch.tensor([
    [42, 17, 93, 8, -1, -1, -1, -1, -1, -1],
    [100, 101, 102, -1, -1, -1, -1, -1, -1, -1],
    [50, 51, 52, 53, 54, -1, -1, -1, -1, -1],
    [200, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [75, 76, 77, 78, -1, -1, -1, -1, -1, -1],
])

token_positions = torch.tensor([25, 10, 50, 5, 35])

physical, offsets = batch_translate(token_positions, block_tables, block_size)
print(f"Physical blocks: {physical}")
print(f"Offsets: {offsets}")
```

### Exercise 3: KV Cache Access

```python
def access_kv_cache_with_block_table(
    kv_cache: torch.Tensor,         # [num_blocks, layers, 2, block_size, heads, dim]
    block_table: torch.Tensor,      # [max_blocks]
    token_position: int,
    layer_idx: int,
    kv_idx: int,                    # 0=K, 1=V
    head_idx: int,
    block_size: int
) -> torch.Tensor:
    """
    TODO: Access KV cache using block table translation

    Returns:
        KV vector of shape [head_dim]
    """
    pass


# Test with real data
num_blocks = 100
num_layers = 4
block_size = 16
num_heads = 8
head_dim = 64

kv_cache = torch.randn(num_blocks, num_layers, 2, block_size, num_heads, head_dim)
block_table = torch.tensor([42, 17, 93, 8])

# Access token 25, layer 2, keys, head 5
kv_vector = access_kv_cache_with_block_table(
    kv_cache, block_table, token_position=25,
    layer_idx=2, kv_idx=0, head_idx=5, block_size=16
)

print(f"KV vector shape: {kv_vector.shape}")  # Should be [64]
```

### Exercise 4: Performance Comparison

```python
import time

def benchmark_access_patterns():
    """
    TODO: Benchmark different access patterns

    Compare:
    1. Direct contiguous access (traditional)
    2. Block table translation (PagedAttention)
    3. Random access (worst case)

    Measure:
    - Throughput (accesses per second)
    - Latency (time per access)
    - Memory bandwidth utilization
    """
    pass

# Run benchmark and plot results
benchmark_access_patterns()
```

### Exercise 5: Explore vLLM Implementation

**Tasks:**

1. Read `/vllm/v1/worker/block_table.py`:
   - How is the block table structured?
   - What operations are supported?
   - How is it integrated with attention?

2. Find the block table tensor creation:
   - Where is it allocated?
   - What's the shape?
   - How is it passed to kernels?

3. Trace an address translation:
   - Start from token position
   - Follow through block table lookup
   - End at KV cache access

**Deliverable:** Write 200-300 words explaining vLLM's block table implementation.

### Exercise 6: Optimize Block Table Access

```python
class OptimizedBlockTable:
    """
    TODO: Implement an optimized block table

    Optimizations to implement:
    1. Cache recently accessed translations
    2. Prefetch likely-to-be-accessed blocks
    3. Batch multiple lookups
    4. Use GPU for parallel translation

    Compare performance with naive implementation
    """
    pass
```

---

## Summary and Key Takeaways

### Block Table Fundamentals

1. **Purpose:** Map logical blocks to physical blocks per sequence
2. **Structure:** Simple array/list of physical block IDs
3. **Scope:** One block table per sequence
4. **Size:** O(sequence_length / block_size) entries

### Address Translation

Three-level translation:
```
Token Position → (Logical Block, Offset) → Physical Block → Memory Address
```

Complexity: O(1) for all operations

### Key Operations

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Append block | O(1) | O(1) |
| Translate address | O(1) | O(1) |
| Remove sequence | O(num_blocks) | O(1) |
| Batch translation | O(batch_size) | O(batch_size) |

### Performance Characteristics

- **Translation overhead:** <0.1% of total computation
- **Memory overhead:** ~4 bytes per block (negligible)
- **Bandwidth impact:** <0.1% (block table << KV cache)
- **Latency:** ~50μs for 1000 translations on A100

### Implementation Highlights

1. **Batched structure:** All block tables in single tensor for GPU efficiency
2. **Precomputed strides:** Avoid repeated calculations
3. **Shared memory caching:** For frequently accessed tables
4. **Vectorized operations:** Parallel translation for entire batches

### Design Principles

1. **Simplicity:** Direct indexing, no complex data structures
2. **Efficiency:** O(1) operations, minimal overhead
3. **Scalability:** Works for any number of sequences
4. **GPU-friendly:** Coalesced access patterns where possible

### Integration with PagedAttention

```
Block Table enables:
✓ Non-contiguous memory allocation
✓ Dynamic sequence length
✓ Efficient memory utilization
✓ Fast allocation/deallocation
✓ Prefix sharing (copy-on-write)
```

### Trade-offs

**Advantages:**
- Simple and fast O(1) translation
- Minimal memory overhead
- Easy to implement and maintain
- Flexible for variable lengths

**Disadvantages:**
- Slight translation overhead vs. direct access
- Block table must fit in memory
- More complex than contiguous allocation

### Preparation for Next Lesson

Next: **Memory Fragmentation Solutions**

Topics:
- How PagedAttention eliminates internal fragmentation
- External fragmentation management
- Memory pool organization
- Allocation strategies

Prerequisites:
- Understand block table structure
- Know address translation mechanism
- Familiar with memory fragmentation concepts

---

## Additional Resources

### Code References
- `/vllm/v1/worker/block_table.py` - Block table implementation
- `/vllm/v1/core/block_pool.py` - Block pool management
- `/vllm/v1/core/kv_cache_utils.py` - KV cache utilities

### Papers
- **vLLM Paper:** PagedAttention details
- **Virtual Memory:** OS textbooks for page table concepts
- **GPU Memory Management:** CUDA programming guides

### Related Topics
- Virtual memory and page tables
- Hash tables and indexing
- GPU memory hierarchy
- Cache optimization techniques

---

**Next Lesson:** [04_memory_fragmentation_solutions.md](./04_memory_fragmentation_solutions.md)

---

*Last Updated: 2025-11-19*
*Estimated Completion Time: 2-3 hours*
*Difficulty: Advanced*
