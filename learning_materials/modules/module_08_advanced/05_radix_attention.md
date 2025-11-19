# Module 8.5: RadixAttention - Efficient KV Cache Management

## Overview

RadixAttention is vLLM's sophisticated attention mechanism that combines Radix trees with PagedAttention to enable efficient prefix caching and KV cache sharing. This module provides a deep dive into the implementation, optimizations, and production deployment of RadixAttention.

**Learning Objectives:**
- Understand the integration of Radix trees with PagedAttention
- Implement efficient KV cache sharing across requests
- Master block-level cache management and eviction
- Optimize RadixAttention for different workload patterns
- Deploy RadixAttention in production systems

**Prerequisites:**
- Completion of Module 2 (PagedAttention)
- Module 8.4 (Prefix Caching)
- Understanding of attention mechanisms
- Knowledge of memory management

**Estimated Time:** 3-4 hours

---

## 1. RadixAttention Architecture

### 1.1 Combining Radix Trees with PagedAttention

```python
"""
RadixAttention Architecture

Traditional PagedAttention:
- KV cache stored in physical blocks
- Each request gets its own blocks
- No sharing between requests

RadixAttention Enhancement:
- Radix tree tracks common prefixes
- Blocks shared for common prefix parts
- Copy-on-write for diverging suffixes
"""

class RadixAttentionOverview:
    """
    Conceptual overview of RadixAttention
    """

    def __init__(self):
        self.block_size = 16  # Tokens per block
        self.num_blocks = 1000  # Total available blocks

    def explain_integration(self):
        """
        Explain how Radix tree integrates with PagedAttention
        """
        explanation = """
        Integration Points:

        1. Block Allocation:
           - PagedAttention: Allocate new blocks for each request
           - RadixAttention: Check Radix tree first, reuse if prefix exists

        2. Block Mapping:
           - PagedAttention: Virtual → Physical block mapping per sequence
           - RadixAttention: Multiple sequences can map to same physical blocks (read-only)

        3. Copy-on-Write:
           - When sequences diverge, allocate new blocks only for diverging parts
           - Prefix blocks remain shared

        4. Memory Efficiency:
           - Shared blocks counted only once
           - Total memory = (unique blocks) × (block size)
        """
        return explanation

    def visualize_block_sharing(self):
        """
        Visualize how blocks are shared in RadixAttention
        """
        print("Block Sharing Example:")
        print("=" * 60)

        # Three requests with shared prefix
        requests = {
            "Request 1": "Common prefix [Blocks 0-9] + Unique suffix [Blocks 10-11]",
            "Request 2": "Common prefix [Blocks 0-9] + Unique suffix [Blocks 12-13]",
            "Request 3": "Common prefix [Blocks 0-9] + Unique suffix [Blocks 14-15]"
        }

        print("\nWithout Sharing (Traditional PagedAttention):")
        print("  Request 1: Blocks 0-11 (12 blocks)")
        print("  Request 2: Blocks 12-23 (12 blocks)")
        print("  Request 3: Blocks 24-35 (12 blocks)")
        print("  Total: 36 blocks")

        print("\nWith Sharing (RadixAttention):")
        print("  Shared prefix: Blocks 0-9 (10 blocks)")
        print("  Request 1 suffix: Blocks 10-11 (2 blocks)")
        print("  Request 2 suffix: Blocks 12-13 (2 blocks)")
        print("  Request 3 suffix: Blocks 14-15 (2 blocks)")
        print("  Total: 16 blocks")

        print(f"\nMemory saved: {(36-16)/36*100:.1f}%")

overview = RadixAttentionOverview()
print(overview.explain_integration())
overview.visualize_block_sharing()
```

### 1.2 Block-Level Radix Tree

```python
import torch
from typing import List, Optional, Dict, Tuple

class RadixBlockNode:
    """
    Node in block-level Radix tree

    Each node represents a sequence of KV cache blocks
    """

    def __init__(self, block_ids: List[int], is_leaf: bool = False):
        """
        Args:
            block_ids: Physical block IDs for this edge
            is_leaf: Whether this is a leaf node (complete sequence)
        """
        self.block_ids = block_ids
        self.is_leaf = is_leaf
        self.children: Dict[int, 'RadixBlockNode'] = {}  # Map from first block ID to child

        # Metadata
        self.ref_count = 0  # Number of active sequences using this node
        self.last_access_time = 0
        self.total_accesses = 0

        # Cache status
        self.blocks_resident = True  # Whether blocks are in GPU memory

    def __repr__(self):
        return f"RadixBlockNode(blocks={self.block_ids[:3]}..., leaf={self.is_leaf}, refs={self.ref_count})"

class BlockLevelRadixTree:
    """
    Radix tree for managing KV cache blocks
    """

    def __init__(self, block_manager):
        """
        Args:
            block_manager: PagedAttention block manager
        """
        self.root = RadixBlockNode(block_ids=[], is_leaf=False)
        self.block_manager = block_manager

        # Statistics
        self.total_nodes = 1
        self.shared_blocks = 0
        self.unique_blocks = 0

    def insert_sequence(self, block_ids: List[int], sequence_id: str) -> Tuple[List[int], List[int]]:
        """
        Insert sequence into Radix tree

        Returns:
            (shared_blocks, new_blocks): Blocks reused vs newly allocated
        """
        current = self.root
        shared_blocks = []
        new_blocks = []
        remaining_blocks = block_ids[:]

        while remaining_blocks:
            first_block = remaining_blocks[0]

            if first_block not in current.children:
                # No existing child - create new path
                new_node = RadixBlockNode(
                    block_ids=remaining_blocks,
                    is_leaf=True
                )
                new_node.ref_count = 1
                current.children[first_block] = new_node

                new_blocks.extend(remaining_blocks)
                self.total_nodes += 1
                break

            # Found matching child
            child = current.children[first_block]

            # Find common prefix between remaining and child's blocks
            common_len = self._find_common_prefix_length(
                remaining_blocks,
                child.block_ids
            )

            if common_len == len(child.block_ids):
                # Full match - continue down tree
                shared_blocks.extend(child.block_ids)
                remaining_blocks = remaining_blocks[common_len:]
                current = child
                child.ref_count += 1

            else:
                # Partial match - need to split
                split_shared, split_new = self._split_node(
                    current, child, first_block,
                    common_len, remaining_blocks
                )
                shared_blocks.extend(split_shared)
                new_blocks.extend(split_new)
                break

        return shared_blocks, new_blocks

    def _find_common_prefix_length(self, blocks1: List[int], blocks2: List[int]) -> int:
        """Find length of common block prefix"""
        common_len = 0
        for i in range(min(len(blocks1), len(blocks2))):
            if blocks1[i] == blocks2[i]:
                common_len += 1
            else:
                break
        return common_len

    def _split_node(self, parent: RadixBlockNode, child: RadixBlockNode,
                   first_block: int, split_point: int,
                   new_blocks: List[int]) -> Tuple[List[int], List[int]]:
        """
        Split node when partial match occurs

        Returns:
            (shared_blocks, new_blocks)
        """
        # Create middle node with common prefix
        common_blocks = child.block_ids[:split_point]
        middle_node = RadixBlockNode(block_ids=common_blocks, is_leaf=False)
        middle_node.ref_count = child.ref_count + 1

        # Update child to only have suffix
        child.block_ids = child.block_ids[split_point:]

        # Create new sibling for diverging path
        new_suffix = new_blocks[split_point:]
        new_sibling = RadixBlockNode(block_ids=new_suffix, is_leaf=True)
        new_sibling.ref_count = 1

        # Reconnect tree
        parent.children[first_block] = middle_node

        if child.block_ids:
            middle_node.children[child.block_ids[0]] = child

        if new_suffix:
            middle_node.children[new_suffix[0]] = new_sibling

        self.total_nodes += 2

        return common_blocks, new_suffix

    def lookup_sequence(self, block_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Find longest matching prefix in tree

        Returns:
            (cached_blocks, remaining_blocks)
        """
        current = self.root
        cached_blocks = []
        remaining_blocks = block_ids[:]

        while remaining_blocks:
            first_block = remaining_blocks[0]

            if first_block not in current.children:
                break

            child = current.children[first_block]

            common_len = self._find_common_prefix_length(
                remaining_blocks,
                child.block_ids
            )

            if common_len > 0:
                cached_blocks.extend(child.block_ids[:common_len])
                remaining_blocks = remaining_blocks[common_len:]

                if common_len == len(child.block_ids):
                    current = child
                    # Update access stats
                    import time
                    child.last_access_time = time.time()
                    child.total_accesses += 1
                else:
                    break
            else:
                break

        return cached_blocks, remaining_blocks

    def remove_sequence(self, block_ids: List[int], sequence_id: str):
        """
        Remove sequence from tree (decrement ref counts)

        If ref count reaches 0, blocks can be freed
        """
        current = self.root
        path = [current]

        # Traverse to find the sequence
        remaining = block_ids[:]

        while remaining:
            first_block = remaining[0]

            if first_block not in current.children:
                break

            child = current.children[first_block]
            path.append(child)

            common_len = self._find_common_prefix_length(remaining, child.block_ids)

            if common_len == len(child.block_ids):
                remaining = remaining[common_len:]
                current = child
            else:
                break

        # Decrement ref counts along path
        for node in path[1:]:  # Skip root
            node.ref_count -= 1

            if node.ref_count == 0:
                # Can free blocks
                self._free_blocks(node.block_ids)

    def _free_blocks(self, block_ids: List[int]):
        """Free blocks back to block manager"""
        for block_id in block_ids:
            self.block_manager.free_block(block_id)

    def get_statistics(self) -> Dict:
        """Get tree statistics"""
        def count_blocks(node: RadixBlockNode) -> Tuple[int, int]:
            """Count shared and unique blocks"""
            shared = len(node.block_ids) if node.ref_count > 1 else 0
            unique = len(node.block_ids) if node.ref_count == 1 else 0

            for child in node.children.values():
                child_shared, child_unique = count_blocks(child)
                shared += child_shared
                unique += child_unique

            return shared, unique

        shared, unique = count_blocks(self.root)

        return {
            "total_nodes": self.total_nodes,
            "shared_blocks": shared,
            "unique_blocks": unique,
            "total_blocks": shared + unique,
            "sharing_ratio": shared / (shared + unique) if (shared + unique) > 0 else 0
        }

def radix_tree_block_example():
    """
    Example of block-level Radix tree usage
    """
    from unittest.mock import Mock

    # Mock block manager
    block_manager = Mock()
    block_manager.allocate_block = Mock(side_effect=lambda: 1)
    block_manager.free_block = Mock()

    tree = BlockLevelRadixTree(block_manager)

    # Insert sequences
    sequences = [
        ([1, 2, 3, 4, 5], "seq1"),
        ([1, 2, 3, 6, 7], "seq2"),
        ([1, 2, 8, 9], "seq3"),
    ]

    print("Inserting sequences into block-level Radix tree:")
    print("=" * 60)

    for blocks, seq_id in sequences:
        shared, new = tree.insert_sequence(blocks, seq_id)
        print(f"\n{seq_id}: {blocks}")
        print(f"  Shared blocks: {shared}")
        print(f"  New blocks: {new}")

    # Statistics
    stats = tree.get_statistics()
    print("\n\nTree Statistics:")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if "ratio" in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

radix_tree_block_example()
```

---

## 2. Integration with PagedAttention

### 2.1 Modified Block Allocation

```python
class RadixPagedAttentionBlockManager:
    """
    Block manager integrating Radix tree with PagedAttention
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # PagedAttention block pool
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}  # block_id -> owner info

        # RadixAttention Radix tree
        self.radix_tree = BlockLevelRadixTree(self)

        # Statistics
        self.allocation_stats = {
            "total_allocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "blocks_saved": 0
        }

    def allocate_sequence(self, token_ids: List[int], sequence_id: str) -> List[int]:
        """
        Allocate blocks for sequence with prefix caching

        Returns:
            List of physical block IDs
        """
        # Calculate required blocks
        num_tokens = len(token_ids)
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        # Try to find cached prefix
        # (Simplified: would tokenize and hash in practice)
        prefix_hash = hash(tuple(token_ids[:num_tokens//2]))  # Use first half as "prefix"

        # Look up in Radix tree
        # For simplicity, assume each block maps to block_size tokens
        block_sequence = list(range(num_blocks_needed))  # Placeholder

        cached_blocks, remaining = self.radix_tree.lookup_sequence(block_sequence)

        self.allocation_stats["total_allocations"] += 1

        if cached_blocks:
            self.allocation_stats["cache_hits"] += 1
            self.allocation_stats["blocks_saved"] += len(cached_blocks)

            # Allocate only remaining blocks
            new_blocks = self._allocate_new_blocks(len(remaining))
            all_blocks = cached_blocks + new_blocks

        else:
            self.allocation_stats["cache_misses"] += 1

            # Allocate all blocks
            all_blocks = self._allocate_new_blocks(num_blocks_needed)

        # Insert into Radix tree
        self.radix_tree.insert_sequence(all_blocks, sequence_id)

        return all_blocks

    def _allocate_new_blocks(self, num_blocks: int) -> List[int]:
        """Allocate new physical blocks"""
        if len(self.free_blocks) < num_blocks:
            # Need to evict
            self._evict_blocks(num_blocks - len(self.free_blocks))

        allocated = []
        for _ in range(num_blocks):
            if self.free_blocks:
                block_id = self.free_blocks.pop(0)
                allocated.append(block_id)

        return allocated

    def free_sequence(self, block_ids: List[int], sequence_id: str):
        """
        Free blocks for sequence

        Blocks are only freed if no other sequence is using them
        """
        self.radix_tree.remove_sequence(block_ids, sequence_id)

    def free_block(self, block_id: int):
        """Free a single block back to pool"""
        if block_id not in self.free_blocks:
            self.free_blocks.append(block_id)

        if block_id in self.allocated_blocks:
            del self.allocated_blocks[block_id]

    def _evict_blocks(self, num_needed: int):
        """Evict blocks using LRU policy"""
        # Collect all nodes and sort by access time
        nodes_to_evict = []

        def collect_nodes(node: RadixBlockNode):
            if node.ref_count > 0 and node.block_ids:
                nodes_to_evict.append(node)
            for child in node.children.values():
                collect_nodes(child)

        collect_nodes(self.radix_tree.root)

        # Sort by last access time (oldest first)
        nodes_to_evict.sort(key=lambda n: n.last_access_time)

        # Evict until we have enough blocks
        freed = 0
        for node in nodes_to_evict:
            if freed >= num_needed:
                break

            # Free this node's blocks
            for block_id in node.block_ids:
                self.free_block(block_id)
                freed += 1

            node.block_ids = []
            node.blocks_resident = False

    def get_statistics(self):
        """Get allocation statistics"""
        radix_stats = self.radix_tree.get_statistics()

        return {
            **self.allocation_stats,
            "free_blocks": len(self.free_blocks),
            "total_blocks": self.num_blocks,
            "utilization": 1 - (len(self.free_blocks) / self.num_blocks),
            **radix_stats
        }

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()

        print("\nRadixPagedAttention Block Manager Statistics:")
        print("=" * 60)

        print(f"\nAllocations:")
        print(f"  Total: {stats['total_allocations']}")
        print(f"  Cache Hits: {stats['cache_hits']}")
        print(f"  Cache Misses: {stats['cache_misses']}")
        if stats['total_allocations'] > 0:
            hit_rate = stats['cache_hits'] / stats['total_allocations']
            print(f"  Hit Rate: {hit_rate:.2%}")

        print(f"\nBlocks:")
        print(f"  Total: {stats['total_blocks']}")
        print(f"  Free: {stats['free_blocks']}")
        print(f"  Utilization: {stats['utilization']:.2%}")
        print(f"  Blocks Saved: {stats['blocks_saved']}")

        print(f"\nSharing:")
        print(f"  Shared Blocks: {stats['shared_blocks']}")
        print(f"  Unique Blocks: {stats['unique_blocks']}")
        print(f"  Sharing Ratio: {stats['sharing_ratio']:.2%}")

        print("=" * 60)

def benchmark_radix_block_manager():
    """
    Benchmark RadixPagedAttention block manager
    """
    import time
    import random

    manager = RadixPagedAttentionBlockManager(num_blocks=1000, block_size=16)

    # Common prefix (simulating shared document)
    common_prefix = list(range(500))  # 500 tokens = ~31 blocks

    # Generate sequences with shared prefix
    sequences = []
    for i in range(100):
        # Each sequence has common prefix + unique suffix
        suffix = [random.randint(1000, 2000) for _ in range(100)]
        sequences.append((common_prefix + suffix, f"seq_{i}"))

    print("Benchmarking RadixPagedAttention Block Manager:")
    print("=" * 60)

    # Allocate all sequences
    start = time.time()
    allocated_blocks = []

    for tokens, seq_id in sequences:
        blocks = manager.allocate_sequence(tokens, seq_id)
        allocated_blocks.append((blocks, seq_id))

    allocation_time = time.time() - start

    print(f"\nAllocation Time: {allocation_time * 1000:.2f} ms")
    print(f"Sequences Allocated: {len(sequences)}")

    # Print statistics
    manager.print_statistics()

    # Free sequences
    for blocks, seq_id in allocated_blocks:
        manager.free_sequence(blocks, seq_id)

    print("\nAfter freeing all sequences:")
    manager.print_statistics()

benchmark_radix_block_manager()
```

### 2.2 Copy-on-Write Optimization

```python
class CopyOnWriteBlockManager:
    """
    Implement copy-on-write for KV cache blocks

    Key idea: Shared blocks are read-only
    When modification needed, copy to new block
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Block metadata
        self.block_data = {}  # block_id -> actual KV cache data
        self.block_ref_counts = {}  # block_id -> ref count
        self.block_write_locks = {}  # block_id -> is_writable

        self.free_blocks = list(range(num_blocks))

    def allocate_shared_block(self, block_id: int, sequence_id: str) -> int:
        """
        Allocate shared block (read-only)

        Returns:
            block_id (can be shared with other sequences)
        """
        if block_id not in self.block_ref_counts:
            self.block_ref_counts[block_id] = 0
            self.block_write_locks[block_id] = True  # Initially writable

        self.block_ref_counts[block_id] += 1

        # Block becomes read-only when shared
        if self.block_ref_counts[block_id] > 1:
            self.block_write_locks[block_id] = False

        return block_id

    def write_to_block(self, block_id: int, data, sequence_id: str) -> int:
        """
        Write to block (may trigger copy-on-write)

        Returns:
            Actual block_id written to (may be different if COW occurred)
        """
        if self.block_write_locks.get(block_id, True):
            # Block is writable - write directly
            self.block_data[block_id] = data
            return block_id

        else:
            # Block is shared - need copy-on-write
            print(f"Copy-on-write triggered for block {block_id}")

            # Allocate new block
            if not self.free_blocks:
                raise RuntimeError("Out of blocks!")

            new_block_id = self.free_blocks.pop(0)

            # Copy data to new block
            self.block_data[new_block_id] = self.block_data[block_id].copy()

            # Write new data
            self.block_data[new_block_id] = data

            # Update metadata
            self.block_ref_counts[new_block_id] = 1
            self.block_write_locks[new_block_id] = True

            # Decrement ref count on old block
            self.block_ref_counts[block_id] -= 1

            if self.block_ref_counts[block_id] == 1:
                # Block no longer shared - make writable again
                self.block_write_locks[block_id] = True

            return new_block_id

    def read_from_block(self, block_id: int):
        """Read from block (always allowed)"""
        return self.block_data.get(block_id, None)

def copy_on_write_example():
    """
    Demonstrate copy-on-write behavior
    """
    import numpy as np

    manager = CopyOnWriteBlockManager(num_blocks=100, block_size=16)

    # Simulate KV cache data
    kv_data_1 = np.random.randn(16, 128)  # 16 tokens, 128 dim

    print("Copy-on-Write Example:")
    print("=" * 60)

    # Sequence 1 writes to block 0
    print("\nSequence 1 writes to block 0:")
    block_0 = manager.allocate_shared_block(0, "seq1")
    actual_block = manager.write_to_block(block_0, kv_data_1, "seq1")
    print(f"  Written to block: {actual_block}")
    print(f"  Block 0 ref count: {manager.block_ref_counts[0]}")

    # Sequence 2 shares block 0 (read-only)
    print("\nSequence 2 shares block 0 (read-only):")
    block_0_shared = manager.allocate_shared_block(0, "seq2")
    print(f"  Sharing block: {block_0_shared}")
    print(f"  Block 0 ref count: {manager.block_ref_counts[0]}")
    print(f"  Block 0 writable: {manager.block_write_locks[0]}")

    # Sequence 2 tries to write (triggers COW)
    print("\nSequence 2 writes (triggers copy-on-write):")
    kv_data_2 = np.random.randn(16, 128)
    actual_block = manager.write_to_block(block_0_shared, kv_data_2, "seq2")
    print(f"  Written to new block: {actual_block}")
    print(f"  Block 0 ref count: {manager.block_ref_counts[0]}")
    print(f"  Block {actual_block} ref count: {manager.block_ref_counts[actual_block]}")

    # Verify data independence
    data_0 = manager.read_from_block(0)
    data_new = manager.read_from_block(actual_block)

    print(f"\n  Block 0 and Block {actual_block} are different: {not np.array_equal(data_0, data_new)}")

copy_on_write_example()
```

---

## 3. Performance Optimization

### 3.1 Batched Radix Lookups

```python
class BatchedRadixLookup:
    """
    Optimize Radix tree lookups for batched requests
    """

    def __init__(self, radix_tree: BlockLevelRadixTree):
        self.radix_tree = radix_tree

    def lookup_batch(self, sequences: List[Tuple[List[int], str]]) -> List[Tuple[List[int], List[int]]]:
        """
        Lookup multiple sequences in parallel

        Args:
            sequences: List of (block_ids, sequence_id) tuples

        Returns:
            List of (cached_blocks, remaining_blocks) for each sequence
        """
        results = []

        # Process each sequence
        for block_ids, seq_id in sequences:
            cached, remaining = self.radix_tree.lookup_sequence(block_ids)
            results.append((cached, remaining))

        return results

    def optimize_lookup_order(self, sequences: List[Tuple[List[int], str]]) -> List[Tuple[List[int], str]]:
        """
        Optimize lookup order for better cache locality

        Strategy: Process sequences with common prefixes together
        """
        # Group sequences by prefix
        prefix_groups = {}

        for block_ids, seq_id in sequences:
            if not block_ids:
                continue

            prefix_key = tuple(block_ids[:min(5, len(block_ids))])  # First 5 blocks as prefix

            if prefix_key not in prefix_groups:
                prefix_groups[prefix_key] = []

            prefix_groups[prefix_key].append((block_ids, seq_id))

        # Flatten groups (sequences with same prefix together)
        optimized_order = []
        for group in prefix_groups.values():
            optimized_order.extend(group)

        return optimized_order

def benchmark_batched_lookups():
    """
    Benchmark batched vs individual lookups
    """
    import time
    import random

    # Create Radix tree and populate
    block_manager = Mock()
    tree = BlockLevelRadixTree(block_manager)

    # Insert some sequences
    base_prefix = list(range(100))
    for i in range(50):
        suffix = [random.randint(1000, 2000) for _ in range(20)]
        tree.insert_sequence(base_prefix + suffix, f"seq_{i}")

    # Create lookup batch
    lookup_sequences = []
    for i in range(100):
        suffix = [random.randint(1000, 2000) for _ in range(20)]
        lookup_sequences.append((base_prefix + suffix, f"lookup_{i}"))

    # Individual lookups
    start = time.time()
    for block_ids, seq_id in lookup_sequences:
        _ = tree.lookup_sequence(block_ids)
    individual_time = time.time() - start

    # Batched lookups
    batched_lookup = BatchedRadixLookup(tree)

    # Without optimization
    start = time.time()
    _ = batched_lookup.lookup_batch(lookup_sequences)
    batched_time = time.time() - start

    # With optimization
    start = time.time()
    optimized_sequences = batched_lookup.optimize_lookup_order(lookup_sequences)
    _ = batched_lookup.lookup_batch(optimized_sequences)
    optimized_time = time.time() - start

    print("Batched Lookup Benchmark:")
    print("=" * 60)
    print(f"Individual lookups: {individual_time * 1000:.2f} ms")
    print(f"Batched lookups: {batched_time * 1000:.2f} ms")
    print(f"Optimized batched: {optimized_time * 1000:.2f} ms")
    print(f"\nSpeedup (batched): {individual_time / batched_time:.2f}x")
    print(f"Speedup (optimized): {individual_time / optimized_time:.2f}x")

benchmark_batched_lookups()
```

### 3.2 Prefix Compression

```python
class CompressedRadixTree:
    """
    Radix tree with prefix compression for memory efficiency
    """

    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.root = RadixBlockNode(block_ids=[], is_leaf=False)

        # Compression metadata
        self.compressed_prefixes = {}  # hash -> compressed representation

    def compress_prefix(self, block_ids: List[int]) -> bytes:
        """
        Compress block ID sequence

        Use delta encoding + variable-length encoding
        """
        if not block_ids:
            return b''

        # Delta encoding: store differences instead of absolute values
        deltas = [block_ids[0]]
        for i in range(1, len(block_ids)):
            deltas.append(block_ids[i] - block_ids[i-1])

        # Variable-length encoding (simplified)
        import struct
        compressed = struct.pack(f'{len(deltas)}i', *deltas)

        return compressed

    def decompress_prefix(self, compressed: bytes) -> List[int]:
        """Decompress block ID sequence"""
        if not compressed:
            return []

        import struct
        num_values = len(compressed) // 4  # 4 bytes per int
        deltas = struct.unpack(f'{num_values}i', compressed)

        # Reconstruct absolute values
        block_ids = [deltas[0]]
        for i in range(1, len(deltas)):
            block_ids.append(block_ids[-1] + deltas[i])

        return block_ids

    def calculate_compression_ratio(self, block_ids: List[int]) -> float:
        """Calculate compression ratio"""
        original_size = len(block_ids) * 4  # 4 bytes per int
        compressed = self.compress_prefix(block_ids)
        compressed_size = len(compressed)

        return compressed_size / original_size if original_size > 0 else 1.0

def compression_example():
    """
    Example of prefix compression
    """
    compressor = CompressedRadixTree(None)

    # Sequential block IDs (should compress well)
    sequential_blocks = list(range(100))

    # Random block IDs (should compress poorly)
    import random
    random_blocks = [random.randint(0, 10000) for _ in range(100)]

    print("Prefix Compression Example:")
    print("=" * 60)

    for name, blocks in [("Sequential", sequential_blocks), ("Random", random_blocks)]:
        compressed = compressor.compress_prefix(blocks)
        decompressed = compressor.decompress_prefix(compressed)
        ratio = compressor.calculate_compression_ratio(blocks)

        print(f"\n{name} Blocks:")
        print(f"  Original size: {len(blocks) * 4} bytes")
        print(f"  Compressed size: {len(compressed)} bytes")
        print(f"  Compression ratio: {ratio:.2%}")
        print(f"  Decompression correct: {decompressed == blocks}")

compression_example()
```

---

## 4. Production Deployment

### 4.1 Enabling RadixAttention in vLLM

```python
from vllm import LLM, SamplingParams

def deploy_radix_attention():
    """
    Deploy vLLM with RadixAttention enabled
    """
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",

        # Enable RadixAttention (automatic prefix caching)
        enable_prefix_caching=True,

        # Block configuration
        block_size=16,  # Tokens per block
        max_num_batched_tokens=8192,

        # Memory configuration
        gpu_memory_utilization=0.85,  # Leave room for cached blocks

        # Eviction policy
        swap_space=4,  # GB of CPU swap space
    )

    return llm

def production_radix_example():
    """
    Production example with RadixAttention
    """
    llm = deploy_radix_attention()

    # Shared document
    document = "Long document content... " * 1000

    # Multiple questions
    questions = [
        "What is the main topic?",
        "Who are mentioned?",
        "What is the conclusion?",
    ]

    sampling_params = SamplingParams(max_tokens=100, temperature=0.8)

    import time

    print("RadixAttention Production Example:")
    print("=" * 60)

    for i, question in enumerate(questions):
        prompt = f"{document}\n\nQuestion: {question}\nAnswer:"

        start = time.time()
        output = llm.generate(prompt, sampling_params)
        elapsed = time.time() - start

        print(f"\nQuestion {i+1}: {elapsed:.2f}s")
        print(f"  Expected: {'2-3s' if i == 0 else '0.3-0.5s'}")
```

### 4.2 Monitoring RadixAttention

```python
class RadixAttentionMonitor:
    """
    Monitor RadixAttention performance in production
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            "total_requests": 0,
            "total_blocks_allocated": 0,
            "total_blocks_reused": 0,
            "radix_tree_size": 0,
            "avg_prefix_match_length": [],
            "block_evictions": 0,
        }

    def record_allocation(self, blocks_allocated: int, blocks_reused: int,
                         prefix_match_length: int):
        """Record allocation metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["total_blocks_allocated"] += blocks_allocated
        self.metrics["total_blocks_reused"] += blocks_reused
        self.metrics["avg_prefix_match_length"].append(prefix_match_length)

    def record_eviction(self, num_blocks: int):
        """Record block eviction"""
        self.metrics["block_evictions"] += num_blocks

    def get_summary(self):
        """Get summary statistics"""
        import numpy as np

        total_blocks = self.metrics["total_blocks_allocated"] + self.metrics["total_blocks_reused"]

        summary = {
            "total_requests": self.metrics["total_requests"],
            "block_reuse_rate": self.metrics["total_blocks_reused"] / total_blocks if total_blocks > 0 else 0,
            "avg_blocks_per_request": total_blocks / self.metrics["total_requests"] if self.metrics["total_requests"] > 0 else 0,
            "avg_prefix_match": np.mean(self.metrics["avg_prefix_match_length"]) if self.metrics["avg_prefix_match_length"] else 0,
            "eviction_rate": self.metrics["block_evictions"] / total_blocks if total_blocks > 0 else 0,
        }

        return summary

    def print_report(self):
        """Print monitoring report"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("RADIXATTENTION MONITORING REPORT")
        print("=" * 60)

        print(f"\nRequests: {summary['total_requests']}")
        print(f"Block Reuse Rate: {summary['block_reuse_rate']:.2%}")
        print(f"Avg Blocks/Request: {summary['avg_blocks_per_request']:.1f}")
        print(f"Avg Prefix Match: {summary['avg_prefix_match']:.1f} blocks")
        print(f"Eviction Rate: {summary['eviction_rate']:.2%}")

        print("=" * 60)
```

---

## 5. Summary

### 5.1 Key Takeaways

1. **RadixAttention = Radix Tree + PagedAttention**
   - Efficient prefix sharing via Radix tree
   - Block-level granularity for memory efficiency
   - Copy-on-write for safety

2. **Performance Benefits:**
   - 2-5x speedup for workloads with shared prefixes
   - 50-80% memory reduction with high sharing
   - Minimal overhead for diverse workloads

3. **Best Practices:**
   - Enable for document Q&A, chatbots with system prompts
   - Monitor block reuse rates
   - Tune eviction policies based on workload

---

## 6. Exercises

**Exercise 8.5.1:** Implement block-level Radix tree from scratch

**Exercise 8.5.2:** Benchmark copy-on-write overhead

**Exercise 8.5.3:** Optimize Radix tree for specific workload pattern

**Exercise 8.5.4:** Implement custom eviction policy

**Exercise 8.5.5:** Deploy and monitor RadixAttention in production

---

**Next:** [Module 8.6: Chunked Prefill Optimization](06_chunked_prefill_optimization.md)
