# Distributed KV Cache Management

## Table of Contents
1. [KV Cache Basics](#basics)
2. [Partitioning Strategies](#partitioning)
3. [Memory Management](#memory)
4. [Cache Synchronization](#synchronization)
5. [Transfer Mechanisms](#transfer)
6. [Performance Optimization](#optimization)
7. [vLLM Implementation](#vllm-implementation)
8. [Advanced Techniques](#advanced)

---

## KV Cache Basics {#basics}

### What is KV Cache?

```python
"""
KV Cache in Transformer Attention:

During inference, attention mechanism computes:
Attention(Q, K, V) = softmax(QK^T / √d_k) @ V

For autoregressive generation:
- Query (Q): only for new token
- Key (K), Value (V): for all previous tokens + new token

KV Cache stores K and V for all previous tokens
to avoid recomputation.

Memory per sequence:
- 2 tensors (K and V) per layer
- Size: [num_layers, seq_len, hidden_dim]
- For Llama-70B at 2048 tokens: ~10GB per sequence
"""


def calculate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_length: int,
    batch_size: int,
    dtype_bytes: int = 2,  # FP16
):
    """
    Calculate KV cache memory requirements.
    """

    # K and V cache per layer
    cache_per_layer = 2 * seq_length * num_kv_heads * head_dim * dtype_bytes

    # Total across all layers
    cache_per_sequence = cache_per_layer * num_layers

    # Total for batch
    total_cache = cache_per_sequence * batch_size

    print(f"KV Cache Memory:")
    print(f"  Per layer: {cache_per_layer / 1e9:.2f} GB")
    print(f"  Per sequence: {cache_per_sequence / 1e9:.2f} GB")
    print(f"  Total (batch={batch_size}): {total_cache / 1e9:.2f} GB")

    return total_cache / 1e9


# Example: Llama-70B
calculate_kv_cache_memory(
    num_layers=80,
    num_kv_heads=8,  # Grouped-query attention
    head_dim=128,
    seq_length=2048,
    batch_size=32,
)
```

---

## Partitioning Strategies {#partitioning}

### Tensor Parallel KV Cache

```python
class TensorParallelKVCache:
    """
    KV cache partitioned across tensor parallel ranks.

    Each GPU stores KV cache for its subset of attention heads.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_length: int,
        tensor_parallel_size: int,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self.tp_size = tensor_parallel_size

        # Heads per GPU
        assert num_kv_heads % tp_size == 0
        self.num_kv_heads_per_gpu = num_kv_heads // tp_size

        # Allocate cache
        self.cache = self._allocate_cache()

    def _allocate_cache(self):
        """Allocate KV cache on this GPU."""

        cache = []

        for _ in range(self.num_layers):
            # K and V cache for this layer
            # Shape: [max_batch_size, max_seq_len, num_heads_per_gpu, head_dim]
            k_cache = torch.empty(
                (MAX_BATCH_SIZE, self.max_seq_length,
                 self.num_kv_heads_per_gpu, self.head_dim),
                dtype=torch.float16,
                device='cuda'
            )

            v_cache = torch.empty_like(k_cache)

            cache.append((k_cache, v_cache))

        return cache

    def update(self, layer_id, seq_id, position, k, v):
        """
        Update KV cache for a sequence.

        Args:
            layer_id: Layer index
            seq_id: Sequence index in batch
            position: Token position
            k: New key [num_heads_per_gpu, head_dim]
            v: New value [num_heads_per_gpu, head_dim]
        """

        k_cache, v_cache = self.cache[layer_id]

        # Update at position
        k_cache[seq_id, position] = k
        v_cache[seq_id, position] = v

    def get(self, layer_id, seq_id, start_pos, end_pos):
        """
        Retrieve KV cache for a sequence.

        Returns:
            k: [seq_len, num_heads_per_gpu, head_dim]
            v: [seq_len, num_heads_per_gpu, head_dim]
        """

        k_cache, v_cache = self.cache[layer_id]

        k = k_cache[seq_id, start_pos:end_pos]
        v = v_cache[seq_id, start_pos:end_pos]

        return k, v


### PagedKV Cache (vLLM's Approach)

```python
class PagedKVCache:
    """
    Paged KV cache for efficient memory management.

    KV cache is divided into blocks (pages) that can be
    allocated/deallocated dynamically.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,  # Tokens per block
        num_blocks: int,
        tensor_parallel_size: int,
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.tp_size = tensor_parallel_size

        # Heads per GPU
        self.num_kv_heads_per_gpu = num_kv_heads // tp_size
        self.head_dim = head_dim

        # Physical blocks
        # Shape: [num_layers, num_blocks, block_size, num_heads_per_gpu, head_dim]
        self.key_cache = self._allocate_physical_cache()
        self.value_cache = self._allocate_physical_cache()

        # Block manager
        self.free_blocks = set(range(num_blocks))
        self.block_tables = {}  # seq_id -> list of block indices

    def _allocate_physical_cache(self):
        """Allocate physical KV cache blocks."""

        cache = []

        for _ in range(self.num_layers):
            layer_cache = torch.empty(
                (self.num_blocks, self.block_size,
                 self.num_kv_heads_per_gpu, self.head_dim),
                dtype=torch.float16,
                device='cuda'
            )
            cache.append(layer_cache)

        return cache

    def allocate_sequence(self, seq_id, num_tokens):
        """Allocate blocks for a new sequence."""

        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of KV cache memory")

        # Allocate blocks
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated_blocks.append(block_id)

        self.block_tables[seq_id] = allocated_blocks

    def free_sequence(self, seq_id):
        """Free blocks for a sequence."""

        if seq_id in self.block_tables:
            for block_id in self.block_tables[seq_id]:
                self.free_blocks.add(block_id)

            del self.block_tables[seq_id]

    def append_tokens(self, layer_id, seq_id, k, v):
        """
        Append new tokens' KV to cache.

        Args:
            k: [num_new_tokens, num_heads_per_gpu, head_dim]
            v: [num_new_tokens, num_heads_per_gpu, head_dim]
        """

        block_table = self.block_tables[seq_id]
        num_new_tokens = k.shape[0]

        # Current position in sequence
        # (Would track this separately in real implementation)
        current_pos = self._get_sequence_length(seq_id)

        # Write to blocks
        for i in range(num_new_tokens):
            token_pos = current_pos + i

            block_idx = token_pos // self.block_size
            offset = token_pos % self.block_size

            physical_block_id = block_table[block_idx]

            self.key_cache[layer_id][physical_block_id, offset] = k[i]
            self.value_cache[layer_id][physical_block_id, offset] = v[i]
```

---

## Memory Management {#memory}

### Cross-GPU KV Cache Transfer

```python
class DistributedKVCacheManager:
    """
    Manage KV cache across multiple GPUs.

    Handles cache migration for load balancing.
    """

    def __init__(self, local_cache, device_id):
        self.local_cache = local_cache
        self.device_id = device_id

    def migrate_sequence(
        self,
        seq_id: int,
        target_device: int,
    ):
        """
        Migrate sequence's KV cache to another GPU.
        """

        # Get sequence cache
        cache_data = self._serialize_sequence_cache(seq_id)

        # Transfer to target device
        if target_device == self.device_id:
            # Local, no transfer needed
            return

        # Send to target GPU
        cache_data_gpu = cache_data.to(f'cuda:{target_device}')

        # Notify target to deserialize
        self._send_cache_metadata(target_device, seq_id, cache_data_gpu)

        # Free local cache
        self.local_cache.free_sequence(seq_id)

    def _serialize_sequence_cache(self, seq_id):
        """Serialize KV cache for a sequence."""

        # Collect all KV tensors for this sequence
        block_table = self.local_cache.block_tables[seq_id]

        kv_tensors = []

        for layer_id in range(self.local_cache.num_layers):
            for block_id in block_table:
                k = self.local_cache.key_cache[layer_id][block_id]
                v = self.local_cache.value_cache[layer_id][block_id]

                kv_tensors.extend([k, v])

        # Concatenate for transfer
        serialized = torch.cat([t.flatten() for t in kv_tensors])

        return serialized


### CPU Offloading

```python
class CPUOffloadKVCache:
    """
    KV cache with CPU offloading for memory management.

    Offload cold sequences to CPU to free GPU memory.
    """

    def __init__(self, gpu_cache, cpu_memory_gb=64):
        self.gpu_cache = gpu_cache
        self.cpu_memory_gb = cpu_memory_gb

        # CPU storage
        self.cpu_cache = {}

        # Track which sequences are on GPU vs CPU
        self.gpu_sequences = set()
        self.cpu_sequences = set()

    def offload_to_cpu(self, seq_id):
        """Move sequence KV cache from GPU to CPU."""

        if seq_id not in self.gpu_sequences:
            return

        # Get from GPU
        cache_data = self._get_gpu_cache(seq_id)

        # Move to CPU
        cache_data_cpu = cache_data.cpu()
        self.cpu_cache[seq_id] = cache_data_cpu

        # Free GPU memory
        self.gpu_cache.free_sequence(seq_id)

        # Update tracking
        self.gpu_sequences.remove(seq_id)
        self.cpu_sequences.add(seq_id)

        print(f"Offloaded sequence {seq_id} to CPU")

    def prefetch_to_gpu(self, seq_id):
        """Prefetch sequence KV cache from CPU to GPU."""

        if seq_id not in self.cpu_sequences:
            return

        # Get from CPU
        cache_data_cpu = self.cpu_cache[seq_id]

        # Move to GPU asynchronously
        cache_data_gpu = cache_data_cpu.to('cuda', non_blocking=True)

        # Restore to GPU cache
        self._restore_gpu_cache(seq_id, cache_data_gpu)

        # Free CPU memory
        del self.cpu_cache[seq_id]

        # Update tracking
        self.cpu_sequences.remove(seq_id)
        self.gpu_sequences.add(seq_id)

        print(f"Prefetched sequence {seq_id} to GPU")

    def evict_cold_sequences(self, keep_top_k=10):
        """
        Evict least recently used sequences to CPU.
        """

        # Get LRU sequences
        sequences_by_access = sorted(
            self.gpu_sequences,
            key=lambda s: self._get_last_access_time(s)
        )

        # Keep top-k most recent
        to_evict = sequences_by_access[:-keep_top_k] if len(sequences_by_access) > keep_top_k else []

        for seq_id in to_evict:
            self.offload_to_cpu(seq_id)
```

---

## Cache Synchronization {#synchronization}

### Distributed Cache Coherence

```python
class CoherentKVCache:
    """
    Maintain cache coherence across distributed workers.
    """

    def __init__(self, local_cache, worker_id, num_workers):
        self.local_cache = local_cache
        self.worker_id = worker_id
        self.num_workers = num_workers

        # Sequence ownership
        self.owned_sequences = set()

        # Shared sequences (read-only replicas)
        self.shared_sequences = {}

    def write_cache(self, seq_id, layer_id, k, v):
        """
        Write to KV cache with coherence.
        """

        # Check ownership
        owner = self._get_owner(seq_id)

        if owner == self.worker_id:
            # Local write
            self.local_cache.update(layer_id, seq_id, k, v)

            # Invalidate replicas on other workers
            self._invalidate_replicas(seq_id)

        else:
            # Forward write to owner
            self._remote_write(owner, seq_id, layer_id, k, v)

    def read_cache(self, seq_id, layer_id):
        """
        Read from KV cache with coherence.
        """

        owner = self._get_owner(seq_id)

        if owner == self.worker_id:
            # Local read
            return self.local_cache.get(layer_id, seq_id)

        else:
            # Check if have valid replica
            if seq_id in self.shared_sequences:
                return self.shared_sequences[seq_id][layer_id]

            # Fetch from owner
            cache_data = self._remote_read(owner, seq_id, layer_id)

            # Cache replica
            if seq_id not in self.shared_sequences:
                self.shared_sequences[seq_id] = {}
            self.shared_sequences[seq_id][layer_id] = cache_data

            return cache_data

    def _invalidate_replicas(self, seq_id):
        """Invalidate replicas on other workers."""

        for worker_id in range(self.num_workers):
            if worker_id != self.worker_id:
                self._send_invalidate(worker_id, seq_id)
```

---

## Transfer Mechanisms {#transfer}

### Efficient KV Transfer

```python
import torch.distributed as dist

def transfer_kv_cache(
    kv_tensor: torch.Tensor,
    src_rank: int,
    dst_rank: int,
):
    """
    Transfer KV cache between GPUs efficiently.

    Uses NCCL for intra-node, direct P2P for same device.
    """

    # Get current rank
    rank = dist.get_rank()

    if rank == src_rank:
        # Sender
        dist.send(kv_tensor, dst=dst_rank)

    elif rank == dst_rank:
        # Receiver
        received = torch.empty_like(kv_tensor)
        dist.recv(received, src=src_rank)
        return received


def broadcast_kv_cache(
    kv_tensor: torch.Tensor,
    src_rank: int,
):
    """
    Broadcast KV cache from one GPU to all others.

    Useful for shared prefix caching.
    """

    dist.broadcast(kv_tensor, src=src_rank)

    return kv_tensor


# Async transfer for overlapping
def async_transfer_kv_cache(
    kv_tensor: torch.Tensor,
    src_rank: int,
    dst_rank: int,
):
    """Asynchronous KV cache transfer."""

    rank = dist.get_rank()

    if rank == src_rank:
        work = dist.isend(kv_tensor, dst=dst_rank)
        return work

    elif rank == dst_rank:
        received = torch.empty_like(kv_tensor)
        work = dist.irecv(received, src=src_rank)
        return received, work
```

---

## Performance Optimization {#optimization}

### Prefix Caching

```python
class PrefixKVCache:
    """
    KV cache with prefix sharing.

    Multiple sequences can share KV cache for common prefixes.
    """

    def __init__(self, base_cache):
        self.base_cache = base_cache

        # Prefix tree
        self.prefix_tree = PrefixTree()

        # Reference counting
        self.ref_counts = {}

    def allocate_with_prefix(
        self,
        seq_id: int,
        prefix_tokens: List[int],
        total_length: int,
    ):
        """
        Allocate sequence with shared prefix.
        """

        # Find existing prefix
        prefix_node = self.prefix_tree.find(prefix_tokens)

        if prefix_node:
            # Reuse prefix blocks
            prefix_blocks = prefix_node.blocks

            # Increment reference count
            for block in prefix_blocks:
                self.ref_counts[block] = self.ref_counts.get(block, 0) + 1

            # Allocate only for new tokens
            num_prefix_tokens = len(prefix_tokens)
            num_new_tokens = total_length - num_prefix_tokens

            new_blocks = self.base_cache.allocate(num_new_tokens)

            # Combine
            all_blocks = prefix_blocks + new_blocks

            self.base_cache.block_tables[seq_id] = all_blocks

        else:
            # No prefix match, allocate normally
            self.base_cache.allocate_sequence(seq_id, total_length)

            # Add to prefix tree
            blocks = self.base_cache.block_tables[seq_id]
            self.prefix_tree.insert(prefix_tokens, blocks)

    def free_with_prefix(self, seq_id):
        """Free sequence, respecting reference counts."""

        blocks = self.base_cache.block_tables[seq_id]

        for block in blocks:
            if block in self.ref_counts:
                self.ref_counts[block] -= 1

                # Only free if no references
                if self.ref_counts[block] == 0:
                    self.base_cache.free_blocks.add(block)
                    del self.ref_counts[block]
            else:
                # Not shared, free immediately
                self.base_cache.free_blocks.add(block)

        del self.base_cache.block_tables[seq_id]
```

---

## vLLM Implementation {#vllm-implementation}

### vLLM's KV Cache

```python
# From vllm/attention/backends/

class PagedAttentionBackend:
    """
    vLLM's paged attention with distributed KV cache.
    """

    def __init__(self, ...):
        # Physical KV cache blocks
        # Partitioned across tensor parallel ranks
        self.key_cache = [
            torch.empty(...) for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.empty(...) for _ in range(num_layers)
        ]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ):
        """
        Paged attention with partitioned KV cache.

        Args:
            block_tables: [num_seqs, max_num_blocks_per_seq]
                Maps logical blocks to physical blocks
            context_lens: [num_seqs]
                Length of context for each sequence
        """

        # Append new K, V to cache
        # (Implementation uses custom CUDA kernels)
        paged_attention_ops.append_kv(
            key, value,
            self.key_cache, self.value_cache,
            block_tables,
        )

        # Compute attention with paged KV
        output = paged_attention_ops.forward(
            query,
            self.key_cache, self.value_cache,
            block_tables,
            context_lens,
        )

        return output
```

---

## Advanced Techniques {#advanced}

### Disaggregated KV Cache

```python
"""
Disaggregated serving: Separate prefill and decode workers.

Prefill workers: Compute KV cache for prompts
Decode workers: Use KV cache for generation

KV cache is transferred from prefill to decode workers.
"""

class DisaggregatedKVCache:
    """KV cache for disaggregated serving."""

    def __init__(self, role):
        self.role = role  # 'prefill' or 'decode'

    def prefill_and_transfer(self, prompt_tokens, decode_worker_id):
        """Prefill KV cache and transfer to decode worker."""

        assert self.role == 'prefill'

        # Compute KV cache
        kv_cache = self.compute_kv_cache(prompt_tokens)

        # Transfer to decode worker
        transfer_kv_cache(
            kv_cache,
            src_rank=self.worker_id,
            dst_rank=decode_worker_id,
        )

    def receive_and_decode(self, prefill_worker_id):
        """Receive KV cache and start decoding."""

        assert self.role == 'decode'

        # Receive KV cache
        kv_cache = receive_kv_cache(src_rank=prefill_worker_id)

        # Generate tokens
        generated_tokens = self.decode(kv_cache)

        return generated_tokens
```

---

## Summary

Distributed KV cache management is crucial for scalable inference:

**Key Strategies:**
- Tensor parallel partitioning (heads across GPUs)
- Paged memory management (vLLM's approach)
- CPU offloading for overflow
- Prefix sharing for efficiency

**Challenges:**
- Memory overhead (10+ GB per long sequence)
- Transfer latency for migration
- Coherence across replicas
- Load balancing with cache locality

**Best Practices:**
- Use paged KV cache for flexibility
- Leverage prefix caching for shared prompts
- Profile memory usage carefully
- Consider disaggregated architectures for very large scale

Continue to the next module on load balancing strategies.
