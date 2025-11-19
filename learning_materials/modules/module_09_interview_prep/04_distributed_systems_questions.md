# Distributed Systems Interview Questions

## Introduction

This document contains 12 detailed questions on distributed systems for LLM inference, focusing on parallelism strategies, communication optimization, and scaling challenges.

**Target Roles**: Distributed Systems Engineer, ML Infrastructure Engineer (Multi-GPU)

---

## Question 1: Explain Tensor Parallelism

### Question
"How does tensor parallelism work for LLM inference? Walk through the implementation and communication pattern."

### Detailed Answer

**Core Concept**:
Tensor parallelism splits individual layers across multiple GPUs. Each GPU holds a portion of the weights and computes a portion of the output.

**Column Parallelism (Linear layer)**:
```python
# Standard: Y = XW (X: [B, D_in], W: [D_in, D_out])
# On single GPU: Y = X @ W

# Tensor Parallel (split columns):
# GPU 0: W_0 = W[:, :D_out/2]
# GPU 1: W_1 = W[:, D_out/2:]

# Each GPU computes:
# GPU 0: Y_0 = X @ W_0  # [B, D_out/2]
# GPU 1: Y_1 = X @ W_1  # [B, D_out/2]

# Result: Concatenate [Y_0, Y_1] → Y [B, D_out]
# No communication needed! (f-split in Megatron-LM)
```

**Row Parallelism (Linear layer)**:
```python
# Split rows of weight matrix:
# GPU 0: W_0 = W[:D_in/2, :]
# GPU 1: W_1 = W[D_in/2:, :]

# Need to split input X:
# GPU 0: X_0 = X[:, :D_in/2]
# GPU 1: X_1 = X[:, D_in/2:]

# Each GPU computes:
# GPU 0: Y_0 = X_0 @ W_0
# GPU 1: Y_1 = X_1 @ W_1

# Combine with All-Reduce: Y = Y_0 + Y_1
# Communication: AllReduce (g-split in Megatron-LM)
```

**Transformer Block with TP**:
```python
class TransformerBlockTP:
    def forward(self, x):
        # Multi-Head Attention (TP across heads)
        # Each GPU: subset of attention heads
        # Q, K, V: column-parallel (no communication)
        q, k, v = self.qkv_proj(x)  # Column-parallel

        # Attention computation (local to each GPU)
        attn_out = self.attention(q, k, v)

        # Output projection: row-parallel (AllReduce)
        attn_out = self.out_proj(attn_out)  # All-reduce here

        # Feed-Forward Network
        # First layer: column-parallel (no communication)
        ff = self.ff1(attn_out)  # Column-parallel
        ff = self.activation(ff)

        # Second layer: row-parallel (AllReduce)
        ff_out = self.ff2(ff)  # All-reduce here

        return ff_out + x  # Residual
```

**Communication Pattern (TP=4)**:
```
Per transformer layer:
1. QKV projection: No communication
2. Attention output: AllReduce (1 communication)
3. FF1: No communication
4. FF2: AllReduce (1 communication)

Total: 2 AllReduce operations per layer

For 32-layer model:
64 AllReduce operations per forward pass

Data size per AllReduce: [batch_size, seq_len, hidden_dim]
For batch=32, seq=100, hidden=4096:
32 * 100 * 4096 * 2 bytes = ~26 MB per AllReduce
Total communication: 64 * 26 MB = ~1.66 GB per batch
```

**Follow-up**: Scaling efficiency analysis:
```
Ideal speedup (TP=N): N× faster
Actual speedup: ~0.8-0.9 × N (80-90% efficiency)

Overhead sources:
1. AllReduce communication: ~10-15% of time
2. Load imbalance: ~2-5%
3. Synchronization: ~2-5%

Efficiency improves with:
- Larger models (computation dominates communication)
- Faster interconnect (NVLink > PCIe)
- Larger batch sizes (amortize communication)
```

---

## Question 2: Compare Tensor vs Pipeline Parallelism

### Condensed Answer

| Aspect | Tensor Parallelism | Pipeline Parallelism |
|--------|-------------------|---------------------|
| **Granularity** | Within layer | Across layers |
| **Communication** | High (every layer) | Low (at boundaries) |
| **Latency** | Low (parallel) | Higher (sequential stages) |
| **Throughput** | Good for small batches | Best for large batches |
| **Bubble Overhead** | None | 10-30% idle time |
| **Scaling** | Up to ~8 GPUs | Scales to 100s of GPUs |
| **Complexity** | Moderate | High (scheduling) |

**When to use**:
- **TP**: ≤8 GPUs, low latency needed, real-time serving
- **PP**: >8 GPUs, batch workloads, very large models
- **Hybrid (TP+PP)**: Best of both - TP within node, PP across nodes

---

## Question 3: Explain NCCL and Collective Operations

### Condensed Answer

**NCCL**: NVIDIA Collective Communication Library - optimized multi-GPU communication

**Key Operations**:
1. **AllReduce**: Sum across all GPUs, result to all
2. **AllGather**: Concatenate data from all GPUs
3. **ReduceScatter**: Sum + distribute partitions
4. **Broadcast**: One → all
5. **Point-to-Point**: Send/receive between pairs

**AllReduce Algorithms**:
- **Ring AllReduce**: O(N) bandwidth, N GPUs
- **Tree AllReduce**: O(log N) latency
- **Double Binary Tree**: Used by NCCL

**Performance**: NVLink: ~300 GB/s per GPU, PCIe: ~24 GB/s

---

## Questions 4-12: Additional Topics

*Condensed for brevity*

---

## Question 4: How to Optimize Communication Overhead?

**Strategies**:
1. **Overlap communication and computation**: Start AllReduce before kernel finishes
2. **Fuse operations**: Combine multiple small communications
3. **Compression**: FP16 instead of FP32 for gradients
4. **Topology-aware**: Place GPUs optimally (NVLink)

---

## Question 5: Explain Pipeline Parallelism Bubbles

**Problem**: GPUs idle while waiting for previous stage
**Solution**: Micro-batching, GPipe, 1F1B schedule
**Overhead**: ~(PP_size - 1) / micro_batches

---

## Question 6: How Does vLLM Handle Multi-GPU?

**Answer**: Ray + Tensor Parallelism
- Ray for distributed orchestration
- TP for model parallelism
- Each worker group = TP set

---

## Question 7: Distributed KV Cache Management

**Challenges**: Each GPU has partial KV cache
**Solution**: TP splits heads, each GPU stores subset
**Communication**: During attention, no extra communication needed

---

## Question 8: How to Handle GPU Failures?

**Detection**: Health checks, CUDA errors
**Recovery**: Checkpoint-restart, job re-queue
**Mitigation**: Redundancy, graceful degradation

---

## Question 9: Network Requirements for Multi-Node

**Bandwidth**: 100 Gbps+ for multi-node TP
**Latency**: <10μs for NVLink, <100μs for InfiniBand
**Topology**: Fat-tree, rail-optimized

---

## Question 10: Explain Data Parallelism vs Model Parallelism

**Data Parallel**: Replicate model, split data
**Model Parallel**: Split model (TP or PP)
**Hybrid**: Combine both

---

## Question 11: Scaling Efficiency Metrics

**Formulas**:
- **Strong scaling**: Speedup = T(1) / T(N)
- **Weak scaling**: Keep per-GPU work constant
- **Communication efficiency**: (Compute time) / (Compute + Comm time)

---

## Question 12: When to Use Sequence Parallelism?

**Use**: Very long sequences (>10K tokens)
**Method**: Split sequence dimension
**Challenge**: Attention requires all-to-all communication

---

*Last Updated: 2025-11-19*
*Module 9: Distributed Systems Questions*
