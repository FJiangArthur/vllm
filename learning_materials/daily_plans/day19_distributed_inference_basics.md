# Day 19: Distributed Inference Basics

> **Goal**: Understand multi-GPU distributed inference strategies and communication primitives
> **Time**: 4-6 hours
> **Prerequisites**: Day 18 complete, understanding of GPU execution
> **Deliverables**: Distributed execution diagrams, communication analysis, multi-GPU setup guide

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): Distributed Inference Overview

**9:00-9:30** - Why Distributed Inference?
**9:30-10:30** - Parallelism Strategies Overview
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Communication Primitives (NCCL)

### Afternoon Session (2-3 hours): Implementation Deep Dive

**14:00-15:00** - vLLM Distributed Architecture
**15:00-16:00** - Worker Model & Execution
**16:00-16:30** - Break
**16:30-17:30** - Hands-On: Multi-GPU Setup

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Review distributed concepts, diagram flows, preview Day 20

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain when and why to use distributed inference
- [ ] Understand different parallelism strategies (TP, PP, DP)
- [ ] Use NCCL communication primitives
- [ ] Trace distributed execution in vLLM
- [ ] Set up multi-GPU inference

---

## 📂 Morning: Distributed Inference Overview (9:00-12:00)

### Task 1: Why Distributed Inference? (30 min)

**Problem: Model Too Large for Single GPU**

```
Example: Llama-2 70B
- Parameters: 70 billion
- Memory (FP16): 70B × 2 bytes = 140 GB
- KV cache (1024 tokens): ~40 GB
- Total: ~180 GB

Single GPU:
- A100 40GB: ✗ Can't fit
- A100 80GB: ✗ Can't fit
- H100 80GB: ✗ Can't fit

Solution: Split across multiple GPUs!
```

**Benefits of Distributed Inference**:

```
1. Memory Capacity:
   - Fit large models (70B, 175B+)
   - 4× A100 40GB = 160GB total
   - Can serve Llama-2 70B

2. Throughput:
   - Parallel computation
   - Multiple GPUs work together
   - Higher tokens/second

3. Flexibility:
   - Trade latency for throughput
   - Scale to larger models
   - Better resource utilization

Challenges:
✗ Communication overhead
✗ Load balancing
✗ Complexity
✗ Need fast interconnect (NVLink, InfiniBand)
```

### Task 2: Parallelism Strategies Overview (60 min)

**Strategy 1: Data Parallelism (DP)**

```
Each GPU has complete model copy, processes different data.

GPU 0: Model copy 1 → Requests [A, B]
GPU 1: Model copy 2 → Requests [C, D]
GPU 2: Model copy 3 → Requests [E, F]
GPU 3: Model copy 4 → Requests [G, H]

Pros:
✓ Simple implementation
✓ No communication during forward pass
✓ Linear throughput scaling

Cons:
✗ Each GPU needs full model (memory inefficient)
✗ Doesn't solve "model too large" problem
✗ Used mainly for training, not inference

Use case: Multiple instances of same model
```

**Strategy 2: Tensor Parallelism (TP)**

```
Split model tensors (weights, activations) across GPUs.

Example: Linear layer (Y = XW)
- Input X: [batch, d_in]
- Weight W: [d_in, d_out]
- Split W column-wise across GPUs

GPU 0: W[:, 0:d_out/4]     → Y[:, 0:d_out/4]
GPU 1: W[:, d_out/4:2*d_out/4] → Y[:, d_out/4:2*d_out/4]
GPU 2: W[:, 2*d_out/4:3*d_out/4] → Y[:, 2*d_out/4:3*d_out/4]
GPU 3: W[:, 3*d_out/4:d_out] → Y[:, 3*d_out/4:d_out]

Then: All-gather Y to get complete output

Pros:
✓ Reduces memory per GPU (model split)
✓ Can fit larger models
✓ Low latency (single forward pass)

Cons:
✗ Communication every layer (all-reduce/all-gather)
✗ Requires fast GPU interconnect
✗ Limited scaling (typically 2-8 GPUs)

Use case: Large models, low latency required
```

**Strategy 3: Pipeline Parallelism (PP)**

```
Split model layers across GPUs (vertical split).

GPU 0: Layers 0-7   (Embedding + first layers)
GPU 1: Layers 8-15  (Middle layers)
GPU 2: Layers 16-23 (Middle layers)
GPU 3: Layers 24-31 (Final layers + LM head)

Data flows: GPU 0 → GPU 1 → GPU 2 → GPU 3

Pros:
✓ Reduces memory per GPU (layer split)
✓ Simpler communication (point-to-point)
✓ Scales to more GPUs

Cons:
✗ Pipeline bubbles (idle time)
✗ Higher latency (sequential stages)
✗ Load balancing challenges

Use case: Very deep models, high throughput
```

**Hybrid: TP + PP**

```
Combine both strategies for best of both worlds.

Example: 8 GPUs for 70B model
- TP=4 (4-way tensor parallel)
- PP=2 (2-stage pipeline)

Stage 0 (Layers 0-15):
  GPU 0, 1, 2, 3 (TP group)

Stage 1 (Layers 16-31):
  GPU 4, 5, 6, 7 (TP group)

Benefits:
✓ Memory efficiency (both splits)
✓ Balanced latency/throughput
✓ Flexible scaling
```

**Comparison Table**:

| Strategy | Memory/GPU | Latency | Throughput | Communication | Best For |
|----------|------------|---------|------------|---------------|----------|
| DP | High (full model) | Low | High | Low | Multiple instances |
| TP | Low (split) | Low | Medium | High | Large models, low latency |
| PP | Low (split) | High | High | Medium | Very deep models |
| TP+PP | Low | Medium | High | Medium | Largest models |

### Task 3: Communication Primitives (60 min)

**NCCL (NVIDIA Collective Communications Library)**

```python
# NCCL provides efficient GPU-to-GPU communication

import torch.distributed as dist

# Initialize process group
dist.init_process_group(
    backend='nccl',  # Use NCCL for GPU
    init_method='env://',
    world_size=4,    # Total GPUs
    rank=0,          # This GPU's ID
)
```

**Key Collectives**:

**1. All-Reduce**

```
Reduce values from all GPUs, broadcast result.

GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]
GPU 3: [10, 11, 12]

After all_reduce(sum):
All GPUs: [22, 26, 30]  # Element-wise sum

Code:
tensor = torch.tensor([1, 2, 3], device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# tensor now contains sum across all GPUs

Use in TP: Combine partial results from split computation
```

**2. All-Gather**

```
Gather tensors from all GPUs to all GPUs.

GPU 0: [1, 2]
GPU 1: [3, 4]
GPU 2: [5, 6]
GPU 3: [7, 8]

After all_gather:
All GPUs: [1, 2, 3, 4, 5, 6, 7, 8]

Code:
tensor = torch.tensor([1, 2], device='cuda')
tensor_list = [torch.zeros_like(tensor) for _ in range(4)]
dist.all_gather(tensor_list, tensor)
# tensor_list contains all tensors

Use in TP: Collect results from all GPUs
```

**3. Reduce-Scatter**

```
Reduce then scatter (opposite of all-gather).

GPU 0: [1, 2, 3, 4]
GPU 1: [5, 6, 7, 8]
GPU 2: [9, 10, 11, 12]
GPU 3: [13, 14, 15, 16]

After reduce_scatter(sum):
GPU 0: [28, 32]        # Sum of [0:2] from all
GPU 1: [36, 40]        # Sum of [2:4] from all
GPU 2: [44, 48]
GPU 3: [52, 56]

Use in TP: Distribute reduced results
```

**4. Broadcast**

```
Send tensor from one GPU to all others.

Before:
GPU 0 (root): [1, 2, 3]
GPU 1: [?, ?, ?]
GPU 2: [?, ?, ?]
GPU 3: [?, ?, ?]

After broadcast:
All GPUs: [1, 2, 3]

Code:
tensor = torch.tensor([1, 2, 3], device='cuda')
dist.broadcast(tensor, src=0)  # GPU 0 is source

Use: Share configuration, initial data
```

**Communication Patterns in Attention**:

```python
# Tensor Parallel Attention

def tensor_parallel_attention(
    query, key, value,
    tp_group,  # Tensor parallel process group
):
    """
    Attention with tensor parallelism.

    Split: Heads divided across GPUs
    GPU 0: Heads 0-7
    GPU 1: Heads 8-15
    GPU 2: Heads 16-23
    GPU 3: Heads 24-31
    """

    # Each GPU computes attention for its heads (local)
    local_output = compute_attention_local(
        query[:, local_heads, :],
        key[:, local_heads, :],
        value[:, local_heads, :]
    )

    # No communication needed - outputs are independent!

    # Later: All-gather for next layer if needed
    if need_full_output:
        full_output = all_gather(local_output, tp_group)
    else:
        full_output = local_output

    return full_output


# Tensor Parallel MLP
def tensor_parallel_mlp(input, weight, tp_group):
    """
    MLP with tensor parallelism.

    Split weight column-wise: W = [W0 | W1 | W2 | W3]
    """

    # Each GPU has part of weight matrix
    # Y = X @ W → Y = X @ [W0 | W1 | W2 | W3]
    #           = [X@W0 | X@W1 | X@W2 | X@W3]

    # Local computation
    local_output = torch.matmul(input, local_weight)

    # All-reduce to combine (if needed) OR keep split for next layer
    # Depends on next layer's requirements

    return local_output
```

**Bandwidth Requirements**:

```python
def calculate_communication_volume(
    model_size_b: int,
    num_gpus: int,
    batch_size: int,
    seq_len: int,
):
    """
    Estimate communication volume for tensor parallelism.
    """

    hidden_size = 4096  # Typical for 7B model
    num_layers = 32

    # Per layer: 2 all-reduce (attention out + MLP out)
    bytes_per_token = 2 * hidden_size * 2  # FP16

    # Total per forward pass
    total_bytes = (
        bytes_per_token *
        batch_size *
        seq_len *
        num_layers *
        2  # 2 all-reduces per layer
    )

    # Bandwidth requirement (assuming 100ms target latency)
    required_bandwidth_gbps = total_bytes / 0.1 / 1e9

    print(f"Communication Analysis:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num GPUs (TP): {num_gpus}")
    print(f"  Total communication: {total_bytes / 1e9:.2f} GB")
    print(f"  Required bandwidth: {required_bandwidth_gbps:.2f} GB/s")

    # Compare to hardware
    nvlink_bw = 600  # GB/s for NVLink
    pcie_bw = 32     # GB/s for PCIe 4.0 x16

    print(f"\n  NVLink (600 GB/s): {'✓ OK' if required_bandwidth_gbps < nvlink_bw else '✗ Too slow'}")
    print(f"  PCIe (32 GB/s): {'✓ OK' if required_bandwidth_gbps < pcie_bw else '✗ Too slow'}")

# Example
calculate_communication_volume(
    model_size_b=70,
    num_gpus=4,
    batch_size=8,
    seq_len=512
)
```

---

## 🔬 Afternoon: Implementation Deep Dive (14:00-17:30)

### Task 4: vLLM Distributed Architecture (60 min)

**File**: `vllm/distributed/parallel_state.py`

**Distributed Process Group**:

```python
# vLLM distributed initialization

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    """
    Initialize model parallel groups.

    Args:
        tensor_model_parallel_size: TP group size
        pipeline_model_parallel_size: PP group size
    """

    world_size = torch.distributed.get_world_size()
    assert world_size == tensor_model_parallel_size * pipeline_model_parallel_size

    # Create tensor parallel groups
    # Example: 8 GPUs, TP=4, PP=2
    # TP groups: [0,1,2,3], [4,5,6,7]
    # PP groups: [0,4], [1,5], [2,6], [3,7]

    for pp_rank in range(pipeline_model_parallel_size):
        for tp_rank in range(tensor_model_parallel_size):
            rank = pp_rank * tensor_model_parallel_size + tp_rank

            # Tensor parallel group
            tp_group_ranks = [
                pp_rank * tensor_model_parallel_size + i
                for i in range(tensor_model_parallel_size)
            ]
            tp_group = dist.new_group(tp_group_ranks)

            # Pipeline parallel group
            pp_group_ranks = [
                pp * tensor_model_parallel_size + tp_rank
                for pp in range(pipeline_model_parallel_size)
            ]
            pp_group = dist.new_group(pp_group_ranks)
```

**Worker Architecture**:

```python
# vllm/worker/worker.py

class Worker:
    """
    Worker runs on each GPU process.

    Responsibilities:
    - Load model shard
    - Execute forward pass
    - Communicate with other workers (TP/PP)
    - Manage local KV cache
    """

    def __init__(self, model_config, parallel_config, ...):
        self.model_config = model_config
        self.parallel_config = parallel_config

        # Determine this worker's role
        self.tensor_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_parallel_rank = get_pipeline_model_parallel_rank()

        # Load model shard
        self.model = self._load_model_shard()

        # Initialize KV cache
        self.kv_cache = self._init_kv_cache()

    def _load_model_shard(self):
        """
        Load appropriate shard of model weights.

        With TP=4:
        - GPU 0: Load columns 0:d/4 of all weight matrices
        - GPU 1: Load columns d/4:2d/4
        - GPU 2: Load columns 2d/4:3d/4
        - GPU 3: Load columns 3d/4:d
        """

        # This is handled by ParallelLinear layers
        model = load_model(self.model_config)

        return model

    def execute_model(self, input_metadata):
        """
        Execute model forward pass.

        Communication happens inside model layers.
        """

        output = self.model(
            input_ids=input_metadata.input_ids,
            positions=input_metadata.positions,
            kv_caches=self.kv_cache,
        )

        return output
```

### Task 5: Distributed Execution Flow (60 min)

**Single Step with Tensor Parallelism**:

```
Step execution with TP=4:

1. Driver (GPU 0) coordinates:
   ┌────────────────────────────────────┐
   │ Driver: Schedule requests          │
   │  - Select batch                    │
   │  - Prepare metadata                │
   └───────────┬────────────────────────┘
               │
               ├─ Broadcast to all workers
               ↓

2. All workers execute in parallel:
   ┌─────────────────────────────────────────────────┐
   │ Worker 0        Worker 1        Worker 2   Worker 3│
   │ (GPU 0)         (GPU 1)         (GPU 2)    (GPU 3) │
   │                                                      │
   │ Embedding       Embedding       Embedding  Embedding│
   │ (duplicated)    (duplicated)    (duplicated) (dup.) │
   │     │               │               │          │     │
   │     ↓               ↓               ↓          ↓     │
   │ Layer 0         Layer 0         Layer 0    Layer 0  │
   │ Heads 0-7       Heads 8-15      Heads 16-23 24-31   │
   │     │               │               │          │     │
   │     └───────────────┴───────all_gather───────┘     │
   │                      │                              │
   │                      ↓                              │
   │                  All have                           │
   │                  full output                        │
   │                      │                              │
   │     ┌────────────────┴───────────────┐             │
   │     ↓               ↓               ↓          ↓    │
   │ Layer 1         Layer 1         Layer 1    Layer 1 │
   │ ... (repeat for all layers)                        │
   └─────────────────────────────────────────────────────┘

3. Gather final results:
   - All workers have logits for their head splits
   - All-gather to get complete logits
   - Sample tokens (each worker does same sampling)

4. Return to driver:
   - Workers send results to driver
   - Driver updates sequences
```

**Communication Timeline**:

```
Timeline for one layer with TP=4:

|<──────────────  Layer N computation ────────────────>|

├─ Compute ─┤├─Comm─┤├─ Compute ─┤├─Comm─┤├─Compute─┤
   QKV proj   AllReduce  Attn+Proj  AllReduce  MLP

Total time = Compute + Communication
  Compute: ~80% (actual work)
  Communication: ~20% (overhead)

With NVLink (600 GB/s):
  Communication: 1-2ms per layer
  Acceptable overhead

With PCIe (32 GB/s):
  Communication: 20-40ms per layer
  High overhead - TP not practical
```

### Task 6: Multi-GPU Setup (60 min)

**Setup Multi-GPU vLLM**:

```python
#!/usr/bin/env python3
"""
Run vLLM with tensor parallelism.
"""

from vllm import LLM, SamplingParams

# Launch with tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=4,  # Use 4 GPUs
    dtype="float16",
    max_model_len=2048,
)

# Use normally
prompts = ["Hello, my name is"] * 10
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

for output in outputs:
    print(output.outputs[0].text)
```

**Command-line Server**:

```bash
# Start API server with TP
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-hf \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len 2048

# Server will automatically distribute across 4 GPUs
```

**Verify Multi-GPU Usage**:

```bash
# In another terminal, monitor GPUs
watch -n 0.5 nvidia-smi

# Expected output:
# GPU 0: 95% util, 35GB memory
# GPU 1: 95% util, 35GB memory
# GPU 2: 95% util, 35GB memory
# GPU 3: 95% util, 35GB memory
# All GPUs actively used!
```

**Debug Distributed Issues**:

```python
# Enable distributed debugging
import os
os.environ['NCCL_DEBUG'] = 'INFO'  # Detailed NCCL logging
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# Run vLLM
# Logs will show:
# - Process group creation
# - NCCL operations
# - Communication volumes
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Distributed Strategies**
- Data, Tensor, Pipeline Parallelism
- Trade-offs and use cases
- Hybrid approaches

✅ **Communication**
- NCCL collectives (all-reduce, all-gather)
- Bandwidth requirements
- Communication patterns

✅ **vLLM Implementation**
- Worker architecture
- Process group setup
- Multi-GPU execution

### Knowledge Check Questions

**Q1**: When would you choose Tensor Parallelism over Pipeline Parallelism?
<details>
<summary>Answer</summary>
**Choose TP when**:
- Model fits on few GPUs (2-8)
- Low latency critical
- Have fast interconnect (NVLink)
- Model is wide (large hidden size)

**Choose PP when**:
- Very deep model (many layers)
- Higher latency acceptable
- Slower interconnect OK
- Need to scale to many GPUs (8+)

**Or use both** (hybrid) for largest models!
</details>

**Q2**: What does all-reduce do and why is it needed in TP?
<details>
<summary>Answer</summary>
**All-reduce**: Reduce values across GPUs, broadcast result to all

**Example**: Sum partial results
- GPU 0: [1, 2]
- GPU 1: [3, 4]
- After all_reduce(SUM): Both have [4, 6]

**Used in TP** to combine partial outputs from split computation:
- Each GPU computes part of layer output
- All-reduce sums partial results
- All GPUs get full result for next layer
</details>

**Q3**: Calculate memory per GPU for Llama-2 70B with TP=4.
<details>
<summary>Answer</summary>
Model: 70B parameters
Precision: FP16 (2 bytes/param)

Total memory: 70B × 2 = 140 GB

With TP=4:
**Memory per GPU: 140 GB / 4 = 35 GB**

Plus KV cache (~10-20 GB) and activations (~5 GB):
**Total per GPU: ~50-60 GB**

Fits on A100 80GB ✓
</details>

---

## 🚀 Preview: Day 20

Tomorrow you'll dive deep into:
- **Tensor Parallelism Implementation**: Layer-by-layer breakdown
- **Pipeline Parallelism**: Staging and scheduling
- **Communication Optimization**: Overlapping, fusion
- **Scaling Analysis**: Performance at different TP/PP configurations

**Preparation**:
- Review distributed concepts from today
- Understand matrix partitioning
- Think about communication bottlenecks

---

## 📚 Additional Resources

**Key Files**:
- [ ] `vllm/distributed/parallel_state.py`
- [ ] `vllm/worker/worker.py`
- [ ] `vllm/model_executor/layers/parallel_linear.py`

**Concepts**:
- [ ] NCCL documentation
- [ ] Megatron-LM (inspiration for vLLM's TP)
- [ ] Ring all-reduce algorithm

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
