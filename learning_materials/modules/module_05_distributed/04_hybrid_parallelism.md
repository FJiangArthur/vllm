# Hybrid Parallelism

## Table of Contents
1. [Introduction to Hybrid Parallelism](#introduction)
2. [3D Parallelism: DP + TP + PP](#3d-parallelism)
3. [Process Group Organization](#process-groups)
4. [Optimal Parallelism Configuration](#optimal-config)
5. [Memory and Communication Analysis](#analysis)
6. [vLLM Hybrid Parallelism](#vllm-implementation)
7. [Scaling Laws](#scaling-laws)
8. [Configuration Strategies](#strategies)
9. [Advanced Techniques](#advanced-techniques)
10. [Troubleshooting](#troubleshooting)

---

## Introduction to Hybrid Parallelism {#introduction}

Hybrid parallelism combines multiple parallelization strategies to efficiently train and serve extremely large models. The most common approach is **3D parallelism**: combining Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline Parallelism (PP).

### Why Hybrid Parallelism?

**Single Strategy Limitations:**

```
Data Parallelism (DP) only:
- Model must fit in single GPU
- Limited by GPU memory (~80GB for A100)
- Max model: ~70B parameters in FP16

Tensor Parallelism (TP) only:
- High communication overhead
- Limited scalability (typically ≤8 GPUs)
- Bandwidth becomes bottleneck

Pipeline Parallelism (PP) only:
- Pipeline bubble overhead
- Load imbalance across stages
- Limited by model depth
```

**Hybrid Approach Benefits:**

```
TP + PP + DP:
- Models: 100B - 1T+ parameters
- Efficient use of 100s-1000s of GPUs
- Balanced communication patterns
- Flexible configuration for different hardware
```

### Parallelism Comparison

| Strategy | Splits | Communication | Best For |
|----------|--------|---------------|----------|
| Data Parallelism | Data across GPUs | All-reduce gradients | Small models, many GPUs |
| Tensor Parallelism | Layers across GPUs | All-reduce per layer | Wide layers, NVLink |
| Pipeline Parallelism | Stages across GPUs | Point-to-point | Deep models, slower interconnect |
| Hybrid (3D) | All dimensions | Optimized | Very large models |

---

## 3D Parallelism: DP + TP + PP {#3d-parallelism}

### Conceptual Model

```python
"""
3D Parallelism Visualization

Total GPUs: 64
TP size: 4 (tensor parallelism)
PP size: 4 (pipeline parallelism)
DP size: 4 (data parallelism)

Total = TP × PP × DP = 4 × 4 × 4 = 64 GPUs

GPU Organization:
- 4 data parallel groups
- Each DP group has 16 GPUs (4 TP × 4 PP)
- Each TP group has 4 GPUs working on same layers
- Each PP group has 4 stages, each stage has 4 GPUs (TP)

Visualization:

Data Parallel Replica 0:
  Pipeline Stage 0: [GPU0, GPU1, GPU2, GPU3] <- TP group
  Pipeline Stage 1: [GPU4, GPU5, GPU6, GPU7] <- TP group
  Pipeline Stage 2: [GPU8, GPU9, GPU10, GPU11] <- TP group
  Pipeline Stage 3: [GPU12, GPU13, GPU14, GPU15] <- TP group

Data Parallel Replica 1:
  Pipeline Stage 0: [GPU16, GPU17, GPU18, GPU19]
  Pipeline Stage 1: [GPU20, GPU21, GPU22, GPU23]
  Pipeline Stage 2: [GPU24, GPU25, GPU26, GPU27]
  Pipeline Stage 3: [GPU28, GPU29, GPU30, GPU31]

... (2 more DP replicas)
"""


class HybridParallelConfig:
    """Configuration for 3D parallelism."""

    def __init__(
        self,
        data_parallel_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        self.dp_size = data_parallel_size
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size

        # Total GPUs required
        self.world_size = dp_size * tp_size * pp_size

    def get_rank_mapping(self, global_rank: int):
        """
        Map global GPU rank to (dp_rank, tp_rank, pp_rank).

        Organization: DP is outermost, then PP, then TP (innermost)
        """
        # TP changes fastest
        tp_rank = global_rank % self.tp_size

        # PP changes next
        pp_rank = (global_rank // self.tp_size) % self.pp_size

        # DP changes slowest
        dp_rank = global_rank // (self.tp_size * self.pp_size)

        return dp_rank, tp_rank, pp_rank

    def get_global_rank(self, dp_rank: int, tp_rank: int, pp_rank: int):
        """Convert (dp_rank, tp_rank, pp_rank) to global rank."""
        global_rank = (
            dp_rank * (self.pp_size * self.tp_size) +
            pp_rank * self.tp_size +
            tp_rank
        )
        return global_rank

    def visualize(self):
        """Print 3D parallelism configuration."""
        print(f"3D Parallelism Configuration")
        print(f"=" * 60)
        print(f"Data Parallel Size: {self.dp_size}")
        print(f"Tensor Parallel Size: {self.tp_size}")
        print(f"Pipeline Parallel Size: {self.pp_size}")
        print(f"Total GPUs: {self.world_size}")
        print()

        # Show first DP replica
        print("GPU Layout (Data Parallel Replica 0):")
        for pp_rank in range(self.pp_size):
            gpus = []
            for tp_rank in range(self.tp_size):
                global_rank = self.get_global_rank(0, tp_rank, pp_rank)
                gpus.append(global_rank)
            print(f"  Pipeline Stage {pp_rank}: {gpus}")


# Example
config = HybridParallelConfig(
    data_parallel_size=4,
    tensor_parallel_size=4,
    pipeline_parallel_size=4,
)
config.visualize()
```

### Communication Patterns

```python
def analyze_communication_patterns(config: HybridParallelConfig):
    """
    Analyze communication patterns in 3D parallelism.
    """

    print("Communication Patterns in 3D Parallelism")
    print("=" * 60)

    # Tensor Parallelism Communication
    print("\n1. Tensor Parallelism (within each TP group):")
    print(f"   - Groups: {config.dp_size * config.pp_size}")
    print(f"   - Size: {config.tp_size} GPUs per group")
    print(f"   - Pattern: All-reduce after each layer")
    print(f"   - Frequency: ~2× per transformer layer")
    print(f"   - Data size: O(batch_size × seq_len × hidden_dim)")
    print(f"   - Bandwidth requirement: HIGH (needs NVLink)")

    # Pipeline Parallelism Communication
    print("\n2. Pipeline Parallelism (between PP stages):")
    print(f"   - Groups: {config.dp_size * config.tp_size}")
    print(f"   - Size: {config.pp_size} stages")
    print(f"   - Pattern: Point-to-point (send/recv)")
    print(f"   - Frequency: Once per micro-batch per stage")
    print(f"   - Data size: O(micro_batch × seq_len × hidden_dim)")
    print(f"   - Bandwidth requirement: MEDIUM")

    # Data Parallelism Communication
    print("\n3. Data Parallelism (across DP replicas):")
    print(f"   - Groups: 1 (all DP replicas)")
    print(f"   - Size: {config.dp_size} replicas")
    print(f"   - Pattern: All-reduce gradients")
    print(f"   - Frequency: Once per global batch")
    print(f"   - Data size: O(model_parameters)")
    print(f"   - Bandwidth requirement: LOW (infrequent)")


config = HybridParallelConfig(dp_size=4, tp_size=4, pp_size=4)
analyze_communication_patterns(config)
```

---

## Process Group Organization {#process-groups}

### Creating Process Groups

```python
import torch.distributed as dist
from typing import List, Tuple


class ProcessGroupManager:
    """
    Manage process groups for 3D parallelism.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        dp_size: int,
        tp_size: int,
        pp_size: int,
    ):
        self.world_size = world_size
        self.rank = rank
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size

        # Initialize all process groups
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None

        self._initialize_process_groups()

    def _initialize_process_groups(self):
        """Create all necessary process groups."""

        # Tensor Parallel Groups
        # Each TP group contains tp_size consecutive ranks
        self.tp_groups = []
        for i in range(0, self.world_size, self.tp_size):
            ranks = list(range(i, min(i + self.tp_size, self.world_size)))
            group = dist.new_group(ranks)
            self.tp_groups.append(group)

            if self.rank in ranks:
                self.tp_group = group
                self.tp_rank = ranks.index(self.rank)

        # Pipeline Parallel Groups
        # Each PP group contains ranks with same (dp_rank, tp_rank)
        self.pp_groups = []
        for dp_rank in range(self.dp_size):
            for tp_rank in range(self.tp_size):
                ranks = []
                for pp_rank in range(self.pp_size):
                    global_rank = (
                        dp_rank * (self.pp_size * self.tp_size) +
                        pp_rank * self.tp_size +
                        tp_rank
                    )
                    ranks.append(global_rank)

                group = dist.new_group(ranks)
                self.pp_groups.append(group)

                if self.rank in ranks:
                    self.pp_group = group
                    self.pp_rank = ranks.index(self.rank)

        # Data Parallel Groups
        # Each DP group contains all replicas of same (pp_rank, tp_rank)
        self.dp_groups = []
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                ranks = []
                for dp_rank in range(self.dp_size):
                    global_rank = (
                        dp_rank * (self.pp_size * self.tp_size) +
                        pp_rank * self.tp_size +
                        tp_rank
                    )
                    ranks.append(global_rank)

                group = dist.new_group(ranks)
                self.dp_groups.append(group)

                if self.rank in ranks:
                    self.dp_group = group
                    self.dp_rank = ranks.index(self.rank)

    def get_tensor_parallel_group(self):
        """Get tensor parallel process group for this rank."""
        return self.tp_group

    def get_pipeline_parallel_group(self):
        """Get pipeline parallel process group for this rank."""
        return self.pp_group

    def get_data_parallel_group(self):
        """Get data parallel process group for this rank."""
        return self.dp_group


# Usage example
def initialize_hybrid_parallelism():
    """
    Initialize 3D parallelism with PyTorch distributed.
    """
    # Initialize PyTorch distributed
    dist.init_process_group(backend='nccl')

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Configure parallelism
    dp_size = 4
    tp_size = 4
    pp_size = 4

    assert world_size == dp_size * tp_size * pp_size

    # Create process groups
    pg_manager = ProcessGroupManager(
        world_size=world_size,
        rank=rank,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
    )

    # Now can use process groups for communication
    tp_group = pg_manager.get_tensor_parallel_group()
    pp_group = pg_manager.get_pipeline_parallel_group()
    dp_group = pg_manager.get_data_parallel_group()

    return pg_manager
```

---

## Optimal Parallelism Configuration {#optimal-config}

### Choosing Parallelism Dimensions

```python
def recommend_parallelism_config(
    model_params_b: float,  # Model parameters in billions
    num_gpus: int,
    gpu_memory_gb: float,
    gpu_interconnect: str,  # 'nvlink', 'infiniband', 'ethernet'
    model_depth: int,  # Number of layers
):
    """
    Recommend optimal 3D parallelism configuration.

    Args:
        model_params_b: Model size in billions of parameters
        num_gpus: Total number of available GPUs
        gpu_memory_gb: Memory per GPU (GB)
        gpu_interconnect: Type of interconnect
        model_depth: Number of transformer layers

    Returns:
        Recommended (dp_size, tp_size, pp_size)
    """

    print(f"Configuration Recommendation")
    print(f"=" * 60)
    print(f"Model: {model_params_b}B parameters, {model_depth} layers")
    print(f"Hardware: {num_gpus} GPUs × {gpu_memory_gb}GB, {gpu_interconnect}")
    print()

    # Step 1: Determine minimum TP size (based on memory)
    model_memory_gb = model_params_b * 2  # FP16: 2 bytes per param
    overhead_factor = 1.5  # Activations, optimizer states, etc.
    required_memory = model_memory_gb * overhead_factor

    min_tp_size = math.ceil(required_memory / gpu_memory_gb)

    # Adjust based on interconnect
    if gpu_interconnect == 'nvlink':
        max_tp_size = 8  # Can use up to 8-way TP with NVLink
        recommended_tp_size = min(max_tp_size, 2 ** math.ceil(math.log2(min_tp_size)))
    elif gpu_interconnect == 'infiniband':
        max_tp_size = 4  # Limit TP with IB
        recommended_tp_size = min(max_tp_size, min_tp_size)
    else:  # ethernet
        max_tp_size = 2  # Very limited TP
        recommended_tp_size = min(max_tp_size, min_tp_size)

    print(f"Step 1 - Tensor Parallelism:")
    print(f"  Minimum TP size (memory): {min_tp_size}")
    print(f"  Recommended TP size: {recommended_tp_size}")
    print()

    # Step 2: Determine PP size (based on remaining GPUs and model depth)
    remaining_gpus = num_gpus // recommended_tp_size

    # Good rule: at least 4 layers per PP stage
    max_pp_size = model_depth // 4
    max_pp_size = min(max_pp_size, remaining_gpus)

    # Prefer smaller PP to reduce bubble overhead
    # Use PP only if necessary
    if remaining_gpus <= 8:
        recommended_pp_size = 1
    elif remaining_gpus <= 16:
        recommended_pp_size = min(2, max_pp_size)
    else:
        recommended_pp_size = min(4, max_pp_size)

    print(f"Step 2 - Pipeline Parallelism:")
    print(f"  Remaining GPUs: {remaining_gpus}")
    print(f"  Max PP size (layers): {max_pp_size}")
    print(f"  Recommended PP size: {recommended_pp_size}")
    print()

    # Step 3: Use remaining GPUs for DP
    recommended_dp_size = num_gpus // (recommended_tp_size * recommended_pp_size)

    print(f"Step 3 - Data Parallelism:")
    print(f"  Recommended DP size: {recommended_dp_size}")
    print()

    # Verify
    total_gpus = recommended_dp_size * recommended_tp_size * recommended_pp_size
    assert total_gpus <= num_gpus

    # Summary
    print(f"Final Recommendation:")
    print(f"  TP size: {recommended_tp_size}")
    print(f"  PP size: {recommended_pp_size}")
    print(f"  DP size: {recommended_dp_size}")
    print(f"  Total GPUs used: {total_gpus}/{num_gpus}")
    print()

    # Estimate efficiency
    tp_efficiency = 0.85 if recommended_tp_size <= 4 else 0.75
    pp_efficiency = 0.9 if recommended_pp_size == 1 else 0.8
    dp_efficiency = 0.95

    overall_efficiency = tp_efficiency * pp_efficiency * dp_efficiency
    print(f"Estimated efficiency: {overall_efficiency * 100:.1f}%")

    return recommended_tp_size, recommended_pp_size, recommended_dp_size


# Examples
print("Example 1: Llama-70B on 32× A100 (80GB) with NVLink")
recommend_parallelism_config(
    model_params_b=70,
    num_gpus=32,
    gpu_memory_gb=80,
    gpu_interconnect='nvlink',
    model_depth=80,
)

print("\n" + "=" * 60 + "\n")

print("Example 2: 175B model on 128× A100 with InfiniBand")
recommend_parallelism_config(
    model_params_b=175,
    num_gpus=128,
    gpu_memory_gb=80,
    gpu_interconnect='infiniband',
    model_depth=96,
)
```

---

## Memory and Communication Analysis {#analysis}

### Memory Breakdown

```python
def analyze_memory_usage(
    model_params_b: float,
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    num_layers: int,
    tp_size: int,
    pp_size: int,
):
    """
    Detailed memory analysis for hybrid parallelism.
    """

    print(f"Memory Analysis")
    print(f"=" * 60)

    # Model parameters
    total_params = model_params_b * 1e9
    params_per_gpu = total_params / (tp_size * pp_size)
    param_memory_gb = params_per_gpu * 2 / 1e9  # FP16

    print(f"1. Model Parameters:")
    print(f"   Total: {total_params / 1e9:.1f}B parameters")
    print(f"   Per GPU: {params_per_gpu / 1e9:.2f}B parameters")
    print(f"   Memory: {param_memory_gb:.2f} GB")
    print()

    # Activations (depends on PP stage)
    # Simplified: assuming uniform layer distribution
    layers_per_stage = num_layers / pp_size
    activation_per_layer = batch_size * seq_length * hidden_dim * 2 / 1e9  # FP16

    # TP reduces activation memory
    activation_per_layer_per_gpu = activation_per_layer / tp_size

    total_activation = activation_per_layer_per_gpu * layers_per_stage

    print(f"2. Activations:")
    print(f"   Per layer: {activation_per_layer:.2f} GB")
    print(f"   Per layer per GPU (TP): {activation_per_layer_per_gpu:.2f} GB")
    print(f"   Total per GPU: {total_activation:.2f} GB")
    print()

    # KV Cache (for inference)
    # KV cache is partitioned by TP
    kv_cache_per_token = hidden_dim * 2 * 2 / 1e9  # K and V, FP16
    kv_cache_per_seq = kv_cache_per_token * seq_length
    kv_cache_total = kv_cache_per_seq * batch_size / tp_size

    print(f"3. KV Cache (inference):")
    print(f"   Per sequence: {kv_cache_per_seq:.2f} GB")
    print(f"   Total per GPU: {kv_cache_total:.2f} GB")
    print()

    # Optimizer states (only for training, partitioned by DP)
    # Adam: 2× parameters (momentum + variance)
    optimizer_memory = param_memory_gb * 2

    print(f"4. Optimizer States (training):")
    print(f"   Per GPU: {optimizer_memory:.2f} GB")
    print()

    # Total
    total_inference = param_memory_gb + total_activation + kv_cache_total
    total_training = total_inference + optimizer_memory

    print(f"Total Memory Per GPU:")
    print(f"   Inference: {total_inference:.2f} GB")
    print(f"   Training: {total_training:.2f} GB")

    return {
        'param_memory': param_memory_gb,
        'activation_memory': total_activation,
        'kv_cache_memory': kv_cache_total,
        'optimizer_memory': optimizer_memory,
        'total_inference': total_inference,
        'total_training': total_training,
    }


# Example: Llama-70B
analyze_memory_usage(
    model_params_b=70,
    batch_size=32,
    seq_length=2048,
    hidden_dim=8192,
    num_layers=80,
    tp_size=4,
    pp_size=2,
)
```

### Communication Volume Analysis

```python
def analyze_communication_volume(
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    num_layers: int,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    num_micro_batches: int,
):
    """
    Analyze communication volume for different parallelism strategies.
    """

    print(f"Communication Volume Analysis")
    print(f"=" * 60)

    # Tensor Parallelism
    # All-reduce after each layer, twice per transformer layer
    all_reduce_size = batch_size * seq_length * hidden_dim * 2 / 1e9  # FP16
    num_all_reduces = num_layers * 2  # Attention + MLP
    tp_comm_per_layer = all_reduce_size * 2  # Send + receive
    tp_total_comm = tp_comm_per_layer * num_all_reduces

    print(f"1. Tensor Parallelism:")
    print(f"   All-reduce size: {all_reduce_size:.2f} GB")
    print(f"   All-reduces per forward: {num_all_reduces}")
    print(f"   Total TP communication: {tp_total_comm:.2f} GB")
    print()

    # Pipeline Parallelism
    # Point-to-point between stages, per micro-batch
    pp_activation_size = (batch_size // num_micro_batches) * seq_length * hidden_dim * 2 / 1e9
    pp_comm_per_microbatch = pp_activation_size * (pp_size - 1)  # Between stages
    pp_total_comm = pp_comm_per_microbatch * num_micro_batches

    print(f"2. Pipeline Parallelism:")
    print(f"   Activation size per micro-batch: {pp_activation_size:.2f} GB")
    print(f"   Stages: {pp_size}")
    print(f"   Micro-batches: {num_micro_batches}")
    print(f"   Total PP communication: {pp_total_comm:.2f} GB")
    print()

    # Data Parallelism
    # All-reduce gradients once per batch
    # Gradients = model parameters
    model_params_b = 70  # Example
    gradient_size = model_params_b * 2  # FP16, in GB
    dp_comm = gradient_size * 2  # Send + receive

    print(f"3. Data Parallelism:")
    print(f"   Gradient size: {gradient_size:.2f} GB")
    print(f"   Total DP communication: {dp_comm:.2f} GB")
    print()

    # Total
    total_comm = tp_total_comm + pp_total_comm + dp_comm

    print(f"Total Communication: {total_comm:.2f} GB")
    print()
    print(f"Breakdown:")
    print(f"   TP: {tp_total_comm / total_comm * 100:.1f}%")
    print(f"   PP: {pp_total_comm / total_comm * 100:.1f}%")
    print(f"   DP: {dp_comm / total_comm * 100:.1f}%")


analyze_communication_volume(
    batch_size=32,
    seq_length=2048,
    hidden_dim=8192,
    num_layers=80,
    tp_size=4,
    pp_size=2,
    dp_size=4,
    num_micro_batches=8,
)
```

---

## vLLM Hybrid Parallelism {#vllm-implementation}

### Configuration Example

```python
from vllm import LLM, SamplingParams

# Configure 3D parallelism in vLLM
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",

    # Tensor parallelism (for wide layers)
    tensor_parallel_size=4,

    # Pipeline parallelism (for deep models)
    pipeline_parallel_size=2,

    # Data parallelism is implicit
    # Total GPUs = TP × PP = 4 × 2 = 8

    # Other configurations
    dtype="float16",
    max_num_seqs=128,
    max_num_batched_tokens=32768,
)

# Generate
prompts = ["Your prompt here"] * 64
sampling_params = SamplingParams(temperature=0.8, max_tokens=512)

outputs = llm.generate(prompts, sampling_params)
```

### Initialization

```python
# vLLM's hybrid parallelism initialization (simplified)

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    """
    Initialize model parallelism for hybrid configuration.
    """
    import torch.distributed as dist

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Verify configuration
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    assert world_size % model_parallel_size == 0, (
        f"World size {world_size} not divisible by model parallel size "
        f"{model_parallel_size}"
    )

    # Data parallel size
    data_parallel_size = world_size // model_parallel_size

    # Calculate ranks
    data_parallel_rank = rank // model_parallel_size
    model_parallel_rank = rank % model_parallel_size

    tensor_model_parallel_rank = model_parallel_rank % tensor_model_parallel_size
    pipeline_model_parallel_rank = model_parallel_rank // tensor_model_parallel_size

    # Create process groups
    # ... (similar to ProcessGroupManager above)

    return {
        'dp_rank': data_parallel_rank,
        'tp_rank': tensor_model_parallel_rank,
        'pp_rank': pipeline_model_parallel_rank,
        'dp_size': data_parallel_size,
        'tp_size': tensor_model_parallel_size,
        'pp_size': pipeline_model_parallel_size,
    }
```

---

## Scaling Laws {#scaling-laws}

### Efficiency vs Scale

```python
def model_scaling_efficiency(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    num_micro_batches: int,
):
    """
    Estimate overall efficiency of 3D parallelism.
    """

    # TP efficiency (decreases with size due to communication)
    if tp_size == 1:
        tp_eff = 1.0
    elif tp_size <= 4:
        tp_eff = 0.9
    elif tp_size <= 8:
        tp_eff = 0.8
    else:
        tp_eff = 0.7

    # PP efficiency (depends on bubble overhead)
    if pp_size == 1:
        pp_eff = 1.0
    else:
        # Pipeline efficiency = m / (m + p - 1)
        pp_eff = num_micro_batches / (num_micro_batches + pp_size - 1)

    # DP efficiency (high, infrequent communication)
    dp_eff = 0.95

    # Overall efficiency
    overall_eff = tp_eff * pp_eff * dp_eff

    print(f"Scaling Efficiency Analysis")
    print(f"=" * 60)
    print(f"Configuration: TP={tp_size}, PP={pp_size}, DP={dp_size}")
    print(f"Micro-batches: {num_micro_batches}")
    print()
    print(f"TP efficiency: {tp_eff * 100:.1f}%")
    print(f"PP efficiency: {pp_eff * 100:.1f}%")
    print(f"DP efficiency: {dp_eff * 100:.1f}%")
    print(f"Overall efficiency: {overall_eff * 100:.1f}%")
    print()

    # Effective throughput
    ideal_throughput = tp_size * pp_size * dp_size  # Perfect scaling
    actual_throughput = ideal_throughput * overall_eff

    print(f"Scaling factor:")
    print(f"   Ideal: {ideal_throughput:.1f}×")
    print(f"   Actual: {actual_throughput:.1f}×")
    print(f"   Efficiency: {actual_throughput / ideal_throughput * 100:.1f}%")

    return overall_eff


# Compare different configurations
configs = [
    (2, 1, 1, 4),   # 2-way TP only
    (4, 1, 1, 4),   # 4-way TP only
    (2, 2, 1, 8),   # TP + PP
    (4, 2, 1, 8),   # TP + PP
    (4, 2, 2, 8),   # TP + PP + DP
]

for tp, pp, dp, mb in configs:
    eff = model_scaling_efficiency(tp, pp, dp, mb)
    print()
```

---

## Configuration Strategies {#strategies}

### Strategy 1: Memory-Constrained

```python
def memory_constrained_strategy(
    model_size_gb: float,
    available_memory_gb: float,
    num_gpus: int,
):
    """
    Choose configuration when memory is the primary constraint.

    Priority: Fit model first, then optimize performance.
    """

    # Use TP to fit model
    min_tp = math.ceil(model_size_gb / available_memory_gb)

    # Use power of 2 for efficiency
    tp_size = 2 ** math.ceil(math.log2(min_tp))
    tp_size = min(tp_size, 8)  # Cap at 8-way TP

    # Use PP if still doesn't fit
    if model_size_gb / tp_size > available_memory_gb:
        pp_size = math.ceil(model_size_gb / (available_memory_gb * tp_size))
    else:
        pp_size = 1

    # Remaining GPUs for DP
    dp_size = num_gpus // (tp_size * pp_size)

    return tp_size, pp_size, dp_size
```

### Strategy 2: Throughput-Optimized

```python
def throughput_optimized_strategy(
    num_gpus: int,
    interconnect_type: str,
    model_depth: int,
):
    """
    Choose configuration to maximize throughput.

    Priority: Minimize communication overhead.
    """

    if interconnect_type == 'nvlink':
        # Can use more aggressive TP
        if num_gpus >= 8:
            tp_size = 8
        elif num_gpus >= 4:
            tp_size = 4
        else:
            tp_size = num_gpus

        # Avoid PP if possible (has bubbles)
        pp_size = 1

    elif interconnect_type == 'infiniband':
        # Moderate TP, consider PP
        tp_size = min(4, num_gpus)

        remaining = num_gpus // tp_size
        if remaining > 4 and model_depth >= 64:
            pp_size = min(4, remaining)
        else:
            pp_size = 1

    else:  # ethernet
        # Minimize communication: use DP mainly
        tp_size = 1
        pp_size = 1

    dp_size = num_gpus // (tp_size * pp_size)

    return tp_size, pp_size, dp_size
```

### Strategy 3: Latency-Optimized

```python
def latency_optimized_strategy(num_gpus: int, model_fits_single_gpu: bool):
    """
    Choose configuration to minimize latency.

    Priority: Minimize pipeline bubbles and communication.
    """

    if model_fits_single_gpu:
        # No parallelism needed
        return 1, 1, 1

    # Use TP only (no PP bubbles)
    tp_size = num_gpus
    pp_size = 1
    dp_size = 1

    return tp_size, pp_size, dp_size
```

---

## Advanced Techniques {#advanced-techniques}

### Sequence Parallelism

```python
"""
Sequence Parallelism: Additional dimension of parallelism.

Splits sequence dimension across GPUs within TP group.
Reduces activation memory further.

Used in Megatron-LM for very long sequences.
"""

class SequenceParallelAttention(nn.Module):
    """
    Attention with sequence parallelism.

    Sequences are split across TP group.
    Each GPU processes partial sequence.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        # Same as regular TP attention
        self.qkv_proj = ColumnParallelLinear(hidden_size, 3 * hidden_size)
        self.o_proj = RowParallelLinear(hidden_size, hidden_size)

    def forward(self, x):
        # x: [batch, seq_len / tp_size, hidden] - sequence already split

        # QKV projection
        qkv = self.qkv_proj(x)

        # All-gather sequence dimension for attention
        # Now each GPU has full sequence
        qkv_full = all_gather_sequence(qkv, self.tp_size)

        # Attention computation
        attn_out = self.attention(qkv_full)

        # Scatter back to sequence parallelism
        attn_out_split = scatter_sequence(attn_out, self.tp_size)

        # Output projection
        out = self.o_proj(attn_out_split)

        return out
```

### Expert Parallelism (MoE)

```python
"""
Expert Parallelism: For Mixture of Experts models.

Each GPU hosts subset of experts.
Tokens are routed to appropriate GPUs.

Example: GPT-4 reportedly uses MoE with ~8 experts per layer.
"""

class ExpertParallelMoE(nn.Module):
    """
    Mixture of Experts with expert parallelism.
    """

    def __init__(self, hidden_size, num_experts, expert_size):
        super().__init__()
        self.num_experts = num_experts
        self.ep_size = get_expert_parallel_world_size()

        # Each GPU hosts num_experts / ep_size experts
        self.experts_per_gpu = num_experts // self.ep_size

        # Local experts
        self.experts = nn.ModuleList([
            FFN(hidden_size, expert_size)
            for _ in range(self.experts_per_gpu)
        ])

        # Router
        self.router = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # Route tokens to experts
        router_logits = self.router(x)
        routing_weights, selected_experts = torch.topk(router_logits, k=2)

        # All-to-all communication: send tokens to expert GPUs
        token_expert_assignments = all_to_all_routing(x, selected_experts)

        # Process tokens through local experts
        expert_outputs = []
        for expert_id, tokens in token_expert_assignments:
            local_expert_id = expert_id % self.experts_per_gpu
            expert_out = self.experts[local_expert_id](tokens)
            expert_outputs.append(expert_out)

        # All-to-all communication: return results
        final_output = all_to_all_gather(expert_outputs)

        return final_output
```

---

## Troubleshooting {#troubleshooting}

### Issue 1: OOM with Hybrid Parallelism

**Symptom:** Out of memory despite using TP and PP

**Diagnosis:**
```python
def diagnose_oom():
    """Diagnose OOM in hybrid parallelism."""

    rank_info = get_rank_info()
    memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    memory_reserved = torch.cuda.max_memory_reserved() / 1e9

    print(f"Rank {rank_info['global_rank']}: "
          f"DP={rank_info['dp_rank']}, "
          f"TP={rank_info['tp_rank']}, "
          f"PP={rank_info['pp_rank']}")
    print(f"  Allocated: {memory_allocated:.2f} GB")
    print(f"  Reserved: {memory_reserved:.2f} GB")
```

**Solutions:**
1. Increase TP size
2. Increase PP size
3. Reduce batch size
4. Enable activation checkpointing
5. Use sequence parallelism

### Issue 2: Low Efficiency

**Symptom:** GPU utilization < 60%

**Diagnosis:**
```python
def diagnose_efficiency():
    """Profile to find bottlenecks."""

    # Check TP communication overhead
    tp_comm_time = profile_all_reduce_time()

    # Check PP bubble time
    pp_efficiency = num_micro_batches / (num_micro_batches + pp_size - 1)

    # Check load balance
    check_load_balance_across_stages()
```

**Solutions:**
1. Increase micro-batches (for PP)
2. Reduce TP size if communication dominates
3. Balance layers across PP stages
4. Optimize NCCL settings

### Issue 3: Hanging or Deadlock

**Symptom:** Process hangs during distributed operations

**Common Causes:**
- Mismatched process groups
- Missing synchronization
- NCCL timeout

**Debug:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## Summary

Hybrid parallelism (3D parallelism) enables extreme-scale model serving:

**Key Points:**
1. **Combine DP, TP, PP** for different parallelization dimensions
2. **TP for width**: Split wide layers, needs high bandwidth
3. **PP for depth**: Split deep models, lower communication
4. **DP for throughput**: Process more data, infrequent sync

**Configuration Guidelines:**
- Start with TP to fit model in memory
- Add PP if model is very deep
- Use remaining GPUs for DP
- Consider interconnect bandwidth
- Balance communication overhead

**Optimization:**
- Monitor efficiency metrics
- Profile communication patterns
- Balance memory across stages
- Use appropriate micro-batch count

Continue to the next module to learn about NCCL collective operations that power distributed communication.
