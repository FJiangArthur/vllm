# Tensor Parallelism Fundamentals

## Table of Contents
1. [Introduction to Tensor Parallelism](#introduction)
2. [Why Tensor Parallelism?](#why-tensor-parallelism)
3. [Column vs Row Parallelism](#column-vs-row-parallelism)
4. [Mathematical Foundations](#mathematical-foundations)
5. [vLLM Implementation](#vllm-implementation)
6. [Multi-GPU Setup](#multi-gpu-setup)
7. [Communication Patterns](#communication-patterns)
8. [Performance Analysis](#performance-analysis)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Introduction to Tensor Parallelism {#introduction}

Tensor Parallelism (TP) is a model parallelism technique that splits individual layers of a neural network across multiple GPUs. Unlike data parallelism where each GPU has a complete copy of the model, tensor parallelism distributes the model's weights themselves across devices.

### Key Concepts

**Model Parallelism vs Data Parallelism:**
- **Data Parallelism**: Each GPU has full model, processes different data batches
- **Model Parallelism**: Model is split across GPUs, each processes same data
- **Tensor Parallelism**: Specific type of model parallelism splitting at tensor level

**When to Use Tensor Parallelism:**
1. Model doesn't fit in single GPU memory (e.g., 70B+ parameter models)
2. Need to reduce per-GPU memory footprint
3. Want to scale inference throughput with multiple GPUs
4. Working with attention and feedforward layers

### Historical Context

Tensor parallelism became critical with the advent of large language models:
- **GPT-3 (175B parameters)**: ~350GB in FP16, requiring multi-GPU
- **LLaMA-65B**: ~130GB in FP16
- **LLaMA-70B**: ~140GB in FP16

Single GPU memory limits (A100: 80GB, H100: 80GB) necessitate model splitting.

---

## Why Tensor Parallelism? {#why-tensor-parallelism}

### Memory Constraints

Modern LLMs have massive parameter counts:

```
Model Size Calculation:
- Parameters: 70 billion
- FP16 precision: 2 bytes per parameter
- Total: 70B × 2 = 140GB

Additional Memory for Inference:
- Activations: ~20-40GB (depends on batch size)
- KV Cache: ~10-30GB per request (depends on sequence length)
- CUDA kernels and overhead: ~5-10GB

Total: 175-220GB for 70B model inference
```

A single 80GB GPU cannot fit this model, requiring tensor parallelism.

### Throughput Scaling

Even if a model fits on one GPU, tensor parallelism can improve throughput:
- Parallel computation across GPUs
- Reduced per-GPU memory pressure allows larger batch sizes
- Better GPU utilization with balanced workloads

### Latency Considerations

**Trade-offs:**
- TP adds communication overhead (all-reduce, all-gather operations)
- Single request latency may increase slightly
- Throughput (requests/second) typically improves
- Optimal for batch inference scenarios

---

## Column vs Row Parallelism {#column-vs-row-parallelism}

Tensor parallelism splits weight matrices along different dimensions. Understanding column vs row parallelism is fundamental.

### Column Parallelism

Split weight matrix along column dimension:

```
Original Matrix Multiplication:
Y = X @ W
where:
  X: [batch, seq_len, hidden]
  W: [hidden, output_dim]
  Y: [batch, seq_len, output_dim]

Column Parallel Split:
W = [W_1 | W_2 | ... | W_n]  (split along columns)

Each GPU computes:
  GPU_1: Y_1 = X @ W_1
  GPU_2: Y_2 = X @ W_2
  ...
  GPU_n: Y_n = X @ W_n

Final output: Y = [Y_1 | Y_2 | ... | Y_n] (concatenate)
```

**Characteristics:**
- Input X is replicated to all GPUs (requires all-gather or broadcast)
- Each GPU computes partial output
- Outputs are concatenated (no reduction needed)
- Used for: QKV projection, first FFN layer

### Row Parallelism

Split weight matrix along row dimension:

```
Original Matrix Multiplication:
Y = X @ W

Row Parallel Split:
W = [W_1]  (split along rows)
    [W_2]
    [...]
    [W_n]

X = [X_1 | X_2 | ... | X_n]  (split along hidden dim)

Each GPU computes:
  GPU_1: Y_1 = X_1 @ W_1
  GPU_2: Y_2 = X_2 @ W_2
  ...
  GPU_n: Y_n = X_n @ W_n

Final output: Y = Y_1 + Y_2 + ... + Y_n (all-reduce sum)
```

**Characteristics:**
- Input X is split across GPUs (already partitioned)
- Each GPU computes partial output
- Outputs are summed (requires all-reduce)
- Used for: Output projection, second FFN layer

### Combining Column and Row Parallelism

Transformers use both strategically to minimize communication:

```python
# Attention Layer
class ParallelAttention:
    def __init__(self):
        # Column parallel: splits output heads
        self.qkv_proj = ColumnParallelLinear(hidden, 3 * hidden)
        # Row parallel: combines head outputs
        self.o_proj = RowParallelLinear(hidden, hidden)

    def forward(self, x):
        # qkv_proj: input replicated, output partitioned
        qkv = self.qkv_proj(x)  # No all-reduce needed

        # ... attention computation ...

        # o_proj: input partitioned, output reduced
        output = self.o_proj(attn_out)  # All-reduce inside
        return output

# Feedforward Layer
class ParallelMLP:
    def __init__(self):
        # Column parallel: splits intermediate dimension
        self.gate_up_proj = ColumnParallelLinear(hidden, intermediate)
        # Row parallel: combines outputs
        self.down_proj = RowParallelLinear(intermediate, hidden)

    def forward(self, x):
        # gate_up_proj: input replicated, output partitioned
        intermediate = self.gate_up_proj(x)  # No all-reduce
        intermediate = activation(intermediate)

        # down_proj: input partitioned, output reduced
        output = self.down_proj(intermediate)  # All-reduce inside
        return output
```

**Communication Pattern:**
- Column → Row combination requires only 1 all-reduce
- Alternative approaches would need 2+ all-reduces
- Minimizes communication overhead

---

## Mathematical Foundations {#mathematical-foundations}

### Linear Layer Decomposition

Given a linear transformation Y = XW + b:

**Column Parallel Decomposition:**
```
W ∈ R^(d_in × d_out) split into [W_1, W_2, ..., W_n]
Each W_i ∈ R^(d_in × d_out/n)

Y = X[W_1 | W_2 | ... | W_n] + b
  = [XW_1 | XW_2 | ... | XW_n] + [b_1 | b_2 | ... | b_n]
  = [Y_1 | Y_2 | ... | Y_n]
```

**Row Parallel Decomposition:**
```
W ∈ R^(d_in × d_out) split into [W_1; W_2; ...; W_n]
Each W_i ∈ R^(d_in/n × d_out)

X split into [X_1, X_2, ..., X_n] where X_i ∈ R^(batch × d_in/n)

Y = [X_1 | X_2 | ... | X_n][W_1; W_2; ...; W_n] + b
  = X_1W_1 + X_2W_2 + ... + X_nW_n + b
  = ∑(X_iW_i) + b  (requires all-reduce)
```

### Attention Mechanism Parallelization

Multi-head attention with tensor parallelism:

```
Standard Multi-Head Attention:
Q, K, V = X @ W_q, X @ W_k, X @ W_v
where W_q, W_k, W_v ∈ R^(d × d)

Split into n heads per GPU:
Total heads: h (e.g., 32)
Heads per GPU: h/n (e.g., 32/4 = 8)

GPU_i computes heads [i*(h/n) : (i+1)*(h/n)]

Attention_i = softmax(Q_i @ K_i^T / √d_k) @ V_i
Output_i = Attention_i @ W_o_i

Final = all_reduce(∑ Output_i)
```

### Communication Cost Analysis

For a linear layer Y = XW with:
- Batch size: b
- Sequence length: s
- Hidden dimension: d
- Number of GPUs: n

**Column Parallel:**
- Computation per GPU: O(bsd²/n)
- Communication: None (output is partitioned)
- Memory per GPU: O(d²/n)

**Row Parallel:**
- Computation per GPU: O(bsd²/n)
- Communication: All-reduce of size O(bsd) - one message per GPU
- Memory per GPU: O(d²/n)

**Total Communication per Transformer Layer:**
- 2 all-reduce operations (one after attention, one after FFN)
- Size: O(bsd) each
- With optimized communication: can overlap with computation

---

## vLLM Implementation {#vllm-implementation}

### Parallel State Management

vLLM uses `parallel_state.py` to manage distributed state:

```python
# From vllm/distributed/parallel_state.py

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    """
    Initialize model parallel groups.

    Args:
        tensor_model_parallel_size: Number of GPUs for tensor parallelism
        pipeline_model_parallel_size: Number of GPUs for pipeline parallelism
    """
    # Initialize process groups for TP
    # Each TP group contains tensor_model_parallel_size GPUs
    # that work together on the same model shard

def get_tensor_model_parallel_world_size() -> int:
    """Return the tensor model parallel world size."""
    return torch.distributed.get_world_size(
        group=get_tensor_model_parallel_group()
    )

def get_tensor_model_parallel_rank() -> int:
    """Return the tensor model parallel rank."""
    return torch.distributed.get_rank(
        group=get_tensor_model_parallel_group()
    )
```

### ColumnParallelLinear Implementation

```python
# Simplified from vLLM's implementation

class ColumnParallelLinear(torch.nn.Module):
    """
    Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension (columns).

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, gather output and return full tensor
        skip_bias_add: If true, return bias separately for fusion
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
    ):
        super().__init__()

        # Keep input dimension same
        self.input_size = input_size

        # Divide output dimension by world size
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = output_size // world_size
        self.output_size = output_size

        # Parameters
        self.weight = torch.nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size)
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(self.output_size_per_partition)
            )
        else:
            self.register_parameter('bias', None)

        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional output gathering."""

        # Matrix multiplication
        output_parallel = F.linear(input_, self.weight)

        # Gather output if needed
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        # Add bias
        if self.bias is not None:
            if self.skip_bias_add:
                return output, self.bias
            else:
                return output + self.bias
        else:
            return output
```

### RowParallelLinear Implementation

```python
class RowParallelLinear(torch.nn.Module):
    """
    Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension (rows).

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        input_is_parallel: If true, input is already partitioned
        skip_bias_add: If true, return bias separately
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
    ):
        super().__init__()

        # Divide input dimension by world size
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size // world_size
        self.input_size = input_size

        # Keep output dimension same
        self.output_size = output_size

        # Parameters
        self.weight = torch.nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition)
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with all-reduce."""

        # Scatter input if needed
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        else:
            input_parallel = input_

        # Matrix multiplication (partial result)
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce across tensor parallel group
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        # Add bias (only on rank 0 to avoid duplication)
        if self.bias is not None:
            if self.skip_bias_add:
                return output, self.bias
            else:
                return output + self.bias
        else:
            return output
```

### Communication Primitives

```python
# From vllm/distributed/utils.py

def gather_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors and concatenate along the last dimension.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    # Get the size and dimension
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    # Allocate output tensor
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_

    # All-gather
    torch.distributed.all_gather(
        tensor_list,
        input_,
        group=get_tensor_model_parallel_group()
    )

    # Concatenate
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """
    All-reduce the input tensor across model parallel group.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    # All-reduce sum
    torch.distributed.all_reduce(
        input_,
        group=get_tensor_model_parallel_group()
    )

    return input_


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """
    Split the tensor along its last dimension and keep the
    corresponding slice for current rank.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    # Get rank and split
    rank = get_tensor_model_parallel_rank()
    last_dim = input_.dim() - 1

    # Split input tensor
    input_list = torch.split(
        input_,
        input_.shape[last_dim] // world_size,
        dim=last_dim
    )

    # Return slice for this rank
    output = input_list[rank].contiguous()
    return output
```

---

## Multi-GPU Setup {#multi-gpu-setup}

### Basic Configuration

```python
# Example: Running Llama-70B with 4-way tensor parallelism

from vllm import LLM, SamplingParams

# Initialize LLM with tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # Use 4 GPUs
    dtype="float16",
    trust_remote_code=True,
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# Generate
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print()
```

### Environment Setup

```bash
# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL environment variables for optimal performance
export NCCL_DEBUG=INFO  # For debugging
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Network interface
export NCCL_P2P_DISABLE=0  # Enable P2P transfers

# Run with torchrun for multi-GPU
torchrun --nproc_per_node=4 inference_script.py
```

### Verifying GPU Distribution

```python
import torch
import torch.distributed as dist
from vllm.distributed import (
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# Initialize distributed
dist.init_process_group(backend='nccl')

# Initialize model parallel
initialize_model_parallel(tensor_model_parallel_size=4)

# Print GPU assignment
local_rank = torch.distributed.get_rank()
tp_rank = get_tensor_model_parallel_rank()
tp_world_size = get_tensor_model_parallel_world_size()

print(f"Local Rank: {local_rank}")
print(f"TP Rank: {tp_rank}/{tp_world_size}")
print(f"GPU Device: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name()}")
```

---

## Communication Patterns {#communication-patterns}

### All-Reduce Operation

```python
# All-reduce sums partial results from all GPUs

# Before all-reduce (each GPU has partial result):
# GPU 0: [1, 2, 3]
# GPU 1: [4, 5, 6]
# GPU 2: [7, 8, 9]
# GPU 3: [10, 11, 12]

# After all-reduce:
# GPU 0: [22, 26, 30]  # Sum of all
# GPU 1: [22, 26, 30]
# GPU 2: [22, 26, 30]
# GPU 3: [22, 26, 30]

def all_reduce_example():
    tensor = torch.randn(1024, 4096, device='cuda')

    # Time the all-reduce
    torch.cuda.synchronize()
    start = time.time()

    torch.distributed.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
        group=get_tensor_model_parallel_group()
    )

    torch.cuda.synchronize()
    end = time.time()

    # Calculate bandwidth
    tensor_size_bytes = tensor.numel() * tensor.element_size()
    bandwidth_gbps = (tensor_size_bytes / (end - start)) / 1e9

    print(f"All-reduce time: {(end - start) * 1000:.2f} ms")
    print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
```

### All-Gather Operation

```python
# All-gather collects partitioned data from all GPUs

# Before all-gather (each GPU has different data):
# GPU 0: [1, 2]
# GPU 1: [3, 4]
# GPU 2: [5, 6]
# GPU 3: [7, 8]

# After all-gather:
# GPU 0: [1, 2, 3, 4, 5, 6, 7, 8]
# GPU 1: [1, 2, 3, 4, 5, 6, 7, 8]
# GPU 2: [1, 2, 3, 4, 5, 6, 7, 8]
# GPU 3: [1, 2, 3, 4, 5, 6, 7, 8]

def all_gather_example():
    # Each GPU has different partial tensor
    local_tensor = torch.randn(1024, 1024, device='cuda')
    world_size = get_tensor_model_parallel_world_size()

    # Allocate output
    output_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]

    # All-gather
    torch.distributed.all_gather(
        output_tensors,
        local_tensor,
        group=get_tensor_model_parallel_group()
    )

    # Concatenate results
    full_tensor = torch.cat(output_tensors, dim=-1)
    return full_tensor
```

### Reduce-Scatter Operation

```python
# Reduce-scatter combines all-reduce + scatter

# Before reduce-scatter:
# GPU 0: [1, 2, 3, 4]
# GPU 1: [5, 6, 7, 8]
# GPU 2: [9, 10, 11, 12]
# GPU 3: [13, 14, 15, 16]

# After reduce-scatter (sum, then split):
# GPU 0: [28, 32]  # Sum of first halves
# GPU 1: [36, 40]  # Sum of second halves
# GPU 2: [44, 48]
# GPU 3: [52, 56]

def reduce_scatter_example():
    tensor = torch.randn(1024, 4096, device='cuda')
    world_size = get_tensor_model_parallel_world_size()

    # Output will be 1/world_size of input
    output = torch.empty(
        1024, 4096 // world_size,
        device='cuda',
        dtype=tensor.dtype
    )

    torch.distributed.reduce_scatter(
        output,
        list(tensor.chunk(world_size, dim=-1)),
        group=get_tensor_model_parallel_group()
    )

    return output
```

---

## Performance Analysis {#performance-analysis}

### Scaling Efficiency

```python
def measure_scaling_efficiency():
    """
    Measure weak and strong scaling efficiency.
    """
    results = {}

    # Test different TP sizes
    for tp_size in [1, 2, 4, 8]:
        # Initialize model
        llm = LLM(
            model="meta-llama/Llama-2-70b-hf",
            tensor_parallel_size=tp_size,
            dtype="float16",
        )

        # Benchmark
        prompts = ["Test prompt"] * 100

        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end = time.time()

        throughput = len(prompts) / (end - start)

        results[tp_size] = {
            'time': end - start,
            'throughput': throughput,
            'efficiency': throughput / tp_size  # Tokens/sec/GPU
        }

    # Calculate scaling efficiency
    baseline = results[1]['throughput']
    for tp_size, metrics in results.items():
        ideal_speedup = tp_size
        actual_speedup = metrics['throughput'] / baseline
        efficiency = (actual_speedup / ideal_speedup) * 100

        print(f"TP Size: {tp_size}")
        print(f"  Throughput: {metrics['throughput']:.2f} req/s")
        print(f"  Speedup: {actual_speedup:.2f}x (ideal: {ideal_speedup}x)")
        print(f"  Efficiency: {efficiency:.1f}%")
```

### Communication Overhead

```python
def profile_communication_overhead():
    """
    Profile communication vs computation time.
    """
    import torch.cuda.profiler as profiler

    # Enable profiling
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        # Run inference
        outputs = llm.generate(prompts, sampling_params)

    # Analyze results
    events = prof.key_averages()

    total_time = 0
    comm_time = 0
    compute_time = 0

    for evt in events:
        total_time += evt.self_cuda_time_total

        if 'nccl' in evt.key.lower() or 'all_reduce' in evt.key.lower():
            comm_time += evt.self_cuda_time_total
        elif 'gemm' in evt.key.lower() or 'matmul' in evt.key.lower():
            compute_time += evt.self_cuda_time_total

    print(f"Total time: {total_time / 1000:.2f} ms")
    print(f"Communication: {comm_time / 1000:.2f} ms ({comm_time/total_time*100:.1f}%)")
    print(f"Computation: {compute_time / 1000:.2f} ms ({compute_time/total_time*100:.1f}%)")
```

### Memory Usage Analysis

```python
def analyze_memory_usage():
    """
    Compare memory usage across different TP configurations.
    """
    for tp_size in [1, 2, 4, 8]:
        torch.cuda.reset_peak_memory_stats()

        llm = LLM(
            model="meta-llama/Llama-2-70b-hf",
            tensor_parallel_size=tp_size,
        )

        # Measure memory
        memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        memory_reserved = torch.cuda.max_memory_reserved() / 1e9

        print(f"TP Size: {tp_size}")
        print(f"  Memory Allocated: {memory_allocated:.2f} GB")
        print(f"  Memory Reserved: {memory_reserved:.2f} GB")
        print(f"  Per GPU: {memory_allocated / tp_size:.2f} GB")
```

---

## Best Practices {#best-practices}

### 1. Choose Appropriate TP Size

```python
# Rule of thumb for TP size selection:
def recommend_tp_size(model_size_gb, gpu_memory_gb):
    """
    Recommend tensor parallel size based on model and GPU memory.
    """
    # Account for overhead (activations, KV cache, etc.)
    safety_factor = 1.5
    required_memory = model_size_gb * safety_factor

    tp_size = math.ceil(required_memory / gpu_memory_gb)

    # Round to power of 2 for optimal NCCL performance
    tp_size = 2 ** math.ceil(math.log2(tp_size))

    return min(tp_size, 8)  # Max 8-way TP typically

# Examples:
print(recommend_tp_size(140, 80))  # 70B model, A100: 4
print(recommend_tp_size(280, 80))  # 140B model, A100: 8
print(recommend_tp_size(70, 80))   # 35B model, A100: 2
```

### 2. Optimize Communication

```python
# Enable optimal NCCL settings
import os

def set_nccl_optimizations():
    """Set environment variables for optimal NCCL performance."""

    # Use InfiniBand if available
    os.environ['NCCL_IB_DISABLE'] = '0'

    # Enable GPU Direct RDMA
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'

    # Use fastest network interface
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Adjust to your NIC

    # Enable P2P transfers
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'  # NVLink if available

    # Tune buffer sizes
    os.environ['NCCL_BUFFSIZE'] = '8388608'  # 8MB

    # Enable tree algorithm for large messages
    os.environ['NCCL_ALGO'] = 'Tree'
```

### 3. Balance Computation and Communication

```python
# Prefer larger batch sizes to amortize communication cost
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=256,
)

# Good: Larger batch amortizes all-reduce overhead
large_batch_prompts = ["Prompt"] * 32

# Less optimal: Small batch, communication overhead dominates
small_batch_prompts = ["Prompt"] * 2
```

### 4. Monitor GPU Utilization

```bash
# Monitor GPUs during inference
watch -n 0.5 nvidia-smi

# Look for:
# - Balanced memory usage across GPUs
# - High GPU utilization (>80%)
# - Similar power consumption across GPUs
```

---

## Troubleshooting {#troubleshooting}

### Issue 1: NCCL Timeout

**Symptom:**
```
NCCL error: unhandled system error, NCCL version 2.18.1
Watchdog caught collective operation timeout: WorkNCCL(...)
```

**Solution:**
```bash
# Increase timeout
export NCCL_TIMEOUT=1800000  # 30 minutes in ms

# Enable debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check network connectivity between GPUs
nvidia-smi topo -m
```

### Issue 2: Imbalanced GPU Usage

**Symptom:**
Some GPUs show high utilization while others are idle.

**Solution:**
```python
# Verify tensor parallel groups are correctly initialized
def verify_tp_groups():
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()

    # Should print 0,1,2,3 for 4-way TP
    print(f"TP Rank: {rank}/{world_size}")

    # All ranks should see the same world size
    assert world_size == expected_tp_size

# Check model weight distribution
def check_weight_distribution(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} on GPU {torch.cuda.current_device()}")
        # Shapes should be smaller by factor of TP size
```

### Issue 3: Out of Memory with TP

**Symptom:**
OOM even with tensor parallelism enabled.

**Solution:**
```python
# Increase TP size
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,  # Increase from 4
    max_num_seqs=64,  # Reduce batch size
)

# Or enable CPU offloading for KV cache
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    swap_space=20,  # 20GB CPU swap space
)
```

### Issue 4: Slower than Expected

**Symptom:**
Multi-GPU slower than single GPU for small models.

**Analysis:**
```python
# Communication overhead dominates for small models
model_flops = calculate_model_flops(model_size)
comm_bandwidth = measure_nccl_bandwidth()

# If communication time > computation time, don't use TP
if model_fits_in_single_gpu and comm_overhead_high:
    # Use data parallelism instead
    llm = LLM(model="small-model", tensor_parallel_size=1)
```

---

## Summary

Tensor parallelism is essential for serving large language models:

**Key Takeaways:**
1. TP splits model weights across GPUs, reducing per-GPU memory
2. Column and row parallelism minimize communication (only 2 all-reduces per layer)
3. vLLM implements TP using PyTorch distributed primitives
4. Optimal TP size balances memory constraints and communication overhead
5. Monitor scaling efficiency and communication patterns

**When to Use:**
- Model doesn't fit in single GPU
- Need to serve 70B+ parameter models
- Have high-bandwidth GPU interconnect (NVLink, InfiniBand)

**When to Avoid:**
- Model fits comfortably in single GPU
- Limited GPU interconnect bandwidth
- Latency is critical over throughput

**Next Steps:**
- Experiment with different TP sizes
- Profile communication overhead
- Combine with pipeline parallelism (next module)
- Optimize NCCL settings for your hardware

Continue to the next module to learn about Megatron-style model parallelism and advanced partitioning strategies.
