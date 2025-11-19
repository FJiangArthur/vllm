# Day 20: Tensor & Pipeline Parallelism Deep Dive

> **Goal**: Master tensor and pipeline parallelism implementation details and optimization strategies
> **Time**: 4-6 hours
> **Prerequisites**: Day 19 complete, understanding of distributed basics
> **Deliverables**: TP/PP implementation analysis, performance comparison, scaling guide

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): Tensor Parallelism Deep Dive

**9:00-9:30** - TP Theory & Mathematics
**9:30-10:30** - Layer-by-Layer TP Implementation
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Communication Optimization

### Afternoon Session (2-3 hours): Pipeline Parallelism & Optimization

**14:00-15:00** - Pipeline Parallelism Implementation
**15:00-16:00** - Hybrid TP+PP Strategies
**16:00-16:30** - Break
**16:30-17:30** - Hands-On: Scaling Experiments

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Review parallelism strategies, analyze results, preview Day 21

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Implement tensor parallel layers from scratch
- [ ] Understand pipeline parallelism scheduling
- [ ] Optimize communication in distributed models
- [ ] Choose optimal TP/PP configuration for workloads
- [ ] Debug distributed training/inference issues

---

## 📂 Morning: Tensor Parallelism Deep Dive (9:00-12:00)

### Task 1: TP Theory & Mathematics (30 min)

**Column-wise Parallelism (Linear Layers)**

```
Standard linear layer: Y = XW + b
  X: [batch, d_in]
  W: [d_in, d_out]
  Y: [batch, d_out]

Column-wise split across GPUs:
  W = [W_1 | W_2 | W_3 | W_4]

  GPU 1: Y_1 = XW_1  →  [batch, d_out/4]
  GPU 2: Y_2 = XW_2  →  [batch, d_out/4]
  GPU 3: Y_3 = XW_3  →  [batch, d_out/4]
  GPU 4: Y_4 = XW_4  →  [batch, d_out/4]

  Y = [Y_1 | Y_2 | Y_3 | Y_4]  →  [batch, d_out]

Communication: None needed if next layer also column-split!
  Next layer can use Y_i locally
```

**Row-wise Parallelism**

```
Row-wise split across GPUs:
  W = [W_1; W_2; W_3; W_4]  (vertical stack)

Each GPU receives full input X:
  GPU 1: Y_1 = XW_1  →  [batch, d_out]
  GPU 2: Y_2 = XW_2  →  [batch, d_out]
  GPU 3: Y_3 = XW_3  →  [batch, d_out]
  GPU 4: Y_4 = XW_4  →  [batch, d_out]

Then all-reduce to combine:
  Y = Y_1 + Y_2 + Y_3 + Y_4  →  [batch, d_out]

Communication: All-reduce required
```

**Attention TP Strategy**

```
Multi-Head Attention with TP:

Split heads across GPUs:
  Total heads: 32
  TP=4  →  8 heads per GPU

GPU 0: Heads 0-7
GPU 1: Heads 8-15
GPU 2: Heads 16-23
GPU 3: Heads 24-31

Each GPU computes:
  Q_local = X @ W_Q_local  (8 heads)
  K_local = X @ W_K_local  (8 heads)
  V_local = X @ W_V_local  (8 heads)

  Attn_local = Attention(Q_local, K_local, V_local)
  Out_local = Attn_local @ W_O_local

Then: All-reduce(Out_local) to get final output

Communication points:
  1. After attention output projection: All-reduce
  Total: 1 all-reduce per attention layer
```

**MLP TP Strategy**

```
Two-layer MLP with TP:

Layer 1 (column-parallel): Y_1 = X @ W_1
  W_1: [d_model, d_ff] split column-wise
  Each GPU: [d_model, d_ff/4]
  Output: Y_1_local [batch, d_ff/4]
  Communication: None

Activation: Y_1_activated = GELU(Y_1_local)
  Independent per GPU

Layer 2 (row-parallel): Y_2 = Y_1_activated @ W_2
  W_2: [d_ff, d_model] split row-wise
  Each GPU: [d_ff/4, d_model]
  Output: Y_2_local [batch, d_model]
  Communication: All-reduce

Total: 1 all-reduce per MLP
```

### Task 2: Layer-by-Layer TP Implementation (60 min)

**File**: `vllm/model_executor/layers/linear.py`

**Column-Parallel Linear**

```python
class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise parallelism.

    Split output dimension across GPUs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()

        # Get tensor parallel info
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Output size per GPU
        assert output_size % self.tp_size == 0
        self.output_size_per_partition = output_size // self.tp_size

        # Weight: [input_size, output_size_per_partition]
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            input_size,
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition
            ))

        self.gather_output = gather_output

    def forward(self, input: torch.Tensor):
        """
        Forward pass.

        Input: [batch, seq_len, input_size]
        Output: [batch, seq_len, output_size] or
                [batch, seq_len, output_size_per_partition]
        """

        # Local computation
        output_parallel = F.linear(input, self.weight, self.bias)

        if self.gather_output:
            # All-gather across TP group
            output = gather_from_tensor_model_parallel_region(
                output_parallel
            )
        else:
            # Keep partitioned (for next layer)
            output = output_parallel

        return output
```

**Row-Parallel Linear**

```python
class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise parallelism.

    Split input dimension across GPUs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Input size per GPU
        assert input_size % self.tp_size == 0
        self.input_size_per_partition = input_size // self.tp_size

        # Weight: [output_size, input_size_per_partition]
        self.weight = nn.Parameter(torch.empty(
            output_size,
            self.input_size_per_partition,
        ))

        if bias:
            # Only rank 0 has bias (to avoid duplicates after all-reduce)
            if self.tp_rank == 0:
                self.bias = nn.Parameter(torch.empty(output_size))
            else:
                self.register_buffer('bias', torch.zeros(output_size))

        self.input_is_parallel = input_is_parallel

    def forward(self, input: torch.Tensor):
        """
        Forward pass.

        Input: [batch, seq_len, input_size_per_partition]
        Output: [batch, seq_len, output_size]
        """

        if not self.input_is_parallel:
            # Scatter input across TP group
            input = scatter_to_tensor_model_parallel_region(input)

        # Local computation
        output_parallel = F.linear(input, self.weight)

        # All-reduce across TP group
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        # Add bias (only on rank 0 to avoid duplication)
        if self.bias is not None:
            output = output + self.bias

        return output
```

**Tensor Parallel Attention**

```python
class TensorParallelAttention(nn.Module):
    """
    Multi-head attention with tensor parallelism.
    """

    def __init__(self, hidden_size, num_heads, ...):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.tp_size = get_tensor_model_parallel_world_size()

        # Divide heads across GPUs
        assert num_heads % self.tp_size == 0
        self.num_heads_per_partition = num_heads // self.tp_size

        self.head_dim = hidden_size // num_heads

        # QKV projection (column-parallel)
        # Each GPU handles subset of heads
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,  # Q, K, V
            bias=False,
            gather_output=False,  # Keep partitioned
        )

        # Output projection (row-parallel)
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,  # Input already partitioned
        )

    def forward(self, hidden_states, kv_cache, ...):
        """
        Forward pass.

        Each GPU computes attention for its subset of heads.
        """

        # QKV projection (column-parallel, no communication)
        qkv = self.qkv_proj(hidden_states)
        # qkv: [batch, seq, 3 * hidden_size_per_partition]

        # Split into Q, K, V
        q, k, v = qkv.split(
            self.hidden_size_per_partition,
            dim=-1
        )

        # Reshape for multi-head
        # q: [batch, seq, num_heads_per_partition, head_dim]

        # Compute attention (independent per GPU)
        attn_output = self._compute_attention(q, k, v, kv_cache)
        # attn_output: [batch, seq, num_heads_per_partition, head_dim]

        # Reshape back
        attn_output = attn_output.view(
            batch, seq, -1
        )  # [batch, seq, hidden_size_per_partition]

        # Output projection (row-parallel, includes all-reduce)
        output = self.o_proj(attn_output)
        # output: [batch, seq, hidden_size]

        return output
        # Communication: 1 all-reduce in o_proj
```

### Task 3: Communication Optimization (60 min)

**Overlapping Communication and Computation**

```python
class OptimizedTensorParallelLinear(nn.Module):
    """
    TP linear with communication-computation overlap.
    """

    def forward(self, input):
        # Start computation immediately
        output_parallel = F.linear(input, self.weight, self.bias)

        # Launch all-reduce (async)
        handle = dist.all_reduce(
            output_parallel,
            async_op=True  # Don't wait
        )

        # Do other work while communication happens
        # (e.g., prepare next layer's input)

        # Wait for communication to finish
        handle.wait()

        return output_parallel
```

**Communication Fusion**

```python
# Bad: Multiple small all-reduces
output1 = all_reduce(layer1_output)  # Small communication
output2 = all_reduce(layer2_output)  # Small communication
output3 = all_reduce(layer3_output)  # Small communication

# Good: Fuse into single large all-reduce
combined = torch.cat([layer1_output, layer2_output, layer3_output])
reduced = all_reduce(combined)  # Single large communication (more efficient)
output1, output2, output3 = reduced.split([size1, size2, size3])

# Benefit: Amortize communication overhead
```

**Smart Partitioning**

```python
# Attention + MLP sequence optimization

class OptimizedTransformerLayer(nn.Module):
    def forward(self, x):
        # Attention
        attn_qkv = self.attn_qkv_proj(x)        # Column-parallel (no comm)
        attn_out_local = self.attention(attn_qkv)  # Local compute
        attn_out = self.attn_o_proj(attn_out_local)  # Row-parallel (all-reduce)

        # Residual
        x = x + attn_out

        # MLP
        mlp_1 = self.mlp_fc1(x)                 # Column-parallel (no comm)
        mlp_1_act = self.activation(mlp_1)      # Local
        mlp_out = self.mlp_fc2(mlp_1_act)       # Row-parallel (all-reduce)

        # Residual
        x = x + mlp_out

        return x

        # Total communication: 2 all-reduces per layer
        # (attn_o_proj + mlp_fc2)
```

---

## 🔬 Afternoon: Pipeline Parallelism (14:00-17:30)

### Task 4: Pipeline Parallelism Implementation (60 min)

**Pipeline Stages**

```
Example: 32-layer model, PP=4

Stage 0 (GPU 0): Layers 0-7
Stage 1 (GPU 1): Layers 8-15
Stage 2 (GPU 2): Layers 16-23
Stage 3 (GPU 3): Layers 24-31

Data flow:
  Input → GPU 0 → GPU 1 → GPU 2 → GPU 3 → Output
```

**Pipeline Schedule (GPipe)**

```
Naive pipeline (bubble problem):

Time  GPU 0      GPU 1      GPU 2      GPU 3
  0   Batch1     -          -          -
  1   Batch2     Batch1     -          -
  2   Batch3     Batch2     Batch1     -
  3   Batch4     Batch3     Batch2     Batch1
  4   -          Batch4     Batch3     Batch2
  5   -          -          Batch4     Batch3
  6   -          -          -          Batch4

Problem: GPUs idle during pipeline fill/drain
Efficiency: 4/(4+3) = 57%
```

**Micro-batching (Improved)**

```
Split batch into micro-batches for better pipelining:

Batch size: 32
Micro-batches: 8 (4 samples each)

Time  GPU 0      GPU 1      GPU 2      GPU 3
  0   MB1        -          -          -
  1   MB2        MB1        -          -
  2   MB3        MB2        MB1        -
  3   MB4        MB3        MB2        MB1
  4   MB5        MB4        MB3        MB2
  5   MB6        MB5        MB4        MB3
  6   MB7        MB6        MB5        MB4
  7   MB8        MB7        MB6        MB5
  8   -          MB8        MB7        MB6
  9   -          -          MB8        MB7
 10   -          -          -          MB8

Efficiency: 8/(8+3) = 73%
Better!
```

**Implementation**

```python
class PipelineParallelModel(nn.Module):
    """
    Model with pipeline parallelism.
    """

    def __init__(self, num_layers, num_stages):
        super().__init__()

        self.num_stages = num_stages
        self.stage_id = get_pipeline_model_parallel_rank()

        # Determine which layers this stage handles
        layers_per_stage = num_layers // num_stages
        start_layer = self.stage_id * layers_per_stage
        end_layer = start_layer + layers_per_stage

        # Create only layers for this stage
        self.layers = nn.ModuleList([
            TransformerLayer(...)
            for i in range(start_layer, end_layer)
        ])

    def forward(self, input):
        """
        Forward pass for this pipeline stage.
        """

        # Receive input from previous stage (if not first)
        if self.stage_id > 0:
            input = recv_from_prev_pipeline_stage()

        # Process through local layers
        hidden = input
        for layer in self.layers:
            hidden = layer(hidden)

        # Send to next stage (if not last)
        if self.stage_id < self.num_stages - 1:
            send_to_next_pipeline_stage(hidden)
        else:
            # Last stage returns output
            return hidden
```

**Communication Primitives**

```python
def send_to_next_pipeline_stage(tensor):
    """Send activation to next stage."""

    next_rank = get_next_pipeline_stage_rank()

    # Point-to-point send
    dist.send(tensor, dst=next_rank)


def recv_from_prev_pipeline_stage():
    """Receive activation from previous stage."""

    prev_rank = get_prev_pipeline_stage_rank()

    # Allocate buffer
    tensor = torch.empty(expected_shape, device='cuda')

    # Point-to-point receive
    dist.recv(tensor, src=prev_rank)

    return tensor
```

### Task 5: Hybrid TP+PP (60 min)

**Configuration Example**

```
8 GPUs total, Llama-2 70B:

Configuration: TP=4, PP=2

Pipeline Stage 0 (Layers 0-39):
  GPU 0, 1, 2, 3 (TP group)
  Each GPU: 1/4 of layers 0-39

Pipeline Stage 1 (Layers 40-79):
  GPU 4, 5, 6, 7 (TP group)
  Each GPU: 1/4 of layers 40-79

Memory per GPU: ~35 GB (model) + ~20 GB (KV cache)
Fits on A100 80GB ✓
```

**Process Group Setup**

```python
def init_hybrid_parallel(world_size=8, tp_size=4, pp_size=2):
    """
    Initialize hybrid TP+PP.

    8 GPUs: [0, 1, 2, 3, 4, 5, 6, 7]

    TP groups:
      - [0, 1, 2, 3] (PP stage 0)
      - [4, 5, 6, 7] (PP stage 1)

    PP groups:
      - [0, 4] (TP rank 0 across stages)
      - [1, 5] (TP rank 1 across stages)
      - [2, 6] (TP rank 2 across stages)
      - [3, 7] (TP rank 3 across stages)
    """

    rank = dist.get_rank()

    # Determine TP and PP ranks
    tp_rank = rank % tp_size
    pp_rank = rank // tp_size

    # Create TP groups
    for pp in range(pp_size):
        tp_group_ranks = [pp * tp_size + i for i in range(tp_size)]
        tp_group = dist.new_group(tp_group_ranks)

        if pp == pp_rank:
            set_tensor_model_parallel_group(tp_group)

    # Create PP groups
    for tp in range(tp_size):
        pp_group_ranks = [pp * tp_size + tp for pp in range(pp_size)]
        pp_group = dist.new_group(pp_group_ranks)

        if tp == tp_rank:
            set_pipeline_model_parallel_group(pp_group)

    return tp_rank, pp_rank
```

**Execution Flow**

```
Hybrid TP+PP execution:

Step 1: Micro-batch 1 enters stage 0
  GPUs 0-3 (TP group):
    - All-reduce for attention (across 0-3)
    - All-reduce for MLP (across 0-3)
  GPU 3 sends to GPU 7 (next PP stage, same TP rank)

Step 2: Micro-batch 1 in stage 1, Micro-batch 2 in stage 0
  GPUs 0-3: Process MB2 (with TP all-reduces)
  GPUs 4-7: Process MB1 (with TP all-reduces)

Pipeline continues...

Communication types:
1. TP all-reduce: Within TP group (frequent, small)
2. PP point-to-point: Between stages (less frequent, larger)
```

### Task 6: Scaling Experiments (60 min)

**Performance Comparison**

```python
#!/usr/bin/env python3
"""
Compare different parallelism configurations.
"""

import time
from vllm import LLM, SamplingParams

def benchmark_config(model, tp_size, pp_size=1):
    """Benchmark specific configuration."""

    print(f"\nTesting TP={tp_size}, PP={pp_size}")
    print("=" * 60)

    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        dtype="float16",
    )

    # Test workload
    prompts = ["Generate a story about AI: "] * 100
    params = SamplingParams(max_tokens=100)

    # Warmup
    _ = llm.generate(prompts[:10], params)

    # Benchmark
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    # Calculate metrics
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Latency: {elapsed/len(prompts)*1000:.2f} ms/request")

    return {
        'tp_size': tp_size,
        'pp_size': pp_size,
        'throughput': throughput,
        'latency': elapsed/len(prompts),
    }

# Test configurations
configs = [
    (1, 1),  # Single GPU
    (2, 1),  # TP=2
    (4, 1),  # TP=4
    (2, 2),  # TP=2, PP=2
    (2, 4),  # TP=2, PP=4
]

results = []
for tp, pp in configs:
    result = benchmark_config("meta-llama/Llama-2-13b-hf", tp, pp)
    results.append(result)

# Print summary
print("\n" + "=" * 60)
print("CONFIGURATION COMPARISON")
print("=" * 60)
print(f"{'TP':>4} {'PP':>4} {'Throughput (tok/s)':>20} {'Latency (ms)':>15}")
for r in results:
    print(f"{r['tp_size']:>4} {r['pp_size']:>4} "
          f"{r['throughput']:>20.2f} {r['latency']*1000:>15.2f}")
```

**Scaling Analysis**

```python
def analyze_scaling_efficiency(results):
    """Analyze scaling efficiency."""

    baseline = results[0]  # Single GPU

    print("\nScaling Efficiency Analysis")
    print("=" * 60)
    print(f"{'Config':>10} {'GPUs':>6} {'Speedup':>10} {'Efficiency':>12}")

    for r in results:
        total_gpus = r['tp_size'] * r['pp_size']
        speedup = r['throughput'] / baseline['throughput']
        efficiency = speedup / total_gpus

        print(f"TP={r['tp_size']},PP={r['pp_size']:>6} "
              f"{total_gpus:>6} {speedup:>10.2f}x {efficiency*100:>11.1f}%")

    # Ideal: Efficiency = 100% (linear scaling)
    # Reality: 70-90% due to communication overhead
```

**Optimal Configuration Finder**

```python
def find_optimal_config(
    model_params_b: int,
    max_latency_ms: float,
    min_throughput: float,
):
    """
    Find optimal TP/PP configuration.

    Args:
        model_params_b: Model size in billions
        max_latency_ms: Maximum acceptable latency
        min_throughput: Minimum required throughput

    Returns:
        Optimal (tp_size, pp_size)
    """

    # Rules of thumb
    if model_params_b <= 13:
        # Small model: Single GPU or TP=2
        configs = [(1, 1), (2, 1)]

    elif model_params_b <= 70:
        # Medium model: TP=4 or TP=8
        configs = [(4, 1), (8, 1), (4, 2)]

    else:
        # Large model: Hybrid TP+PP
        configs = [(8, 2), (8, 4), (8, 8)]

    print(f"\nFinding optimal config for {model_params_b}B model")
    print(f"  Max latency: {max_latency_ms} ms")
    print(f"  Min throughput: {min_throughput} tok/s")

    # Test each config
    best_config = None
    best_score = 0

    for tp, pp in configs:
        # Simulate (in practice, run actual benchmark)
        latency = simulate_latency(model_params_b, tp, pp)
        throughput = simulate_throughput(model_params_b, tp, pp)

        if latency <= max_latency_ms and throughput >= min_throughput:
            # Valid config, score by efficiency
            total_gpus = tp * pp
            score = throughput / total_gpus  # throughput per GPU

            if score > best_score:
                best_config = (tp, pp)
                best_score = score

            print(f"  TP={tp}, PP={pp}: "
                  f"Latency={latency:.1f}ms, "
                  f"Throughput={throughput:.1f} tok/s "
                  f"{'✓' if score == best_score else ''}")

    return best_config
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Tensor Parallelism**
- Column and row parallelism
- Layer-by-layer implementation
- Communication optimization

✅ **Pipeline Parallelism**
- Pipeline stages and scheduling
- Micro-batching strategies
- Point-to-point communication

✅ **Hybrid Strategies**
- TP+PP configuration
- Process group setup
- Scaling analysis

### Knowledge Check Questions

**Q1**: Why use column-parallel for first MLP layer and row-parallel for second?
<details>
<summary>Answer</summary>
**Pattern**: Column → Row
- **Column-parallel** (first): Splits output, no communication needed
- **Row-parallel** (second): Takes split input, combines with all-reduce

**Benefit**: Only 1 all-reduce for entire 2-layer MLP

Alternative (both column): Would need all-gather after first layer + all-reduce after second = 2 communications
</details>

**Q2**: What causes pipeline bubbles and how does micro-batching help?
<details>
<summary>Answer</summary>
**Bubble**: GPUs idle during pipeline fill (start) and drain (end)

**Example**: 4 stages, 1 batch → 3 idle periods (43% wasted)

**Micro-batching**: Split batch into N micro-batches
- Fills pipeline faster
- Reduces idle time
- Efficiency: N/(N+stages-1)

8 micro-batches, 4 stages: 8/(8+3) = 73% efficiency ✓
</details>

**Q3**: For Llama-2 70B on 8× A100 80GB, choose between TP=8 or TP=4,PP=2. Why?
<details>
<summary>Answer</summary>
**TP=8**:
- Memory: 70B/8 = 8.75B params per GPU (~18 GB)
- Latency: Low (single pipeline stage)
- Communication: High (7 all-reduces per layer)

**TP=4, PP=2**:
- Memory: 70B/4 = 17.5B params per GPU (~35 GB), half layers
- Latency: Higher (2 pipeline stages)
- Communication: Lower (3 all-reduces per layer in TP group)

**Choose TP=4,PP=2** if:
- Can tolerate slight latency increase
- Want lower communication overhead
- Planning to scale to more GPUs

**Choose TP=8** if:
- Latency critical
- Have NVLink (fast interconnect)
- Not scaling beyond 8 GPUs
</details>

---

## 🚀 Preview: Day 21

Tomorrow is Week 3 Review:
- **Consolidate Learning**: Scheduler, batching, executor, distributed
- **Integration Project**: End-to-end system tracing
- **Knowledge Assessment**: Comprehensive quiz
- **Week 4 Preview**: Advanced optimizations

**Preparation**:
- Review all Week 3 concepts
- Organize notes and diagrams
- Prepare questions

---

## 📚 Additional Resources

**Key Files**:
- [ ] `vllm/model_executor/layers/linear.py`
- [ ] `vllm/distributed/parallel_state.py`
- [ ] `vllm/distributed/communication_op.py`

**Papers**:
- [ ] Megatron-LM (NVIDIA's TP implementation)
- [ ] GPipe (Google's PP implementation)
- [ ] PipeDream (Microsoft's PP)

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
