# Pipeline Parallelism Strategies

## Table of Contents
1. [Introduction to Pipeline Parallelism](#introduction)
2. [Pipeline Parallelism Basics](#basics)
3. [Micro-Batching](#micro-batching)
4. [Pipeline Schedules](#pipeline-schedules)
5. [Bubble Analysis](#bubble-analysis)
6. [vLLM Pipeline Implementation](#vllm-implementation)
7. [GPipe and PipeDream](#gpipe-pipedream)
8. [Interleaved Pipeline Parallelism](#interleaved-pp)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Introduction to Pipeline Parallelism {#introduction}

Pipeline Parallelism (PP) partitions a model's layers across multiple GPUs, with each GPU processing a subset of consecutive layers. This is particularly useful for very large models that cannot fit on a single device even with tensor parallelism.

### Key Concepts

**Pipeline Parallelism vs Tensor Parallelism:**

```
Tensor Parallelism (TP):
- Splits individual layers across GPUs
- All GPUs work on same layers simultaneously
- High communication overhead (all-reduce per layer)
- Better for: Wide models, high-bandwidth interconnect

Pipeline Parallelism (PP):
- Splits layers across GPUs sequentially
- Each GPU works on different layers
- Lower communication (point-to-point between stages)
- Better for: Deep models, lower-bandwidth networks
```

### Why Pipeline Parallelism?

1. **Model Size**: Enables models too large for single GPU
2. **Reduced Communication**: Only activations passed between stages
3. **Scalability**: Can scale to many GPUs with lower bandwidth requirements
4. **Complementary to TP**: Combine PP + TP for very large models

### Challenges

1. **Pipeline Bubbles**: GPU idle time during pipeline fill/drain
2. **Memory Imbalance**: Different stages may have different memory requirements
3. **Complexity**: More complex than data or tensor parallelism
4. **Activation Checkpointing**: Need to store activations for backward pass

---

## Pipeline Parallelism Basics {#basics}

### Simple Pipeline Example

```python
# Example: 12-layer model on 4 GPUs

GPU 0: Layers 0-2   (3 layers)
GPU 1: Layers 3-5   (3 layers)
GPU 2: Layers 6-8   (3 layers)
GPU 3: Layers 9-11  (3 layers)

# Data flow:
Input -> GPU0 -> GPU1 -> GPU2 -> GPU3 -> Output

# Naive pipeline (sequential):
# Time step 1: GPU0 processes batch, others idle
# Time step 2: GPU1 processes, GPU0 and others idle
# Time step 3: GPU2 processes, GPU0-1 and GPU3 idle
# Time step 4: GPU3 processes, GPU0-2 idle
# Efficiency: 25% (only 1/4 GPUs active at a time)
```

### Pipeline Stages

```python
class PipelineStage:
    """
    A pipeline stage contains a subset of model layers.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
    ):
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages

        # Communication setup
        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this stage's layers.
        """
        hidden_states = input_tensor

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states

    def send_forward(self, tensor: torch.Tensor):
        """Send activation to next stage."""
        if not self.is_last_stage:
            dist.send(tensor, dst=self.stage_id + 1)

    def recv_forward(self) -> torch.Tensor:
        """Receive activation from previous stage."""
        if not self.is_first_stage:
            tensor = torch.empty(...)  # Allocate buffer
            dist.recv(tensor, src=self.stage_id - 1)
            return tensor
        else:
            return None  # First stage receives real input


def create_pipeline_stages(
    model: nn.Module,
    num_stages: int,
) -> List[PipelineStage]:
    """
    Split model into pipeline stages.
    """
    num_layers = len(model.layers)
    layers_per_stage = num_layers // num_stages

    stages = []
    for stage_id in range(num_stages):
        start_layer = stage_id * layers_per_stage
        end_layer = (stage_id + 1) * layers_per_stage

        stage_layers = model.layers[start_layer:end_layer]
        stage = PipelineStage(stage_layers, stage_id, num_stages)
        stages.append(stage)

    return stages
```

---

## Micro-Batching {#micro-batching}

Micro-batching splits each batch into smaller micro-batches to improve pipeline utilization.

### Micro-Batch Pipeline

```python
# Example: Batch size 32, 4 micro-batches -> 8 samples per micro-batch

# Without micro-batching (pipeline bubble):
# T0: [GPU0 m0] [    ] [    ] [    ]
# T1: [    ] [GPU1 m0] [    ] [    ]
# T2: [    ] [    ] [GPU2 m0] [    ]
# T3: [    ] [    ] [    ] [GPU3 m0]
# Efficiency: 25%

# With 4 micro-batches:
# T0: [GPU0 m0] [    ] [    ] [    ]
# T1: [GPU0 m1] [GPU1 m0] [    ] [    ]
# T2: [GPU0 m2] [GPU1 m1] [GPU2 m0] [    ]
# T3: [GPU0 m3] [GPU1 m2] [GPU2 m1] [GPU3 m0]  <- Full pipeline
# T4: [    ] [GPU1 m3] [GPU2 m2] [GPU3 m1]
# T5: [    ] [    ] [GPU2 m3] [GPU3 m2]
# T6: [    ] [    ] [    ] [GPU3 m3]
# Efficiency: 4/7 ≈ 57%
```

### Micro-Batch Implementation

```python
class MicroBatchScheduler:
    """
    Schedule micro-batches through pipeline.
    """

    def __init__(
        self,
        num_micro_batches: int,
        num_stages: int,
    ):
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages

        # Calculate pipeline schedule
        self.schedule = self._create_schedule()

    def _create_schedule(self) -> List[Tuple[str, int, int]]:
        """
        Create pipeline schedule.

        Returns list of (action, stage_id, micro_batch_id) tuples.
        """
        schedule = []

        # Fill phase: pipeline fills up
        for step in range(self.num_stages - 1):
            for stage in range(step + 1):
                micro_batch = step - stage
                schedule.append(('forward', stage, micro_batch))

        # Steady phase: all stages active
        for step in range(self.num_micro_batches - self.num_stages + 1):
            for stage in range(self.num_stages):
                micro_batch = step + self.num_stages - 1 - stage
                schedule.append(('forward', stage, micro_batch))

        # Drain phase: pipeline drains
        for step in range(self.num_stages - 1):
            for stage in range(step + 1, self.num_stages):
                micro_batch = (
                    self.num_micro_batches - self.num_stages + step + 1 +
                    self.num_stages - 1 - stage
                )
                schedule.append(('forward', stage, micro_batch))

        return schedule

    def run_schedule(
        self,
        stage: PipelineStage,
        input_batches: List[torch.Tensor],
    ):
        """
        Execute pipeline schedule for this stage.
        """
        outputs = []
        activations = {}

        for action, stage_id, micro_batch_id in self.schedule:
            if stage_id != stage.stage_id:
                continue

            if action == 'forward':
                # Receive input (if not first stage)
                if stage.is_first_stage:
                    input_tensor = input_batches[micro_batch_id]
                else:
                    input_tensor = stage.recv_forward()

                # Forward pass
                output_tensor = stage.forward(input_tensor)

                # Send output (if not last stage)
                if stage.is_last_stage:
                    outputs.append(output_tensor)
                else:
                    stage.send_forward(output_tensor)

                # Store activation for backward pass
                activations[micro_batch_id] = input_tensor

        return outputs


def calculate_pipeline_efficiency(
    num_micro_batches: int,
    num_stages: int,
) -> float:
    """
    Calculate pipeline efficiency (fraction of time GPUs are active).
    """
    # Total time steps
    total_steps = num_micro_batches + num_stages - 1

    # Ideal time (all GPUs always busy)
    ideal_steps = num_micro_batches

    # Efficiency
    efficiency = ideal_steps / total_steps

    return efficiency


# Examples
for m in [1, 2, 4, 8, 16]:
    eff = calculate_pipeline_efficiency(m, num_stages=4)
    print(f"Micro-batches: {m:2d}, Efficiency: {eff*100:.1f}%")

# Output:
# Micro-batches:  1, Efficiency: 25.0%
# Micro-batches:  2, Efficiency: 40.0%
# Micro-batches:  4, Efficiency: 57.1%
# Micro-batches:  8, Efficiency: 72.7%
# Micro-batches: 16, Efficiency: 84.2%
```

---

## Pipeline Schedules {#pipeline-schedules}

### GPipe Schedule (Fill-Drain)

```python
class GPipeSchedule:
    """
    GPipe: Fill-drain pipeline schedule.

    Forward pass fills pipeline, then drains.
    Then backward pass fills and drains.
    """

    def __init__(self, num_micro_batches: int, num_stages: int):
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages

    def forward_schedule(self) -> List[Tuple[int, int]]:
        """
        Generate forward pass schedule.

        Returns: [(stage_id, micro_batch_id), ...]
        """
        schedule = []

        # Each stage processes micro-batches in order
        for micro_batch in range(self.num_micro_batches):
            for stage in range(self.num_stages):
                schedule.append((stage, micro_batch))

        return schedule

    def backward_schedule(self) -> List[Tuple[int, int]]:
        """
        Generate backward pass schedule (reverse order).
        """
        schedule = []

        # Process in reverse order
        for micro_batch in reversed(range(self.num_micro_batches)):
            for stage in reversed(range(self.num_stages)):
                schedule.append((stage, micro_batch))

        return schedule

    def visualize(self):
        """Print schedule visualization."""
        print("GPipe Schedule Visualization")
        print("=" * 60)

        # Forward pass
        print("\nForward Pass:")
        for t in range(self.num_micro_batches + self.num_stages - 1):
            time_step = f"T{t:2d}: "
            for stage in range(self.num_stages):
                micro_batch = t - stage
                if 0 <= micro_batch < self.num_micro_batches:
                    time_step += f"[S{stage}M{micro_batch}] "
                else:
                    time_step += "[    ] "
            print(time_step)

        # Calculate bubble time
        total_steps = self.num_micro_batches + self.num_stages - 1
        bubble_steps = (self.num_stages - 1) * 2  # Fill + drain
        efficiency = (total_steps - bubble_steps) / total_steps * 100

        print(f"\nEfficiency: {efficiency:.1f}%")


# Example
schedule = GPipeSchedule(num_micro_batches=4, num_stages=4)
schedule.visualize()
```

### 1F1B Schedule (One-Forward-One-Backward)

```python
class OneFOneB Schedule:
    """
    1F1B: Interleave forward and backward passes.

    More memory efficient than GPipe.
    Reduces peak activation memory.
    """

    def __init__(self, num_micro_batches: int, num_stages: int):
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages

    def create_schedule(self) -> List[Tuple[str, int, int]]:
        """
        Create 1F1B schedule.

        Returns: [(action, stage_id, micro_batch_id), ...]
        where action is 'F' (forward) or 'B' (backward)
        """
        schedule = []

        # Each stage has different warm-up time
        for stage in range(self.num_stages):
            warm_up_steps = self.num_stages - stage - 1

            # Warm-up: forward only
            for i in range(warm_up_steps):
                schedule.append(('F', stage, i))

            # Steady state: 1F1B
            for i in range(self.num_micro_batches - warm_up_steps):
                # Forward
                micro_batch_f = warm_up_steps + i
                schedule.append(('F', stage, micro_batch_f))

                # Backward
                micro_batch_b = i
                schedule.append(('B', stage, micro_batch_b))

            # Cool-down: backward only
            for i in range(warm_up_steps):
                micro_batch = self.num_micro_batches - warm_up_steps + i
                schedule.append(('B', stage, micro_batch))

        return schedule

    def visualize(self):
        """Visualize 1F1B schedule."""
        print("1F1B Schedule Visualization")
        print("=" * 80)

        # Create timeline
        max_steps = (self.num_micro_batches + self.num_stages - 1) * 2

        for t in range(max_steps):
            time_step = f"T{t:2d}: "

            for stage in range(self.num_stages):
                # Determine what this stage is doing at time t
                warm_up_steps = self.num_stages - stage - 1

                if t < warm_up_steps * 2:
                    # Warm-up phase
                    if t % 2 == 0:
                        mb = t // 2
                        time_step += f"[S{stage}F{mb}] "
                    else:
                        time_step += "[     ] "
                elif t < (self.num_micro_batches + warm_up_steps) * 2:
                    # Steady phase: 1F1B
                    if t % 2 == 0:
                        mb = t // 2
                        time_step += f"[S{stage}F{mb}] "
                    else:
                        mb = (t - warm_up_steps * 2 - 1) // 2
                        time_step += f"[S{stage}B{mb}] "
                else:
                    # Cool-down phase
                    if t % 2 == 1:
                        mb = (t - self.num_micro_batches * 2 + 1) // 2
                        time_step += f"[S{stage}B{mb}] "
                    else:
                        time_step += "[     ] "

            print(time_step)


# Compare memory usage
def compare_memory_usage():
    """
    Compare peak activation memory: GPipe vs 1F1B.
    """
    num_micro_batches = 8
    num_stages = 4
    activation_size_mb = 100  # MB per micro-batch

    # GPipe: stores all activations until backward pass
    gpipe_memory = num_micro_batches * activation_size_mb
    print(f"GPipe peak memory: {gpipe_memory} MB")

    # 1F1B: stores at most (num_stages) activations
    # During steady state, pipeline has num_stages activations in flight
    one_f_one_b_memory = num_stages * activation_size_mb
    print(f"1F1B peak memory: {one_f_one_b_memory} MB")

    savings = (gpipe_memory - one_f_one_b_memory) / gpipe_memory * 100
    print(f"Memory savings: {savings:.1f}%")

# Output:
# GPipe peak memory: 800 MB
# 1F1B peak memory: 400 MB
# Memory savings: 50.0%
```

---

## Bubble Analysis {#bubble-analysis}

### Understanding Pipeline Bubbles

```python
def analyze_pipeline_bubbles(
    num_micro_batches: int,
    num_stages: int,
    forward_time: float,
    backward_time: float = None,
):
    """
    Analyze pipeline bubble overhead.

    Args:
        num_micro_batches: Number of micro-batches
        num_stages: Number of pipeline stages
        forward_time: Time for forward pass (ms)
        backward_time: Time for backward pass (ms), defaults to 2x forward
    """
    if backward_time is None:
        backward_time = 2 * forward_time

    # Total computation time (ideal, no bubbles)
    total_compute_time = num_micro_batches * (forward_time + backward_time)

    # Pipeline fill time (bubble)
    fill_time = (num_stages - 1) * forward_time

    # Pipeline drain time (bubble)
    drain_time = (num_stages - 1) * backward_time

    # Total bubble time
    bubble_time = fill_time + drain_time

    # Total elapsed time
    total_time = total_compute_time + bubble_time

    # Efficiency
    efficiency = total_compute_time / total_time

    print(f"Pipeline Bubble Analysis")
    print(f"=" * 60)
    print(f"Configuration:")
    print(f"  Micro-batches: {num_micro_batches}")
    print(f"  Pipeline stages: {num_stages}")
    print(f"  Forward time: {forward_time:.1f} ms")
    print(f"  Backward time: {backward_time:.1f} ms")
    print()
    print(f"Timing:")
    print(f"  Ideal compute time: {total_compute_time:.1f} ms")
    print(f"  Fill bubble: {fill_time:.1f} ms")
    print(f"  Drain bubble: {drain_time:.1f} ms")
    print(f"  Total bubble: {bubble_time:.1f} ms")
    print(f"  Total time: {total_time:.1f} ms")
    print()
    print(f"Efficiency: {efficiency * 100:.1f}%")
    print(f"Bubble overhead: {(1 - efficiency) * 100:.1f}%")

    return efficiency


# Example: Analyze different configurations
print("Effect of number of micro-batches:\n")
for m in [4, 8, 16, 32]:
    eff = analyze_pipeline_bubbles(
        num_micro_batches=m,
        num_stages=4,
        forward_time=10.0,
    )
    print()

print("\nEffect of number of stages:\n")
for s in [2, 4, 8]:
    eff = analyze_pipeline_bubbles(
        num_micro_batches=16,
        num_stages=s,
        forward_time=10.0,
    )
    print()
```

### Minimizing Bubbles

```python
def optimal_micro_batch_count(
    num_stages: int,
    target_efficiency: float = 0.9,
) -> int:
    """
    Calculate optimal number of micro-batches for target efficiency.

    Pipeline efficiency = m / (m + p - 1)
    where m = micro-batches, p = stages

    Solve for m:
    eff = m / (m + p - 1)
    eff * (m + p - 1) = m
    eff * m + eff * (p - 1) = m
    m * (1 - eff) = eff * (p - 1)
    m = eff * (p - 1) / (1 - eff)
    """
    m = target_efficiency * (num_stages - 1) / (1 - target_efficiency)
    return int(math.ceil(m))


# Examples
for p in [2, 4, 8, 16]:
    m = optimal_micro_batch_count(p, target_efficiency=0.9)
    actual_eff = m / (m + p - 1)
    print(f"Stages: {p:2d}, Optimal micro-batches: {m:3d}, "
          f"Efficiency: {actual_eff * 100:.1f}%")

# Output:
# Stages:  2, Optimal micro-batches:   9, Efficiency: 90.0%
# Stages:  4, Optimal micro-batches:  27, Efficiency: 90.0%
# Stages:  8, Optimal micro-batches:  63, Efficiency: 90.0%
# Stages: 16, Optimal micro-batches: 135, Efficiency: 90.0%
```

---

## vLLM Pipeline Implementation {#vllm-implementation}

### Pipeline Parallel Configuration

```python
# vLLM pipeline parallelism example

from vllm import LLM, SamplingParams

# Configure pipeline parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,  # 2-way tensor parallelism
    pipeline_parallel_size=4,  # 4-way pipeline parallelism
    # Total GPUs: 2 * 4 = 8
)

# In vLLM, inference is typically prefill + decode
# Pipeline parallelism helps with large batch prefills

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

prompts = ["Your prompt here"] * 32

# vLLM automatically handles pipeline scheduling
outputs = llm.generate(prompts, sampling_params)
```

### Pipeline Stage Implementation

```python
# From vllm perspective (simplified)

class PipelineParallelLLM:
    """
    Model with pipeline parallelism.
    """

    def __init__(self, config):
        self.pp_size = get_pipeline_model_parallel_world_size()
        self.pp_rank = get_pipeline_model_parallel_rank()

        # Determine which layers this stage owns
        self.start_layer, self.end_layer = self._get_layer_range(config)

        # Build stage
        self.stage = self._build_stage(config)

    def _get_layer_range(self, config):
        """Determine layer range for this pipeline stage."""
        total_layers = config.num_hidden_layers
        layers_per_stage = total_layers // self.pp_size

        start = self.pp_rank * layers_per_stage
        end = start + layers_per_stage

        return start, end

    def _build_stage(self, config):
        """Build pipeline stage with assigned layers."""
        layers = nn.ModuleList()

        # First stage includes embedding
        if self.pp_rank == 0:
            self.embed_tokens = VocabParallelEmbedding(...)

        # Middle stages have transformer layers
        for layer_id in range(self.start_layer, self.end_layer):
            layers.append(TransformerLayer(config))

        # Last stage includes final norm and LM head
        if self.pp_rank == self.pp_size - 1:
            self.norm = LayerNorm(...)
            self.lm_head = ParallelLMHead(...)

        return layers

    def forward_stage(self, hidden_states):
        """Forward through this stage."""
        # First stage: embed tokens
        if self.pp_rank == 0:
            hidden_states = self.embed_tokens(hidden_states)

        # All stages: transformer layers
        for layer in self.stage:
            hidden_states = layer(hidden_states)

        # Last stage: norm and LM head
        if self.pp_rank == self.pp_size - 1:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

        return hidden_states
```

---

## GPipe and PipeDream {#gpipe-pipedream}

### GPipe Implementation

```python
class GPipe:
    """
    GPipe: Efficient pipeline parallelism with re-materialization.

    Key features:
    - Fill-drain pipeline schedule
    - Activation checkpointing to reduce memory
    - Synchronous gradient updates
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_micro_batches: int,
        checkpoint_activations: bool = True,
    ):
        self.model = model
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.checkpoint_activations = checkpoint_activations

        # Split model into stages
        self.stages = self._split_model()

    def forward(self, inputs):
        """
        Forward pass with micro-batching.
        """
        # Split input into micro-batches
        micro_batches = torch.chunk(inputs, self.num_micro_batches, dim=0)

        # Store outputs from each micro-batch
        outputs = []

        # Process each micro-batch through pipeline
        for micro_batch in micro_batches:
            hidden = micro_batch

            # Pass through each stage
            for stage in self.stages:
                if self.checkpoint_activations:
                    # Use gradient checkpointing
                    hidden = checkpoint(stage, hidden)
                else:
                    hidden = stage(hidden)

            outputs.append(hidden)

        # Concatenate outputs
        return torch.cat(outputs, dim=0)
```

### PipeDream Implementation

```python
class PipeDream:
    """
    PipeDream: Asynchronous pipeline parallelism.

    Key features:
    - 1F1B schedule
    - Weight versioning for correct gradients
    - Asynchronous gradient updates
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_micro_batches: int,
    ):
        self.model = model
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches

        # Weight versions for correctness
        self.weight_versions = {}

    def forward_backward(self, inputs, targets):
        """
        Forward and backward with 1F1B schedule.
        """
        micro_batches_inputs = torch.chunk(
            inputs, self.num_micro_batches, dim=0
        )
        micro_batches_targets = torch.chunk(
            targets, self.num_micro_batches, dim=0
        )

        # Get current stage
        stage_id = get_pipeline_model_parallel_rank()
        stage = self.stages[stage_id]

        # Warm-up phase: forward only
        warm_up_steps = self.num_stages - stage_id - 1
        activations = []

        for i in range(warm_up_steps):
            input_mb = self._recv_or_input(micro_batches_inputs[i], stage_id)
            output_mb = stage(input_mb)
            self._send_or_loss(output_mb, stage_id)
            activations.append((input_mb, output_mb))

        # Steady phase: 1F1B
        for i in range(warm_up_steps, self.num_micro_batches):
            # Forward
            input_mb = self._recv_or_input(micro_batches_inputs[i], stage_id)
            output_mb = stage(input_mb)
            self._send_or_loss(output_mb, stage_id)
            activations.append((input_mb, output_mb))

            # Backward
            grad_output = self._recv_grad(stage_id)
            input_mb, output_mb = activations.pop(0)
            grad_input = self._backward(output_mb, grad_output)
            self._send_grad(grad_input, stage_id)

        # Cool-down phase: backward only
        for i in range(warm_up_steps):
            grad_output = self._recv_grad(stage_id)
            input_mb, output_mb = activations.pop(0)
            grad_input = self._backward(output_mb, grad_output)
            self._send_grad(grad_input, stage_id)
```

---

## Interleaved Pipeline Parallelism {#interleaved-pp}

### Interleaved Scheduling

Interleaved pipeline assigns non-consecutive layers to each stage, reducing bubble time.

```python
class InterleavedPipeline:
    """
    Interleaved pipeline parallelism (used in Megatron-LM).

    Instead of:
      GPU0: [L0, L1, L2, L3]
      GPU1: [L4, L5, L6, L7]

    Use:
      GPU0: [L0, L1, L4, L5]
      GPU1: [L2, L3, L6, L7]

    Reduces bubble time significantly.
    """

    def __init__(
        self,
        num_layers: int,
        num_stages: int,
        num_model_chunks: int,  # Number of chunks per stage
    ):
        self.num_layers = num_layers
        self.num_stages = num_stages
        self.num_model_chunks = num_model_chunks

        # Calculate layers per chunk
        self.layers_per_chunk = num_layers // (num_stages * num_model_chunks)

    def get_layer_assignment(self, stage_id: int) -> List[List[int]]:
        """
        Get layer indices for this stage.

        Returns list of chunks, each chunk is list of layer indices.
        """
        chunks = []

        for chunk_id in range(self.num_model_chunks):
            chunk_layers = []
            start_layer = (
                chunk_id * self.num_stages * self.layers_per_chunk +
                stage_id * self.layers_per_chunk
            )

            for i in range(self.layers_per_chunk):
                chunk_layers.append(start_layer + i)

            chunks.append(chunk_layers)

        return chunks


# Example: 16 layers, 2 stages, 2 chunks per stage
pipeline = InterleavedPipeline(num_layers=16, num_stages=2, num_model_chunks=2)

print("Layer Assignment:")
for stage_id in range(2):
    chunks = pipeline.get_layer_assignment(stage_id)
    print(f"GPU {stage_id}: {chunks}")

# Output:
# GPU 0: [[0, 1, 2, 3], [8, 9, 10, 11]]
# GPU 1: [[4, 5, 6, 7], [12, 13, 14, 15]]


def compare_bubble_time():
    """
    Compare bubble time: regular vs interleaved pipeline.
    """
    num_micro_batches = 4
    num_stages = 4

    # Regular pipeline
    regular_bubble = 2 * (num_stages - 1)  # Fill + drain

    # Interleaved pipeline with 2 chunks
    num_chunks = 2
    interleaved_bubble = 2 * (num_stages // num_chunks - 1) * num_chunks

    print(f"Regular bubble steps: {regular_bubble}")
    print(f"Interleaved bubble steps: {interleaved_bubble}")
    print(f"Bubble reduction: {regular_bubble - interleaved_bubble} steps")

# Output:
# Regular bubble steps: 6
# Interleaved bubble steps: 4
# Bubble reduction: 2 steps
```

---

## Performance Optimization {#performance-optimization}

### Communication Optimization

```python
def optimize_pipeline_communication():
    """
    Optimize pipeline communication performance.
    """

    # 1. Use appropriate tensor sizes
    # Larger activations amortize communication overhead
    min_activation_size = 1024 * 1024  # 1M elements minimum

    # 2. Overlap communication and computation
    # Send next micro-batch while computing current one
    use_async_communication = True

    # 3. Optimize micro-batch size
    # Balance between bubble overhead and activation memory
    def choose_micro_batch_size(
        total_batch_size: int,
        num_stages: int,
        target_efficiency: float = 0.9,
    ):
        min_micro_batches = optimal_micro_batch_count(num_stages, target_efficiency)

        # Also consider activation memory
        max_micro_batches = total_batch_size // 8  # At least 8 samples per micro-batch

        # Choose smaller to satisfy both constraints
        num_micro_batches = min(max_micro_batches, min_micro_batches)

        return max(num_micro_batches, 1)

    # 4. Use efficient communication primitives
    # Point-to-point is sufficient for pipeline (vs all-reduce for TP)
    comm_backend = 'nccl'  # NCCL for NVIDIA GPUs

    return {
        'min_activation_size': min_activation_size,
        'async_communication': use_async_communication,
        'comm_backend': comm_backend,
    }
```

### Memory Optimization

```python
def optimize_pipeline_memory():
    """
    Memory optimization strategies for pipeline parallelism.
    """

    # 1. Activation checkpointing
    # Trade computation for memory
    checkpoint_activations = True

    # 2. Use 1F1B schedule
    # Reduces peak activation memory
    use_1f1b = True

    # 3. Optimize stage boundaries
    # Balance memory across stages
    def balance_stage_memory(model, num_stages):
        """
        Distribute layers to balance memory across stages.
        """
        layer_memories = [estimate_layer_memory(layer) for layer in model.layers]
        total_memory = sum(layer_memories)
        target_memory_per_stage = total_memory / num_stages

        stages = []
        current_stage = []
        current_memory = 0

        for i, (layer, memory) in enumerate(zip(model.layers, layer_memories)):
            current_stage.append(layer)
            current_memory += memory

            if current_memory >= target_memory_per_stage or i == len(model.layers) - 1:
                stages.append(current_stage)
                current_stage = []
                current_memory = 0

        return stages

    # 4. Offload optimizer states
    # Keep optimizer states on CPU if memory is tight
    offload_optimizer = True

    return {
        'checkpoint_activations': checkpoint_activations,
        'use_1f1b': use_1f1b,
        'offload_optimizer': offload_optimizer,
    }


def estimate_layer_memory(layer):
    """Estimate memory usage of a layer."""
    param_memory = sum(p.numel() * p.element_size() for p in layer.parameters())

    # Estimate activation memory (rough approximation)
    # Depends on batch size and hidden dimensions
    activation_memory = param_memory * 0.5  # Rough estimate

    return param_memory + activation_memory
```

---

## Troubleshooting {#troubleshooting}

### Issue 1: High Bubble Overhead

**Symptom:** Poor GPU utilization, long training time

**Solution:**
```python
def reduce_bubble_overhead(
    current_micro_batches: int,
    num_stages: int,
):
    """
    Recommend configuration to reduce bubble overhead.
    """
    current_eff = current_micro_batches / (current_micro_batches + num_stages - 1)

    print(f"Current efficiency: {current_eff * 100:.1f}%")

    # Recommend more micro-batches
    for target_eff in [0.9, 0.95, 0.99]:
        recommended = optimal_micro_batch_count(num_stages, target_eff)
        print(f"For {target_eff * 100:.0f}% efficiency, use {recommended} micro-batches")

    # Or use interleaved scheduling
    print(f"\nAlternatively, use interleaved scheduling with 2-4 chunks per stage")
```

### Issue 2: Memory Imbalance

**Symptom:** Some stages OOM while others have spare memory

**Solution:**
```python
def diagnose_memory_imbalance():
    """
    Diagnose and fix memory imbalance across pipeline stages.
    """
    import torch.distributed as dist

    # Measure memory per stage
    stage_id = get_pipeline_model_parallel_rank()
    num_stages = get_pipeline_model_parallel_world_size()

    memory_allocated = torch.cuda.max_memory_allocated() / 1e9

    # Gather from all stages
    all_memories = [torch.tensor(0.0) for _ in range(num_stages)]
    dist.all_gather(all_memories, torch.tensor(memory_allocated))

    if stage_id == 0:
        print("Memory per stage:")
        for i, mem in enumerate(all_memories):
            print(f"  Stage {i}: {mem:.2f} GB")

        max_mem = max(all_memories)
        min_mem = min(all_memories)
        imbalance = (max_mem - min_mem) / max_mem * 100

        print(f"\nMemory imbalance: {imbalance:.1f}%")

        if imbalance > 20:
            print("WARNING: Significant memory imbalance!")
            print("Recommendations:")
            print("  1. Redistribute layers based on memory usage")
            print("  2. Use activation checkpointing for heavy stages")
            print("  3. Adjust layer assignment to balance memory")
```

### Issue 3: Communication Bottleneck

**Symptom:** Low communication bandwidth, stages waiting

**Solution:**
```bash
# Check network bandwidth
ib_write_bw  # InfiniBand bandwidth test

# Optimize NCCL settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Your network interface
export NCCL_IB_DISABLE=0  # Enable IB if available

# Use P2P when possible
export NCCL_P2P_LEVEL=NVL
```

---

## Summary

Pipeline parallelism enables training and inference of very large models:

**Key Concepts:**
- Split model layers across GPUs sequentially
- Use micro-batching to reduce pipeline bubbles
- Choose appropriate schedule (GPipe, 1F1B, interleaved)
- Balance memory and communication

**Best Practices:**
- Use enough micro-batches for >90% efficiency
- Apply 1F1B schedule for memory efficiency
- Consider interleaved scheduling for lower bubble overhead
- Balance memory across stages
- Monitor and optimize communication

**Trade-offs:**
- More stages → more bubbles (need more micro-batches)
- More micro-batches → more activation memory
- Pipeline parallelism complements tensor parallelism

Continue to the next module on hybrid parallelism to learn how to combine pipeline and tensor parallelism for maximum scale.
