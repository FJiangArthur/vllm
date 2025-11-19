# Communication-Computation Overlap

## Table of Contents
1. [Introduction](#introduction)
2. [Overlap Techniques](#overlap-techniques)
3. [Asynchronous Operations](#async-operations)
4. [CUDA Streams](#cuda-streams)
5. [Gradient Accumulation Overlap](#gradient-overlap)
6. [vLLM Implementation](#vllm-implementation)
7. [Performance Analysis](#performance-analysis)
8. [Best Practices](#best-practices)

---

## Introduction {#introduction}

Communication-computation overlap is a critical optimization technique that hides communication latency by performing computation concurrently with communication operations.

### The Problem

```python
"""
Without Overlap (Sequential):

Time →
GPU Compute: [████████] [        ] [████████] [        ]
Communication: [        ] [████████] [        ] [████████]

Total time = Compute time + Communication time

With Overlap (Concurrent):

Time →
GPU Compute: [████████████████████████████████]
Communication: [  ████████  ][  ████████  ]

Total time ≈ max(Compute time, Communication time)

Speedup = (Compute + Comm) / max(Compute, Comm)
"""


def calculate_overlap_speedup(
    compute_time_ms: float,
    comm_time_ms: float,
    overlap_efficiency: float = 1.0,
):
    """
    Calculate speedup from overlapping communication with computation.

    Args:
        compute_time_ms: Time for computation
        comm_time_ms: Time for communication
        overlap_efficiency: How well overlap works (0-1)
    """
    # Sequential time
    sequential_time = compute_time_ms + comm_time_ms

    # Overlapped time (best case: max of the two)
    ideal_overlap_time = max(compute_time_ms, comm_time_ms)

    # Actual overlap time (accounting for inefficiency)
    actual_overlap_time = (
        compute_time_ms +
        comm_time_ms * (1 - overlap_efficiency)
    )

    # Speedup
    ideal_speedup = sequential_time / ideal_overlap_time
    actual_speedup = sequential_time / actual_overlap_time

    print(f"Overlap Analysis:")
    print(f"  Compute time: {compute_time_ms:.2f} ms")
    print(f"  Communication time: {comm_time_ms:.2f} ms")
    print(f"  Sequential total: {sequential_time:.2f} ms")
    print(f"  Overlapped time: {actual_overlap_time:.2f} ms")
    print(f"  Ideal speedup: {ideal_speedup:.2f}x")
    print(f"  Actual speedup: {actual_speedup:.2f}x")
    print(f"  Overlap efficiency: {overlap_efficiency * 100:.1f}%")

    return actual_speedup


# Example: Transformer layer with tensor parallelism
# Computation: 50ms, All-reduce communication: 10ms
calculate_overlap_speedup(50, 10, overlap_efficiency=0.8)
```

---

## Overlap Techniques {#overlap-techniques}

### 1. Async NCCL Operations

```python
import torch
import torch.distributed as dist


def async_all_reduce_overlap():
    """
    Overlap all-reduce with computation using async operations.
    """

    # Initialize distributed
    dist.init_process_group(backend='nccl')

    # Create tensors
    # Gradient to communicate
    gradient = torch.randn(10000, 1000, device='cuda')

    # Next layer computation
    input_tensor = torch.randn(1000, 1000, device='cuda')
    weight = torch.randn(1000, 1000, device='cuda')

    # === Without Overlap ===
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Communication (blocking)
    dist.all_reduce(gradient, async_op=False)

    # Computation
    output = torch.matmul(input_tensor, weight)

    end.record()
    torch.cuda.synchronize()

    sequential_time = start.elapsed_time(end)

    # === With Overlap ===
    start.record()

    # Start async communication
    work = dist.all_reduce(gradient, async_op=True)

    # Do computation while communication happens
    output = torch.matmul(input_tensor, weight)

    # Wait for communication to complete
    work.wait()

    end.record()
    torch.cuda.synchronize()

    overlap_time = start.elapsed_time(end)

    print(f"Sequential: {sequential_time:.2f} ms")
    print(f"Overlapped: {overlap_time:.2f} ms")
    print(f"Speedup: {sequential_time / overlap_time:.2f}x")

    dist.destroy_process_group()


### 2. Layer-wise Overlap

```python
class OverlappedTransformerLayer(nn.Module):
    """
    Transformer layer with communication-computation overlap.

    Overlaps all-reduce from previous layer with computation of current layer.
    """

    def __init__(self, config):
        super().__init__()

        self.attention = MegatronAttention(config)
        self.mlp = MegatronMLP(config)
        self.norm1 = LayerNorm(config.hidden_size)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_all_reduce_work=None,  # Async work from previous layer
    ):
        # Wait for previous all-reduce if any
        if prev_all_reduce_work is not None:
            prev_all_reduce_work.wait()

        # === Attention ===
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # Attention forward (includes column-parallel, no communication)
        attn_output = self.attention(hidden_states)

        # Row-parallel projection (starts all-reduce)
        attn_output, all_reduce_work = self.attention.o_proj_with_async_comm(
            attn_output
        )

        # While all-reduce happens, do:
        # 1. Residual connection (cheap)
        # 2. Start next norm (cheap)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        # Wait for all-reduce to complete
        all_reduce_work.wait()

        # === MLP ===
        # MLP up projection (column-parallel, no communication)
        mlp_hidden = self.mlp.up_proj(hidden_states)
        mlp_hidden = self.mlp.activation(mlp_hidden)

        # MLP down projection (starts all-reduce)
        mlp_output, mlp_all_reduce_work = self.mlp.down_proj_with_async_comm(
            mlp_hidden
        )

        # Return output and async work (next layer will wait)
        return residual + mlp_output, mlp_all_reduce_work


class RowParallelLinearWithAsyncComm(nn.Module):
    """
    Row-parallel linear layer that returns async communication handle.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = RowParallelLinear(input_size, output_size)

    def forward_with_async_comm(self, input_tensor):
        # Local matmul
        output = self.linear.forward_local(input_tensor)

        # Start async all-reduce
        work = dist.all_reduce(output, async_op=True)

        return output, work
```

### 3. Micro-batch Pipeline Overlap

```python
class OverlappedPipelineStage:
    """
    Pipeline stage that overlaps communication with computation.

    Sends next micro-batch while computing current micro-batch.
    """

    def __init__(self, stage_id, num_stages):
        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward_with_overlap(self, micro_batches):
        """
        Process micro-batches with overlapped communication.
        """

        outputs = []
        send_work = None

        for i, micro_batch in enumerate(micro_batches):
            # Wait for previous send to complete
            if send_work is not None:
                send_work.wait()

            # Receive next micro-batch (async)
            if self.stage_id > 0:
                input_tensor = torch.empty_like(micro_batch)
                recv_work = dist.irecv(input_tensor, src=self.stage_id - 1)
            else:
                input_tensor = micro_batch
                recv_work = None

            # Compute current micro-batch
            if recv_work is not None:
                recv_work.wait()

            output = self.compute(input_tensor)

            # Send to next stage (async)
            if self.stage_id < self.num_stages - 1:
                send_work = dist.isend(output, dst=self.stage_id + 1)
            else:
                outputs.append(output)
                send_work = None

        # Wait for final send
        if send_work is not None:
            send_work.wait()

        return outputs
```

---

## Asynchronous Operations {#async-operations}

### PyTorch Distributed Async API

```python
def demonstrate_async_api():
    """
    Demonstrate PyTorch distributed async API.
    """

    dist.init_process_group(backend='nccl')

    # All operations support async_op=True
    tensor = torch.randn(1000, 1000, device='cuda')

    # All-reduce
    work1 = dist.all_reduce(tensor, async_op=True)

    # Broadcast
    work2 = dist.broadcast(tensor, src=0, async_op=True)

    # All-gather
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    work3 = dist.all_gather(tensors, tensor, async_op=True)

    # Send (always async)
    if dist.get_rank() == 0:
        work4 = dist.isend(tensor, dst=1)

    # Recv (always async)
    if dist.get_rank() == 1:
        recv_tensor = torch.empty_like(tensor)
        work5 = dist.irecv(recv_tensor, src=0)

    # Do computation while communication happens
    computation_result = perform_computation()

    # Wait for all communication
    work1.wait()
    work2.wait()
    work3.wait()

    dist.destroy_process_group()


def wait_strategies():
    """
    Different strategies for waiting on async operations.
    """

    # Strategy 1: Wait immediately (no overlap)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()  # Blocks immediately

    # Strategy 2: Wait before using result
    work = dist.all_reduce(tensor, async_op=True)
    result = do_computation()  # Overlapped
    work.wait()  # Wait when needed
    use_tensor(tensor)

    # Strategy 3: Wait on multiple operations
    works = []
    for tensor in tensors:
        work = dist.all_reduce(tensor, async_op=True)
        works.append(work)

    # Do computation
    result = do_computation()

    # Wait for all
    for work in works:
        work.wait()

    # Strategy 4: Check completion without blocking
    work = dist.all_reduce(tensor, async_op=True)

    while not work.is_completed():
        do_some_computation()  # Do work while waiting

    # Now can use result
    use_tensor(tensor)
```

---

## CUDA Streams {#cuda-streams}

### Multi-Stream Overlap

```python
def multi_stream_overlap():
    """
    Use multiple CUDA streams for better overlap.

    Default stream: Computation
    Stream 1: Communication
    Stream 2: Memory copies (if needed)
    """

    # Create streams
    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()

    # Tensors
    input_tensor = torch.randn(1000, 1000, device='cuda')
    grad_tensor = torch.randn(1000, 1000, device='cuda')
    weight = torch.randn(1000, 1000, device='cuda')

    # Computation on default stream
    with torch.cuda.stream(compute_stream):
        output = torch.matmul(input_tensor, weight)

    # Communication on separate stream (overlaps with computation)
    with torch.cuda.stream(comm_stream):
        dist.all_reduce(grad_tensor)

    # Copy on another stream (if needed)
    with torch.cuda.stream(copy_stream):
        cpu_tensor = grad_tensor.cpu()

    # Synchronize all streams
    compute_stream.synchronize()
    comm_stream.synchronize()
    copy_stream.synchronize()


class StreamedModel(nn.Module):
    """
    Model that uses multiple streams for overlap.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([...])

        # Create communication stream
        self.comm_stream = torch.cuda.Stream()

    def forward(self, x):
        # Process layers
        for i, layer in enumerate(self.layers):
            # Compute on default stream
            x = layer(x)

            # If this layer has communication, do it on comm stream
            if hasattr(layer, 'async_comm'):
                with torch.cuda.stream(self.comm_stream):
                    # Start async communication
                    layer.async_comm.start()

        # Wait for all communication to complete
        self.comm_stream.synchronize()

        return x


def stream_synchronization():
    """
    Proper stream synchronization.
    """

    default_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()

    tensor = torch.randn(1000, 1000, device='cuda')

    # Start communication on comm_stream
    with torch.cuda.stream(comm_stream):
        work = dist.all_reduce(tensor, async_op=True)

    # Computation on default stream (happens concurrently)
    result = torch.randn(1000, 1000, device='cuda')
    result = torch.matmul(result, result)

    # Before using tensor, must synchronize
    comm_stream.synchronize()

    # Now safe to use tensor
    final = tensor + result
```

---

## Gradient Accumulation Overlap {#gradient-overlap}

### Overlapping Gradient All-Reduce

```python
class OverlappedGradientAccumulation:
    """
    Overlap gradient all-reduce with backward pass.

    Key idea: Start all-reducing gradients for early layers
    while still computing gradients for later layers.
    """

    def __init__(self, model, num_accumulation_steps):
        self.model = model
        self.num_accumulation_steps = num_accumulation_steps

        # Register hooks for async gradient all-reduce
        self._register_hooks()

    def _register_hooks(self):
        """
        Register backward hooks to start all-reduce as soon as gradient is ready.
        """

        def make_hook(name):
            def hook(grad):
                # As soon as gradient is computed, start all-reduce
                if self.should_reduce_grads():
                    work = dist.all_reduce(grad, async_op=True)
                    self.pending_all_reduces.append((name, work))
                return grad
            return hook

        # Register for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(make_hook(name))

        self.pending_all_reduces = []

    def train_step(self, batch):
        """
        Training step with overlapped gradient reduction.
        """

        # Forward
        loss = self.model(batch)

        # Backward (gradients computed in reverse order)
        # Hooks will start all-reduce for each gradient as ready
        loss.backward()

        # Wait for all gradient reductions
        for name, work in self.pending_all_reduces:
            work.wait()

        self.pending_all_reduces.clear()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()


# PyTorch DDP does this automatically
def pytorch_ddp_overlap():
    """
    PyTorch DistributedDataParallel automatically overlaps
    gradient all-reduce with backward pass.
    """

    from torch.nn.parallel import DistributedDataParallel as DDP

    model = MyModel().cuda()

    # DDP wraps model and adds hooks
    ddp_model = DDP(
        model,
        device_ids=[torch.cuda.current_device()],
        # DDP automatically overlaps gradient reduction with backward
        bucket_cap_mb=25,  # Bucket size for communication
        find_unused_parameters=False,
    )

    # Forward
    output = ddp_model(input)
    loss = criterion(output, target)

    # Backward - DDP automatically overlaps all-reduce
    # Gradients for early layers all-reduced while computing later layers
    loss.backward()

    # All gradients reduced and averaged by this point
    optimizer.step()
```

---

## vLLM Implementation {#vllm-implementation}

### vLLM's Overlap Strategy

```python
# vLLM overlaps in several places:

class VLLMModelRunner:
    """
    vLLM model runner with communication overlap.
    """

    def execute_model(
        self,
        seq_group_metadata_list,
        kv_caches,
    ):
        """
        Execute model with overlapped communication.
        """

        # Prepare inputs
        input_ids, positions, ... = self.prepare_inputs(seq_group_metadata_list)

        # Run model
        # Each layer overlaps all-reduce with next layer's computation
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
        )

        # Sampling (overlaps with any remaining communication)
        next_tokens = self.sample(hidden_states, seq_group_metadata_list)

        return next_tokens


# From vllm/model_executor/models/
class VLLMTransformerLayer(nn.Module):
    """
    Transformer layer with built-in overlap.
    """

    def forward(self, hidden_states, ...):
        # Attention
        residual = hidden_states

        # QKV projection (column-parallel, no comm)
        qkv = self.qkv_proj(hidden_states)

        # Attention computation
        attn_out = self.attn(qkv, ...)

        # Output projection (row-parallel, starts all-reduce)
        # This is where overlap happens - all-reduce starts but doesn't block
        attn_out = self.o_proj(attn_out)  # async all-reduce inside

        # Residual and norm (cheap ops while all-reduce completes)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)

        # MLP (similar pattern)
        mlp_out = self.mlp(hidden_states)  # async all-reduce in down_proj

        hidden_states = residual + mlp_out

        return hidden_states
```

---

## Performance Analysis {#performance-analysis}

### Measuring Overlap Efficiency

```python
def measure_overlap_efficiency():
    """
    Measure how well communication is overlapped with computation.
    """

    import torch.profiler

    # Profile with CUDA activities
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        # Run model
        output = model(input)

    # Analyze timeline
    events = prof.key_averages()

    # Find communication events
    comm_events = [e for e in events if 'nccl' in e.key.lower()]

    # Find computation events
    comp_events = [e for e in events if 'gemm' in e.key.lower() or 'matmul' in e.key.lower()]

    # Calculate overlap
    total_comm_time = sum(e.self_cuda_time_total for e in comm_events)
    total_comp_time = sum(e.self_cuda_time_total for e in comp_events)
    total_time = max(e.self_cuda_time_total for e in events)

    # Theoretical sequential time
    sequential_time = total_comm_time + total_comp_time

    # Actual overlapped time
    actual_time = total_time

    # Overlap efficiency
    overlap_efficiency = (sequential_time - actual_time) / total_comm_time

    print("Overlap Analysis:")
    print(f"  Total computation: {total_comp_time / 1000:.2f} ms")
    print(f"  Total communication: {total_comm_time / 1000:.2f} ms")
    print(f"  Sequential time: {sequential_time / 1000:.2f} ms")
    print(f"  Actual time: {actual_time / 1000:.2f} ms")
    print(f"  Overlap efficiency: {overlap_efficiency * 100:.1f}%")
    print(f"  Speedup: {sequential_time / actual_time:.2f}x")


def profile_overlap_timeline():
    """
    Generate timeline showing overlap.
    """

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        output = model(input)

    # Export Chrome trace for visualization
    prof.export_chrome_trace("overlap_timeline.json")

    print("Timeline exported to overlap_timeline.json")
    print("Open in chrome://tracing to visualize overlap")
```

---

## Best Practices {#best-practices}

### 1. Identify Overlappable Operations

```python
"""
Good candidates for overlap:
✓ All-reduce with next layer computation
✓ Send/recv with local computation
✓ Gradient all-reduce with backward pass

Poor candidates:
✗ Sequential dependencies (can't overlap)
✗ Very fast operations (overhead > benefit)
✗ Memory-bound operations (compete for bandwidth)
"""


def identify_overlap_opportunities(model):
    """
    Analyze model to find overlap opportunities.
    """

    print("Overlap Opportunities:")

    for name, module in model.named_modules():
        if isinstance(module, RowParallelLinear):
            print(f"✓ {name}: All-reduce can overlap with next layer")

        if isinstance(module, PipelineStage):
            print(f"✓ {name}: Send/recv can overlap with computation")

        if isinstance(module, nn.Linear) and is_large(module):
            print(f"✓ {name}: Large GEMM can overlap with communication")
```

### 2. Use Appropriate Stream Count

```python
def choose_stream_count():
    """
    Too few streams: Limited overlap
    Too many streams: Overhead and complexity

    Recommended:
    - 1 default stream (computation)
    - 1 communication stream
    - Optionally 1 memory stream

    Total: 2-3 streams
    """

    # Good
    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()

    # Avoid creating many streams
    # Bad: streams = [torch.cuda.Stream() for _ in range(100)]
```

### 3. Proper Synchronization

```python
def proper_synchronization():
    """
    Always synchronize before using results from async operations.
    """

    # Start async operation
    work = dist.all_reduce(tensor, async_op=True)

    # Do other work
    result = compute()

    # Must wait before using tensor
    work.wait()  # ← Don't forget!

    # Now safe to use
    output = tensor + result
```

### 4. Monitor Overhead

```python
def monitor_async_overhead():
    """
    Async operations have overhead. Monitor to ensure benefit > overhead.
    """

    # Too much overhead
    for small_tensor in many_small_tensors:
        work = dist.all_reduce(small_tensor, async_op=True)  # Bad: overhead too high
        work.wait()

    # Better: batch
    combined = torch.cat(many_small_tensors)
    work = dist.all_reduce(combined, async_op=True)
    work.wait()
    results = combined.split(...)
```

---

## Summary

Communication-computation overlap is essential for distributed inference efficiency:

**Key Techniques:**
1. Async NCCL operations (async_op=True)
2. Multi-stream execution
3. Layer-wise overlap in transformers
4. Gradient all-reduce during backward

**Best Practices:**
- Profile to identify overlap opportunities
- Use 2-3 CUDA streams
- Synchronize properly before using results
- Batch small operations to amortize overhead

**Expected Gains:**
- 20-40% speedup when communication is 10-30% of total time
- Higher gains with slower interconnects
- Critical for scaling to many GPUs

Continue to the next module on Ray distributed framework for multi-node orchestration.
