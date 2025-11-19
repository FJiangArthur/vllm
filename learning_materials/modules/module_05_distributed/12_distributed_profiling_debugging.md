# Distributed Profiling and Debugging

## Table of Contents
1. [Introduction](#introduction)
2. [Profiling Tools](#profiling-tools)
3. [Performance Bottlenecks](#bottlenecks)
4. [Debugging Distributed Issues](#debugging)
5. [Logging and Tracing](#logging)
6. [Visualization Tools](#visualization)
7. [Common Issues](#common-issues)
8. [Best Practices](#best-practices)

---

## Introduction {#introduction}

Profiling and debugging distributed systems is challenging due to concurrency, communication, and the sheer scale of operations. This module covers tools and techniques for identifying and resolving issues.

### Challenges in Distributed Profiling

```python
"""
Distributed System Challenges:

1. Visibility:
   - Multiple processes across nodes
   - Async operations difficult to trace
   - Race conditions and timing issues

2. Complexity:
   - Communication overhead
   - Load imbalance
   - Resource contention

3. Scale:
   - 100s of GPUs
   - 1000s of operations/second
   - TBs of data transferred

Tools Needed:
- System-wide profiling
- Cross-process tracing
- Performance visualization
- Distributed debugging
"""
```

---

## Profiling Tools {#profiling-tools}

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

def profile_distributed_inference():
    """Profile distributed inference with PyTorch profiler."""

    # Setup distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # Create model
    model = create_distributed_model()

    # Profile
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        # Enable distributed profiling
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f'./log/rank_{rank}'
        ),
    ) as prof:
        with record_function("model_inference"):
            # Run inference
            outputs = model(inputs)

        # Communication (will be captured)
        dist.all_reduce(outputs)

    # Print summary (rank 0)
    if rank == 0:
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))

        # Export Chrome trace
        prof.export_chrome_trace("trace.json")


### NVIDIA Nsight Systems

```bash
# Profile distributed application with Nsight Systems

# Single node, multiple GPUs
nsys profile \
    --output=nsys_report \
    --trace=cuda,nvtx,mpi \
    --sample=cpu \
    --cpuctxsw=none \
    torchrun --nproc_per_node=8 inference.py

# Multi-node (run on each node)
mpirun -np 16 \
    -H node1:8,node2:8 \
    nsys profile \
    --output=nsys_node_%h_rank_%q \
    --trace=cuda,nvtx,mpi,ucx \
    python inference.py

# View results
# nsys-ui nsys_report.nsys-rep
```

### NCCL Profiling

```python
import os

def enable_nccl_profiling():
    """Enable detailed NCCL profiling."""

    # Enable NCCL debug output
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'  # Collective operations

    # Log to file
    os.environ['NCCL_DEBUG_FILE'] = '/tmp/nccl_debug_%h_%p.log'

    # Enable NCCL profiling
    os.environ['NCCL_GRAPH_DUMP_FILE'] = '/tmp/nccl_graph.xml'

    print("NCCL profiling enabled")


def benchmark_nccl_operations():
    """Benchmark NCCL collective operations."""

    import torch.distributed as dist

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Test different sizes
    sizes_mb = [1, 4, 16, 64, 256]
    operations = ['all_reduce', 'all_gather', 'broadcast']

    results = []

    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1024 * 1024 / 4)  # FP32
        tensor = torch.randn(num_elements, device='cuda')

        for op in operations:
            # Warmup
            for _ in range(10):
                if op == 'all_reduce':
                    dist.all_reduce(tensor.clone())
                elif op == 'all_gather':
                    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
                    dist.all_gather(tensors, tensor)
                elif op == 'broadcast':
                    dist.broadcast(tensor.clone(), src=0)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            num_iters = 100
            for _ in range(num_iters):
                if op == 'all_reduce':
                    dist.all_reduce(tensor.clone())
                elif op == 'all_gather':
                    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
                    dist.all_gather(tensors, tensor)
                elif op == 'broadcast':
                    dist.broadcast(tensor.clone(), src=0)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            avg_time_ms = elapsed * 1000 / num_iters
            bandwidth_gbps = (size_mb / 1024) / (avg_time_ms / 1000)

            if rank == 0:
                results.append({
                    'operation': op,
                    'size_mb': size_mb,
                    'time_ms': avg_time_ms,
                    'bandwidth_gbps': bandwidth_gbps,
                })

    if rank == 0:
        import pandas as pd
        df = pd.DataFrame(results)
        print(df.to_string())
        df.to_csv('nccl_benchmark.csv', index=False)
```

---

## Performance Bottlenecks {#bottlenecks}

### Identifying Communication Bottlenecks

```python
def analyze_communication_overhead():
    """
    Analyze communication vs computation time.
    """

    import torch.profiler

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        # Run model
        output = model(input)

    # Analyze events
    events = prof.key_averages()

    # Categorize
    compute_time = 0
    comm_time = 0
    other_time = 0

    for event in events:
        key = event.key.lower()

        if 'nccl' in key or 'all_reduce' in key or 'all_gather' in key:
            comm_time += event.self_cuda_time_total
        elif 'gemm' in key or 'matmul' in key or 'conv' in key:
            compute_time += event.self_cuda_time_total
        else:
            other_time += event.self_cuda_time_total

    total_time = compute_time + comm_time + other_time

    print("\nTime Breakdown:")
    print(f"  Computation: {compute_time / 1000:.2f} ms ({compute_time/total_time*100:.1f}%)")
    print(f"  Communication: {comm_time / 1000:.2f} ms ({comm_time/total_time*100:.1f}%)")
    print(f"  Other: {other_time / 1000:.2f} ms ({other_time/total_time*100:.1f}%)")

    # Communication overhead ratio
    comm_overhead = comm_time / compute_time

    print(f"\nCommunication overhead: {comm_overhead:.2f}x computation")

    if comm_overhead > 0.3:
        print("⚠️  High communication overhead!")
        print("Recommendations:")
        print("  - Increase batch size to amortize communication")
        print("  - Use communication-computation overlap")
        print("  - Consider reducing tensor parallel size")


### Detecting Load Imbalance

```python
def detect_load_imbalance():
    """
    Detect load imbalance across workers.
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Measure local computation time
    torch.cuda.synchronize()
    start = time.time()

    # Do computation
    for _ in range(100):
        output = model(input)

    torch.cuda.synchronize()
    local_time = time.time() - start

    # Gather times from all ranks
    all_times = [torch.tensor(0.0) for _ in range(world_size)]
    dist.all_gather_object(all_times, local_time)

    if rank == 0:
        times = [t.item() for t in all_times]

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        # Calculate imbalance
        imbalance = (max_time - min_time) / mean_time

        print("\nLoad Balance Analysis:")
        print(f"  Mean time: {mean_time:.3f}s")
        print(f"  Std dev: {std_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s (rank {np.argmin(times)})")
        print(f"  Max time: {max_time:.3f}s (rank {np.argmax(times)})")
        print(f"  Imbalance: {imbalance*100:.1f}%")

        if imbalance > 0.1:
            print("⚠️  Significant load imbalance detected!")
            print("Recommendations:")
            print("  - Check model partitioning (layers per rank)")
            print("  - Verify data distribution")
            print("  - Look for stragglers (slow GPUs/nodes)")


### Memory Profiling

```python
def profile_memory_usage():
    """
    Profile memory usage across distributed system.
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Track memory
    torch.cuda.reset_peak_memory_stats()

    # Run inference
    output = model(input)

    # Collect memory stats
    stats = {
        'rank': rank,
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'peak': torch.cuda.max_memory_allocated() / 1e9,
    }

    # Gather from all ranks
    all_stats = [None] * world_size
    dist.all_gather_object(all_stats, stats)

    if rank == 0:
        print("\nMemory Usage (GB):")
        print(f"{'Rank':>6} {'Allocated':>12} {'Reserved':>12} {'Peak':>12}")
        print("-" * 48)

        for s in all_stats:
            print(f"{s['rank']:>6} {s['allocated']:>12.2f} {s['reserved']:>12.2f} {s['peak']:>12.2f}")

        # Check balance
        peaks = [s['peak'] for s in all_stats]
        peak_imbalance = (max(peaks) - min(peaks)) / np.mean(peaks)

        if peak_imbalance > 0.2:
            print(f"\n⚠️  Memory imbalance: {peak_imbalance*100:.1f}%")
```

---

## Debugging Distributed Issues {#debugging}

### Debugging Hangs

```python
def debug_hang():
    """
    Debug distributed hangs.

    Common causes:
    - Mismatched collective operations
    - Deadlocks
    - Network issues
    """

    # Enable detailed logging
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Enable NCCL debug
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    # Enable PyTorch distributed debug
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # Set timeout
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(seconds=30),  # Short timeout to detect hangs
    )

    rank = dist.get_rank()

    print(f"Rank {rank}: Initialized")

    # Test collective operations
    try:
        tensor = torch.randn(100, device='cuda')

        print(f"Rank {rank}: Starting all-reduce")
        dist.all_reduce(tensor)
        print(f"Rank {rank}: All-reduce complete")

    except Exception as e:
        print(f"Rank {rank}: Error - {e}")

        # Dump stack traces
        import traceback
        traceback.print_exc()


def check_collective_sync():
    """
    Verify all ranks call collectives in sync.
    """

    rank = dist.get_rank()

    # Add barrier before each collective
    dist.barrier()
    print(f"Rank {rank}: Before all-reduce")

    tensor = torch.randn(100, device='cuda')
    dist.all_reduce(tensor)

    dist.barrier()
    print(f"Rank {rank}: After all-reduce")


### Debugging Communication Errors

```python
def debug_nccl_errors():
    """
    Debug NCCL communication errors.
    """

    # Test network connectivity
    def test_network():
        """Test network between all pairs of ranks."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        for other_rank in range(world_size):
            if other_rank == rank:
                continue

            try:
                tensor = torch.tensor([rank], device='cuda')

                # Send to other rank
                dist.send(tensor, dst=other_rank)

                # Receive from other rank
                recv_tensor = torch.tensor([0], device='cuda')
                dist.recv(recv_tensor, src=other_rank)

                print(f"Rank {rank} ↔ Rank {other_rank}: OK")

            except Exception as e:
                print(f"Rank {rank} ↔ Rank {other_rank}: FAILED - {e}")

    test_network()


    # Test NCCL operations
    def test_nccl_ops():
        """Test different NCCL operations."""

        rank = dist.get_rank()
        tensor = torch.randn(1000, device='cuda')

        ops = {
            'all_reduce': lambda t: dist.all_reduce(t),
            'broadcast': lambda t: dist.broadcast(t, src=0),
            'all_gather': lambda t: dist.all_gather(
                [torch.empty_like(t) for _ in range(dist.get_world_size())], t
            ),
        }

        for op_name, op_func in ops.items():
            try:
                op_func(tensor.clone())
                print(f"Rank {rank}: {op_name} - OK")
            except Exception as e:
                print(f"Rank {rank}: {op_name} - FAILED - {e}")

    test_nccl_ops()
```

---

## Logging and Tracing {#logging}

### Distributed Logging

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_distributed_logging(log_dir='/tmp/logs'):
    """
    Setup logging for distributed system.

    Each rank logs to separate file.
    """

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.DEBUG)

    # File handler (separate file per rank)
    fh = RotatingFileHandler(
        f'{log_dir}/rank_{rank}.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5,
    )
    fh.setLevel(logging.DEBUG)

    # Console handler (only rank 0)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - Rank %(rank)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    # Add rank to log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)

    logger.addHandler(fh)

    return logger


# Usage
logger = setup_distributed_logging()

logger.info("Starting inference")
logger.debug(f"Batch size: {batch_size}")
logger.error("OOM error occurred")


### Distributed Tracing

```python
import time
from contextlib import contextmanager

class DistributedTracer:
    """
    Distributed tracing for performance analysis.
    """

    def __init__(self):
        self.traces = []
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    @contextmanager
    def trace(self, name):
        """Trace a code section."""

        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            duration = end - start

            self.traces.append({
                'rank': self.rank,
                'name': name,
                'start': start,
                'end': end,
                'duration': duration,
            })

    def export_traces(self, filename='traces.json'):
        """Export traces in Chrome trace format."""

        chrome_traces = []

        for trace in self.traces:
            chrome_traces.append({
                'name': trace['name'],
                'cat': f'rank_{trace["rank"]}',
                'ph': 'X',  # Complete event
                'ts': trace['start'] * 1e6,  # Microseconds
                'dur': trace['duration'] * 1e6,
                'pid': trace['rank'],
                'tid': 0,
            })

        # Save to file
        import json
        with open(filename, 'w') as f:
            json.dump(chrome_traces, f)

        print(f"Traces exported to {filename}")
        print(f"View in chrome://tracing")


# Usage
tracer = DistributedTracer()

with tracer.trace("forward_pass"):
    output = model(input)

with tracer.trace("backward_pass"):
    loss.backward()

with tracer.trace("optimizer_step"):
    optimizer.step()

tracer.export_traces()
```

---

## Visualization Tools {#visualization}

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

def log_distributed_metrics():
    """
    Log distributed metrics to TensorBoard.
    """

    rank = dist.get_rank()

    # Create writer for this rank
    writer = SummaryWriter(f'runs/rank_{rank}')

    iteration = 0

    while training:
        # Log metrics
        writer.add_scalar('Loss/train', loss, iteration)
        writer.add_scalar('GPU/utilization', gpu_util, iteration)
        writer.add_scalar('GPU/memory', gpu_memory, iteration)
        writer.add_scalar('Communication/time', comm_time, iteration)

        # Log histograms
        for name, param in model.named_parameters():
            writer.add_histogram(f'Params/{name}', param, iteration)

        iteration += 1

    writer.close()


# View with:
# tensorboard --logdir=runs
```

### Custom Dashboards

```python
def create_monitoring_dashboard():
    """
    Create real-time monitoring dashboard.
    """

    import dash
    from dash import dcc, html
    import plotly.graph_objs as go

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1('Distributed Inference Monitor'),

        # GPU utilization
        dcc.Graph(id='gpu-utilization'),

        # Communication time
        dcc.Graph(id='comm-time'),

        # Request throughput
        dcc.Graph(id='throughput'),

        # Auto-refresh
        dcc.Interval(id='interval', interval=1000),  # 1 second
    ])

    @app.callback(
        [dash.dependencies.Output('gpu-utilization', 'figure'),
         dash.dependencies.Output('comm-time', 'figure'),
         dash.dependencies.Output('throughput', 'figure')],
        [dash.dependencies.Input('interval', 'n_intervals')]
    )
    def update_graphs(n):
        # Fetch metrics from all ranks
        metrics = collect_metrics_from_all_ranks()

        # GPU utilization graph
        gpu_fig = go.Figure()
        for rank, util in enumerate(metrics['gpu_util']):
            gpu_fig.add_trace(go.Scatter(
                y=util,
                name=f'Rank {rank}',
                mode='lines'
            ))

        # Similar for other graphs...

        return gpu_fig, comm_fig, throughput_fig

    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

---

## Common Issues {#common-issues}

### Issue Checklist

```python
def debug_checklist():
    """
    Systematic debugging checklist.
    """

    print("Distributed Debugging Checklist:")
    print("=" * 60)

    checks = [
        ("All ranks initialized?", check_ranks_initialized),
        ("Network connectivity OK?", check_network),
        ("NCCL working?", test_nccl),
        ("GPU memory available?", check_gpu_memory),
        ("Collectives synchronized?", check_collective_sync),
        ("No stragglers?", check_for_stragglers),
        ("Load balanced?", check_load_balance),
    ]

    for check_name, check_func in checks:
        try:
            result = check_func()
            status = "✓" if result else "✗"
            print(f"{status} {check_name}")
        except Exception as e:
            print(f"✗ {check_name} - Error: {e}")


def check_ranks_initialized():
    """Check if all ranks are initialized."""
    return dist.is_initialized() and dist.get_world_size() > 0


def check_network():
    """Check network connectivity."""
    try:
        tensor = torch.tensor([1.0], device='cuda')
        dist.all_reduce(tensor)
        return True
    except:
        return False
```

---

## Best Practices {#best-practices}

### 1. Profile Early and Often

```python
"""
Profile at different scales:
✓ Single GPU (baseline)
✓ Single node (8 GPUs)
✓ Multiple nodes

Identify bottlenecks early before scaling further.
"""
```

### 2. Use Appropriate Tools

```python
"""
Tool Selection:

System-level: Nsight Systems
Kernel-level: Nsight Compute
Python-level: PyTorch Profiler
Communication: NCCL profiling
Application: Custom logging/tracing
"""
```

### 3. Monitor Continuously

```python
"""
Production Monitoring:

✓ GPU utilization per device
✓ Communication time percentage
✓ Request latency (P50, P95, P99)
✓ Memory usage trends
✓ Error rates

Alert on anomalies.
"""
```

---

## Summary

Effective profiling and debugging is essential for distributed inference:

**Key Tools:**
- PyTorch Profiler for Python-level profiling
- NVIDIA Nsight for system-level analysis
- NCCL profiling for communication
- Custom logging and tracing

**Common Bottlenecks:**
- Communication overhead (>30% of time)
- Load imbalance (>10% variance)
- Memory pressure
- Network issues

**Best Practices:**
- Profile at multiple scales
- Use systematic debugging approach
- Monitor continuously in production
- Maintain debug logs
- Test failure scenarios

**Debugging Workflow:**
1. Reproduce issue reliably
2. Collect logs from all ranks
3. Profile to identify bottleneck
4. Isolate problematic component
5. Fix and verify
6. Test at scale

This completes Module 5 on Distributed Inference!
