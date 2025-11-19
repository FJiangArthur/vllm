# NCCL Collective Operations

## Table of Contents
1. [Introduction to NCCL](#introduction)
2. [NCCL Architecture](#architecture)
3. [Collective Operations](#collective-operations)
4. [Communication Algorithms](#algorithms)
5. [Performance Optimization](#performance-optimization)
6. [Benchmarking NCCL](#benchmarking)
7. [NCCL in vLLM](#vllm-nccl)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction to NCCL {#introduction}

NCCL (NVIDIA Collective Communications Library) is a library of multi-GPU collective communication primitives optimized for NVIDIA GPUs. It's the foundation for distributed deep learning and inference on NVIDIA hardware.

### What is NCCL?

**NCCL provides:**
- Optimized collective operations (all-reduce, broadcast, etc.)
- Multi-GPU and multi-node support
- Automatic topology detection and optimization
- Support for NVLink, PCIe, and InfiniBand
- Integration with PyTorch, TensorFlow, and other frameworks

**Key Features:**
- **GPU-direct**: Direct GPU-to-GPU communication
- **Topology-aware**: Optimizes for hardware topology
- **Efficient**: Near-linear scaling on modern interconnects
- **Easy integration**: Simple API

### Why NCCL Matters for vLLM

```
vLLM's distributed operations rely on NCCL:

Tensor Parallelism:
- All-reduce after attention projection
- All-reduce after MLP layers
- ~2× all-reduce per transformer layer

Pipeline Parallelism:
- Send/recv activations between stages
- Point-to-point communication

Data Parallelism:
- All-reduce gradients (training)
- Broadcast parameters

Performance Impact:
- Communication can be 20-40% of total time
- NCCL optimization crucial for scaling
```

---

## NCCL Architecture {#architecture}

### Communication Topology

```python
"""
NCCL automatically detects and optimizes for hardware topology.

Example topologies:

1. Single Node with NVLink:
   GPU0 <--NVLink--> GPU1
    |                 |
   NVLink          NVLink
    |                 |
   GPU2 <--NVLink--> GPU3

   Bandwidth: ~300-600 GB/s (NVLink 3.0/4.0)

2. Multi-Node with InfiniBand:
   Node 0:           Node 1:
   GPU0 - GPU1       GPU2 - GPU3
     |      |          |      |
     +--IB--+----------+--IB--+

   Bandwidth: ~200 GB/s (HDR InfiniBand)

3. PCIe-only:
   CPU
    |
   PCIe switch
    |
   GPU0 - GPU1 - GPU2 - GPU3

   Bandwidth: ~32 GB/s (PCIe 4.0 x16)
"""


def detect_gpu_topology():
    """
    Detect GPU topology using nvidia-smi.
    """
    import subprocess
    import re

    # Run nvidia-smi to get topology
    result = subprocess.run(
        ['nvidia-smi', 'topo', '-m'],
        capture_output=True,
        text=True
    )

    print("GPU Topology:")
    print(result.stdout)

    # Parse for NVLink connections
    nvlink_pattern = r'NV(\d+)'
    nvlink_count = len(re.findall(nvlink_pattern, result.stdout))

    if nvlink_count > 0:
        print(f"\nNVLink detected: {nvlink_count} connections")
        print("Recommendation: Use aggressive tensor parallelism (TP=4-8)")
    else:
        print("\nNo NVLink detected (PCIe only)")
        print("Recommendation: Limit tensor parallelism (TP=2), prefer pipeline/data parallelism")

    return result.stdout


# Example usage
# detect_gpu_topology()
```

### NCCL Rings and Trees

```python
"""
NCCL uses different algorithms based on operation and topology:

1. Ring Algorithm:
   - Data flows in ring: GPU0 -> GPU1 -> GPU2 -> GPU3 -> GPU0
   - Good for all-reduce with uniform bandwidth
   - Steps: 2(N-1) where N = number of GPUs

   Example All-Reduce on 4 GPUs:
   Step 1: GPU0 sends to GPU1, GPU1 to GPU2, GPU2 to GPU3, GPU3 to GPU0
   Step 2: Each GPU reduces received data
   Step 3-6: Continue until all GPUs have full reduction

2. Tree Algorithm:
   - Hierarchical reduction/broadcast
   - Better for non-uniform topologies
   - Fewer steps: 2*log2(N)

   Example Tree Reduce:
           GPU0
          /    \
       GPU1    GPU2
                 |
               GPU3

3. Double Binary Tree:
   - NCCL's optimized algorithm
   - Two trees for send/receive
   - Better bandwidth utilization
"""

def demonstrate_ring_allreduce():
    """
    Demonstrate ring all-reduce algorithm.
    """
    num_gpus = 4
    chunks_per_gpu = 4

    print("Ring All-Reduce Algorithm")
    print("=" * 60)
    print(f"GPUs: {num_gpus}, Chunks per GPU: {chunks_per_gpu}")
    print()

    # Initial data
    # Each GPU has chunks: A, B, C, D
    data = {
        0: ['A0', 'B0', 'C0', 'D0'],
        1: ['A1', 'B1', 'C1', 'D1'],
        2: ['A2', 'B2', 'C2', 'D2'],
        3: ['A3', 'B3', 'C3', 'D3'],
    }

    print("Initial state:")
    for gpu, chunks in data.items():
        print(f"  GPU{gpu}: {chunks}")
    print()

    # Reduce-scatter phase
    print("Phase 1: Reduce-Scatter")
    for step in range(num_gpus - 1):
        print(f"Step {step + 1}:")
        for gpu in range(num_gpus):
            send_to = (gpu + 1) % num_gpus
            chunk_idx = (gpu - step) % num_gpus
            print(f"  GPU{gpu} sends chunk {chunk_idx} to GPU{send_to}")
        print()

    # All-gather phase
    print("Phase 2: All-Gather")
    for step in range(num_gpus - 1):
        print(f"Step {step + 1}:")
        for gpu in range(num_gpus):
            send_to = (gpu + 1) % num_gpus
            chunk_idx = (gpu - step - 1) % num_gpus
            print(f"  GPU{gpu} sends reduced chunk {chunk_idx} to GPU{send_to}")
        print()

    print("Final: All GPUs have fully reduced data")


# demonstrate_ring_allreduce()
```

---

## Collective Operations {#collective-operations}

### All-Reduce

```python
import torch
import torch.distributed as dist


def nccl_all_reduce_example():
    """
    All-reduce: Sum tensors across all GPUs, result on all GPUs.

    Before:
    GPU0: [1, 2, 3]
    GPU1: [4, 5, 6]
    GPU2: [7, 8, 9]

    After (sum):
    GPU0: [12, 15, 18]
    GPU1: [12, 15, 18]
    GPU2: [12, 15, 18]
    """

    # Initialize process group
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create tensor
    tensor = torch.arange(3, dtype=torch.float32, device='cuda') + rank * 3

    print(f"Rank {rank} before all-reduce: {tensor}")

    # All-reduce with sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after all-reduce: {tensor}")

    # Cleanup
    dist.destroy_process_group()


def benchmark_all_reduce(tensor_size_mb: float, num_iterations: int = 100):
    """
    Benchmark all-reduce performance.
    """
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create tensor
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # FP32 = 4 bytes
    tensor = torch.randn(num_elements, device='cuda')

    # Warm-up
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    if rank == 0:
        avg_time_ms = elapsed_ms / num_iterations
        tensor_size_bytes = num_elements * 4
        bandwidth_gbps = (tensor_size_bytes * 2 * (world_size - 1) / world_size) / (avg_time_ms / 1000) / 1e9

        print(f"All-Reduce Benchmark:")
        print(f"  Tensor size: {tensor_size_mb:.2f} MB")
        print(f"  World size: {world_size}")
        print(f"  Average time: {avg_time_ms:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"  Algorithm bandwidth: {bandwidth_gbps / 2:.2f} GB/s")  # Effective

    dist.destroy_process_group()
```

### Broadcast

```python
def nccl_broadcast_example():
    """
    Broadcast: Send tensor from one GPU to all others.

    Before:
    GPU0: [1, 2, 3]
    GPU1: [0, 0, 0]
    GPU2: [0, 0, 0]

    After (broadcast from GPU0):
    GPU0: [1, 2, 3]
    GPU1: [1, 2, 3]
    GPU2: [1, 2, 3]
    """

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # Source rank
    src_rank = 0

    # Create tensor
    if rank == src_rank:
        tensor = torch.arange(3, dtype=torch.float32, device='cuda')
    else:
        tensor = torch.zeros(3, dtype=torch.float32, device='cuda')

    print(f"Rank {rank} before broadcast: {tensor}")

    # Broadcast from rank 0
    dist.broadcast(tensor, src=src_rank)

    print(f"Rank {rank} after broadcast: {tensor}")

    dist.destroy_process_group()
```

### All-Gather

```python
def nccl_all_gather_example():
    """
    All-gather: Collect tensors from all GPUs to all GPUs.

    Before:
    GPU0: [1, 2]
    GPU1: [3, 4]
    GPU2: [5, 6]

    After:
    GPU0: [[1, 2], [3, 4], [5, 6]]
    GPU1: [[1, 2], [3, 4], [5, 6]]
    GPU2: [[1, 2], [3, 4], [5, 6]]
    """

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create local tensor
    local_tensor = torch.arange(2, dtype=torch.float32, device='cuda') + rank * 2

    print(f"Rank {rank} local tensor: {local_tensor}")

    # Allocate output
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]

    # All-gather
    dist.all_gather(gathered_tensors, local_tensor)

    # Concatenate
    full_tensor = torch.cat(gathered_tensors)

    print(f"Rank {rank} after all-gather: {full_tensor}")

    dist.destroy_process_group()
```

### Reduce-Scatter

```python
def nccl_reduce_scatter_example():
    """
    Reduce-scatter: Reduce and split across GPUs.

    Combines all-reduce + scatter.

    Before:
    GPU0: [1, 2, 3, 4]
    GPU1: [5, 6, 7, 8]
    GPU2: [9, 10, 11, 12]

    After (sum, then split):
    GPU0: [15, 18]      # Sum of first half
    GPU1: [21, 24]      # Sum of second half
    GPU2: ...
    """

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create input tensor
    input_tensor = torch.arange(4, dtype=torch.float32, device='cuda') + rank * 4

    print(f"Rank {rank} input: {input_tensor}")

    # Output will be 1/world_size of input
    output = torch.zeros(4 // world_size, dtype=torch.float32, device='cuda')

    # Reduce-scatter (sum operation)
    # Split input into chunks
    input_list = list(input_tensor.chunk(world_size))

    dist.reduce_scatter(output, input_list)

    print(f"Rank {rank} output: {output}")

    dist.destroy_process_group()
```

### Send/Recv (Point-to-Point)

```python
def nccl_send_recv_example():
    """
    Send/Recv: Point-to-point communication between two GPUs.

    Used for pipeline parallelism.
    """

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        # Sender
        tensor = torch.arange(5, dtype=torch.float32, device='cuda')
        print(f"Rank {rank} sending: {tensor}")
        dist.send(tensor, dst=1)
        print(f"Rank {rank} sent to rank 1")

    elif rank == 1:
        # Receiver
        tensor = torch.zeros(5, dtype=torch.float32, device='cuda')
        print(f"Rank {rank} before receive: {tensor}")
        dist.recv(tensor, src=0)
        print(f"Rank {rank} after receive: {tensor}")

    dist.destroy_process_group()
```

---

## Communication Algorithms {#algorithms}

### Algorithm Selection

```python
"""
NCCL automatically selects algorithm based on:
1. Operation type
2. Message size
3. Number of GPUs
4. Topology

Manual selection via environment variable:
export NCCL_ALGO=RING    # Force ring algorithm
export NCCL_ALGO=TREE    # Force tree algorithm
"""


def compare_nccl_algorithms():
    """
    Compare different NCCL algorithms.
    """
    import os

    algorithms = ['RING', 'TREE', 'AUTO']
    tensor_size_mb = 64
    num_iterations = 100

    results = {}

    for algo in algorithms:
        # Set algorithm
        if algo != 'AUTO':
            os.environ['NCCL_ALGO'] = algo
        else:
            os.environ.pop('NCCL_ALGO', None)

        # Benchmark
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()

        num_elements = int(tensor_size_mb * 1024 * 1024 / 4)
        tensor = torch.randn(num_elements, device='cuda')

        # Warm-up
        for _ in range(10):
            dist.all_reduce(tensor)

        # Time
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_iterations):
            dist.all_reduce(tensor)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if rank == 0:
            avg_time_ms = elapsed * 1000 / num_iterations
            results[algo] = avg_time_ms
            print(f"{algo:6s}: {avg_time_ms:.2f} ms")

        dist.destroy_process_group()

    if rank == 0:
        best_algo = min(results, key=results.get)
        print(f"\nBest algorithm: {best_algo}")

    return results
```

### Ring vs Tree Analysis

```python
def analyze_ring_vs_tree():
    """
    Analyze when to use ring vs tree algorithms.
    """

    print("Ring vs Tree Algorithm Analysis")
    print("=" * 60)

    # Ring algorithm
    print("\nRing Algorithm:")
    print("  Pros:")
    print("    - Optimal bandwidth utilization")
    print("    - Good for uniform bandwidth (NVLink, IB)")
    print("    - Scales well to many GPUs")
    print("  Cons:")
    print("    - More steps: 2(N-1)")
    print("    - Higher latency for small messages")
    print("  Best for:")
    print("    - Large messages (>1MB)")
    print("    - Uniform topology")
    print("    - 4-16 GPUs with NVLink")

    # Tree algorithm
    print("\nTree Algorithm:")
    print("  Pros:")
    print("    - Fewer steps: 2*log2(N)")
    print("    - Lower latency for small messages")
    print("    - Good for hierarchical topology")
    print("  Cons:")
    print("    - Root can be bottleneck")
    print("    - Lower bandwidth utilization")
    print("  Best for:")
    print("    - Small messages (<1MB)")
    print("    - Non-uniform topology")
    print("    - Many GPUs (>16)")

    # Recommendations
    print("\nRecommendations:")
    print("  NVLink single node (8 GPUs): RING")
    print("  Multi-node InfiniBand: RING or AUTO")
    print("  PCIe-only: TREE for small, RING for large")
    print("  100+ GPUs: Consider hierarchical all-reduce")
```

---

## Performance Optimization {#performance-optimization}

### NCCL Environment Variables

```python
def configure_nccl_for_performance():
    """
    Configure NCCL environment variables for optimal performance.
    """
    import os

    print("Configuring NCCL for Performance")
    print("=" * 60)

    # Network interface
    # Specify which network interface to use
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Change to your interface
    print("✓ Set NCCL_SOCKET_IFNAME=eth0")

    # Enable InfiniBand if available
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_IB_HCA'] = 'mlx5'  # InfiniBand adapter
    print("✓ Enabled InfiniBand")

    # Enable GPU Direct RDMA (if supported)
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'
    print("✓ Enabled GPU Direct RDMA")

    # P2P and NVLink settings
    os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'  # Use NVLink
    print("✓ Enabled NVLink P2P")

    # Buffer sizes
    os.environ['NCCL_BUFFSIZE'] = '8388608'  # 8MB buffers
    print("✓ Set buffer size to 8MB")

    # Number of threads
    os.environ['NCCL_NTHREADS'] = '512'
    print("✓ Set 512 NCCL threads")

    # Timeout (for debugging)
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes
    print("✓ Set timeout to 30 minutes")

    # Protocol (for debugging)
    # os.environ['NCCL_PROTO'] = 'Simple'  # Simple, LL, or LL128
    # print("✓ Set protocol to Simple")

    # Enable tuning
    os.environ['NCCL_TUNER_PLUGIN'] = 'libnccl-tuner.so'
    print("✓ Enabled NCCL tuner")

    print("\nNCCL configuration complete!")


# For debugging
def configure_nccl_for_debugging():
    """
    Configure NCCL for debugging.
    """
    import os

    os.environ['NCCL_DEBUG'] = 'INFO'  # Or 'TRACE' for verbose
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['NCCL_DEBUG_FILE'] = '/tmp/nccl_debug_%h_%p.log'

    print("NCCL debugging enabled")
    print("  Debug level: INFO")
    print("  Log file: /tmp/nccl_debug_<hostname>_<pid>.log")
```

### Optimizing Message Sizes

```python
def optimize_message_size():
    """
    Analyze optimal message size for NCCL operations.
    """

    print("Message Size Optimization")
    print("=" * 60)

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # Test different message sizes
    message_sizes_kb = [1, 4, 16, 64, 256, 1024, 4096, 16384]

    results = []

    for size_kb in message_sizes_kb:
        num_elements = size_kb * 256  # 4 bytes per float32
        tensor = torch.randn(num_elements, device='cuda')

        # Warm-up
        for _ in range(5):
            dist.all_reduce(tensor)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        num_iters = max(10, 1000 // size_kb)  # More iterations for small messages
        for _ in range(num_iters):
            dist.all_reduce(tensor)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time_ms = elapsed * 1000 / num_iters
        bandwidth_gbps = (size_kb / 1024) / (avg_time_ms / 1000)

        results.append({
            'size_kb': size_kb,
            'time_ms': avg_time_ms,
            'bandwidth_gbps': bandwidth_gbps,
        })

        if rank == 0:
            print(f"{size_kb:6d} KB: {avg_time_ms:6.2f} ms, {bandwidth_gbps:6.2f} GB/s")

    if rank == 0:
        # Find optimal size (best bandwidth)
        optimal = max(results, key=lambda x: x['bandwidth_gbps'])
        print(f"\nOptimal message size: {optimal['size_kb']} KB")
        print(f"Peak bandwidth: {optimal['bandwidth_gbps']:.2f} GB/s")

        # Recommendations
        print("\nRecommendations:")
        print("  - Batch small operations to reach optimal size")
        print("  - Avoid very small messages (<4KB)")
        print("  - Use asynchronous operations when possible")

    dist.destroy_process_group()
```

---

## Benchmarking NCCL {#benchmarking}

### NCCL Tests Suite

```bash
# Official NCCL tests
# https://github.com/NVIDIA/nccl-tests

# Build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda

# Run all-reduce test
# 4 GPUs, 1GB message size
mpirun -np 4 ./build/all_reduce_perf -b 1G -e 1G -f 2 -g 1

# Run all-to-all test
mpirun -np 4 ./build/alltoall_perf -b 128M -e 1G -f 2 -g 1

# Run bandwidth test
mpirun -np 4 ./build/sendrecv_perf -b 1M -e 1G -f 2 -g 1
```

### Custom Benchmarks

```python
def comprehensive_nccl_benchmark():
    """
    Comprehensive NCCL benchmark for all operations.
    """

    operations = [
        ('all_reduce', dist.all_reduce),
        ('broadcast', lambda t: dist.broadcast(t, src=0)),
        ('all_gather', lambda t: dist.all_gather([torch.empty_like(t) for _ in range(dist.get_world_size())], t)),
    ]

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("Comprehensive NCCL Benchmark")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name()}")
        print()

    # Test sizes: 1KB to 1GB
    sizes_kb = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]

    for op_name, op_func in operations:
        if rank == 0:
            print(f"\n{op_name.upper()}")
            print("-" * 80)
            print(f"{'Size':>10s} {'Time (ms)':>12s} {'Bandwidth (GB/s)':>20s}")

        for size_kb in sizes_kb:
            num_elements = size_kb * 256
            tensor = torch.randn(num_elements, device='cuda')

            # Warm-up
            for _ in range(5):
                try:
                    op_func(tensor.clone())
                except:
                    pass

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            num_iters = max(10, 10000 // size_kb)
            for _ in range(num_iters):
                op_func(tensor.clone())

            torch.cuda.synchronize()
            elapsed = time.time() - start

            avg_time_ms = elapsed * 1000 / num_iters
            size_gb = size_kb / (1024 * 1024)
            bandwidth_gbps = size_gb / (avg_time_ms / 1000)

            if rank == 0:
                print(f"{size_kb:8d} KB {avg_time_ms:12.3f} {bandwidth_gbps:20.2f}")

    dist.destroy_process_group()
```

---

## NCCL in vLLM {#vllm-nccl}

### vLLM's NCCL Usage

```python
# From vllm/distributed/communication_op.py

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """
    All-reduce within tensor model parallel group.

    Used in row-parallel linear layers.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    # All-reduce across TP group
    torch.distributed.all_reduce(
        input_,
        group=get_tensor_model_parallel_group()
    )

    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor) -> torch.Tensor:
    """
    All-gather within tensor model parallel group.

    Used in column-parallel linear layers when gathering output.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    # Get dimension to gather along
    last_dim = input_.dim() - 1

    # Allocate output
    gather_list = [torch.empty_like(input_) for _ in range(world_size)]

    # All-gather
    torch.distributed.all_gather(
        gather_list,
        input_,
        group=get_tensor_model_parallel_group()
    )

    # Concatenate
    output = torch.cat(gather_list, dim=last_dim)

    return output


def pipeline_parallel_send_recv(
    output: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Send/recv for pipeline parallelism.

    Implements point-to-point communication between pipeline stages.
    """
    pp_rank = get_pipeline_model_parallel_rank()
    pp_world_size = get_pipeline_model_parallel_world_size()

    # Previous stage
    if pp_rank > 0:
        prev_rank = pp_rank - 1
    else:
        prev_rank = None

    # Next stage
    if pp_rank < pp_world_size - 1:
        next_rank = pp_rank + 1
    else:
        next_rank = None

    # Send to next stage
    if next_rank is not None and output is not None:
        torch.distributed.send(output, dst=next_rank)

    # Receive from previous stage
    input_tensor = None
    if prev_rank is not None and recv_prev:
        input_tensor = torch.empty_like(output)
        torch.distributed.recv(input_tensor, src=prev_rank)

    return input_tensor
```

### Monitoring NCCL Performance

```python
def monitor_nccl_performance():
    """
    Monitor NCCL performance in vLLM.
    """
    import torch.profiler

    dist.init_process_group(backend='nccl')

    # Create model (simplified)
    model = YourModel()

    # Profile with NCCL operations
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Run forward pass
        output = model(inputs)

    # Analyze results
    events = prof.key_averages()

    # Filter NCCL operations
    nccl_events = [e for e in events if 'nccl' in e.key.lower()]

    total_time = sum(e.self_cuda_time_total for e in events)
    nccl_time = sum(e.self_cuda_time_total for e in nccl_events)

    print("NCCL Performance Analysis")
    print("=" * 60)
    print(f"Total time: {total_time / 1000:.2f} ms")
    print(f"NCCL time: {nccl_time / 1000:.2f} ms")
    print(f"NCCL overhead: {nccl_time / total_time * 100:.1f}%")
    print()

    print("Top NCCL Operations:")
    sorted_events = sorted(nccl_events, key=lambda e: e.self_cuda_time_total, reverse=True)
    for event in sorted_events[:5]:
        print(f"  {event.key}: {event.self_cuda_time_total / 1000:.2f} ms")

    dist.destroy_process_group()
```

---

## Advanced Features {#advanced-features}

### NCCL Groups

```python
def use_multiple_nccl_groups():
    """
    Use multiple NCCL groups for different operations.

    Useful for hybrid parallelism.
    """

    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create TP group (ranks 0-3)
    tp_ranks = list(range(4))
    tp_group = dist.new_group(tp_ranks)

    # Create DP group (ranks 0, 4, 8, ...)
    dp_ranks = list(range(0, world_size, 4))
    dp_group = dist.new_group(dp_ranks)

    # Example: TP all-reduce
    if rank in tp_ranks:
        tensor = torch.randn(1024, device='cuda')
        dist.all_reduce(tensor, group=tp_group)
        print(f"Rank {rank}: TP all-reduce complete")

    # Example: DP all-reduce (less frequent)
    if rank in dp_ranks:
        tensor = torch.randn(1024, device='cuda')
        dist.all_reduce(tensor, group=dp_group)
        print(f"Rank {rank}: DP all-reduce complete")

    dist.destroy_process_group()
```

### Asynchronous NCCL

```python
def use_async_nccl():
    """
    Use asynchronous NCCL operations to overlap communication and computation.
    """

    dist.init_process_group(backend='nccl')

    # Create tensors
    comm_tensor = torch.randn(1024 * 1024, device='cuda')
    compute_tensor = torch.randn(1024, 1024, device='cuda')

    # Start async all-reduce
    work = dist.all_reduce(comm_tensor, async_op=True)

    # Do computation while communication happens
    for _ in range(100):
        compute_tensor = torch.matmul(compute_tensor, compute_tensor)

    # Wait for communication to complete
    work.wait()

    print("Communication and computation overlapped successfully")

    dist.destroy_process_group()
```

---

## Troubleshooting {#troubleshooting}

### Common NCCL Errors

```python
def troubleshoot_nccl_errors():
    """
    Common NCCL errors and solutions.
    """

    print("NCCL Troubleshooting Guide")
    print("=" * 60)

    issues = [
        {
            'error': 'NCCL error: unhandled system error',
            'causes': [
                'Network connectivity issues',
                'Firewall blocking NCCL ports',
                'Incompatible NCCL versions',
            ],
            'solutions': [
                'Check network with ping between nodes',
                'Disable firewall or open ports 1024-65535',
                'Ensure same NCCL version on all nodes',
                'Set NCCL_DEBUG=INFO for more details',
            ],
        },
        {
            'error': 'NCCL timeout',
            'causes': [
                'Slow network',
                'Large message size',
                'Deadlock in user code',
            ],
            'solutions': [
                'Increase timeout: NCCL_TIMEOUT=1800',
                'Check network bandwidth',
                'Verify all ranks call collectives',
            ],
        },
        {
            'error': 'NCCL invalid usage',
            'causes': [
                'Mismatched tensor sizes',
                'Different number of collectives',
                'Incorrect process group',
            ],
            'solutions': [
                'Ensure all ranks use same tensor size',
                'Check all ranks call same collectives',
                'Verify process group membership',
            ],
        },
    ]

    for issue in issues:
        print(f"\nError: {issue['error']}")
        print("Possible causes:")
        for cause in issue['causes']:
            print(f"  - {cause}")
        print("Solutions:")
        for solution in issue['solutions']:
            print(f"  - {solution}")
```

### Debugging Tools

```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Test NCCL connectivity
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Check GPU topology
nvidia-smi topo -m

# Test network bandwidth
iperf3 -c <remote_host> -P 4  # TCP test
ib_write_bw <remote_host>     # InfiniBand test
```

---

## Best Practices {#best-practices}

### 1. Choose Right Collective

```python
# Use the most efficient collective for your use case

# Instead of all-gather + sum:
dist.all_gather(tensors, local_tensor)
result = sum(tensors)

# Use all-reduce directly:
result = local_tensor.clone()
dist.all_reduce(result)  # More efficient
```

### 2. Batch Communications

```python
# Avoid:
for tensor in tensors:
    dist.all_reduce(tensor)  # Many small operations

# Prefer:
combined = torch.cat(tensors)
dist.all_reduce(combined)  # Single large operation
results = combined.split([t.numel() for t in tensors])
```

### 3. Use Async When Possible

```python
# Overlap communication and computation
work = dist.all_reduce(tensor, async_op=True)
computation_result = do_computation()
work.wait()
```

### 4. Monitor Performance

```python
# Regularly profile NCCL operations
# Aim for <20% of total time in communication
# If higher, consider:
#   - Larger batch sizes
#   - Better network
#   - Less parallelism
```

---

## Summary

NCCL is the foundation of efficient distributed inference in vLLM:

**Key Points:**
1. NCCL provides optimized collective operations for GPUs
2. Automatically adapts to hardware topology
3. Supports NVLink, InfiniBand, and Ethernet
4. Critical for tensor parallelism (all-reduce) and pipeline parallelism (send/recv)

**Performance Tips:**
- Configure environment variables appropriately
- Use asynchronous operations when possible
- Batch small messages
- Monitor and profile communication overhead

**Common Operations:**
- All-reduce: Tensor parallelism
- Send/Recv: Pipeline parallelism
- Broadcast: Parameter distribution
- All-gather: Gathering distributed results

Continue to the next module on communication-computation overlap to learn advanced optimization techniques.
