# Ray Distributed Framework

## Table of Contents
1. [Introduction to Ray](#introduction)
2. [Ray Core Concepts](#core-concepts)
3. [Ray in vLLM](#ray-vllm)
4. [Multi-Node Deployment](#multi-node)
5. [Resource Management](#resources)
6. [Actor-Based Parallelism](#actors)
7. [Performance Optimization](#optimization)
8. [Monitoring and Debugging](#monitoring)

---

## Introduction to Ray {#introduction}

Ray is a distributed computing framework that vLLM uses for multi-GPU and multi-node deployment. It provides simple abstractions for distributed execution without complex cluster management.

### Why Ray for vLLM?

```python
"""
Benefits of Ray:
✓ Easy multi-node deployment
✓ Automatic resource management
✓ Fault tolerance
✓ Dynamic scaling
✓ Simple API for distributed computing

vLLM uses Ray for:
- Distributing model across GPUs
- Managing distributed workers
- Coordinating inference requests
- Load balancing across nodes
"""
```

---

## Ray Core Concepts {#core-concepts}

### Ray Tasks and Actors

```python
import ray

# Initialize Ray
ray.init()

# === Tasks (Stateless) ===
@ray.remote
def remote_function(x):
    """Ray task: stateless remote function."""
    return x * 2

# Execute remotely
future = remote_function.remote(5)
result = ray.get(future)  # result = 10


# === Actors (Stateful) ===
@ray.remote
class Counter:
    """Ray actor: stateful remote object."""

    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

# Create actor
counter = Counter.remote()

# Call methods
future1 = counter.increment.remote()
future2 = counter.increment.remote()

print(ray.get([future1, future2]))  # [1, 2]


### GPU Resources

```python
# Specify GPU requirements
@ray.remote(num_gpus=1)
def gpu_task():
    import torch
    return torch.cuda.is_available()

# Actor with multiple GPUs
@ray.remote(num_gpus=4)
class MultiGPUWorker:
    def __init__(self):
        self.device_count = torch.cuda.device_count()

    def get_devices(self):
        return self.device_count

# Create worker with 4 GPUs
worker = MultiGPUWorker.remote()
result = ray.get(worker.get_devices.remote())  # 4
```

---

## Ray in vLLM {#ray-vllm}

### vLLM's Ray Architecture

```python
"""
vLLM Ray Architecture:

┌─────────────────────────────────────────┐
│           vLLM Server (Driver)           │
│  - Request scheduling                    │
│  - Load balancing                       │
│  - Response aggregation                 │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌────▼────┐
│ Ray    │          │  Ray    │
│ Worker │          │  Worker │
│ GPU 0-3│          │  GPU 4-7│
└────────┘          └─────────┘

Each Ray worker manages a subset of GPUs with tensor/pipeline parallelism.
"""


# Simplified vLLM Ray integration
class VLLMRayWorker:
    """Ray actor wrapping vLLM model."""

    def __init__(
        self,
        model_config,
        parallel_config,
        worker_id,
    ):
        self.worker_id = worker_id

        # Initialize model on designated GPUs
        self.model = self._initialize_model(
            model_config,
            parallel_config,
        )

    def execute_model(self, inputs):
        """Execute model inference."""
        return self.model(inputs)


def create_ray_workers(
    num_workers: int,
    gpus_per_worker: int,
):
    """
    Create Ray workers for distributed vLLM.
    """
    workers = []

    for i in range(num_workers):
        # Create Ray actor with GPU allocation
        worker = ray.remote(
            num_gpus=gpus_per_worker
        )(VLLMRayWorker).remote(
            model_config=model_config,
            parallel_config=parallel_config,
            worker_id=i,
        )

        workers.append(worker)

    return workers


# Usage in vLLM
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    # Ray automatically manages workers
)
```

### Distributed Inference with Ray

```python
class RayDistributedLLM:
    """
    Distributed LLM using Ray.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        num_nodes: int = 1,
    ):
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(address='auto')  # Connect to existing cluster

        self.num_workers = num_nodes
        self.tp_size = tensor_parallel_size

        # Create workers
        self.workers = self._create_workers()

    def _create_workers(self):
        """Create Ray workers on different nodes."""

        @ray.remote(num_gpus=self.tp_size)
        class ModelWorker:
            def __init__(self, model_name, tp_size):
                from vllm import LLM
                self.llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                )

            def generate(self, prompts):
                return self.llm.generate(prompts)

        # Create workers (Ray distributes across cluster)
        workers = []
        for i in range(self.num_workers):
            worker = ModelWorker.remote(
                model_name=self.model_name,
                tp_size=self.tp_size,
            )
            workers.append(worker)

        return workers

    def generate(self, prompts, sampling_params):
        """
        Generate with load balancing across workers.
        """

        # Split prompts across workers
        prompts_per_worker = len(prompts) // len(self.workers)

        futures = []
        for i, worker in enumerate(self.workers):
            start = i * prompts_per_worker
            end = start + prompts_per_worker if i < len(self.workers) - 1 else len(prompts)

            worker_prompts = prompts[start:end]

            future = worker.generate.remote(worker_prompts)
            futures.append(future)

        # Wait for all workers
        results = ray.get(futures)

        # Combine results
        all_outputs = []
        for result in results:
            all_outputs.extend(result)

        return all_outputs


# Example usage
llm = RayDistributedLLM(
    model_name="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    num_nodes=2,  # Use 2 nodes, 8 GPUs total
)

prompts = ["Hello"] * 100
outputs = llm.generate(prompts, sampling_params)
```

---

## Multi-Node Deployment {#multi-node}

### Setting Up Ray Cluster

```bash
# === Head Node ===
# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Output:
# Started Ray on this node at 192.168.1.100:6379


# === Worker Nodes ===
# On each worker node, connect to head
ray start --address='192.168.1.100:6379'

# Verify cluster status
ray status

# Output:
# ======== Cluster Resources ========
# CPU: 128.0
# GPU: 16.0
# Memory: 512.0 GB
# ...
```

### Ray Cluster Configuration

```yaml
# cluster.yaml - Ray cluster configuration

cluster_name: vllm-cluster

provider:
    type: aws
    region: us-west-2

auth:
    ssh_user: ubuntu

available_node_types:
    ray_head:
        node_config:
            InstanceType: p4d.24xlarge  # 8x A100
            ImageId: ami-xxx  # Deep Learning AMI
        resources: {"CPU": 96, "GPU": 8}
        min_workers: 0
        max_workers: 0

    ray_worker:
        node_config:
            InstanceType: p4d.24xlarge
            ImageId: ami-xxx
        resources: {"CPU": 96, "GPU": 8}
        min_workers: 1
        max_workers: 10

head_node_type: ray_head

setup_commands:
    - pip install vllm
    - pip install ray[default]

# Start cluster:
# ray up cluster.yaml
```

### Programmatic Cluster Management

```python
def deploy_vllm_on_ray():
    """
    Deploy vLLM on Ray cluster programmatically.
    """

    # Connect to Ray cluster
    ray.init(address='auto')  # Connect to existing cluster

    # Get cluster resources
    resources = ray.cluster_resources()
    num_gpus = resources.get('GPU', 0)
    num_nodes = resources.get('node:__internal_head__', 0) + resources.get('node', 0)

    print(f"Cluster resources:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Total GPUs: {num_gpus}")

    # Configure parallelism
    tensor_parallel_size = 8  # GPUs per model replica
    num_replicas = int(num_gpus // tensor_parallel_size)

    print(f"\nDeployment configuration:")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Number of replicas: {num_replicas}")

    # Deploy
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=tensor_parallel_size,
    )

    return llm
```

---

## Resource Management {#resources}

### Resource Allocation

```python
@ray.remote(num_gpus=1, num_cpus=4, memory=16*1024**3)
class ResourceAwareWorker:
    """Worker with specific resource requirements."""

    def __init__(self):
        pass

    def work(self):
        pass


# Custom resources
@ray.remote(resources={"special_hardware": 1})
def task_with_custom_resource():
    pass


# Dynamic resource allocation
class DynamicResourceManager:
    """Manage resources dynamically based on load."""

    def __init__(self):
        ray.init()
        self.workers = []

    def scale_up(self, num_workers, gpus_per_worker):
        """Add more workers."""

        for _ in range(num_workers):
            worker = ray.remote(
                num_gpus=gpus_per_worker
            )(ModelWorker).remote()

            self.workers.append(worker)

        print(f"Scaled up to {len(self.workers)} workers")

    def scale_down(self, num_workers):
        """Remove workers."""

        for _ in range(num_workers):
            if self.workers:
                worker = self.workers.pop()
                ray.kill(worker)

        print(f"Scaled down to {len(self.workers)} workers")


# Resource monitoring
def monitor_resources():
    """Monitor Ray cluster resources."""

    while True:
        # Get current resources
        available = ray.available_resources()
        total = ray.cluster_resources()

        # Calculate utilization
        gpu_used = total.get('GPU', 0) - available.get('GPU', 0)
        gpu_total = total.get('GPU', 0)
        gpu_util = gpu_used / gpu_total * 100 if gpu_total > 0 else 0

        print(f"GPU Utilization: {gpu_util:.1f}%")
        print(f"  Used: {gpu_used}/{gpu_total}")

        time.sleep(10)
```

---

## Actor-Based Parallelism {#actors}

### Worker Pool Pattern

```python
class WorkerPool:
    """Pool of Ray actors for parallel processing."""

    def __init__(self, num_workers, gpus_per_worker):
        @ray.remote(num_gpus=gpus_per_worker)
        class Worker:
            def __init__(self):
                self.model = load_model()

            def process(self, data):
                return self.model(data)

        # Create worker pool
        self.workers = [Worker.remote() for _ in range(num_workers)]
        self.next_worker = 0

    def submit(self, data):
        """Submit task to next available worker (round-robin)."""
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % len(self.workers)

        return worker.process.remote(data)

    def map(self, data_list):
        """Map data across all workers."""

        # Submit all tasks
        futures = []
        for i, data in enumerate(data_list):
            worker = self.workers[i % len(self.workers)]
            future = worker.process.remote(data)
            futures.append(future)

        # Wait for all results
        results = ray.get(futures)

        return results


# Usage
pool = WorkerPool(num_workers=4, gpus_per_worker=2)

# Process batch
batch = [data1, data2, data3, data4]
results = pool.map(batch)
```

---

## Performance Optimization {#optimization}

### Batch Processing

```python
class BatchedRayWorker:
    """Worker that processes batches for efficiency."""

    def __init__(self, batch_size=32, timeout=0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending = []

    def add_request(self, request):
        """Add request to batch."""
        self.pending.append(request)

        # Process if batch is full
        if len(self.pending) >= self.batch_size:
            return self._process_batch()

        return None

    def _process_batch(self):
        """Process accumulated batch."""
        if not self.pending:
            return []

        # Get batch
        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]

        # Process
        results = self.model(batch)

        return results


# Async batching
class AsyncBatcher:
    """Asynchronously batch requests."""

    def __init__(self, worker, batch_size, max_wait_ms):
        self.worker = worker
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self.pending = []
        self.pending_futures = []

        # Start background batching task
        self.batching_task = asyncio.create_task(self._batch_loop())

    async def submit(self, request):
        """Submit request and get future."""
        future = asyncio.Future()

        self.pending.append(request)
        self.pending_futures.append(future)

        return await future

    async def _batch_loop(self):
        """Background loop that batches requests."""
        while True:
            # Wait for batch to fill or timeout
            await asyncio.sleep(self.max_wait_ms / 1000)

            if len(self.pending) == 0:
                continue

            # Get batch
            batch = self.pending[:self.batch_size]
            futures = self.pending_futures[:self.batch_size]

            self.pending = self.pending[self.batch_size:]
            self.pending_futures = self.pending_futures[self.batch_size:]

            # Process batch
            results = await self.worker.process.remote(batch)

            # Resolve futures
            for future, result in zip(futures, results):
                future.set_result(result)
```

---

## Monitoring and Debugging {#monitoring}

### Ray Dashboard

```python
"""
Ray Dashboard: http://localhost:8265

Features:
- Cluster overview
- Resource utilization
- Task timeline
- Actor status
- Logs and metrics
"""


# Enable dashboard
ray.init(dashboard_host='0.0.0.0', dashboard_port=8265)
```

### Logging and Debugging

```python
@ray.remote
class LoggingWorker:
    """Worker with logging."""

    def __init__(self):
        # Ray redirects logs to driver by default
        import logging
        self.logger = logging.getLogger(__name__)

    def process(self, data):
        self.logger.info(f"Processing {data}")

        try:
            result = self.model(data)
            self.logger.info(f"Success: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise


# Debugging tips
def debug_ray_workers():
    """Debug Ray workers."""

    # 1. Check worker logs
    # Logs are in /tmp/ray/session_*/logs/

    # 2. Use ray.timeline()
    ray.timeline()  # Generate timeline trace

    # 3. Use ray.get() with timeout
    try:
        result = ray.get(future, timeout=10)
    except ray.exceptions.GetTimeoutError:
        print("Worker timed out")

    # 4. Check worker status
    workers = ray.actors()
    for worker in workers:
        print(f"Worker {worker}: {worker.state}")
```

---

## Summary

Ray provides powerful distributed computing capabilities for vLLM:

**Key Benefits:**
- Simple multi-node deployment
- Automatic resource management
- Fault tolerance and recovery
- Easy scaling
- Built-in monitoring

**vLLM Integration:**
- Ray actors for model workers
- Distributed inference across nodes
- Load balancing and request routing
- Efficient resource utilization

**Best Practices:**
- Use actors for stateful workers
- Batch requests for efficiency
- Monitor resource utilization
- Handle failures gracefully
- Use Ray dashboard for debugging

Continue to the next module on multi-node setup and configuration.
