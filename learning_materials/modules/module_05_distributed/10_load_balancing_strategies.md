# Load Balancing Strategies for Distributed Inference

## Table of Contents
1. [Introduction to Load Balancing](#introduction)
2. [Request Distribution](#distribution)
3. [Dynamic Load Balancing](#dynamic)
4. [Locality-Aware Scheduling](#locality)
5. [Adaptive Batching](#batching)
6. [Multi-Model Serving](#multi-model)
7. [Performance Metrics](#metrics)
8. [Best Practices](#best-practices)

---

## Introduction to Load Balancing {#introduction}

Load balancing distributes inference requests across multiple GPUs/nodes to maximize throughput and minimize latency while avoiding resource bottlenecks.

### Why Load Balancing Matters

```python
"""
Without Load Balancing:
GPU 0: ████████████████ (overloaded, 100% utilization)
GPU 1: ████ (idle, 25% utilization)
GPU 2: ████ (idle, 25% utilization)
GPU 3: ████ (idle, 25% utilization)
Result: High latency, poor resource utilization

With Load Balancing:
GPU 0: ████████ (75% utilization)
GPU 1: ████████ (75% utilization)
GPU 2: ████████ (75% utilization)
GPU 3: ████████ (75% utilization)
Result: Lower latency, optimal resource usage
"""

def calculate_load_imbalance(gpu_utilizations):
    """Calculate load imbalance metric."""
    import numpy as np

    mean_util = np.mean(gpu_utilizations)
    max_util = np.max(gpu_utilizations)
    min_util = np.min(gpu_utilizations)

    # Coefficient of variation
    std_util = np.std(gpu_utilizations)
    cv = std_util / mean_util if mean_util > 0 else 0

    # Imbalance ratio
    imbalance = (max_util - min_util) / max_util if max_util > 0 else 0

    print(f"Load Balance Metrics:")
    print(f"  Mean utilization: {mean_util:.1f}%")
    print(f"  Std deviation: {std_util:.1f}%")
    print(f"  Coefficient of variation: {cv:.2f}")
    print(f"  Imbalance ratio: {imbalance:.2f}")

    return {'mean': mean_util, 'cv': cv, 'imbalance': imbalance}

# Example
gpu_utils = [95, 30, 25, 20]  # Imbalanced
calculate_load_imbalance(gpu_utils)
```

---

## Request Distribution {#distribution}

### Round-Robin Distribution

```python
class RoundRobinLoadBalancer:
    """
    Simple round-robin request distribution.

    Pros: Simple, fair distribution
    Cons: Ignores actual load, sequence length
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.next_worker = 0

    def assign_request(self, request):
        """Assign request to next worker in rotation."""
        worker_id = self.next_worker
        self.next_worker = (self.next_worker + 1) % self.num_workers
        return worker_id

    def get_worker(self, request):
        """Get worker for request (stateless)."""
        # Based on request ID for consistency
        return hash(request.id) % self.num_workers


### Weighted Round-Robin

```python
class WeightedRoundRobinBalancer:
    """
    Weighted round-robin based on worker capacity.

    Workers with more GPUs get proportionally more requests.
    """

    def __init__(self, worker_capacities):
        """
        Args:
            worker_capacities: {worker_id: num_gpus}
        """
        self.worker_capacities = worker_capacities
        self.total_capacity = sum(worker_capacities.values())

        # Build weighted list
        self.weighted_workers = []
        for worker_id, capacity in worker_capacities.items():
            weight = capacity / self.total_capacity
            count = int(weight * 100)  # Scale for granularity
            self.weighted_workers.extend([worker_id] * count)

        self.next_idx = 0

    def assign_request(self, request):
        """Assign based on weighted distribution."""
        worker_id = self.weighted_workers[self.next_idx]
        self.next_idx = (self.next_idx + 1) % len(self.weighted_workers)
        return worker_id


# Example
balancer = WeightedRoundRobinBalancer({
    0: 8,  # Worker 0 has 8 GPUs
    1: 4,  # Worker 1 has 4 GPUs
    2: 4,  # Worker 2 has 4 GPUs
})

# Worker 0 gets 50% of requests, workers 1&2 get 25% each
```

### Least-Loaded Distribution

```python
class LeastLoadedBalancer:
    """
    Assign requests to least-loaded worker.

    Tracks current load (active requests) per worker.
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.active_requests = {i: 0 for i in range(num_workers)}

    def assign_request(self, request):
        """Assign to worker with fewest active requests."""

        # Find least loaded worker
        worker_id = min(
            self.active_requests.keys(),
            key=lambda w: self.active_requests[w]
        )

        # Track assignment
        self.active_requests[worker_id] += 1

        return worker_id

    def complete_request(self, worker_id):
        """Mark request as complete on worker."""
        self.active_requests[worker_id] -= 1


### Token-Aware Balancing

```python
class TokenAwareBalancer:
    """
    Balance based on total tokens (prompt + generation).

    More accurate than request count.
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.worker_tokens = {i: 0 for i in range(num_workers)}

    def assign_request(self, request):
        """Assign based on estimated token load."""

        # Estimate total tokens
        prompt_tokens = len(request.prompt_tokens)
        max_gen_tokens = request.max_tokens
        estimated_total = prompt_tokens + max_gen_tokens

        # Find worker with fewest tokens
        worker_id = min(
            self.worker_tokens.keys(),
            key=lambda w: self.worker_tokens[w]
        )

        # Track tokens
        self.worker_tokens[worker_id] += estimated_total

        return worker_id, estimated_total

    def complete_request(self, worker_id, actual_tokens):
        """Update with actual tokens used."""
        # Remove estimated, add actual
        self.worker_tokens[worker_id] += (actual_tokens - estimated_total)


# Example
balancer = TokenAwareBalancer(num_workers=4)

# Short prompt, long generation
request1 = Request(prompt_tokens=[1,2,3], max_tokens=500)
worker1, est1 = balancer.assign_request(request1)

# Long prompt, short generation
request2 = Request(prompt_tokens=list(range(1000)), max_tokens=50)
worker2, est2 = balancer.assign_request(request2)

print(f"Request 1 -> Worker {worker1} ({est1} tokens)")
print(f"Request 2 -> Worker {worker2} ({est2} tokens)")
```

---

## Dynamic Load Balancing {#dynamic}

### Real-Time Load Monitoring

```python
class DynamicLoadBalancer:
    """
    Dynamically balance based on real-time metrics.

    Monitors GPU utilization, memory, queue length.
    """

    def __init__(self, workers):
        self.workers = workers
        self.metrics_history = {w: [] for w in workers}

    def get_worker_metrics(self, worker_id):
        """Get current metrics for worker."""

        metrics = {
            'gpu_util': get_gpu_utilization(worker_id),
            'gpu_memory': get_gpu_memory_used(worker_id),
            'queue_length': get_queue_length(worker_id),
            'avg_latency': get_avg_latency(worker_id),
        }

        return metrics

    def calculate_load_score(self, worker_id):
        """
        Calculate composite load score (lower is better).
        """

        metrics = self.get_worker_metrics(worker_id)

        # Weighted score
        score = (
            metrics['gpu_util'] * 0.3 +
            metrics['gpu_memory'] * 0.3 +
            metrics['queue_length'] * 10 * 0.2 +
            metrics['avg_latency'] * 0.2
        )

        return score

    def assign_request(self, request):
        """Assign to worker with lowest load score."""

        # Get load scores for all workers
        scores = {
            w: self.calculate_load_score(w)
            for w in self.workers
        }

        # Choose worker with lowest score
        worker_id = min(scores.keys(), key=lambda w: scores[w])

        return worker_id


### Predictive Load Balancing

```python
class PredictiveLoadBalancer:
    """
    Predict future load and balance proactively.

    Uses historical data to forecast resource needs.
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.request_history = []
        self.completion_times = {}

    def predict_completion_time(self, request, worker_id):
        """
        Predict when request will complete on worker.

        Based on:
        - Current queue on worker
        - Historical completion times for similar requests
        """

        # Get similar past requests
        similar_requests = self._find_similar_requests(request)

        if similar_requests:
            # Average completion time
            avg_time = np.mean([
                self.completion_times[r]
                for r in similar_requests
                if r in self.completion_times
            ])
        else:
            # Estimate from tokens
            avg_time = self._estimate_from_tokens(request)

        # Add queue wait time
        queue_wait = self._estimate_queue_wait(worker_id)

        return avg_time + queue_wait

    def assign_request(self, request):
        """Assign to worker with earliest predicted completion."""

        predicted_times = {
            w: self.predict_completion_time(request, w)
            for w in range(self.num_workers)
        }

        # Choose worker with earliest completion
        worker_id = min(
            predicted_times.keys(),
            key=lambda w: predicted_times[w]
        )

        return worker_id

    def record_completion(self, request, actual_time):
        """Record actual completion time for learning."""
        self.completion_times[request.id] = actual_time
        self.request_history.append(request)

        # Keep history bounded
        if len(self.request_history) > 10000:
            self.request_history = self.request_history[-5000:]
```

---

## Locality-Aware Scheduling {#locality}

### KV Cache Locality

```python
class CacheLocalityBalancer:
    """
    Balance requests considering KV cache locality.

    Send requests with shared prefixes to same worker
    to maximize cache hits.
    """

    def __init__(self, num_workers):
        self.num_workers = num_workers

        # Track which prefixes are cached on which workers
        self.prefix_cache = {}  # prefix_hash -> worker_id

    def compute_prefix_hash(self, tokens, prefix_length=128):
        """Hash first N tokens as prefix."""
        prefix = tuple(tokens[:prefix_length])
        return hash(prefix)

    def assign_request(self, request):
        """Assign based on cache locality."""

        prefix_hash = self.compute_prefix_hash(request.prompt_tokens)

        if prefix_hash in self.prefix_cache:
            # Send to worker that has this prefix cached
            worker_id = self.prefix_cache[prefix_hash]

            # Verify worker not overloaded
            if self._worker_available(worker_id):
                return worker_id

        # No cache hit or worker busy, use load balancing
        worker_id = self._least_loaded_worker()

        # Update prefix cache
        self.prefix_cache[prefix_hash] = worker_id

        return worker_id

    def _worker_available(self, worker_id):
        """Check if worker can accept more requests."""
        current_load = self.get_worker_load(worker_id)
        return current_load < MAX_LOAD_THRESHOLD


### Data Locality (Multi-Node)

```python
class DataLocalityBalancer:
    """
    Balance considering data locality in multi-node setup.

    Keep requests from same user/session on same node
    to leverage local cache and state.
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.user_node_mapping = {}

    def assign_request(self, request):
        """Assign based on user/session locality."""

        user_id = request.user_id

        if user_id in self.user_node_mapping:
            # Send to user's assigned node
            node_id = self.user_node_mapping[user_id]

            # Check capacity
            if self._node_available(node_id):
                return node_id
            else:
                # Node full, migrate user
                pass

        # Assign to least loaded node
        node_id = self._least_loaded_node()

        # Track assignment
        self.user_node_mapping[user_id] = node_id

        return node_id

    def _least_loaded_node(self):
        """Find node with most available capacity."""
        return min(
            self.nodes,
            key=lambda n: self.get_node_load(n)
        )
```

---

## Adaptive Batching {#batching}

### Dynamic Batch Formation

```python
class AdaptiveBatcher:
    """
    Dynamically form batches based on load.

    - High load: larger batches (better throughput)
    - Low load: smaller batches (better latency)
    """

    def __init__(
        self,
        min_batch_size=1,
        max_batch_size=128,
        target_latency_ms=100,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms

        self.pending_requests = []
        self.current_batch_size = min_batch_size

    def add_request(self, request):
        """Add request to pending queue."""
        self.pending_requests.append(request)

    def should_process_batch(self):
        """Decide whether to process current batch."""

        if len(self.pending_requests) == 0:
            return False

        # Force processing if batch is full
        if len(self.pending_requests) >= self.max_batch_size:
            return True

        # Or if oldest request has waited too long
        oldest_wait = time.time() - self.pending_requests[0].arrival_time
        if oldest_wait > self.target_latency_ms / 1000:
            return True

        # Or if we have enough for current target batch size
        if len(self.pending_requests) >= self.current_batch_size:
            return True

        return False

    def get_next_batch(self):
        """Get next batch to process."""

        batch_size = min(
            len(self.pending_requests),
            self.current_batch_size
        )

        batch = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]

        return batch

    def update_batch_size(self, measured_latency_ms, queue_length):
        """
        Adjust batch size based on latency and queue.
        """

        if measured_latency_ms > self.target_latency_ms:
            # Too slow, reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size - 8
            )

        elif queue_length > 20:
            # Queue building up, increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size + 8
            )

        return self.current_batch_size
```

---

## Multi-Model Serving {#multi-model}

### Model-Specific Load Balancing

```python
class MultiModelBalancer:
    """
    Load balance across multiple models/replicas.

    Each model may have different resource requirements.
    """

    def __init__(self):
        # Model instances: {model_name: [worker_ids]}
        self.model_instances = {}

        # Model resource profiles
        self.model_profiles = {}

    def register_model(self, model_name, worker_ids, profile):
        """Register model deployment."""
        self.model_instances[model_name] = worker_ids
        self.model_profiles[model_name] = profile

    def assign_request(self, request):
        """Assign request to appropriate model instance."""

        model_name = request.model_name

        if model_name not in self.model_instances:
            raise ValueError(f"Model {model_name} not found")

        # Get workers serving this model
        workers = self.model_instances[model_name]

        # Balance among these workers
        worker_id = self._balance_among_workers(workers, request)

        return worker_id

    def _balance_among_workers(self, workers, request):
        """Balance request among workers serving same model."""

        # Consider model-specific metrics
        scores = {}

        for worker in workers:
            # Account for model size and current load
            profile = self.model_profiles[request.model_name]

            load = self.get_worker_load(worker)
            capacity = self.get_worker_capacity(worker)

            # Score: remaining capacity
            score = capacity - load

            scores[worker] = score

        # Choose worker with most remaining capacity
        return max(scores.keys(), key=lambda w: scores[w])
```

---

## Performance Metrics {#metrics}

### Monitoring Load Balance Quality

```python
class LoadBalanceMonitor:
    """Monitor and report load balance metrics."""

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.metrics_history = []

    def collect_metrics(self):
        """Collect current metrics from all workers."""

        metrics = {}

        for worker_id in range(self.num_workers):
            metrics[worker_id] = {
                'requests_processed': get_request_count(worker_id),
                'tokens_processed': get_token_count(worker_id),
                'gpu_utilization': get_gpu_util(worker_id),
                'avg_latency': get_avg_latency(worker_id),
                'p99_latency': get_p99_latency(worker_id),
            }

        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics,
        })

        return metrics

    def analyze_balance_quality(self):
        """Analyze how well load is balanced."""

        if not self.metrics_history:
            return

        latest = self.metrics_history[-1]['metrics']

        # Extract key metrics
        gpu_utils = [m['gpu_utilization'] for m in latest.values()]
        latencies = [m['avg_latency'] for m in latest.values()]

        # Calculate statistics
        util_mean = np.mean(gpu_utils)
        util_std = np.std(gpu_utils)
        util_cv = util_std / util_mean if util_mean > 0 else 0

        latency_mean = np.mean(latencies)
        latency_std = np.std(latencies)

        print("\nLoad Balance Quality:")
        print(f"  GPU Utilization:")
        print(f"    Mean: {util_mean:.1f}%")
        print(f"    Std Dev: {util_std:.1f}%")
        print(f"    CV: {util_cv:.2f}")
        print(f"  Latency:")
        print(f"    Mean: {latency_mean:.1f} ms")
        print(f"    Std Dev: {latency_std:.1f} ms")

        # Quality score (lower CV is better)
        if util_cv < 0.1:
            quality = "Excellent"
        elif util_cv < 0.2:
            quality = "Good"
        elif util_cv < 0.3:
            quality = "Fair"
        else:
            quality = "Poor"

        print(f"  Overall Quality: {quality}")

        return {
            'util_cv': util_cv,
            'quality': quality,
        }
```

---

## Best Practices {#best-practices}

### 1. Choose Right Strategy

```python
"""
Strategy Selection Guide:

Simple Deployment (1-4 GPUs):
→ Round-robin or least-loaded

Medium Scale (8-32 GPUs):
→ Token-aware with dynamic adjustment

Large Scale (32+ GPUs):
→ Predictive with locality awareness

Multi-Model Serving:
→ Model-specific balancing

Long Sequences (>4K tokens):
→ KV cache locality balancing
"""
```

### 2. Monitor and Adapt

```python
def adaptive_load_balancing():
    """Continuously monitor and adapt strategy."""

    balancer = DynamicLoadBalancer(num_workers=8)

    while True:
        # Collect metrics
        metrics = balancer.collect_metrics()

        # Analyze balance quality
        quality = balancer.analyze_quality(metrics)

        # Adapt if needed
        if quality['util_cv'] > 0.3:
            # Rebalance: migrate some requests
            balancer.rebalance()

        time.sleep(10)  # Check every 10 seconds
```

### 3. Handle Stragglers

```python
def handle_stragglers():
    """Deal with slow requests (stragglers)."""

    MAX_PROCESSING_TIME = 30  # seconds

    for request in active_requests:
        processing_time = time.time() - request.start_time

        if processing_time > MAX_PROCESSING_TIME:
            # Straggler detected

            # Option 1: Duplicate to another worker
            backup_worker = assign_to_backup(request)
            # Use whichever finishes first

            # Option 2: Cancel and retry
            cancel_request(request)
            retry_on_different_worker(request)
```

---

## Summary

Effective load balancing is critical for distributed inference:

**Key Strategies:**
- Round-robin: Simple, fair
- Least-loaded: Reactive, adaptive
- Token-aware: More accurate than request count
- Predictive: Proactive, uses history
- Locality-aware: Maximizes cache hits

**Metrics to Monitor:**
- GPU utilization variance
- Request latency distribution
- Queue lengths
- Token throughput

**Best Practices:**
- Start simple, add complexity as needed
- Monitor continuously
- Adapt dynamically to load changes
- Consider cache locality for efficiency
- Handle stragglers and failures

Continue to the next module on fault tolerance and recovery.
