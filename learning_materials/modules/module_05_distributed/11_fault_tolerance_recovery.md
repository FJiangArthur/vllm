# Fault Tolerance and Recovery in Distributed Inference

## Table of Contents
1. [Introduction to Fault Tolerance](#introduction)
2. [Failure Modes](#failure-modes)
3. [Detection Mechanisms](#detection)
4. [Recovery Strategies](#recovery)
5. [Checkpointing](#checkpointing)
6. [Request Retry Logic](#retry)
7. [High Availability](#high-availability)
8. [Best Practices](#best-practices)

---

## Introduction to Fault Tolerance {#introduction}

In distributed systems, failures are inevitable. Fault tolerance ensures the system continues operating despite failures, with minimal impact on users.

### Types of Failures

```python
"""
Common Failure Types in Distributed Inference:

1. GPU Failures:
   - Out of memory (OOM)
   - CUDA errors
   - Hardware faults

2. Network Failures:
   - NCCL timeouts
   - Connection drops
   - Network partitions

3. Process Failures:
   - Worker crashes
   - Deadlocks
   - Out of memory (system RAM)

4. Node Failures:
   - Power outages
   - Kernel panics
   - Hardware failures

Impact:
- Single GPU: Partial degradation
- Single Worker: Loss of requests on that worker
- Single Node: Loss of multiple GPUs
- Network: Communication failures, hangs
"""


def categorize_failure(error):
    """Categorize failure type."""

    error_msg = str(error).lower()

    if 'cuda' in error_msg or 'out of memory' in error_msg:
        return 'GPU_FAILURE'
    elif 'nccl' in error_msg or 'timeout' in error_msg:
        return 'NETWORK_FAILURE'
    elif 'connection' in error_msg or 'refused' in error_msg:
        return 'PROCESS_FAILURE'
    else:
        return 'UNKNOWN_FAILURE'
```

---

## Failure Modes {#failure-modes}

### GPU Failures

```python
import torch

def detect_gpu_failure():
    """Detect GPU failures."""

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            return 'CUDA_UNAVAILABLE'

        # Try simple operation
        device = torch.cuda.current_device()
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.matmul(test_tensor, test_tensor)

        # Check for errors
        torch.cuda.synchronize()

        return None  # No failure

    except RuntimeError as e:
        if 'out of memory' in str(e):
            return 'OOM'
        elif 'CUDA' in str(e):
            return 'CUDA_ERROR'
        else:
            return 'UNKNOWN_GPU_ERROR'


def handle_gpu_oom():
    """Handle GPU OOM."""

    # Clear cache
    torch.cuda.empty_cache()

    # Reduce batch size
    global current_batch_size
    current_batch_size = max(1, current_batch_size // 2)

    print(f"OOM detected, reduced batch size to {current_batch_size}")

    # Restart worker if persistent
    if oom_count > 3:
        restart_worker()
```

### Network Failures

```python
import torch.distributed as dist

def detect_network_failure(timeout_seconds=30):
    """Detect network communication failures."""

    try:
        # Test communication
        tensor = torch.randn(1, device='cuda')

        # Set timeout
        work = dist.all_reduce(tensor, async_op=True)

        # Wait with timeout
        import time
        start = time.time()

        while not work.is_completed():
            if time.time() - start > timeout_seconds:
                return 'TIMEOUT'
            time.sleep(0.1)

        return None  # No failure

    except Exception as e:
        if 'NCCL' in str(e):
            return 'NCCL_ERROR'
        else:
            return 'COMMUNICATION_ERROR'


def handle_nccl_timeout():
    """Handle NCCL timeout."""

    # Attempt recovery
    try:
        # Destroy current process group
        if dist.is_initialized():
            dist.destroy_process_group()

        # Reinitialize
        time.sleep(5)  # Wait before retry

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(seconds=1800),
        )

        print("Process group reinitialized")
        return True

    except Exception as e:
        print(f"Recovery failed: {e}")
        return False
```

### Worker Failures

```python
class WorkerHealthMonitor:
    """Monitor worker health with heartbeats."""

    def __init__(self, workers, heartbeat_interval=10):
        self.workers = workers
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat = {w: time.time() for w in workers}

    def start_monitoring(self):
        """Start background monitoring thread."""

        def monitor_loop():
            while True:
                self.check_workers()
                time.sleep(self.heartbeat_interval)

        import threading
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    def check_workers(self):
        """Check all workers."""

        current_time = time.time()

        for worker_id in self.workers:
            # Check heartbeat
            last_seen = self.last_heartbeat[worker_id]
            time_since_heartbeat = current_time - last_seen

            if time_since_heartbeat > self.heartbeat_interval * 3:
                # Worker appears dead
                print(f"Worker {worker_id} failed (no heartbeat)")
                self.handle_worker_failure(worker_id)

    def record_heartbeat(self, worker_id):
        """Record heartbeat from worker."""
        self.last_heartbeat[worker_id] = time.time()

    def handle_worker_failure(self, worker_id):
        """Handle worker failure."""

        # Mark worker as failed
        self.failed_workers.add(worker_id)

        # Redistribute its requests
        self.redistribute_requests(worker_id)

        # Attempt restart
        self.restart_worker(worker_id)
```

---

## Detection Mechanisms {#detection}

### Health Checks

```python
class HealthChecker:
    """Comprehensive health checking."""

    def __init__(self, worker_id):
        self.worker_id = worker_id

    def check_health(self):
        """Run all health checks."""

        checks = {
            'gpu': self.check_gpu(),
            'memory': self.check_memory(),
            'network': self.check_network(),
            'disk': self.check_disk(),
        }

        # All checks must pass
        healthy = all(checks.values())

        return {
            'healthy': healthy,
            'checks': checks,
            'timestamp': time.time(),
        }

    def check_gpu(self):
        """Check GPU health."""
        try:
            # Test GPU operation
            test = torch.randn(100, 100, device='cuda')
            result = test @ test
            torch.cuda.synchronize()
            return True
        except:
            return False

    def check_memory(self):
        """Check memory usage."""
        import psutil

        # System memory
        mem = psutil.virtual_memory()
        if mem.percent > 95:
            return False

        # GPU memory
        gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        if gpu_mem > 0.95:
            return False

        return True

    def check_network(self):
        """Check network connectivity."""
        if not dist.is_initialized():
            return False

        try:
            # Test communication
            tensor = torch.tensor([1.0], device='cuda')
            dist.all_reduce(tensor)
            return True
        except:
            return False

    def check_disk(self):
        """Check disk space."""
        import shutil

        usage = shutil.disk_usage('/')
        percent_used = usage.used / usage.total

        return percent_used < 0.95
```

### Watchdog Timers

```python
class WatchdogTimer:
    """Watchdog timer to detect hangs."""

    def __init__(self, timeout_seconds=60):
        self.timeout_seconds = timeout_seconds
        self.last_reset = time.time()
        self.active = False

    def start(self):
        """Start watchdog."""
        self.active = True
        self.last_reset = time.time()

        def watchdog_loop():
            while self.active:
                time_since_reset = time.time() - self.last_reset

                if time_since_reset > self.timeout_seconds:
                    # Timeout! Process may be hung
                    print("WATCHDOG TIMEOUT - Process may be hung")
                    self.on_timeout()

                time.sleep(1)

        import threading
        thread = threading.Thread(target=watchdog_loop, daemon=True)
        thread.start()

    def reset(self):
        """Reset watchdog timer."""
        self.last_reset = time.time()

    def stop(self):
        """Stop watchdog."""
        self.active = False

    def on_timeout(self):
        """Handle timeout."""
        # Log stack traces of all threads
        import sys
        import traceback

        for thread_id, frame in sys._current_frames().items():
            print(f"\nThread {thread_id}:")
            traceback.print_stack(frame)

        # Could also trigger process restart
```

---

## Recovery Strategies {#recovery}

### Graceful Degradation

```python
class GracefulDegradation:
    """Degrade service gracefully when failures occur."""

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.failed_workers = set()
        self.degraded_mode = False

    def handle_worker_failure(self, worker_id):
        """Handle worker failure with degradation."""

        self.failed_workers.add(worker_id)

        # Calculate remaining capacity
        remaining_capacity = (
            (self.num_workers - len(self.failed_workers)) /
            self.num_workers
        )

        print(f"Worker {worker_id} failed")
        print(f"Remaining capacity: {remaining_capacity * 100:.1f}%")

        if remaining_capacity < 0.5:
            # Less than 50% capacity, enter degraded mode
            self.enter_degraded_mode()

    def enter_degraded_mode(self):
        """Enter degraded service mode."""

        self.degraded_mode = True

        # Reduce quality of service
        self.reduce_batch_size()
        self.increase_response_timeout()
        self.reject_low_priority_requests()

        print("Entered degraded mode")

    def reduce_batch_size(self):
        """Reduce batch size to lower memory pressure."""
        global max_batch_size
        max_batch_size = max_batch_size // 2

    def reject_low_priority_requests(self):
        """Reject low-priority requests."""
        global accept_low_priority
        accept_low_priority = False
```

### Request Redistribution

```python
class RequestRedistributor:
    """Redistribute requests when workers fail."""

    def __init__(self, workers):
        self.workers = workers
        self.request_assignments = {}  # request_id -> worker_id

    def redistribute_on_failure(self, failed_worker_id):
        """Redistribute requests from failed worker."""

        # Find requests assigned to failed worker
        failed_requests = [
            req_id for req_id, worker_id in self.request_assignments.items()
            if worker_id == failed_worker_id
        ]

        print(f"Redistributing {len(failed_requests)} requests")

        # Reassign to healthy workers
        healthy_workers = [
            w for w in self.workers
            if w != failed_worker_id
        ]

        for req_id in failed_requests:
            # Round-robin assignment
            new_worker = healthy_workers[
                hash(req_id) % len(healthy_workers)
            ]

            self.request_assignments[req_id] = new_worker

            # Restart request on new worker
            self.restart_request(req_id, new_worker)

    def restart_request(self, req_id, worker_id):
        """Restart request on new worker."""

        # Get request details
        request = self.get_request(req_id)

        # Resubmit to new worker
        self.submit_to_worker(request, worker_id)
```

### Automatic Restart

```python
class AutoRestartManager:
    """Automatically restart failed workers."""

    def __init__(self, max_restarts=3):
        self.max_restarts = max_restarts
        self.restart_counts = {}

    def attempt_restart(self, worker_id):
        """Attempt to restart failed worker."""

        # Check restart limit
        restart_count = self.restart_counts.get(worker_id, 0)

        if restart_count >= self.max_restarts:
            print(f"Worker {worker_id} exceeded restart limit")
            return False

        print(f"Restarting worker {worker_id} (attempt {restart_count + 1})")

        # Attempt restart
        success = self.restart_worker(worker_id)

        if success:
            self.restart_counts[worker_id] = restart_count + 1
            return True
        else:
            return False

    def restart_worker(self, worker_id):
        """Restart worker process."""

        try:
            # Kill existing process
            if worker_id in self.worker_processes:
                self.worker_processes[worker_id].terminate()
                self.worker_processes[worker_id].wait(timeout=10)

            # Start new process
            import subprocess

            process = subprocess.Popen([
                'python', 'worker.py',
                '--worker-id', str(worker_id),
                '--master-addr', self.master_addr,
            ])

            self.worker_processes[worker_id] = process

            # Wait for worker to be ready
            if self.wait_for_worker_ready(worker_id, timeout=60):
                print(f"Worker {worker_id} restarted successfully")
                return True
            else:
                print(f"Worker {worker_id} failed to start")
                return False

        except Exception as e:
            print(f"Failed to restart worker {worker_id}: {e}")
            return False
```

---

## Checkpointing {#checkpointing}

### State Checkpointing

```python
class CheckpointManager:
    """Manage checkpoints for recovery."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state, checkpoint_id):
        """Save system state."""

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pt"

        # Save state
        torch.save(state, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        # Keep only recent checkpoints
        self.cleanup_old_checkpoints(keep_last=5)

    def load_checkpoint(self, checkpoint_id=None):
        """Load checkpoint."""

        if checkpoint_id is None:
            # Load latest
            checkpoint_id = self.get_latest_checkpoint_id()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pt"

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return None

        # Load state
        state = torch.load(checkpoint_path)

        print(f"Loaded checkpoint: {checkpoint_path}")

        return state

    def cleanup_old_checkpoints(self, keep_last=5):
        """Remove old checkpoints."""

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old checkpoints
        for checkpoint in checkpoints[keep_last:]:
            checkpoint.unlink()


### Request-Level Checkpointing

```python
class RequestCheckpointer:
    """Checkpoint individual requests for recovery."""

    def __init__(self):
        self.request_states = {}

    def checkpoint_request(self, request_id, state):
        """Save request state."""

        self.request_states[request_id] = {
            'state': state,
            'timestamp': time.time(),
        }

    def recover_request(self, request_id):
        """Recover request from checkpoint."""

        if request_id not in self.request_states:
            return None

        checkpoint = self.request_states[request_id]

        # Check if checkpoint is recent
        age = time.time() - checkpoint['timestamp']
        if age > 3600:  # 1 hour
            # Too old, discard
            del self.request_states[request_id]
            return None

        return checkpoint['state']

    def cleanup_completed(self, request_id):
        """Remove checkpoint after completion."""

        if request_id in self.request_states:
            del self.request_states[request_id]
```

---

## Request Retry Logic {#retry}

### Retry with Exponential Backoff

```python
class RetryManager:
    """Manage request retries with exponential backoff."""

    def __init__(
        self,
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self.retry_counts = {}

    def should_retry(self, request_id, error):
        """Determine if request should be retried."""

        retry_count = self.retry_counts.get(request_id, 0)

        if retry_count >= self.max_retries:
            return False

        # Check if error is retryable
        if self.is_retryable_error(error):
            return True

        return False

    def is_retryable_error(self, error):
        """Check if error is retryable."""

        error_msg = str(error).lower()

        # Retryable errors
        if 'timeout' in error_msg:
            return True
        if 'connection' in error_msg:
            return True
        if 'temporary' in error_msg:
            return True

        # Non-retryable errors
        if 'invalid' in error_msg:
            return False
        if 'unauthorized' in error_msg:
            return False

        # Default: retry
        return True

    def get_retry_delay(self, request_id):
        """Calculate retry delay with exponential backoff."""

        retry_count = self.retry_counts.get(request_id, 0)

        # Exponential backoff: base * 2^retry_count
        delay = self.base_delay * (2 ** retry_count)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter
        import random
        jitter = random.uniform(0, delay * 0.1)
        delay += jitter

        return delay

    def retry_request(self, request, error):
        """Retry request."""

        request_id = request.id

        if not self.should_retry(request_id, error):
            print(f"Request {request_id} failed permanently")
            return False

        # Increment retry count
        retry_count = self.retry_counts.get(request_id, 0)
        self.retry_counts[request_id] = retry_count + 1

        # Calculate delay
        delay = self.get_retry_delay(request_id)

        print(f"Retrying request {request_id} in {delay:.1f}s (attempt {retry_count + 1})")

        # Schedule retry
        time.sleep(delay)

        # Resubmit request
        self.submit_request(request)

        return True
```

---

## High Availability {#high-availability}

### Active-Active Configuration

```python
class HighAvailabilityCluster:
    """
    Active-active HA configuration.

    Multiple model replicas serving requests.
    """

    def __init__(self, num_replicas=2):
        self.num_replicas = num_replicas
        self.replicas = []

        # Initialize replicas
        for i in range(num_replicas):
            replica = self.create_replica(i)
            self.replicas.append(replica)

    def submit_request(self, request):
        """Submit request with HA."""

        # Try primary replica
        try:
            result = self.replicas[0].process(request)
            return result

        except Exception as e:
            print(f"Primary replica failed: {e}")

            # Failover to secondary
            for i in range(1, self.num_replicas):
                try:
                    result = self.replicas[i].process(request)
                    print(f"Succeeded on replica {i}")
                    return result

                except Exception as e:
                    print(f"Replica {i} failed: {e}")
                    continue

            # All replicas failed
            raise RuntimeError("All replicas failed")
```

---

## Best Practices {#best-practices}

### 1. Defense in Depth

```python
"""
Multiple layers of fault tolerance:

Layer 1: Error handling in application code
Layer 2: Worker-level recovery (restart)
Layer 3: Cluster-level redundancy (multiple workers)
Layer 4: Request-level retries
Layer 5: Client-side retries
"""
```

### 2. Fail Fast, Recover Fast

```python
"""
Don't wait for timeout - detect failures quickly:

✓ Use short heartbeat intervals (5-10s)
✓ Set reasonable timeouts (30-60s)
✓ Monitor health continuously
✓ Automated recovery (no manual intervention)
"""
```

### 3. Graceful Degradation

```python
"""
Maintain service under partial failures:

✓ Reduce batch size
✓ Lower quality settings
✓ Reject low-priority requests
✓ Inform users of degraded service
"""
```

---

## Summary

Fault tolerance is essential for production distributed inference:

**Key Strategies:**
- Health monitoring and detection
- Automatic worker restart
- Request redistribution
- Retry with backoff
- Graceful degradation

**Best Practices:**
- Multiple layers of defense
- Fast failure detection
- Automated recovery
- Maintain service under degradation
- Test failure scenarios regularly

Continue to the final module on distributed profiling and debugging.
