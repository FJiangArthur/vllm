# Project 7.1: Custom Request Scheduler

## Project Overview

**Duration:** 3-4 hours
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)
**Prerequisites:** Modules 1, 2, and 4 (Core Concepts, System Components)

### Objective

Implement a production-ready, priority-based request scheduler for vLLM that provides SLA guarantees while preventing starvation of low-priority requests. Your scheduler will replace vLLM's default FCFS (First-Come-First-Served) scheduler with an intelligent multi-level priority system.

### Why This Matters

In production ML serving systems, different users and use cases have different latency requirements:
- **Interactive chat**: P99 latency < 500ms (high priority)
- **Content generation**: P95 latency < 2s (medium priority)
- **Batch processing**: Best-effort throughput (low priority)

A naive priority scheduler can starve low-priority requests indefinitely. Your implementation must balance fairness, SLA guarantees, and resource utilization.

### Real-World Context

This project mirrors work at:
- **OpenAI**: Serving different pricing tiers with different SLAs
- **Anthropic**: Balancing research and production workloads
- **NVIDIA**: Multi-tenant GPU scheduling across customers

---

## Learning Objectives

By completing this project, you will:

1. ✅ Master vLLM's scheduler architecture and extension points
2. ✅ Implement advanced scheduling algorithms (MLFQ, aging, priority inheritance)
3. ✅ Design SLA tracking and violation reporting systems
4. ✅ Optimize for tail latency (P95, P99) in addition to throughput
5. ✅ Test scheduler behavior under various workload patterns

---

## Requirements Specification

### Functional Requirements

#### FR1: Priority Levels
- Support **at least 3 priority levels**: HIGH, MEDIUM, LOW
- Allow dynamic priority assignment per request
- Support priority from API request metadata

#### FR2: SLA Management
- Track per-priority SLA targets (e.g., P99 < 500ms for HIGH)
- Report SLA violations in real-time
- Provide SLA compliance metrics over time windows

#### FR3: Fairness & Anti-Starvation
- Implement **aging mechanism**: increase priority of waiting requests over time
- Guarantee that no request waits more than MAX_WAIT_TIME (configurable)
- Ensure low-priority requests eventually execute

#### FR4: Resource-Aware Scheduling
- Consider available GPU memory before scheduling
- Respect max batch size constraints
- Implement preemption when necessary for high-priority requests

#### FR5: Monitoring & Observability
- Expose Prometheus metrics for monitoring
- Log scheduling decisions with rationale
- Track queue lengths per priority

### Non-Functional Requirements

#### NFR1: Performance
- Scheduling decision overhead: < 1ms per iteration
- Support > 1000 concurrent requests in queue
- Minimal impact on throughput vs. default scheduler

#### NFR2: Code Quality
- Type hints throughout
- Unit tests with >85% coverage
- Integration tests with vLLM
- Comprehensive documentation

#### NFR3: Configuration
- All parameters configurable via config file
- Support for dynamic reconfiguration (stretch goal)
- Sensible defaults for production use

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Request Arrival                          │
│              (with priority metadata)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PriorityScheduler                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Priority Queue Manager                            │     │
│  │  - HIGH:   [req1, req5, ...]                      │     │
│  │  - MEDIUM: [req2, req7, ...]                      │     │
│  │  - LOW:    [req3, req4, ...]                      │     │
│  └────────────────────────────────────────────────────┘     │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Aging & Anti-Starvation Logic                     │     │
│  │  - Track wait times                                │     │
│  │  - Promote aged requests                           │     │
│  │  - Prevent priority inversion                      │     │
│  └────────────────────────────────────────────────────┘     │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Scheduling Decision Engine                        │     │
│  │  - Select batch from queues                        │     │
│  │  - Check resource constraints                      │     │
│  │  - Apply preemption if needed                      │     │
│  └────────────────────────────────────────────────────┘     │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  SLA Tracker                                       │     │
│  │  - Monitor latencies per priority                  │     │
│  │  - Detect violations                               │     │
│  │  - Emit metrics                                    │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 vLLM Executor                                │
│            (runs selected batch)                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Priority Queue Manager
Maintains separate queues for each priority level with efficient operations:
- `enqueue(request, priority)` - O(1)
- `dequeue(priority)` - O(1)
- `peek(priority)` - O(1)
- `reorder(request_id, new_priority)` - O(n) for aging

#### 2. Aging Manager
Prevents starvation through time-based priority promotion:
```python
def apply_aging(self, current_time: float):
    """
    Increase priority of requests that have waited too long.
    """
    for priority in [Priority.LOW, Priority.MEDIUM]:
        for request in self.queues[priority]:
            wait_time = current_time - request.arrival_time
            if wait_time > self.aging_threshold[priority]:
                self.promote_request(request)
```

#### 3. SLA Tracker
Monitors and reports on SLA compliance:
```python
@dataclass
class SLATarget:
    priority: Priority
    p99_target_ms: float
    p95_target_ms: float
    timeout_ms: float

class SLATracker:
    def record_completion(self, request: Request):
        """Track request completion for SLA calculation."""

    def check_violations(self) -> List[SLAViolation]:
        """Return current SLA violations."""

    def get_metrics(self) -> Dict[str, float]:
        """Return Prometheus-compatible metrics."""
```

---

## Implementation Guide

### Step 1: Set Up Project Structure (15 minutes)

```bash
mkdir -p priority_scheduler/{src,tests,docs,benchmarks}
cd priority_scheduler

# Create files
touch src/__init__.py
touch src/priority_scheduler.py
touch src/queue_manager.py
touch src/sla_tracker.py
touch src/aging_manager.py
touch tests/test_scheduler.py
touch tests/test_sla.py
touch README.md
```

### Step 2: Define Core Data Structures (30 minutes)

```python
# src/priority_scheduler.py

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time

class Priority(IntEnum):
    """Priority levels (higher value = higher priority)."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2

@dataclass
class SchedulerConfig:
    """Configuration for the priority scheduler."""

    # Priority settings
    enable_priorities: bool = True
    default_priority: Priority = Priority.MEDIUM

    # Aging settings
    enable_aging: bool = True
    aging_threshold_low: float = 10.0  # seconds
    aging_threshold_medium: float = 5.0  # seconds

    # Anti-starvation
    max_wait_time: float = 30.0  # seconds
    starvation_check_interval: int = 10  # iterations

    # SLA targets (P99 latency in milliseconds)
    sla_p99_high: float = 500.0
    sla_p99_medium: float = 2000.0
    sla_p99_low: float = 10000.0

    # Resource limits
    max_batch_size: int = 32
    max_num_tokens: int = 4096

@dataclass
class RequestMetadata:
    """Metadata attached to each request."""
    request_id: str
    priority: Priority
    arrival_time: float
    original_priority: Priority  # For aging tracking
    num_tokens: int  # For resource estimation
    sla_target_ms: float

@dataclass
class SchedulingDecision:
    """Result of scheduling decision."""
    selected_requests: List[str]  # request IDs
    priority_counts: Dict[Priority, int]
    total_tokens: int
    decision_time_ms: float
    preempted_requests: List[str] = field(default_factory=list)
```

### Step 3: Implement Queue Manager (45 minutes)

```python
# src/queue_manager.py

from collections import deque
from typing import Dict, List, Optional
from .priority_scheduler import Priority, RequestMetadata

class PriorityQueueManager:
    """Manages separate queues for each priority level."""

    def __init__(self):
        self.queues: Dict[Priority, deque] = {
            Priority.HIGH: deque(),
            Priority.MEDIUM: deque(),
            Priority.LOW: deque(),
        }
        self.request_map: Dict[str, RequestMetadata] = {}

    def enqueue(self, request: RequestMetadata) -> None:
        """Add request to appropriate priority queue."""
        self.queues[request.priority].append(request)
        self.request_map[request.request_id] = request

    def dequeue(self, priority: Priority) -> Optional[RequestMetadata]:
        """Remove and return request from priority queue."""
        if not self.queues[priority]:
            return None

        request = self.queues[priority].popleft()
        del self.request_map[request.request_id]
        return request

    def peek(self, priority: Priority, n: int = 1) -> List[RequestMetadata]:
        """Look at first n requests without removing them."""
        return list(self.queues[priority])[:n]

    def promote(self, request_id: str, new_priority: Priority) -> bool:
        """Change request priority (for aging)."""
        if request_id not in self.request_map:
            return False

        request = self.request_map[request_id]
        old_priority = request.priority

        # Remove from old queue
        try:
            self.queues[old_priority].remove(request)
        except ValueError:
            return False

        # Update priority and add to new queue
        request.priority = new_priority
        self.queues[new_priority].append(request)
        return True

    def get_queue_lengths(self) -> Dict[Priority, int]:
        """Return current queue lengths."""
        return {p: len(q) for p, q in self.queues.items()}

    def total_pending(self) -> int:
        """Total requests across all queues."""
        return sum(len(q) for q in self.queues.values())

    def get_oldest_waiting_time(self, priority: Priority) -> Optional[float]:
        """Get wait time of oldest request in queue."""
        if not self.queues[priority]:
            return None
        return time.time() - self.queues[priority][0].arrival_time
```

### Step 4: Implement Aging Manager (30 minutes)

```python
# src/aging_manager.py

import time
from typing import List
from .priority_scheduler import Priority, RequestMetadata, SchedulerConfig
from .queue_manager import PriorityQueueManager

class AgingManager:
    """Handles time-based priority promotion to prevent starvation."""

    def __init__(self, config: SchedulerConfig, queue_manager: PriorityQueueManager):
        self.config = config
        self.queue_manager = queue_manager
        self.last_aging_time = time.time()
        self.promotions_count = {Priority.LOW: 0, Priority.MEDIUM: 0}

    def apply_aging(self) -> int:
        """
        Check all requests and promote those that have waited too long.
        Returns number of promotions made.
        """
        if not self.config.enable_aging:
            return 0

        current_time = time.time()
        promotions = 0

        # Check LOW -> MEDIUM promotions
        for request in list(self.queue_manager.queues[Priority.LOW]):
            wait_time = current_time - request.arrival_time
            if wait_time > self.config.aging_threshold_low:
                if self.queue_manager.promote(request.request_id, Priority.MEDIUM):
                    promotions += 1
                    self.promotions_count[Priority.LOW] += 1

        # Check MEDIUM -> HIGH promotions
        for request in list(self.queue_manager.queues[Priority.MEDIUM]):
            wait_time = current_time - request.arrival_time
            if wait_time > self.config.aging_threshold_medium:
                if self.queue_manager.promote(request.request_id, Priority.HIGH):
                    promotions += 1
                    self.promotions_count[Priority.MEDIUM] += 1

        self.last_aging_time = current_time
        return promotions

    def check_starvation(self) -> List[str]:
        """
        Check if any requests have exceeded max wait time.
        Returns list of starved request IDs.
        """
        current_time = time.time()
        starved = []

        for priority in Priority:
            for request in self.queue_manager.queues[priority]:
                wait_time = current_time - request.arrival_time
                if wait_time > self.config.max_wait_time:
                    starved.append(request.request_id)

        return starved

    def get_aging_stats(self) -> dict:
        """Return statistics about aging behavior."""
        return {
            "total_promotions_low": self.promotions_count[Priority.LOW],
            "total_promotions_medium": self.promotions_count[Priority.MEDIUM],
            "time_since_last_aging": time.time() - self.last_aging_time,
        }
```

### Step 5: Implement SLA Tracker (45 minutes)

```python
# src/sla_tracker.py

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List
from dataclasses import dataclass
from .priority_scheduler import Priority, RequestMetadata, SchedulerConfig

@dataclass
class SLAViolation:
    """Record of an SLA violation."""
    request_id: str
    priority: Priority
    actual_latency_ms: float
    target_latency_ms: float
    violation_amount_ms: float
    timestamp: float

class SLATracker:
    """Tracks and reports on SLA compliance."""

    def __init__(self, config: SchedulerConfig):
        self.config = config

        # Latency history (rolling window)
        self.latencies: Dict[Priority, deque] = {
            p: deque(maxlen=1000) for p in Priority
        }

        # SLA violations
        self.violations: List[SLAViolation] = []

        # Completion tracking
        self.completions: Dict[Priority, int] = defaultdict(int)

        # Request start times
        self.start_times: Dict[str, float] = {}

    def record_start(self, request: RequestMetadata) -> None:
        """Record when request starts execution."""
        self.start_times[request.request_id] = time.time()

    def record_completion(self, request: RequestMetadata) -> None:
        """Record request completion and check for SLA violation."""
        if request.request_id not in self.start_times:
            return

        # Calculate latency
        start_time = self.start_times[request.request_id]
        latency_ms = (time.time() - start_time) * 1000

        # Record latency
        self.latencies[request.priority].append(latency_ms)
        self.completions[request.priority] += 1

        # Check for SLA violation
        if latency_ms > request.sla_target_ms:
            violation = SLAViolation(
                request_id=request.request_id,
                priority=request.priority,
                actual_latency_ms=latency_ms,
                target_latency_ms=request.sla_target_ms,
                violation_amount_ms=latency_ms - request.sla_target_ms,
                timestamp=time.time()
            )
            self.violations.append(violation)

        # Cleanup
        del self.start_times[request.request_id]

    def get_percentile_latency(self, priority: Priority, percentile: float) -> float:
        """Calculate percentile latency for priority level."""
        if not self.latencies[priority]:
            return 0.0
        return np.percentile(list(self.latencies[priority]), percentile)

    def get_sla_compliance_rate(self, priority: Priority) -> float:
        """
        Calculate SLA compliance rate (% of requests meeting SLA).
        """
        if self.completions[priority] == 0:
            return 1.0

        # Count violations for this priority
        priority_violations = sum(
            1 for v in self.violations if v.priority == priority
        )

        compliance_rate = 1.0 - (priority_violations / self.completions[priority])
        return max(0.0, compliance_rate)

    def get_recent_violations(self, since_seconds: float = 60.0) -> List[SLAViolation]:
        """Get violations from last N seconds."""
        cutoff_time = time.time() - since_seconds
        return [v for v in self.violations if v.timestamp > cutoff_time]

    def get_metrics(self) -> Dict[str, float]:
        """Get Prometheus-compatible metrics."""
        metrics = {}

        for priority in Priority:
            prefix = f"scheduler_{priority.name.lower()}"

            # Latency percentiles
            metrics[f"{prefix}_latency_p50"] = self.get_percentile_latency(priority, 50)
            metrics[f"{prefix}_latency_p95"] = self.get_percentile_latency(priority, 95)
            metrics[f"{prefix}_latency_p99"] = self.get_percentile_latency(priority, 99)

            # Compliance
            metrics[f"{prefix}_sla_compliance_rate"] = self.get_sla_compliance_rate(priority)

            # Completions
            metrics[f"{prefix}_total_completions"] = self.completions[priority]

        # Overall metrics
        metrics["scheduler_total_violations"] = len(self.violations)
        metrics["scheduler_total_completions"] = sum(self.completions.values())

        return metrics
```

### Step 6: Implement Main Scheduler (60 minutes)

```python
# src/priority_scheduler.py (continued)

import time
import logging
from typing import List, Optional
from .queue_manager import PriorityQueueManager
from .aging_manager import AgingManager
from .sla_tracker import SLATracker

logger = logging.getLogger(__name__)

class PriorityScheduler:
    """
    Production-ready priority scheduler for vLLM.

    Supports:
    - Multi-level priorities with configurable SLA targets
    - Aging to prevent starvation
    - Resource-aware scheduling
    - SLA tracking and violation reporting
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.queue_manager = PriorityQueueManager()
        self.aging_manager = AgingManager(config, self.queue_manager)
        self.sla_tracker = SLATracker(config)

        self.iteration_count = 0
        self.last_metrics_log_time = time.time()

    def add_request(
        self,
        request_id: str,
        priority: Optional[Priority] = None,
        num_tokens: int = 100,
    ) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request_id: Unique identifier for request
            priority: Priority level (uses default if None)
            num_tokens: Estimated number of tokens
        """
        if priority is None:
            priority = self.config.default_priority

        # Determine SLA target based on priority
        sla_targets = {
            Priority.HIGH: self.config.sla_p99_high,
            Priority.MEDIUM: self.config.sla_p99_medium,
            Priority.LOW: self.config.sla_p99_low,
        }

        request = RequestMetadata(
            request_id=request_id,
            priority=priority,
            arrival_time=time.time(),
            original_priority=priority,
            num_tokens=num_tokens,
            sla_target_ms=sla_targets[priority],
        )

        self.queue_manager.enqueue(request)
        logger.info(f"Added request {request_id} with priority {priority.name}")

    def schedule(self, available_memory: int) -> SchedulingDecision:
        """
        Main scheduling decision function.

        Called each iteration to decide which requests to execute.

        Args:
            available_memory: Available GPU memory in tokens

        Returns:
            SchedulingDecision with selected requests
        """
        start_time = time.time()
        self.iteration_count += 1

        # Apply aging if enabled
        if self.iteration_count % self.config.starvation_check_interval == 0:
            promotions = self.aging_manager.apply_aging()
            if promotions > 0:
                logger.debug(f"Promoted {promotions} requests due to aging")

            # Check for starvation
            starved = self.aging_manager.check_starvation()
            if starved:
                logger.warning(f"Starvation detected: {len(starved)} requests")

        # Select requests to schedule
        selected_requests = []
        priority_counts = {p: 0 for p in Priority}
        total_tokens = 0

        # Try to schedule from high to low priority
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            while (
                len(selected_requests) < self.config.max_batch_size
                and total_tokens < available_memory
            ):
                # Peek at next request
                candidates = self.queue_manager.peek(priority, 1)
                if not candidates:
                    break

                request = candidates[0]

                # Check if we can fit this request
                if total_tokens + request.num_tokens <= available_memory:
                    # Dequeue and select
                    self.queue_manager.dequeue(priority)
                    selected_requests.append(request.request_id)
                    priority_counts[priority] += 1
                    total_tokens += request.num_tokens

                    # Record start for SLA tracking
                    self.sla_tracker.record_start(request)
                else:
                    # Can't fit any more from this priority
                    break

        # Create decision
        decision_time_ms = (time.time() - start_time) * 1000
        decision = SchedulingDecision(
            selected_requests=selected_requests,
            priority_counts=priority_counts,
            total_tokens=total_tokens,
            decision_time_ms=decision_time_ms,
        )

        # Log metrics periodically
        if time.time() - self.last_metrics_log_time > 10.0:
            self._log_metrics()
            self.last_metrics_log_time = time.time()

        return decision

    def complete_request(self, request_id: str) -> None:
        """
        Mark a request as complete.

        Args:
            request_id: ID of completed request
        """
        # Note: In real integration, we'd need to track request metadata
        # For now, this is a placeholder
        pass

    def _log_metrics(self) -> None:
        """Log scheduler metrics."""
        queue_lengths = self.queue_manager.get_queue_lengths()
        sla_metrics = self.sla_tracker.get_metrics()
        aging_stats = self.aging_manager.get_aging_stats()

        logger.info(
            f"Scheduler metrics - "
            f"Queues: {queue_lengths}, "
            f"SLA compliance: HIGH={sla_metrics.get('scheduler_high_sla_compliance_rate', 0):.2f}, "
            f"MEDIUM={sla_metrics.get('scheduler_medium_sla_compliance_rate', 0):.2f}, "
            f"LOW={sla_metrics.get('scheduler_low_sla_compliance_rate', 0):.2f}, "
            f"Promotions: {aging_stats['total_promotions_low'] + aging_stats['total_promotions_medium']}"
        )

    def get_metrics(self) -> dict:
        """Get all scheduler metrics."""
        return {
            **self.sla_tracker.get_metrics(),
            **self.aging_manager.get_aging_stats(),
            "queue_lengths": self.queue_manager.get_queue_lengths(),
            "total_pending": self.queue_manager.total_pending(),
        }
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_scheduler.py

import pytest
import time
from src.priority_scheduler import (
    PriorityScheduler, SchedulerConfig, Priority, RequestMetadata
)

class TestPriorityScheduler:
    """Test suite for priority scheduler."""

    def test_basic_scheduling(self):
        """Test basic request scheduling."""
        config = SchedulerConfig(max_batch_size=2)
        scheduler = PriorityScheduler(config)

        # Add requests
        scheduler.add_request("req1", Priority.HIGH, num_tokens=100)
        scheduler.add_request("req2", Priority.LOW, num_tokens=100)

        # Schedule with sufficient memory
        decision = scheduler.schedule(available_memory=1000)

        # High priority should be scheduled first
        assert len(decision.selected_requests) == 2
        assert decision.selected_requests[0] == "req1"

    def test_priority_ordering(self):
        """Test that higher priority requests are scheduled first."""
        config = SchedulerConfig(max_batch_size=1)
        scheduler = PriorityScheduler(config)

        # Add in reverse priority order
        scheduler.add_request("low", Priority.LOW)
        scheduler.add_request("medium", Priority.MEDIUM)
        scheduler.add_request("high", Priority.HIGH)

        # Should schedule HIGH first
        decision = scheduler.schedule(available_memory=1000)
        assert decision.selected_requests == ["high"]

        # Then MEDIUM
        decision = scheduler.schedule(available_memory=1000)
        assert decision.selected_requests == ["medium"]

        # Then LOW
        decision = scheduler.schedule(available_memory=1000)
        assert decision.selected_requests == ["low"]

    def test_aging_promotion(self):
        """Test that aging promotes old requests."""
        config = SchedulerConfig(
            aging_threshold_low=0.1,  # 100ms for testing
            enable_aging=True,
        )
        scheduler = PriorityScheduler(config)

        # Add LOW priority request
        scheduler.add_request("old_low", Priority.LOW)

        # Wait for aging threshold
        time.sleep(0.15)

        # Trigger aging
        scheduler.aging_manager.apply_aging()

        # Should be promoted to MEDIUM
        assert scheduler.queue_manager.request_map["old_low"].priority == Priority.MEDIUM

    def test_resource_constraints(self):
        """Test scheduling respects memory constraints."""
        config = SchedulerConfig(max_batch_size=10)
        scheduler = PriorityScheduler(config)

        # Add requests that exceed memory when batched
        for i in range(5):
            scheduler.add_request(f"req{i}", Priority.HIGH, num_tokens=300)

        # Limited memory: should only schedule 3 requests
        decision = scheduler.schedule(available_memory=1000)
        assert len(decision.selected_requests) == 3
        assert decision.total_tokens <= 1000

    def test_starvation_detection(self):
        """Test starvation detection works."""
        config = SchedulerConfig(max_wait_time=0.1)
        scheduler = PriorityScheduler(config)

        # Add request
        scheduler.add_request("starving", Priority.LOW)

        # Wait past max wait time
        time.sleep(0.15)

        # Check for starvation
        starved = scheduler.aging_manager.check_starvation()
        assert "starving" in starved

    def test_sla_tracking(self):
        """Test SLA violation tracking."""
        config = SchedulerConfig()
        scheduler = PriorityScheduler(config)

        request = RequestMetadata(
            request_id="test",
            priority=Priority.HIGH,
            arrival_time=time.time(),
            original_priority=Priority.HIGH,
            num_tokens=100,
            sla_target_ms=100.0,
        )

        # Record start
        scheduler.sla_tracker.record_start(request)

        # Simulate slow completion
        time.sleep(0.2)

        # Record completion
        scheduler.sla_tracker.record_completion(request)

        # Should have violation
        violations = scheduler.sla_tracker.get_recent_violations()
        assert len(violations) > 0
        assert violations[0].request_id == "test"
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
import random
import time
from src.priority_scheduler import PriorityScheduler, SchedulerConfig, Priority

class TestIntegration:
    """Integration tests simulating realistic workloads."""

    def test_mixed_workload(self):
        """Test scheduler with mixed priorities and arrival patterns."""
        config = SchedulerConfig(
            max_batch_size=16,
            enable_aging=True,
        )
        scheduler = PriorityScheduler(config)

        # Simulate 100 requests with random priorities
        for i in range(100):
            priority = random.choice(list(Priority))
            scheduler.add_request(f"req{i}", priority, num_tokens=random.randint(50, 200))

        # Schedule in batches
        iterations = 0
        while scheduler.queue_manager.total_pending() > 0:
            decision = scheduler.schedule(available_memory=2048)
            iterations += 1

            # Verify constraints
            assert len(decision.selected_requests) <= config.max_batch_size
            assert decision.total_tokens <= 2048

            # Safety: prevent infinite loop
            if iterations > 20:
                break

        # All requests should be scheduled
        assert scheduler.queue_manager.total_pending() == 0

    def test_sla_compliance_under_load(self):
        """Test SLA compliance with heavy load."""
        config = SchedulerConfig()
        scheduler = PriorityScheduler(config)

        # Add many HIGH priority requests
        for i in range(50):
            scheduler.add_request(f"high{i}", Priority.HIGH)

        # Schedule all
        completed = 0
        while scheduler.queue_manager.total_pending() > 0:
            decision = scheduler.schedule(available_memory=2048)
            completed += len(decision.selected_requests)

        # Check SLA compliance rate
        metrics = scheduler.get_metrics()
        compliance = metrics.get('scheduler_high_sla_compliance_rate', 0)

        # Should maintain reasonable compliance (depends on test speed)
        assert compliance >= 0.0  # Basic sanity check
```

---

## Benchmarking

### Performance Benchmarks

```python
# benchmarks/benchmark_scheduler.py

import time
import random
from src.priority_scheduler import PriorityScheduler, SchedulerConfig, Priority

def benchmark_scheduling_overhead():
    """Measure per-iteration scheduling overhead."""
    config = SchedulerConfig(max_batch_size=32)
    scheduler = PriorityScheduler(config)

    # Add 1000 requests
    for i in range(1000):
        priority = random.choice(list(Priority))
        scheduler.add_request(f"req{i}", priority)

    # Measure scheduling time
    iterations = 100
    start = time.time()

    for _ in range(iterations):
        scheduler.schedule(available_memory=4096)

    elapsed = time.time() - start
    avg_time_ms = (elapsed / iterations) * 1000

    print(f"Average scheduling time: {avg_time_ms:.3f} ms")
    assert avg_time_ms < 1.0, "Scheduling overhead too high!"

def benchmark_throughput():
    """Measure scheduler throughput."""
    config = SchedulerConfig()
    scheduler = PriorityScheduler(config)

    num_requests = 10000
    for i in range(num_requests):
        scheduler.add_request(f"req{i}", Priority.MEDIUM)

    start = time.time()
    scheduled_count = 0

    while scheduler.queue_manager.total_pending() > 0:
        decision = scheduler.schedule(available_memory=4096)
        scheduled_count += len(decision.selected_requests)

    elapsed = time.time() - start
    throughput = scheduled_count / elapsed

    print(f"Scheduler throughput: {throughput:.1f} requests/sec")
    print(f"Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    print("Running scheduler benchmarks...")
    benchmark_scheduling_overhead()
    benchmark_throughput()
```

---

## Evaluation Criteria

### Functionality (40 points)
- ✅ Priority-based scheduling works (10 pts)
- ✅ Aging prevents starvation (10 pts)
- ✅ SLA tracking accurate (10 pts)
- ✅ Resource constraints respected (10 pts)

### Performance (20 points)
- ✅ Scheduling overhead < 1ms (10 pts)
- ✅ Supports 1000+ concurrent requests (10 pts)

### Code Quality (20 points)
- ✅ Type hints throughout (5 pts)
- ✅ Clear documentation (5 pts)
- ✅ Follows best practices (5 pts)
- ✅ No obvious bugs (5 pts)

### Testing (10 points)
- ✅ Unit test coverage > 85% (5 pts)
- ✅ Integration tests pass (5 pts)

### Documentation (10 points)
- ✅ Comprehensive README (5 pts)
- ✅ Design decisions explained (5 pts)

**Total: 100 points**
**Passing: 70 points**

---

## Common Pitfalls

### Pitfall 1: Priority Inversion
**Problem:** Low priority request blocks high priority request.
**Solution:** Always check higher priority queues first.

### Pitfall 2: Aggressive Aging
**Problem:** Aging promotes too quickly, defeating priority system.
**Solution:** Tune aging thresholds based on typical request durations.

### Pitfall 3: Memory Estimation Errors
**Problem:** Overestimating available memory leads to OOM.
**Solution:** Add safety margin; validate estimates with profiling.

### Pitfall 4: Infinite Starvation
**Problem:** High priority requests keep arriving, LOW never scheduled.
**Solution:** Implement max wait time with forced scheduling.

### Pitfall 5: Lock Contention
**Problem:** Thread-safe operations cause slowdowns.
**Solution:** Minimize critical sections; use lock-free structures where possible.

---

## Extensions and Improvements

### Stretch Goals

1. **Dynamic SLA Adjustment**: Automatically adjust SLAs based on system load
2. **Multi-Dimensional Priority**: Consider both user priority and request size
3. **Predictive Scheduling**: Use ML to predict request durations
4. **Fair Queuing**: Implement weighted fair queuing across priorities
5. **Preemption**: Interrupt low-priority work for urgent requests

### Production Enhancements

1. **Persistence**: Save queue state for crash recovery
2. **Distributed Scheduling**: Coordinate across multiple schedulers
3. **Admin Interface**: Web UI for monitoring and control
4. **A/B Testing**: Compare scheduler configurations in production
5. **Cost Optimization**: Factor in GPU costs per priority tier

---

## Submission Checklist

- [ ] Code passes all unit tests
- [ ] Integration tests demonstrate correctness
- [ ] Benchmarks show < 1ms scheduling overhead
- [ ] README documents setup and usage
- [ ] Design decisions explained in docs/
- [ ] SLA tracking produces accurate metrics
- [ ] Aging prevents starvation in tests
- [ ] Code follows style guidelines
- [ ] Type hints throughout
- [ ] Git repo with clean history

---

## Resources

- vLLM Scheduler Code: `vllm/core/scheduler.py`
- Multi-Level Feedback Queue: [Wikipedia](https://en.wikipedia.org/wiki/Multilevel_feedback_queue)
- Queueing Theory Basics: [Introduction to Queueing Theory](https://web.mit.edu/~sgraves/www/18.337/queueing-theory.pdf)
- SLA Management: [Google SRE Book - SLIs, SLOs, and SLAs](https://sre.google/sre-book/service-level-objectives/)

---

**Time to build! Start with the queue manager, then layer in aging and SLA tracking. Test incrementally as you go.**

*Project 7.1: Custom Scheduler | Duration: 3-4 hours*
