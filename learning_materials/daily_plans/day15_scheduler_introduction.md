# Day 15: Scheduler Architecture & Policies

> **Goal**: Master the vLLM scheduler's architecture, scheduling policies, and decision-making algorithms
> **Time**: 4-6 hours
> **Prerequisites**: Week 1 & 2 complete, understanding of continuous batching basics
> **Deliverables**: Scheduler state diagrams, policy comparison analysis, custom scheduler prototype

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): Scheduler Architecture

**9:00-9:30** - Scheduler Overview & Core Responsibilities
**9:30-10:30** - Scheduler State Machine & Queues
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Scheduling Policies Deep Dive

### Afternoon Session (2-3 hours): Advanced Scheduling

**14:00-15:00** - Preemption & Memory Management
**15:00-16:00** - Multi-Step Scheduling Scenarios
**16:00-16:30** - Break
**16:30-17:30** - Hands-On: Custom Scheduler Policy

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Review scheduling decisions, create flowcharts, preview Day 16

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain the scheduler's role in the vLLM engine
- [ ] Trace scheduling decisions through multiple steps
- [ ] Compare FCFS, priority-based, and fair scheduling policies
- [ ] Understand preemption strategies (swap vs. recompute)
- [ ] Implement a simplified custom scheduler policy

---

## 📂 Morning: Scheduler Architecture (9:00-12:00)

### Task 1: Scheduler Core Responsibilities (30 min)

**File**: `vllm/core/scheduler.py:Scheduler`

**What the Scheduler Does**:

```python
class Scheduler:
    """
    The Scheduler is responsible for:
    1. Managing three queues: waiting, running, swapped
    2. Deciding which requests to execute next (batching)
    3. Allocating GPU blocks via BlockManager
    4. Preempting requests when out of memory
    5. Handling request priorities and fairness
    """
```

**📊 High-Level Architecture**:

```
Request Lifecycle Through Scheduler:
┌─────────┐
│  NEW    │ → add_seq_group()
└────┬────┘
     ↓
┌─────────────┐
│  WAITING    │ ← Queued, not yet scheduled
└────┬────────┘
     ↓ schedule() - allocate blocks
┌─────────────┐
│  RUNNING    │ ← Active execution
└─┬─────────┬─┘
  ↓         ↓
  │      ┌──────────┐
  │      │ SWAPPED  │ ← Preempted to CPU
  │      └────┬─────┘
  │           ↓ swap_in() - when memory available
  │      ┌────┴─────┐
  │      │ RUNNING  │
  ↓      └──────────┘
┌─────────────┐
│  FINISHED   │ → free_seq()
└─────────────┘
```

**Key Metrics Scheduler Manages**:

```python
# vllm/core/scheduler.py

class SchedulerOutputs:
    scheduled_seq_groups: List[SequenceGroup]  # What to run
    num_batched_tokens: int                     # Total tokens in batch
    num_prefill_groups: int                     # Prefill requests
    blocks_to_swap_in: Dict                     # Swap in operations
    blocks_to_swap_out: Dict                    # Swap out operations
    blocks_to_copy: Dict                        # Copy operations (beam search)
```

### Task 2: Scheduler State Machine (60 min)

**File**: `vllm/core/scheduler.py:Scheduler.schedule()`

**Understanding the State Transitions**:

```python
def schedule(self) -> SchedulerOutputs:
    """
    Main scheduling logic - called every step.

    Process:
    1. Handle swapped requests (bring back from CPU)
    2. Schedule running requests (continue generation)
    3. Schedule waiting requests (new/queued)
    4. Handle preemption if OOM
    5. Return scheduling decisions
    """

    # Budget tracking
    budget = SchedulingBudget(
        token_budget=self.scheduler_config.max_num_batched_tokens,
        max_num_seqs=self.scheduler_config.max_num_seqs,
    )

    # Priority 1: Swap in (if memory recovered)
    swapped_in = self._schedule_swapped(budget)

    # Priority 2: Running (continue existing)
    running = self._schedule_running(budget)

    # Priority 3: Waiting (new requests)
    waiting = self._schedule_waiting(budget)

    # Handle OOM: preempt if necessary
    if self._need_to_preempt():
        preempted = self._preempt()

    return SchedulerOutputs(...)
```

**📝 Exercise: State Transition Tracing**

Trace this scenario through 5 steps:

```
Initial State:
- GPU blocks: 100 total, 20 free
- Waiting: [A(200 tokens), B(150 tokens)]
- Running: []
- Swapped: []

Step 1: schedule()
  _schedule_swapped(): [] (empty)
  _schedule_running(): [] (empty)
  _schedule_waiting():
    - Check A: needs 13 blocks (200/16 = 12.5 → 13)
      → 20 free >= 13 ✓
      → Allocate blocks
      → Move A to running
    - Check B: needs 10 blocks
      → 7 free >= 10 ✗
      → B stays in waiting
  Result: Running=[A], Waiting=[B], Free blocks=7

Step 2: A generates 1 token (201 total)
  _schedule_running():
    - A needs append_slot
    - Current blocks: 13, usage: 201/208 (13×16)
    - No new block needed ✓
  _schedule_waiting():
    - B needs 10 blocks, only 7 free ✗
  Decision: Continue A, B waits
  Result: Running=[A], Waiting=[B]

Step 3: A generates more (223 tokens)
  _schedule_running():
    - A crosses boundary: 223/208
    - Need 1 more block → 14 blocks total
    - 7 free >= 1 ✓
    - Allocate block
  Result: Running=[A], Free blocks=6

Step 4: A generates (240 tokens), OOM scenario
  _schedule_running(): A continues
  _schedule_waiting():
    - B needs 10 blocks, only 6 free ✗
  _need_to_preempt(): False (not critical yet)
  Result: Same state

Step 5: New request C arrives (300 tokens)
  Waiting: [B, C]
  Neither can be scheduled (not enough memory)
  Options:
    a. Wait for A to finish
    b. Preempt A to make room for B and C

  If priority scheduling:
    - If C is high priority → preempt A
    - Otherwise → wait
```

### Task 3: Scheduling Policies (60 min)

**File**: `vllm/core/scheduler.py`

**Policy 1: First-Come-First-Served (FCFS)**

```python
class SchedulerPolicy(enum.Enum):
    FCFS = "fcfs"  # Default policy

# In _schedule_waiting():
def _schedule_waiting(self, budget: SchedulingBudget):
    """Schedule waiting requests in FIFO order."""

    scheduled = []

    # Process in arrival order
    while self.waiting and budget.can_schedule():
        seq_group = self.waiting[0]  # Always take first

        # Can we allocate blocks?
        num_required_blocks = self._get_num_required_blocks(seq_group)

        if not self.block_manager.can_allocate(num_required_blocks):
            break  # Stop if can't allocate

        # Allocate and schedule
        self.block_manager.allocate(seq_group)
        self.waiting.pop(0)
        scheduled.append(seq_group)
        budget.add_seq_group(seq_group)

    return scheduled
```

**Pros/Cons of FCFS**:
- ✅ Simple, fair in arrival order
- ✅ No starvation
- ❌ Head-of-line blocking (large request blocks small ones)
- ❌ No priority handling

**Policy 2: Priority-Based Scheduling**

```python
class SequenceGroup:
    priority: int = 0  # Higher = more important

def _schedule_waiting_priority(self, budget: SchedulingBudget):
    """Schedule by priority, then FIFO."""

    # Sort by priority (descending), then arrival time
    self.waiting.sort(key=lambda x: (-x.priority, x.arrival_time))

    # Schedule highest priority first
    scheduled = []

    for seq_group in self.waiting[:]:
        if not budget.can_schedule():
            break

        if self.block_manager.can_allocate(...):
            # Allocate and schedule
            self.waiting.remove(seq_group)
            scheduled.append(seq_group)
            budget.add_seq_group(seq_group)

    return scheduled
```

**Policy 3: Fair Scheduling (Round-Robin Token Budget)**

```python
def _schedule_waiting_fair(self, budget: SchedulingBudget):
    """
    Fair scheduling: distribute tokens evenly across requests.
    Each request gets equal token budget per step.
    """

    num_waiting = len(self.waiting)
    if num_waiting == 0:
        return []

    # Tokens per request
    tokens_per_request = budget.token_budget // num_waiting

    scheduled = []

    for seq_group in self.waiting[:]:
        # Limit this request's tokens
        seq_tokens = min(
            seq_group.num_tokens(),
            tokens_per_request
        )

        if self.block_manager.can_allocate(...):
            # Schedule with limited tokens
            scheduled.append(seq_group)
            budget.add_num_batched_tokens(seq_tokens)

    return scheduled
```

**📊 Policy Comparison Table**:

| Policy | Fairness | Latency | Throughput | Complexity |
|--------|----------|---------|------------|------------|
| FCFS | Medium | High for waiting | High | Low |
| Priority | Low | Low for high-pri | Medium | Medium |
| Fair RR | High | Medium | Medium | High |
| Shortest Job First | Low | Low average | Highest | Medium |

---

## 🔬 Afternoon: Advanced Scheduling (14:00-17:30)

### Task 4: Preemption Strategies (60 min)

**File**: `vllm/core/scheduler.py:Scheduler._preempt()`

**Why Preemption?**
```
Scenario: Out of GPU memory
- Running: [A(500 tokens), B(300 tokens), C(200 tokens)]
- Waiting: [D(400 tokens, high priority)]
- Free blocks: 3
- D needs: 25 blocks

Options:
1. Wait for one to finish → high latency for D
2. Preempt one → free blocks for D
```

**Strategy 1: Swap Out**

```python
def _preempt_by_swap(self, seq_group: SequenceGroup):
    """
    Move KV cache from GPU to CPU.

    Process:
    1. Copy blocks: GPU → CPU
    2. Free GPU blocks
    3. Move seq_group: running → swapped
    4. Can swap in later when memory available
    """

    # Get blocks to swap
    gpu_blocks = self.block_manager.get_blocks(seq_group)

    # Allocate CPU blocks
    cpu_blocks = self.block_manager.allocate_cpu_blocks(len(gpu_blocks))

    # Schedule copy operation
    self.blocks_to_swap_out[gpu_blocks] = cpu_blocks

    # Update state
    self.running.remove(seq_group)
    self.swapped.append(seq_group)

    # Free GPU blocks
    self.block_manager.free_gpu_blocks(seq_group)
```

**When to use**: Long sequences, sufficient CPU memory

**Strategy 2: Recomputation**

```python
def _preempt_by_recompute(self, seq_group: SequenceGroup):
    """
    Discard KV cache, recompute from scratch later.

    Process:
    1. Free all GPU blocks
    2. Move seq_group: running → waiting
    3. Will be rescheduled and recomputed
    """

    # Free all blocks
    self.block_manager.free(seq_group)

    # Reset to initial state
    seq_group.reset_state()

    # Move to waiting
    self.running.remove(seq_group)
    self.waiting.append(seq_group)  # Back to queue
```

**When to use**: Short sequences (cheap to recompute), no CPU memory

**Preemption Victim Selection**:

```python
def _select_victim(self) -> SequenceGroup:
    """
    Choose which request to preempt.

    Policies:
    1. Lowest priority
    2. Longest running (most KV cache)
    3. Last in (newest)
    """

    # Policy: Lowest priority first
    victim = min(self.running, key=lambda x: x.priority)

    # Policy: Longest sequence (free most memory)
    # victim = max(self.running, key=lambda x: x.get_seqs()[0].get_len())

    return victim
```

**📝 Exercise: Preemption Decision Tree**

```
Given:
- GPU blocks: 100 total, 2 free
- CPU blocks: 50 available
- Running: [A(1000 tokens, pri=5), B(200 tokens, pri=3), C(500 tokens, pri=7)]
- Waiting: [D(400 tokens, pri=10)]

D needs 25 blocks to run.

Question: Which to preempt and how?

Analysis:
1. Need: 25 blocks, have: 2 free → need 23 more blocks

2. Victim selection by priority:
   - B has lowest priority (3)
   - B uses: ⌈200/16⌉ = 13 blocks
   - Still need: 23 - 13 = 10 more blocks

3. Next victim:
   - A (pri=5) or C (pri=7)?
   - A has lower priority → preempt A
   - A uses: ⌈1000/16⌉ = 63 blocks
   - Now have: 2 + 13 + 63 = 78 blocks ✓

4. Preemption strategy:
   - A: 1000 tokens → SWAP (expensive to recompute)
   - B: 200 tokens → RECOMPUTE (cheap)

Decision:
- Recompute B (→ waiting)
- Swap A (→ swapped)
- Schedule D (→ running)
- C continues running
```

### Task 5: Multi-Step Scheduling Simulation (60 min)

**Build a Scheduling Simulator**:

```python
#!/usr/bin/env python3
"""
Scheduler Simulator - Understanding scheduling decisions
"""

from dataclasses import dataclass
from typing import List
from enum import Enum

class QueueState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"

@dataclass
class Request:
    id: str
    prompt_tokens: int
    max_tokens: int
    generated_tokens: int = 0
    priority: int = 0
    state: QueueState = QueueState.WAITING
    allocated_blocks: List[int] = None

    def __post_init__(self):
        if self.allocated_blocks is None:
            self.allocated_blocks = []

    def current_length(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    def blocks_needed(self, block_size: int = 16) -> int:
        import math
        return math.ceil(self.current_length() / block_size)

    def is_finished(self) -> bool:
        return self.generated_tokens >= self.max_tokens

class SimpleScheduler:
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))

        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []

        self.step_count = 0

    def add_request(self, req: Request):
        """Add new request to waiting queue."""
        req.state = QueueState.WAITING
        self.waiting.append(req)
        print(f"  Added {req.id} to waiting queue")

    def allocate_blocks(self, req: Request) -> bool:
        """Try to allocate blocks for request."""
        needed = req.blocks_needed(self.block_size)
        current = len(req.allocated_blocks)
        to_allocate = needed - current

        if to_allocate <= 0:
            return True  # Already has enough

        if len(self.free_blocks) < to_allocate:
            return False  # Not enough free

        # Allocate
        new_blocks = [self.free_blocks.pop(0) for _ in range(to_allocate)]
        req.allocated_blocks.extend(new_blocks)
        return True

    def free_blocks(self, req: Request):
        """Free all blocks from request."""
        self.free_blocks.extend(req.allocated_blocks)
        self.free_blocks.sort()
        req.allocated_blocks = []

    def schedule(self) -> List[Request]:
        """Main scheduling logic."""
        print(f"\n{'='*60}")
        print(f"Step {self.step_count}: Scheduling")
        print(f"{'='*60}")

        # 1. Schedule running (continue generation)
        for req in self.running[:]:
            if not self.allocate_blocks(req):
                print(f"  ⚠️  Cannot allocate for {req.id} - would need preemption")

        # 2. Schedule waiting (FCFS)
        for req in self.waiting[:]:
            if self.allocate_blocks(req):
                self.waiting.remove(req)
                self.running.append(req)
                req.state = QueueState.RUNNING
                print(f"  ✓ Scheduled {req.id}: {len(req.allocated_blocks)} blocks")
            else:
                print(f"  ✗ Cannot schedule {req.id}: need {req.blocks_needed() - len(req.allocated_blocks)} more blocks")
                break  # FCFS: stop here

        print(f"\n  Running: {[r.id for r in self.running]}")
        print(f"  Waiting: {[r.id for r in self.waiting]}")
        print(f"  Free blocks: {len(self.free_blocks)}")

        return self.running

    def execute_step(self):
        """Execute one generation step."""
        print(f"\n  Executing...")

        for req in self.running[:]:
            # Generate one token
            req.generated_tokens += 1
            print(f"    {req.id}: {req.current_length()} tokens ({req.generated_tokens}/{req.max_tokens})")

            # Check if finished
            if req.is_finished():
                print(f"    ✓ {req.id} finished!")
                self.running.remove(req)
                self.finished.append(req)
                self.free_blocks(req)
                req.state = QueueState.FINISHED

    def run_simulation(self):
        """Run until all requests finished."""
        print(f"\n{'='*60}")
        print(f"Starting Simulation")
        print(f"{'='*60}")
        print(f"Total blocks: {self.total_blocks}, Block size: {self.block_size}")

        while self.waiting or self.running:
            self.schedule()
            self.execute_step()
            self.step_count += 1

            if self.step_count > 100:  # Safety
                print("\n⚠️  Simulation limit reached")
                break

        print(f"\n{'='*60}")
        print(f"Simulation Complete")
        print(f"{'='*60}")
        print(f"Total steps: {self.step_count}")
        print(f"Finished requests: {len(self.finished)}")

# Example usage
if __name__ == "__main__":
    scheduler = SimpleScheduler(total_blocks=50, block_size=16)

    # Add requests
    scheduler.add_request(Request("A", prompt_tokens=32, max_tokens=5, priority=5))
    scheduler.add_request(Request("B", prompt_tokens=128, max_tokens=10, priority=3))
    scheduler.add_request(Request("C", prompt_tokens=64, max_tokens=3, priority=7))

    # Run
    scheduler.run_simulation()
```

### Task 6: Custom Scheduler Policy (60 min)

**Exercise**: Implement shortest-job-first scheduling

```python
class SJFScheduler(SimpleScheduler):
    """Shortest Job First - prioritize requests with fewer remaining tokens."""

    def schedule(self):
        """Schedule shortest jobs first."""

        # Sort waiting by remaining tokens
        self.waiting.sort(key=lambda r: r.max_tokens - r.generated_tokens)

        # Call parent schedule (will process in sorted order)
        return super().schedule()
```

**Exercise**: Implement weighted fair queuing

```python
class WeightedFairScheduler(SimpleScheduler):
    """Weighted Fair Queuing - distribute tokens proportionally to priority."""

    def schedule(self):
        """Distribute token budget fairly by priority weights."""

        total_priority = sum(r.priority for r in self.waiting)
        if total_priority == 0:
            return super().schedule()  # Fall back to FCFS

        # Allocate proportionally
        for req in self.waiting[:]:
            # Fair share based on priority
            fair_share = req.priority / total_priority

            # Try to allocate
            if self.allocate_blocks(req):
                self.waiting.remove(req)
                self.running.append(req)
                print(f"  ✓ {req.id}: priority {req.priority} ({fair_share:.1%} share)")

        return self.running
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Scheduler Architecture**
- Three-queue model: waiting, running, swapped
- State transitions and lifecycle
- Budget tracking (tokens, sequences)

✅ **Scheduling Policies**
- FCFS (First-Come-First-Served)
- Priority-based scheduling
- Fair scheduling approaches

✅ **Preemption Strategies**
- Swap out (GPU → CPU)
- Recomputation (discard and redo)
- Victim selection algorithms

✅ **Practical Skills**
- Tracing scheduling decisions
- Simulating scheduler behavior
- Implementing custom policies

### Knowledge Check Questions

**Q1**: What are the three main queues in the vLLM scheduler?
<details>
<summary>Answer</summary>
1. **Waiting**: New requests not yet scheduled
2. **Running**: Active requests being processed
3. **Swapped**: Preempted requests moved to CPU memory
</details>

**Q2**: When would you choose swap over recomputation for preemption?
<details>
<summary>Answer</summary>
Choose **swap** when:
- Sequence is long (expensive to recompute)
- CPU memory is available
- Will likely resume soon

Choose **recompute** when:
- Sequence is short (cheap to redo)
- Limited CPU memory
- Uncertain if/when will resume
</details>

**Q3**: What is head-of-line blocking and how does it affect FCFS?
<details>
<summary>Answer</summary>
**Head-of-line blocking**: A large request at the front of the queue prevents smaller requests behind it from being scheduled, even if there's enough memory for them.

Example:
- Request A: 2000 tokens (needs 125 blocks)
- Request B: 32 tokens (needs 2 blocks)
- Available: 10 blocks

FCFS tries A first, fails, stops. B could run but is blocked.
</details>

**Q4**: How does the scheduler prevent starvation in priority scheduling?
<details>
<summary>Answer</summary>
Techniques:
1. **Aging**: Gradually increase priority of waiting requests
2. **Priority ceiling**: Limit how long high-priority can block low-priority
3. **Hybrid policy**: Priority for initial scheduling, then FCFS for running
4. **Fair share guarantees**: Ensure minimum service for all priorities
</details>

**Q5**: Calculate blocks needed for a request with 250 tokens (block_size=16).
<details>
<summary>Answer</summary>
Blocks = ⌈250 / 16⌉ = ⌈15.625⌉ = **16 blocks**

Usage: 250 tokens
Capacity: 16 × 16 = 256 tokens
Waste: 6 tokens (2.3%)
</details>

---

## 🚀 Preview: Day 16

Tomorrow you'll dive deeper into:
- **Advanced Batching Strategies**: Mixed prefill/decode, chunked prefill
- **Iteration-Level Scheduling**: Token budgets and micro-batching
- **Performance Optimization**: Batch size tuning, latency vs. throughput
- **Real-World Scenarios**: Multi-tenant scheduling, QoS guarantees

**Preparation**:
- Review continuous batching concepts from Week 1
- Think about trade-offs in batch composition
- Consider real-world serving constraints

---

## 📚 Additional Resources

**Key Files to Review**:
- [ ] `vllm/core/scheduler.py` - Main scheduler implementation
- [ ] `vllm/core/policy.py` - Scheduling policies
- [ ] `vllm/config.py` - SchedulerConfig parameters
- [ ] `vllm/sequence.py` - Sequence and SequenceGroup

**Concepts to Master**:
- [ ] Queue management algorithms
- [ ] Memory-aware scheduling
- [ ] Priority queues and heaps
- [ ] Fair scheduling theory

**Optional Reading**:
- [ ] "Orca: A Distributed Serving System for Transformer-Based Generative Models"
- [ ] Fair queuing in networking (similar principles)
- [ ] Linux process scheduler design

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
