# Day 16: Continuous Batching Deep Dive

> **Goal**: Master advanced continuous batching strategies, mixed batch composition, and performance optimization
> **Time**: 4-6 hours
> **Prerequisites**: Day 15 complete, understanding of scheduler architecture
> **Deliverables**: Batch composition analyzer, performance tuning guide, optimization experiments

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): Advanced Batching Strategies

**9:00-9:30** - Continuous Batching Review & Deep Dive
**9:30-10:30** - Mixed Prefill/Decode Batching
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Chunked Prefill & Token Budgets

### Afternoon Session (2-3 hours): Performance Optimization

**14:00-15:00** - Batch Size Tuning & Trade-offs
**15:00-16:00** - Profiling Batch Performance
**16:00-16:30** - Break
**16:30-17:30** - Hands-On: Batch Optimization Experiments

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Analyze experiments, document findings, preview Day 17

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain mixed prefill/decode batch composition
- [ ] Understand chunked prefill for long prompts
- [ ] Calculate optimal batch sizes for different workloads
- [ ] Analyze latency vs. throughput trade-offs
- [ ] Implement batch performance monitoring

---

## 📂 Morning: Advanced Batching Strategies (9:00-12:00)

### Task 1: Continuous Batching Deep Dive (30 min)

**Continuous Batching vs. Static Batching**

```
Static Batching (Traditional):
┌─────────────────────────────────────────────┐
│ Batch 1: [A, B, C, D]                       │
│                                              │
│ A: ████████░░ (finishes at step 8)          │
│ B: ██████████████ (finishes at step 14)     │
│ C: ████████████████ (finishes at step 16)   │
│ D: ██████████████████ (finishes at step 18) │
│                                              │
│ ⚠️ GPU idle after A finishes                │
│ ⚠️ Must wait for D to complete              │
│ ⚠️ New requests wait for entire batch       │
└─────────────────────────────────────────────┘
Batch completion: Step 18
Batch 2 starts: Step 18

Continuous Batching (vLLM):
┌─────────────────────────────────────────────┐
│ Dynamic Batch: Always full                  │
│                                              │
│ Step 1-7:   [A, B, C, D]                    │
│ Step 8:     [B, C, D, E] ← A done, E added  │
│ Step 9-13:  [B, C, D, E]                    │
│ Step 14:    [C, D, E, F] ← B done, F added  │
│ Step 15:    [C, D, E, F]                    │
│ Step 16:    [D, E, F, G] ← C done, G added  │
│                                              │
│ ✓ No idle time                              │
│ ✓ Continuous request processing             │
│ ✓ Better throughput & lower average latency │
└─────────────────────────────────────────────┘
```

**Performance Impact**:

```python
# Static batching metrics
static_throughput = total_requests / max_completion_time
static_avg_latency = sum(completion_times) / num_requests

# Example:
# Batch size 4, all requests 10 tokens
# Completion times: [10, 10, 10, 10]
# Throughput: 4 / 10 = 0.4 req/step
# Avg latency: 10 steps

# Continuous batching metrics
# Same 4 requests, but can add new ones immediately
# Request 5 starts at step 1 instead of step 11
# Throughput: higher (always processing)
# Avg latency: 3-5x lower for queued requests
```

**Key Implementation Details**:

```python
# vllm/core/scheduler.py

def schedule(self) -> SchedulerOutputs:
    """
    Continuous batching implementation.
    """

    # Start with current running batch
    scheduled_seq_groups = list(self.running)

    # Check for finished sequences
    for seq_group in self.running[:]:
        if seq_group.is_finished():
            # Remove immediately
            self.running.remove(seq_group)
            scheduled_seq_groups.remove(seq_group)

            # Free blocks for new requests
            self.block_manager.free(seq_group)

    # Fill empty slots with waiting requests
    budget = self._compute_available_budget(scheduled_seq_groups)

    while self.waiting and budget.can_schedule():
        next_seq_group = self.waiting[0]

        if self._can_allocate(next_seq_group):
            # Add to batch immediately
            self.waiting.pop(0)
            self.running.append(next_seq_group)
            scheduled_seq_groups.append(next_seq_group)

            budget.add_seq_group(next_seq_group)
        else:
            break  # Out of memory

    return SchedulerOutputs(scheduled_seq_groups)
```

### Task 2: Mixed Prefill/Decode Batching (60 min)

**Understanding Mixed Batches**

```
Prefill vs Decode Characteristics:

Prefill:
  Input: All prompt tokens [t0, t1, t2, ..., tn]
  Output: 1 token
  Compute: O(n²) attention over prompt
  Memory: Sequential access (no KV cache yet)
  Profile: Compute-bound

Decode:
  Input: Last generated token [tn]
  Output: 1 token
  Compute: O(n) attention (query vs. cached K/V)
  Memory: Random access (read KV cache)
  Profile: Memory-bound

Can we batch them together? YES!
```

**Mixed Batch Composition**:

```python
# Example mixed batch
batch = {
    'prefill_requests': [
        {'id': 'A', 'prompt_tokens': 200, 'is_prefill': True},
        {'id': 'B', 'prompt_tokens': 150, 'is_prefill': True},
    ],
    'decode_requests': [
        {'id': 'C', 'current_length': 50, 'is_prefill': False},
        {'id': 'D', 'current_length': 120, 'is_prefill': False},
        {'id': 'E', 'current_length': 80, 'is_prefill': False},
    ]
}

# Total tokens in batch
total_tokens = (
    200 + 150 +  # Prefill: process all prompt tokens
    1 + 1 + 1     # Decode: process 1 token each
) = 353 tokens

# All processed in single forward pass!
```

**Attention Kernel Handling**:

```cuda
// Simplified mixed batch attention

__global__ void paged_attention_kernel(
    float* out,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int* seq_lens,
    const bool* is_prefill,
    const int* block_tables
) {
    int seq_id = blockIdx.x;

    if (is_prefill[seq_id]) {
        // Prefill: Self-attention over prompt
        // Q: [num_prompt_tokens, head_dim]
        // K, V: [num_prompt_tokens, head_dim]
        prefill_attention(q, k_cache, v_cache, seq_lens[seq_id]);
    } else {
        // Decode: Attention over cached K/V
        // Q: [1, head_dim]
        // K, V: [cached_tokens, head_dim]
        decode_attention_paged(q, k_cache, v_cache, block_tables, seq_lens[seq_id]);
    }
}
```

**Scheduling Mixed Batches**:

```python
# vllm/core/scheduler.py

def _schedule_mixed_batch(self, budget: SchedulingBudget):
    """
    Schedule mixed prefill and decode requests.

    Strategy:
    1. Prioritize running decode (low latency)
    2. Fill remaining budget with prefill
    3. Respect token budget limit
    """

    scheduled = []
    num_prefill = 0
    num_decode = 0

    # 1. Schedule running (decode)
    for seq_group in self.running:
        decode_tokens = seq_group.num_seqs()  # 1 per sequence
        budget.add_num_batched_tokens(decode_tokens)
        scheduled.append(seq_group)
        num_decode += 1

    # 2. Schedule waiting (prefill)
    for seq_group in self.waiting[:]:
        prefill_tokens = seq_group.get_seqs()[0].get_len()

        # Check budget
        if not budget.can_add_tokens(prefill_tokens):
            break  # Would exceed token budget

        # Check memory
        if not self._can_allocate(seq_group):
            break

        # Schedule
        self.waiting.remove(seq_group)
        self.running.append(seq_group)
        scheduled.append(seq_group)

        budget.add_num_batched_tokens(prefill_tokens)
        num_prefill += 1

    return SchedulerOutputs(
        scheduled_seq_groups=scheduled,
        num_prefill_groups=num_prefill,
        num_decode_groups=num_decode,
    )
```

**Performance Considerations**:

```
Mixed Batch Trade-offs:

Advantages:
✓ Better GPU utilization (always processing)
✓ Lower latency for decode (not blocked by prefill)
✓ Higher throughput (no idle time)

Challenges:
✗ Kernel complexity (handle both paths)
✗ Load imbalance (prefill threads busy, decode idle)
✗ Tuning required (token budget, batch limits)

Optimal ratios:
- Latency-critical: Limit prefill tokens (< 20% of batch)
- Throughput-focused: Allow more prefill (50-70% of batch)
```

### Task 3: Chunked Prefill (60 min)

**Problem: Long Prompts**

```
Scenario: 10,000 token prompt
- Standard prefill: Process all 10,000 tokens in one step
- Memory: 10,000² attention matrix (~400MB for one head)
- Time: Very long single step
- Impact: Blocks all other requests during prefill

Solution: Chunked Prefill
- Split into chunks: [chunk1: 0-512, chunk2: 512-1024, ...]
- Process incrementally over multiple steps
- Other requests can interleave
```

**Chunked Prefill Implementation**:

```python
class ChunkedPrefillScheduler:
    """
    Split large prefills into chunks.
    """

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def _schedule_chunked_prefill(self, seq_group, budget):
        """
        Schedule one chunk of prefill.
        """

        seq = seq_group.get_seqs()[0]

        # Determine chunk to process
        total_tokens = seq.get_len()
        processed_tokens = seq.data.get_num_computed_tokens()
        remaining = total_tokens - processed_tokens

        # Chunk size for this step
        chunk_tokens = min(remaining, self.chunk_size)

        # Check budget
        if not budget.can_add_tokens(chunk_tokens):
            return False  # Can't fit in this step

        # Schedule chunk
        seq.data.set_num_computed_tokens(processed_tokens + chunk_tokens)
        budget.add_num_batched_tokens(chunk_tokens)

        # Is prefill complete?
        if processed_tokens + chunk_tokens >= total_tokens:
            seq_group.mark_prefill_complete()

        return True
```

**Attention with Chunked Prefill**:

```
Chunk 1 (tokens 0-511):
  Compute: Self-attention over tokens 0-511
  Write: K/V cache for tokens 0-511
  Output: Hidden state for token 511

Chunk 2 (tokens 512-1023):
  Compute: Attention for tokens 512-1023
    - Self-attention within chunk
    - Cross-attention to cached K/V (0-511)
  Write: K/V cache for tokens 512-1023
  Output: Hidden state for token 1023

Chunk 3 (tokens 1024-1535):
  ...similar...

Final chunk: Transition to decode mode
```

**Benefits vs. Costs**:

```
Benefits:
✓ Bounded step latency (predictable)
✓ Can interleave with decode
✓ Better for long prompts (10K+ tokens)
✓ Fairer resource allocation

Costs:
✗ More steps to complete prefill
✗ Higher time-to-first-token (TTFT)
✗ Overhead of multiple kernel launches
✗ Complexity in tracking progress

When to use:
- Long prompts (> 2K tokens)
- Multi-tenant with QoS requirements
- Latency-sensitive workloads
```

**📝 Exercise: Chunked vs. Non-Chunked Comparison**

```python
#!/usr/bin/env python3
"""
Compare chunked vs non-chunked prefill performance.
"""

def simulate_prefill(prompt_tokens: int, chunk_size: int = None):
    """
    Simulate prefill strategies.
    """

    if chunk_size is None:
        # Non-chunked: all at once
        print(f"\nNon-Chunked Prefill: {prompt_tokens} tokens")
        print(f"  Step 1: Process {prompt_tokens} tokens")
        print(f"  Total steps: 1")
        print(f"  TTFT: 1 step")
        return 1

    else:
        # Chunked: split into chunks
        print(f"\nChunked Prefill: {prompt_tokens} tokens (chunk_size={chunk_size})")

        steps = 0
        processed = 0

        while processed < prompt_tokens:
            chunk = min(chunk_size, prompt_tokens - processed)
            steps += 1
            processed += chunk
            print(f"  Step {steps}: Process tokens {processed-chunk} to {processed}")

        print(f"  Total steps: {steps}")
        print(f"  TTFT: {steps} steps")
        return steps

# Test cases
simulate_prefill(2000, chunk_size=None)   # Non-chunked
simulate_prefill(2000, chunk_size=512)    # Chunked

# Output:
# Non-Chunked: 1 step (but blocks others)
# Chunked: 4 steps (allows interleaving)
```

---

## 🔬 Afternoon: Performance Optimization (14:00-17:30)

### Task 4: Batch Size Tuning (60 min)

**Understanding Token Budget**

```python
# vllm/config.py

class SchedulerConfig:
    max_num_batched_tokens: int = 2048  # Max tokens per batch
    max_num_seqs: int = 256              # Max sequences per batch
    max_model_len: int = 4096            # Max sequence length

# These parameters control batch composition
```

**Trade-offs in Batch Size**:

```
Small Batch (e.g., max_tokens=512):
  Latency:    Low ✓ (fast iterations)
  Throughput: Low ✗ (underutilized GPU)
  Use case:   Interactive, latency-critical

Medium Batch (e.g., max_tokens=2048):
  Latency:    Medium
  Throughput: High ✓
  Use case:   Balanced workloads

Large Batch (e.g., max_tokens=8192):
  Latency:    High ✗ (long iterations)
  Throughput: Very High ✓
  Use case:   Batch processing, throughput-focused
```

**Finding Optimal Batch Size**:

```python
#!/usr/bin/env python3
"""
Batch size tuning experiment.
"""

import time
from vllm import LLM, SamplingParams

def benchmark_batch_size(max_tokens: int, num_requests: int = 100):
    """Benchmark with specific batch size."""

    print(f"\n{'='*60}")
    print(f"Testing max_num_batched_tokens = {max_tokens}")
    print(f"{'='*60}")

    # Create LLM with specific config
    llm = LLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=max_tokens,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
    )

    # Prepare requests
    prompts = ["Generate text about AI: "] * num_requests
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )

    # Benchmark
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    duration = time.time() - start

    # Calculate metrics
    throughput = num_requests / duration  # requests/second
    avg_latency = duration / num_requests  # seconds/request

    print(f"  Throughput: {throughput:.2f} req/s")
    print(f"  Avg Latency: {avg_latency:.3f} s")
    print(f"  Total Time: {duration:.2f} s")

    return {
        'max_tokens': max_tokens,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'total_time': duration,
    }

# Run experiments
results = []
for batch_size in [512, 1024, 2048, 4096, 8192]:
    result = benchmark_batch_size(batch_size)
    results.append(result)

# Find optimal
best_throughput = max(results, key=lambda x: x['throughput'])
best_latency = min(results, key=lambda x: x['avg_latency'])

print(f"\n{'='*60}")
print(f"Results Summary")
print(f"{'='*60}")
print(f"Best Throughput: {best_throughput['max_tokens']} tokens")
print(f"  → {best_throughput['throughput']:.2f} req/s")
print(f"\nBest Latency: {best_latency['max_tokens']} tokens")
print(f"  → {best_latency['avg_latency']:.3f} s/req")
```

### Task 5: Profiling Batch Performance (60 min)

**Using Nsight Systems to Profile Batching**:

```bash
# Profile vLLM with different batch sizes
nsys profile \
    --trace=cuda,nvtx \
    --output=batch_profile_2048.qdrep \
    python benchmark_serving.py \
    --max-num-batched-tokens 2048

# View in Nsight Systems GUI
nsys-ui batch_profile_2048.qdrep
```

**What to Look For**:

```
Timeline Analysis:
1. Kernel Gaps: Time between kernel launches
   - Small gaps: Good utilization
   - Large gaps: CPU overhead, scheduling issues

2. Kernel Duration: Time per attention kernel
   - Grows with batch size
   - Should saturate GPU (high SM utilization)

3. Memory Operations: D2D copies, cache ops
   - Should be minimal
   - Large copies indicate inefficiency

4. NVTX Markers: Custom ranges
   - "Schedule" - scheduler time
   - "Execute" - model execution
   - "Sample" - token sampling

Ideal profile:
┌────────────────────────────────────┐
│ Schedule | Execute | Sample | Sched│ ...
│   (CPU)  │ (GPU)   │ (CPU)  │      │
└────────────────────────────────────┘
           ▲
           No gaps here!
```

**Batch Composition Analysis**:

```python
#!/usr/bin/env python3
"""
Analyze batch composition over time.
"""

class BatchCompositionTracker:
    """Track batch composition during serving."""

    def __init__(self):
        self.step_logs = []

    def log_step(self, scheduler_outputs):
        """Log batch composition for this step."""

        prefill_tokens = 0
        decode_tokens = 0

        for seq_group in scheduler_outputs.scheduled_seq_groups:
            if seq_group.is_prefill():
                prefill_tokens += seq_group.num_tokens()
            else:
                decode_tokens += seq_group.num_seqs()

        self.step_logs.append({
            'step': len(self.step_logs),
            'prefill_tokens': prefill_tokens,
            'decode_tokens': decode_tokens,
            'total_tokens': prefill_tokens + decode_tokens,
            'num_seqs': len(scheduler_outputs.scheduled_seq_groups),
        })

    def analyze(self):
        """Analyze batch composition."""

        import numpy as np

        prefill = [log['prefill_tokens'] for log in self.step_logs]
        decode = [log['decode_tokens'] for log in self.step_logs]
        total = [log['total_tokens'] for log in self.step_logs]

        print(f"\n{'='*60}")
        print(f"Batch Composition Analysis")
        print(f"{'='*60}")
        print(f"Total steps: {len(self.step_logs)}")
        print(f"\nToken distribution:")
        print(f"  Prefill - Avg: {np.mean(prefill):.1f}, Max: {max(prefill)}, Min: {min(prefill)}")
        print(f"  Decode  - Avg: {np.mean(decode):.1f}, Max: {max(decode)}, Min: {min(decode)}")
        print(f"  Total   - Avg: {np.mean(total):.1f}, Max: {max(total)}, Min: {min(total)}")
        print(f"\nPrefill ratio: {sum(prefill) / sum(total) * 100:.1f}%")
        print(f"Decode ratio:  {sum(decode) / sum(total) * 100:.1f}%")

        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.plot(prefill, label='Prefill', color='blue')
            plt.plot(decode, label='Decode', color='orange')
            plt.xlabel('Step')
            plt.ylabel('Tokens')
            plt.legend()
            plt.title('Batch Composition Over Time')

            plt.subplot(2, 1, 2)
            plt.plot(total, label='Total', color='green')
            plt.axhline(y=np.mean(total), color='r', linestyle='--', label='Average')
            plt.xlabel('Step')
            plt.ylabel('Total Tokens')
            plt.legend()

            plt.tight_layout()
            plt.savefig('batch_composition.png')
            print(f"\n✓ Saved plot to batch_composition.png")

        except ImportError:
            print(f"\n(Install matplotlib for visualization)")
```

### Task 6: Optimization Experiments (60 min)

**Experiment 1: Prefill/Decode Ratio Tuning**

```python
# Test different prefill limits
configs = [
    {'max_prefill_tokens': 512, 'name': 'Low Prefill'},
    {'max_prefill_tokens': 2048, 'name': 'Medium Prefill'},
    {'max_prefill_tokens': 8192, 'name': 'High Prefill'},
]

for config in configs:
    # Run benchmark with config
    # Measure: throughput, p50/p99 latency, TTFT
    pass
```

**Experiment 2: Chunked Prefill Impact**

```python
# Compare chunked vs non-chunked for long prompts
test_cases = [
    {'prompt_len': 1000, 'chunk_size': None},
    {'prompt_len': 1000, 'chunk_size': 256},
    {'prompt_len': 5000, 'chunk_size': None},
    {'prompt_len': 5000, 'chunk_size': 512},
]
```

**Experiment 3: Load-Dependent Scheduling**

```python
class AdaptiveScheduler:
    """
    Adjust scheduling policy based on load.
    """

    def adjust_policy(self, current_load):
        """Adapt to current load."""

        if current_load < 0.3:
            # Low load: optimize for latency
            self.max_num_batched_tokens = 512
            self.enable_chunked_prefill = False

        elif current_load < 0.7:
            # Medium load: balanced
            self.max_num_batched_tokens = 2048
            self.enable_chunked_prefill = True

        else:
            # High load: optimize for throughput
            self.max_num_batched_tokens = 4096
            self.enable_chunked_prefill = True
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Advanced Batching**
- Mixed prefill/decode composition
- Chunked prefill for long prompts
- Token budget management

✅ **Performance Tuning**
- Batch size optimization
- Latency vs. throughput trade-offs
- Load-dependent strategies

✅ **Analysis Skills**
- Profiling batch performance
- Composition tracking
- Experimental methodology

### Knowledge Check Questions

**Q1**: Why can prefill and decode be batched together?
<details>
<summary>Answer</summary>
Both process tokens through the same model, just with different attention patterns. The model forward pass handles both:
- Prefill: Self-attention over prompt tokens
- Decode: Attention over single query + cached K/V

The batch dimension allows processing multiple requests simultaneously, with per-request flags indicating prefill vs. decode.
</details>

**Q2**: What is the main trade-off in chunked prefill?
<details>
<summary>Answer</summary>
**Trade-off**: Time-to-first-token (TTFT) vs. fairness/predictability

- Non-chunked: Fast TTFT (1 step) but blocks other requests
- Chunked: Slower TTFT (multiple steps) but allows interleaving

Choose based on use case:
- Interactive: Prefer non-chunked (fast TTFT)
- Multi-tenant: Prefer chunked (fairness)
</details>

**Q3**: Calculate total tokens for this batch (block_size=16):
- 2 prefill requests: 256 and 512 tokens
- 3 decode requests: 100, 200, 50 tokens each
<details>
<summary>Answer</summary>
Prefill: 256 + 512 = 768 tokens (process all)
Decode: 1 + 1 + 1 = 3 tokens (1 per request)
**Total: 771 tokens in batch**
</details>

**Q4**: How does batch size affect GPU utilization?
<details>
<summary>Answer</summary>
- **Too small**: Underutilizes GPU (low occupancy, wasted compute)
- **Too large**: Longer iterations (higher latency per step)
- **Optimal**: Saturates GPU while meeting latency requirements

Find via profiling: Monitor SM utilization (target 80-90%)
</details>

---

## 🚀 Preview: Day 17

Tomorrow you'll explore:
- **Model Executor Architecture**: How models are loaded and executed
- **Weight Management**: Quantization, sharding, loading strategies
- **Forward Pass Pipeline**: From inputs to logits
- **Integration Points**: Executor ↔ Scheduler ↔ Attention

**Preparation**:
- Review PyTorch model loading
- Understand tensor operations
- Familiarize with vLLM model implementations

---

## 📚 Additional Resources

**Key Files**:
- [ ] `vllm/core/scheduler.py` - Mixed batch scheduling
- [ ] `vllm/config.py` - Scheduler configuration
- [ ] `benchmarks/benchmark_serving.py` - Performance testing

**Concepts**:
- [ ] Compute vs. memory-bound workloads
- [ ] GPU occupancy and utilization
- [ ] Load balancing strategies

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
