# Day 27: Advanced Problem Solving & Code Review Practice

> **Goal**: Master debugging scenarios, code review, and technical problem-solving
> **Time**: 6-8 hours
> **Prerequisites**: Days 1-26 completed, mock interview practice
> **Deliverables**: Solved problems, code reviews, debugging reports

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Debugging & Performance Problems

**9:00-10:00** - Performance Debugging Scenario 1
**10:00-10:15** - Break
**10:15-11:15** - Performance Debugging Scenario 2
**11:15-12:30** - Memory Leak Investigation

### Afternoon Session (3-4 hours): Code Review & Algorithm Problems

**14:00-15:00** - vLLM Code Review Exercise
**15:00-16:00** - Algorithm Problems (Scheduling, Allocation)
**16:00-16:15** - Break
**16:15-17:00** - Design Pattern Analysis
**17:00-18:00** - Technical Presentation Practice

### Evening (Optional, 1-2 hours): Final Prep

**19:00-21:00** - Compile interview cheat sheet, practice elevator pitches

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Debug performance issues systematically
- [ ] Review and critique CUDA kernel code
- [ ] Solve scheduling and allocation algorithms
- [ ] Identify design patterns and anti-patterns
- [ ] Present technical solutions clearly
- [ ] Handle unexpected questions gracefully

---

## 📚 Morning: Debugging Scenarios (9:00-12:30)

### Debugging Problem 1: Throughput Degradation (60 min)

**Scenario**:
> "Your vLLM deployment was serving 5,000 tokens/sec consistently. After adding a new model variant, throughput dropped to 2,000 tokens/sec. The team is confused because the new model has the same architecture (Llama-2-7B). Debug the issue."

**Your Debugging Process**:

```
Step 1: Gather Information (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Questions to Ask:
Q: What changed exactly?
A: Deployed new Llama-2-7B fine-tuned model

Q: Same hardware?
A: Yes, same A100 GPU

Q: Same configuration?
A: "Should be the same"

Q: Any error messages?
A: No errors, just slower

Q: Both models serving simultaneously?
A: No, replaced old with new

Initial Hypothesis:
  - Configuration mismatch?
  - Model format issue?
  - Quantization difference?

Step 2: Compare Configurations (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Old Model Config:
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)
# Throughput: 5,000 tokens/sec
```

New Model Config (user provided):
```python
llm = LLM(
    model="path/to/finetuned-llama",
    dtype="auto",  # ← SUSPICIOUS!
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)
# Throughput: 2,000 tokens/sec
```

Observation: dtype="auto" vs dtype="float16"

Step 3: Investigate dtype (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Check actual model precision:
```bash
# Inspect model config
cat path/to/finetuned-llama/config.json

{
  "torch_dtype": "float32",  # ← FOUND IT!
  ...
}
```

Root Cause:
  - Fine-tuned model saved in FP32
  - dtype="auto" loads as FP32
  - FP32 is 2x slower than FP16 on A100!

Step 4: Verify Hypothesis (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test with explicit FP16:
```python
llm = LLM(
    model="path/to/finetuned-llama",
    dtype="float16",  # Force FP16
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)
# Throughput: 4,900 tokens/sec ✅
```

Confirmed! FP32 → FP16 fix restored performance.

Step 5: Analysis & Recommendations (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why This Happened:
  1. Training saved in FP32 (default PyTorch)
  2. dtype="auto" trusted model config
  3. No performance monitoring caught it
  4. 2.5x throughput loss!

Recommendations:
  1. Always explicitly set dtype in production
  2. Convert models to FP16 before deployment
  3. Add performance regression tests
  4. Monitor tokens/sec metrics with alerts
  5. Document configuration standards

Lesson for Interview:
  "I approach debugging systematically:
   1. Gather info and form hypotheses
   2. Check what changed
   3. Compare configurations
   4. Test hypotheses
   5. Verify fix
   6. Document and prevent recurrence"

Step 6: Follow-up Questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How would you prevent this in the future?"
A: "Add CI/CD checks:
   1. Model validation script checks dtype
   2. Performance benchmark on deployment
   3. Alert if throughput < 80% of baseline
   4. Require explicit dtype in config
   5. Document model preparation process"

Q: "What if FP16 caused accuracy issues?"
A: "Two approaches:
   1. Quantization-aware training (keep FP32 training, deploy FP16)
   2. Mixed precision (FP32 for sensitive layers, FP16 elsewhere)
   3. Use FP32 only if accuracy loss unacceptable
   4. Trade-off: 2.5x slower but accurate"

Q: "How do you monitor performance in production?"
A: "Key metrics:
   - Tokens/sec (absolute and per-model)
   - P50/P95/P99 latency
   - GPU utilization (should be >80%)
   - Queue depth
   - Request success rate

   Alerts:
   - Throughput <80% of baseline → P2
   - P95 latency >2x normal → P1
   - GPU util <60% → Investigate
   - Error rate >1% → P1"
```

### Debugging Problem 2: Memory Leak (60 min)

**Scenario**:
> "Your vLLM server runs fine for a few hours, then crashes with OOM. Memory usage steadily increases over time. Find and fix the leak."

**Your Approach**:

```
Step 1: Reproduce the Issue (15 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Setup monitoring:
```python
import torch
import time
import matplotlib.pyplot as plt

def monitor_memory():
    """Track GPU memory over time"""
    timestamps = []
    memory_used = []

    for i in range(1000):
        # Run inference
        outputs = llm.generate([prompt], sampling_params)

        # Record memory
        mem = torch.cuda.memory_allocated() / 1e9
        timestamps.append(time.time())
        memory_used.append(mem)

        if i % 100 == 0:
            print(f"Iteration {i}: {mem:.2f} GB")

    # Plot
    plt.plot(timestamps, memory_used)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (GB)')
    plt.title('Memory Usage Over Time')
    plt.savefig('memory_leak.png')

monitor_memory()
```

Observation:
```
Iteration 0: 14.2 GB
Iteration 100: 14.8 GB  (+0.6 GB)
Iteration 200: 15.4 GB  (+0.6 GB)
Iteration 300: 16.0 GB  (+0.6 GB)
...
Iteration 800: 19.0 GB
Iteration 900: OOM!
```

Clear leak: ~0.6 GB per 100 iterations = 6 MB per iteration

Step 2: Hypothesize Sources (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Potential Leak Sources:
  1. KV cache blocks not freed
  2. Cached tensors accumulating
  3. Python objects keeping CUDA tensors alive
  4. Intermediate tensors not released

Most likely: KV cache block leak

Step 3: Investigate Block Management (15 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add debugging:
```python
class BlockManagerWithLogging:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocation_history = []

    def allocate(self, seq_id, num_blocks):
        blocks = super().allocate(seq_id, num_blocks)
        self.allocation_history.append({
            'time': time.time(),
            'action': 'allocate',
            'seq_id': seq_id,
            'blocks': len(blocks),
            'free_blocks': len(self.free_blocks),
        })
        return blocks

    def free(self, seq_id):
        self.allocation_history.append({
            'time': time.time(),
            'action': 'free',
            'seq_id': seq_id,
            'free_blocks_before': len(self.free_blocks),
        })
        super().free(seq_id)
        self.allocation_history[-1]['free_blocks_after'] = len(self.free_blocks)

    def analyze_leaks(self):
        """Find sequences that allocated but never freed"""
        allocated = set()
        freed = set()

        for event in self.allocation_history:
            if event['action'] == 'allocate':
                allocated.add(event['seq_id'])
            elif event['action'] == 'free':
                freed.add(event['seq_id'])

        leaked = allocated - freed
        print(f"Leaked sequences: {leaked}")
        print(f"Leaked count: {len(leaked)}")
```

Run and analyze:
```python
# After 100 iterations
block_manager.analyze_leaks()
# Output:
# Leaked sequences: {1, 5, 12, 23, 31, ...}
# Leaked count: 8
```

Found it! Some sequences not being freed.

Step 4: Root Cause Analysis (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Check sequence lifecycle:
```python
# In scheduler.py
def _process_finished_sequences(self, outputs):
    """Process finished sequences"""
    for output in outputs:
        if output.finished:
            seq_group = self.running.get(output.seq_group_id)
            # BUG: What if seq_group is None?
            self.block_manager.free(seq_group.seq_id)
```

Bug Found:
  - If sequence already removed from running queue
  - seq_group will be None
  - Free is never called!
  - Blocks leak!

When does this happen?
  - Preemption edge case
  - Sequence finishes immediately after being preempted
  - Already removed from running, but now finished

Step 5: Fix & Verify (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fixed code:
```python
def _process_finished_sequences(self, outputs):
    """Process finished sequences - FIXED"""
    for output in outputs:
        if output.finished:
            # Track all sequences, not just in running queue
            seq_id = output.seq_id

            # Free blocks regardless of current state
            if self.block_manager.has_blocks(seq_id):
                self.block_manager.free(seq_id)

            # Remove from all queues
            self.running.pop(seq_id, None)
            self.swapped.pop(seq_id, None)
            self.waiting.pop(seq_id, None)
```

Verify fix:
```
Iteration 0: 14.2 GB
Iteration 100: 14.2 GB  (stable!)
Iteration 200: 14.2 GB  (stable!)
Iteration 300: 14.2 GB  (stable!)
...
Iteration 1000: 14.2 GB  ✅ Fixed!
```

Step 6: Prevention & Testing (10 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add test:
```python
def test_no_block_leak_on_preemption():
    """Test that blocks are freed even if sequence preempted"""

    # Setup
    scheduler = Scheduler(...)
    initial_free_blocks = scheduler.block_manager.get_num_free_blocks()

    # Create sequence
    seq = create_sequence()
    scheduler.add_request(seq)

    # Run one step (allocates blocks)
    scheduler.schedule()

    # Preempt the sequence
    scheduler.preempt(seq.seq_id)

    # Mark as finished (the edge case)
    seq.finish()
    scheduler._process_finished_sequences([seq])

    # Verify blocks freed
    final_free_blocks = scheduler.block_manager.get_num_free_blocks()
    assert final_free_blocks == initial_free_blocks, \
        f"Block leak! Initial: {initial_free_blocks}, Final: {final_free_blocks}"
```

Interview Takeaway:
"When debugging memory leaks in CUDA/GPU code:
 1. Monitor memory usage over time
 2. Identify growth pattern (linear, exponential)
 3. Hypothesize sources (KV cache, tensors, etc.)
 4. Add instrumentation to track allocations
 5. Find allocation without matching deallocation
 6. Analyze root cause (often edge cases)
 7. Fix and verify
 8. Add regression test

Key insight: Memory leaks often occur in edge cases like
preemption, errors, or unusual state transitions."
```

### Debugging Problem 3: Sudden Latency Spike (30 min)

**Quick Problem**: P95 latency suddenly jumped from 200ms to 2000ms. Debug quickly.

**Solution Framework**:

```
1. Check recent changes (5 min)
   - Code deployment?
   - Configuration change?
   - Traffic pattern change?

2. Check monitoring (5 min)
   - GPU utilization (if low → problem)
   - Queue depth (if high → overload)
   - Error rate (if high → failures)
   - Batch size (if small → inefficient)

3. Profile one slow request (10 min)
   - Add NVTX markers
   - Profile with nsys
   - Find bottleneck:
     - Prefill phase slow → compute issue
     - Decode phase slow → memory issue
     - Scheduling slow → Python overhead

4. Common causes (10 min)
   - Bad batch composition (many long sequences)
   - Memory fragmentation (need restart)
   - Thermal throttling (check temps)
   - NVLink failure (multi-GPU)
   - Disk I/O (model loading)
```

---

## 💻 Afternoon: Code Review & Algorithms (14:00-18:00)

### Code Review Exercise: Attention Kernel (60 min)

**Task**: Review this PagedAttention kernel implementation. Find bugs, suggest improvements.

```cuda
// Review this kernel
__global__ void paged_attention_kernel(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_table,
    const int* seq_lengths,
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    // Load query
    float q[128];  // Assume head_dim <= 128
    for (int i = 0; i < head_dim; i++) {
        int idx = seq_idx * num_heads * head_dim +
                  head_idx * head_dim + i;
        q[i] = query[idx];
    }

    int seq_len = seq_lengths[seq_idx];
    int num_blocks = seq_len / block_size;

    // Compute attention
    float max_score = -INFINITY;
    float scores[2048];  // Assume max seq_len = 2048

    // Score computation
    for (int block = 0; block < num_blocks; block++) {
        int phys_block = block_table[seq_idx * num_blocks + block];

        for (int i = 0; i < block_size; i++) {
            int token_idx = block * block_size + i;

            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int k_idx = phys_block * block_size * num_heads * head_dim +
                           i * num_heads * head_dim +
                           head_idx * head_dim + d;
                score += q[d] * key_cache[k_idx];
            }

            scores[token_idx] = score;
            max_score = fmaxf(max_score, score);
        }
    }

    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    // Output
    for (int d = 0; d < head_dim; d++) {
        float out = 0.0f;
        for (int block = 0; block < num_blocks; block++) {
            int phys_block = block_table[seq_idx * num_blocks + block];

            for (int i = 0; i < block_size; i++) {
                int token_idx = block * block_size + i;
                int v_idx = phys_block * block_size * num_heads * head_dim +
                           i * num_heads * head_dim +
                           head_idx * head_dim + d;

                out += scores[token_idx] * value_cache[v_idx];
            }
        }
        int out_idx = seq_idx * num_heads * head_dim +
                     head_idx * head_dim + d;
        output[out_idx] = out;
    }
}
```

**Your Code Review**:

```
ISSUES FOUND:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 CRITICAL BUG #1: Incorrect num_blocks calculation
Line: int num_blocks = seq_len / block_size;

Problem: Integer division truncates!
  Example: seq_len=42, block_size=16
  Calculated: 42/16 = 2 blocks
  Correct: ceil(42/16) = 3 blocks
  Result: Last 10 tokens ignored!

Fix:
  int num_blocks = (seq_len + block_size - 1) / block_size;

🔴 CRITICAL BUG #2: Buffer overflow in scores array
Line: float scores[2048];

Problem: Stack allocation with fixed size
  - If seq_len > 2048 → buffer overflow
  - Stack space limited (~1KB)
  - Will crash or corrupt memory

Fix: Use shared memory or dynamic allocation
  __shared__ float scores[MAX_SEQ_LEN];

🔴 CRITICAL BUG #3: Partial last block not handled
Problem: Code assumes all blocks are full
  - Last block may have fewer tokens
  - Will read garbage data beyond sequence

Fix: Check token_idx < seq_len in loops

⚠️ PERFORMANCE ISSUE #1: No parallelism within block
Problem: Only one thread per (seq, head)
  - Low GPU utilization
  - Not leveraging warp parallelism

Fix: Use multiple threads per block, reduce with warps

⚠️ PERFORMANCE ISSUE #2: Shared memory not used
Problem: Query loaded into registers (128 floats)
  - High register pressure
  - May reduce occupancy

Fix: Load query into shared memory

⚠️ PERFORMANCE ISSUE #3: Redundant calculations
Problem: Computing indices repeatedly
  - phys_block lookup in two places
  - Index calculations duplicated

Fix: Hoist invariant calculations

CORRECTED VERSION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__global__ void paged_attention_kernel_fixed(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_table,
    const int* seq_lengths,
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int max_num_blocks
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;  // Use threads within block

    // Shared memory for query and scores
    __shared__ float s_query[128];
    __shared__ float s_scores[2048];

    // Load query (coalesced)
    if (tid < head_dim) {
        int idx = seq_idx * num_heads * head_dim +
                  head_idx * head_dim + tid;
        s_query[tid] = query[idx];
    }
    __syncthreads();

    int seq_len = seq_lengths[seq_idx];
    int num_blocks = (seq_len + block_size - 1) / block_size;  // FIX #1

    // Compute scores (parallel)
    float local_max = -INFINITY;

    for (int block = 0; block < num_blocks; block++) {
        int phys_block = block_table[seq_idx * max_num_blocks + block];

        for (int i = tid; i < block_size; i += blockDim.x) {
            int token_idx = block * block_size + i;

            // FIX #3: Check bounds
            if (token_idx >= seq_len) continue;

            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int k_idx = phys_block * block_size * num_heads * head_dim +
                           i * num_heads * head_dim +
                           head_idx * head_dim + d;
                score += s_query[d] * key_cache[k_idx];
            }

            s_scores[token_idx] = score;
            local_max = fmaxf(local_max, score);
        }
    }

    // Reduce max across threads
    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();

    // Softmax (parallel)
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float exp_score = expf(s_scores[i] - s_max);
        s_scores[i] = exp_score;
        local_sum += exp_score;
    }

    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = local_sum;
    __syncthreads();

    // Compute output (parallel over dimensions)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out = 0.0f;

        for (int block = 0; block < num_blocks; block++) {
            int phys_block = block_table[seq_idx * max_num_blocks + block];

            for (int i = 0; i < block_size; i++) {
                int token_idx = block * block_size + i;
                if (token_idx >= seq_len) break;  // FIX #3

                int v_idx = phys_block * block_size * num_heads * head_dim +
                           i * num_heads * head_dim +
                           head_idx * head_dim + d;

                out += (s_scores[token_idx] / s_sum) * value_cache[v_idx];
            }
        }

        int out_idx = seq_idx * num_heads * head_dim +
                     head_idx * head_dim + d;
        output[out_idx] = out;
    }
}

SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fixed:
  ✅ Correct block count calculation
  ✅ Safe memory allocation
  ✅ Handle partial last block
  ✅ Parallel computation
  ✅ Use shared memory
  ✅ Reduce redundant work

Performance improvement: ~5-8x
Correctness: All edge cases handled
```

### Algorithm Problem: Smart Scheduling (60 min)

**Problem**:
> "Design an algorithm to schedule LLM requests to minimize average latency while maintaining throughput. You have requests with different priorities and lengths. Explain your approach."

**Your Solution**:

```python
"""
Smart Scheduler Algorithm Design

Goal: Minimize average latency while maintaining throughput
Constraints:
  - Different priorities (Premium, Standard, Budget)
  - Different sequence lengths
  - Maximum batch size limit
"""

class SmartScheduler:
    def __init__(self, max_batch_tokens=4096):
        self.max_batch_tokens = max_batch_tokens
        self.queues = {
            'premium': PriorityQueue(),   # Priority 0 (highest)
            'standard': PriorityQueue(),  # Priority 1
            'budget': PriorityQueue(),    # Priority 2
        }
        self.running = []

    def schedule_next_batch(self):
        """
        Scheduling Algorithm:
        1. Shortest-Job-First within each priority
        2. Fill batch greedily starting from highest priority
        3. Mix short and long sequences for better packing
        """

        batch = []
        batch_tokens = 0

        # Strategy: Pack short requests first (SJF)
        # This minimizes average latency

        # Step 1: Get all pending requests
        all_requests = []
        for priority, queue in self.queues.items():
            while not queue.empty():
                req = queue.get()
                all_requests.append((req, priority))

        # Step 2: Sort by (priority, expected_tokens)
        # Shorter jobs first within each priority
        all_requests.sort(key=lambda x: (
            x[1],  # Priority first
            x[0].expected_total_tokens  # Then length (SJF)
        ))

        # Step 3: Pack batch using bin packing
        for req, priority in all_requests:
            estimated_tokens = self._estimate_tokens(req)

            if batch_tokens + estimated_tokens <= self.max_batch_tokens:
                batch.append(req)
                batch_tokens += estimated_tokens
            else:
                # Put back in queue
                self.queues[priority].put(req)

        return batch

    def _estimate_tokens(self, req):
        """Estimate tokens needed for this request"""
        if req.is_prefill:
            # Prefill: all prompt tokens
            return len(req.prompt_tokens)
        else:
            # Decode: 1 token per request
            return 1

    def optimize_for_latency(self):
        """
        Optimization: Shortest Remaining Time First (SRTF)

        Idea: Always schedule request closest to completion
        This minimizes waiting time for others
        """

        pending = self._get_all_pending()

        # Sort by tokens remaining (not total tokens)
        pending.sort(key=lambda r: r.tokens_remaining)

        batch = []
        batch_tokens = 0

        for req in pending:
            if batch_tokens + req.next_tokens <= self.max_batch_tokens:
                batch.append(req)
                batch_tokens += req.next_tokens

        return batch

    def optimize_for_throughput(self):
        """
        Optimization: Largest Batch First

        Idea: Pack as many requests as possible
        Better GPU utilization → higher throughput
        """

        pending = self._get_all_pending()

        # Sort by size (smallest first for better packing)
        pending.sort(key=lambda r: r.next_tokens)

        batch = []
        batch_tokens = 0

        for req in pending:
            if batch_tokens + req.next_tokens <= self.max_batch_tokens:
                batch.append(req)
                batch_tokens += req.next_tokens

        return batch

    def hybrid_scheduler(self, latency_weight=0.5):
        """
        Hybrid: Balance latency and throughput

        Score = latency_weight * (1/remaining_tokens) +
                (1 - latency_weight) * throughput_contribution
        """

        pending = self._get_all_pending()

        # Compute scores
        for req in pending:
            latency_score = 1.0 / req.tokens_remaining  # Higher = finishing soon
            throughput_score = 1.0 / req.next_tokens     # Higher = cheap to process

            req.score = (latency_weight * latency_score +
                        (1 - latency_weight) * throughput_score)

        # Sort by score (highest first)
        pending.sort(key=lambda r: r.score, reverse=True)

        # Pack batch
        batch = []
        batch_tokens = 0

        for req in pending:
            if batch_tokens + req.next_tokens <= self.max_batch_tokens:
                batch.append(req)
                batch_tokens += req.next_tokens

        return batch

# Analysis of Different Strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
1. FCFS (First-Come-First-Served):
   + Simple, fair
   - High average latency (long jobs block short ones)
   - Poor throughput (bad packing)

2. SJF (Shortest Job First):
   + Minimizes average latency (proven optimal)
   + Good packing (small jobs easier to fit)
   - May starve long jobs
   - Need to estimate job length

3. SRTF (Shortest Remaining Time First):
   + Even better average latency than SJF
   + Prioritizes finishing jobs
   - Risk of starvation
   - More complex

4. Priority-based:
   + Meets SLA requirements
   + Fair to paying customers
   - May sacrifice average latency
   - Requires priority assignment

5. Hybrid (Recommended):
   + Balance latency and throughput
   + Configurable trade-off
   + Handles priorities
   - More complex implementation

Recommendation for LLM Serving:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use Hybrid with:
  - Priority enforcement (SLA guarantees)
  - SJF within priority (minimize latency)
  - Continuous batching (maximize throughput)
  - Preemption for low priority (resource efficiency)

Pseudocode:
  1. Separate queues by priority
  2. Within each priority, sort by remaining tokens (SRTF)
  3. Fill batch from high to low priority
  4. Preempt budget if premium SLA at risk
  5. Track metrics and adjust dynamically
"""
```

**Interview Discussion Points**:

```
Q: "Why Shortest Job First?"
A: "It's mathematically proven to minimize average waiting time.
   Intuition: If you have jobs of 1min and 10min, running 1min
   first means both jobs have total wait of 11min (1+10).
   Running 10min first means total wait of 21min (10+11).
   SJF gives better average."

Q: "How do you handle starvation?"
A: "Use aging: Increase priority over time.
   Example: Budget request waiting >5min → promote to Standard
   This ensures all requests eventually complete."

Q: "What about fairness?"
A: "Define fairness metric:
   - Fair queuing: Each customer gets proportional share
   - Weighted fair queuing: Share based on payment tier
   - Token bucket: Rate limiting per customer
   Choose based on business requirements."
```

---

## 📝 End of Day Summary

### Skills Developed Today

✅ **Systematic Debugging**
- Hypothesis-driven investigation
- Performance regression analysis
- Memory leak detection

✅ **Code Review**
- Identifying bugs and edge cases
- Performance optimization opportunities
- Security and correctness issues

✅ **Algorithm Design**
- Scheduling strategies
- Trade-off analysis
- Complexity reasoning

### Interview Readiness Checklist

```
Technical Skills:
□ Can debug performance issues
□ Can review CUDA code
□ Can design scheduling algorithms
□ Can explain trade-offs clearly

Communication Skills:
□ Structured problem-solving approach
□ Clear explanations
□ Handles follow-up questions

Preparation:
□ Cheat sheet prepared
□ Examples ready
□ Stories polished
```

---

## 🚀 Preview: Day 28

Tomorrow's focus:
- **Final Review**: Comprehensive knowledge check
- **Portfolio Preparation**: Organize learnings
- **Next Steps**: Beyond the roadmap
- **Interview Strategy**: Final tips and prep

**Preparation**:
- Review all 27 days
- Compile best examples
- Identify remaining gaps
- Prepare questions to ask interviewers

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
