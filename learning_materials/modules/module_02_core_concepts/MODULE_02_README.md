# Module 2: Core Concepts - Learning Materials

## Overview

This module provides deep technical understanding of vLLM's core innovations:
- PagedAttention algorithm
- Continuous batching
- KV cache management  
- Scheduling algorithms

**Total Time:** 20-25 hours
**Difficulty:** Advanced
**Prerequisites:** Module 1 completion, strong understanding of transformers

---

## Module Structure

### Part 1: PagedAttention Foundation (Lessons 1-4)
**Time:** 10-12 hours

1. **Traditional Attention Problems** - Understanding the challenges
2. **PagedAttention Algorithm** - The core innovation
3. **Block Tables & Address Translation** - The mapping layer
4. **Memory Fragmentation Solutions** - Eliminating waste

### Part 2: Dynamic Scheduling (Lessons 5-8)
**Time:** 8-10 hours

5. **Continuous Batching Explained** - Dynamic request handling
6. **Preemption & Swapping** - Memory pressure management
7. **KV Cache Block Management** - Lifecycle and operations
8. **Copy-on-Write Optimization** - Prefix sharing

### Part 3: Scheduling Theory (Lessons 9-12)
**Time:** 6-8 hours

9. **Scheduling Algorithms** - FCFS, priority, fairness
10. **Request Queueing Theory** - Arrival patterns and analysis
11. **Chunked Prefill & Decode** - Phase separation
12. **Iteration-Level Scheduling** - Fine-grained decisions

---

## Learning Path

### Recommended Sequence

```
Week 1: Lessons 1-4
├─ Day 1-2: Traditional problems + PagedAttention
├─ Day 3-4: Block tables + fragmentation
└─ Day 5: Exercises and code exploration

Week 2: Lessons 5-8
├─ Day 1-2: Continuous batching
├─ Day 3: Preemption and swapping
├─ Day 4: KV cache management
└─ Day 5: Copy-on-write + exercises

Week 3: Lessons 9-12
├─ Day 1-2: Scheduling algorithms
├─ Day 3: Queueing theory
├─ Day 4: Chunked prefill
└─ Day 5: Iteration scheduling + final exercises
```

### Alternative: Intensive Track

```
3 Days (8-10 hours/day):
├─ Day 1: Lessons 1-4 (PagedAttention foundation)
├─ Day 2: Lessons 5-8 (Dynamic scheduling)
└─ Day 3: Lessons 9-12 (Scheduling theory) + Final assessment
```

---

## Lesson Summaries

### Lesson 1: Traditional Attention Problems

**Key Concepts:**
- KV cache memory requirements
- Internal vs. external fragmentation
- Static batching limitations
- Memory bottlenecks

**Skills Gained:**
- Calculate memory footprint
- Quantify fragmentation
- Identify bottlenecks

**Exercises:** 6 hands-on exercises
**vLLM Files:** `/vllm/v1/core/kv_cache_utils.py`

---

### Lesson 2: PagedAttention Algorithm

**Key Concepts:**
- Block-based memory organization
- Virtual memory analogy
- Attention with paged memory
- Flash-paged attention fusion

**Skills Gained:**
- Implement block allocation
- Understand paging mechanics
- Optimize memory layouts

**Exercises:** 6 implementation exercises
**vLLM Files:** `/vllm/v1/core/block_pool.py`, `/vllm/attention/`

---

### Lesson 3: Block Tables & Address Translation

**Key Concepts:**
- Three-level address translation
- Block table data structures
- Multi-dimensional indexing
- GPU implementation

**Skills Gained:**
- Implement block tables
- Optimize address computation
- Understand memory layouts

**Exercises:** 6 advanced exercises
**vLLM Files:** `/vllm/v1/worker/block_table.py`

---

### Lesson 4: Memory Fragmentation Solutions

**Key Concepts:**
- Internal fragmentation elimination
- External fragmentation management
- Block size optimization
- Memory pool organization

**Skills Gained:**
- Analyze fragmentation
- Optimize block sizes
- Measure utilization

**Exercises:** 5 analysis exercises
**vLLM Files:** `/vllm/v1/core/block_pool.py`

---

### Remaining Lessons (5-12)

Each lesson follows similar structure with:
- 400-600 lines of detailed content
- Mathematical explanations
- Code examples from vLLM
- Hands-on exercises
- vLLM file references

---

## Assessment Strategy

### Formative Assessment

**Per-Lesson Quizzes:**
- 5-10 questions
- Immediate feedback
- Must score 80%+ to proceed

**Hands-On Exercises:**
- Code implementations
- Analysis tasks
- vLLM code exploration

### Summative Assessment

**Module 2 Final Project:**

Implement a simplified PagedAttention system:

```python
Requirements:
1. Block pool with allocation/deallocation
2. Block table management
3. Address translation
4. Simple scheduling policy
5. Memory utilization tracking

Evaluation Criteria:
- Correctness (40%)
- Performance (30%)
- Code quality (20%)
- Documentation (10%)
```

**Passing Score:** 85%

---

## Prerequisites Check

Before starting Module 2, ensure you can:

- [ ] Explain transformer attention mechanism
- [ ] Calculate memory requirements for models
- [ ] Understand GPU memory basics
- [ ] Read and understand Python code
- [ ] Navigate large codebases
- [ ] Use Git to explore code history

---

## Required Tools

### Software
- Python 3.8+
- PyTorch 2.0+
- vLLM (built from source)
- Jupyter notebooks
- Git

### Hardware
- GPU with 16GB+ VRAM (recommended)
- CPU-only possible but slower
- 32GB+ system RAM

### Optional
- CUDA toolkit for profiling
- nsight-compute for kernel analysis
- TensorBoard for visualization

---

## Learning Resources

### Primary Materials
- These lesson files (01-12)
- vLLM source code
- vLLM paper (arXiv)

### Supplementary
- FlashAttention papers
- Virtual memory OS textbooks
- CUDA programming guides
- GPU architecture references

### Community
- vLLM GitHub discussions
- vLLM Discord/Slack
- ML Systems papers (MLSys conference)

---

## Common Pitfalls

### Conceptual Misunderstandings

1. **"PagedAttention changes the attention math"**
   - ❌ Wrong: Math is identical
   - ✓ Right: Only memory organization differs

2. **"Blocks must be contiguous in GPU memory"**
   - ❌ Wrong: This defeats the purpose
   - ✓ Right: Blocks can be anywhere, mapped via block tables

3. **"Larger blocks are always better"**
   - ❌ Wrong: Increases internal fragmentation
   - ✓ Right: Block size is a tradeoff

### Implementation Mistakes

1. **Not handling OOM gracefully**
   ```python
   # Bad
   block = block_pool.allocate()  # May crash
   
   # Good
   if block_pool.get_num_free() > 0:
       block = block_pool.allocate()
   else:
       handle_out_of_memory()
   ```

2. **Forgetting reference counting**
   ```python
   # Bad
   block_pool.free(shared_block)  # May free shared block!
   
   # Good
   block_pool.decrement_ref(shared_block)  # Free only if ref=0
   ```

3. **Incorrect address translation**
   ```python
   # Bad
   physical = block_table[token_pos]  # Wrong!
   
   # Good
   logical_block = token_pos // block_size
   physical = block_table[logical_block]
   ```

---

## Success Metrics

### Knowledge Metrics

After completing Module 2, you should be able to:

- [ ] Explain PagedAttention to a senior engineer (10 min presentation)
- [ ] Implement a working block allocator (200+ lines)
- [ ] Calculate memory savings for any workload
- [ ] Debug PagedAttention issues in vLLM
- [ ] Optimize block size for specific scenarios
- [ ] Design custom scheduling policies

### Performance Metrics

- [ ] Score 85%+ on all lesson quizzes
- [ ] Complete 80%+ of hands-on exercises
- [ ] Pass module final project (85%+)
- [ ] Contribute to vLLM (optional but encouraged)

### Interview Readiness

Can you answer:

1. "Explain how PagedAttention works"
2. "Why is block-based allocation better than contiguous?"
3. "How does vLLM eliminate fragmentation?"
4. "Walk me through address translation"
5. "What are the trade-offs of block size?"

---

## Next Steps

### After Module 2

**Module 3: CUDA Kernels & Optimization**
- Attention kernel implementation
- Memory optimization techniques
- Profiling and benchmarking

**Module 4: System Components**
- Scheduler deep dive
- Block manager internals
- Request lifecycle

**Module 5: Distributed Inference**
- Tensor parallelism
- Pipeline parallelism
- Multi-GPU coordination

---

## Support & Help

### Getting Stuck?

1. **Review prerequisite lessons**
2. **Check vLLM source code**
3. **Read vLLM paper**
4. **Ask in vLLM community**
5. **Consult instructor/mentor**

### Debugging Tips

- Use print statements liberally
- Visualize data structures
- Step through with debugger
- Compare with vLLM implementation
- Test with simple cases first

---

## Revision History

- **2025-11-19:** Initial release (v1.0)
- **Future:** Additional exercises, video walkthroughs, interactive visualizations

---

**Happy Learning! Master these core concepts to unlock deep understanding of vLLM.**

