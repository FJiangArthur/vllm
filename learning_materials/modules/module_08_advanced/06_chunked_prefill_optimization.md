# Module 8.6: Chunked Prefill Optimization

## Overview

Chunked prefill is a critical optimization for handling long-context requests efficiently in vLLM. By breaking long prompt processing into smaller chunks, we can better balance prefill and decode phases, reduce latency variance, and improve overall system throughput.

**Learning Objectives:**
- Understand the prefill bottleneck in long-context scenarios
- Master chunked prefill implementation and configuration
- Optimize chunk sizes for different workload patterns
- Balance fairness and throughput in mixed workloads
- Deploy chunked prefill in production systems

**Prerequisites:**
- Understanding of prefill vs decode phases
- Knowledge of vLLM scheduling (Module 4)
- Familiarity with attention mechanisms
- Production deployment experience

**Estimated Time:** 3-4 hours

---

## 1. The Prefill Bottleneck

### 1.1 Understanding Prefill vs Decode

```python
"""
Prefill vs Decode Phases

Prefill:
- Process entire input prompt at once
- Compute KV cache for all prompt tokens
- Quadratic complexity in prompt length (O(n²) for attention)
- High memory bandwidth usage
- Runs once per request

Decode:
- Generate one token at a time
- Linear complexity per token (O(n) for attention on cached KV)
- Lower memory bandwidth
- Runs multiple times (once per output token)

Problem:
Long prefill blocks decode of other requests → high latency variance
"""

import time
import torch
import numpy as np

class PrefillDecodeAnalyzer:
    """
    Analyze prefill vs decode performance characteristics
    """

    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name

    def measure_prefill_time(self, prompt_length):
        """
        Measure prefill time for given prompt length

        Simulated measurements (would use actual model in practice)
        """
        # Simulated: prefill time grows quadratically
        base_time = 0.01  # 10ms base
        quadratic_factor = 0.000001  # Scale factor

        time_ms = base_time + (prompt_length ** 2) * quadratic_factor
        return time_ms

    def measure_decode_time(self, context_length):
        """
        Measure decode time for one token given context length

        Simulated measurements
        """
        # Simulated: decode time grows linearly
        base_time = 0.005  # 5ms base
        linear_factor = 0.00001  # Scale factor

        time_ms = base_time + context_length * linear_factor
        return time_ms

    def analyze_long_context_impact(self):
        """
        Analyze impact of long prompts on system latency
        """
        prompt_lengths = [100, 500, 1000, 2000, 4000, 8000]

        print("Prefill vs Decode Time Analysis:")
        print("=" * 70)
        print(f"{'Prompt Length':<15} {'Prefill (ms)':<15} {'Decode (ms)':<15} {'Ratio':<10}")
        print("-" * 70)

        for length in prompt_lengths:
            prefill_time = self.measure_prefill_time(length)
            decode_time = self.measure_decode_time(length)
            ratio = prefill_time / decode_time

            print(f"{length:<15} {prefill_time:<15.2f} {decode_time:<15.2f} {ratio:<10.1f}x")

        print("=" * 70)
        print("\nKey Insight: Prefill time dominates for long contexts!")
        print("  → Blocks other requests during long prefill")
        print("  → High latency variance in mixed workloads")

analyzer = PrefillDecodeAnalyzer()
analyzer.analyze_long_context_impact()
```

### 1.2 The Blocking Problem

```python
def simulate_blocking_scenario():
    """
    Simulate how long prefill blocks short decode requests
    """
    print("\nBlocking Scenario Simulation:")
    print("=" * 70)

    # Scenario: Two concurrent requests
    # Request 1: Long prefill (8K tokens)
    # Request 2: Short decode (100 tokens context)

    analyzer = PrefillDecodeAnalyzer()

    long_prefill_time = analyzer.measure_prefill_time(8000)
    short_decode_time = analyzer.measure_decode_time(100)

    print(f"Request 1 (long prefill): {long_prefill_time:.2f} ms")
    print(f"Request 2 (short decode): {short_decode_time:.2f} ms")

    print(f"\nWithout chunking:")
    print(f"  Request 2 blocked for: {long_prefill_time:.2f} ms")
    print(f"  Latency inflation: {long_prefill_time / short_decode_time:.1f}x")

    print(f"\nWith chunking (4 chunks):")
    chunk_time = long_prefill_time / 4
    print(f"  Each chunk: {chunk_time:.2f} ms")
    print(f"  Request 2 blocked for: {chunk_time:.2f} ms (max)")
    print(f"  Latency inflation: {chunk_time / short_decode_time:.1f}x")

    print(f"\nImprovement: {long_prefill_time / chunk_time:.1f}x reduction in blocking time")

simulate_blocking_scenario()
```

---

## 2. Chunked Prefill Implementation

### 2.1 Basic Chunking Strategy

```python
class ChunkedPrefillProcessor:
    """
    Process long prefills in chunks to reduce blocking
    """

    def __init__(self, chunk_size=512, enable_chunking=True):
        """
        Args:
            chunk_size: Number of tokens per chunk
            enable_chunking: Whether to enable chunking
        """
        self.chunk_size = chunk_size
        self.enable_chunking = enable_chunking

    def split_into_chunks(self, tokens):
        """
        Split token sequence into chunks

        Args:
            tokens: List of token IDs

        Returns:
            List of chunks
        """
        if not self.enable_chunking or len(tokens) <= self.chunk_size:
            return [tokens]

        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(chunk)

        return chunks

    def process_chunks(self, tokens, process_fn):
        """
        Process token sequence in chunks

        Args:
            tokens: Input tokens
            process_fn: Function to process each chunk

        Returns:
            Combined results from all chunks
        """
        chunks = self.split_into_chunks(tokens)

        print(f"Processing {len(tokens)} tokens in {len(chunks)} chunks")

        results = []
        accumulated_kv_cache = None

        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} tokens")

            # Process chunk (with context from previous chunks)
            result, kv_cache = process_fn(chunk, accumulated_kv_cache)
            results.append(result)

            # Accumulate KV cache for next chunk
            if accumulated_kv_cache is None:
                accumulated_kv_cache = kv_cache
            else:
                accumulated_kv_cache = self._merge_kv_caches(
                    accumulated_kv_cache, kv_cache
                )

        return results, accumulated_kv_cache

    def _merge_kv_caches(self, cache1, cache2):
        """
        Merge two KV caches

        In practice, this concatenates along sequence dimension
        """
        # Simplified merging
        if cache1 is None:
            return cache2
        if cache2 is None:
            return cache1

        # Concatenate (simplified - would handle tensor properly)
        return {"merged": True, "size": cache1.get("size", 0) + cache2.get("size", 0)}

def chunked_prefill_example():
    """
    Example of chunked prefill processing
    """
    processor = ChunkedPrefillProcessor(chunk_size=512)

    # Long prompt
    long_prompt = list(range(2500))  # 2500 tokens

    # Mock process function
    def mock_process(chunk, kv_cache):
        import time
        time.sleep(0.01)  # Simulate processing
        return {"processed": len(chunk)}, {"size": len(chunk)}

    print("Chunked Prefill Example:")
    print("=" * 60)

    import time
    start = time.time()

    results, kv_cache = processor.process_chunks(long_prompt, mock_process)

    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Chunks processed: {len(results)}")
    print(f"Final KV cache size: {kv_cache.get('size', 0)} tokens")

chunked_prefill_example()
```

### 2.2 Interleaved Scheduling

```python
class InterleavedPrefillDecodeScheduler:
    """
    Scheduler that interleaves prefill chunks with decode steps

    Key idea: Process one prefill chunk, then one decode step,
    alternating to maintain fairness
    """

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
        self.prefill_queue = []  # Requests in prefill phase
        self.decode_queue = []   # Requests in decode phase

    def add_request(self, request):
        """Add new request to appropriate queue"""
        if request.is_prefill:
            # Split into chunks
            chunks = self._split_into_chunks(request.tokens)
            request.chunks = chunks
            request.current_chunk = 0
            self.prefill_queue.append(request)
        else:
            self.decode_queue.append(request)

    def _split_into_chunks(self, tokens):
        """Split tokens into chunks"""
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunks.append(tokens[i:i + self.chunk_size])
        return chunks

    def schedule_next_batch(self):
        """
        Schedule next batch with interleaving

        Strategy:
        1. Pick one prefill chunk (if available)
        2. Pick multiple decode requests (to fill batch)
        3. Alternate between prefill and decode
        """
        batch = []

        # Add one prefill chunk (if available)
        if self.prefill_queue:
            prefill_req = self.prefill_queue[0]

            if prefill_req.current_chunk < len(prefill_req.chunks):
                chunk = prefill_req.chunks[prefill_req.current_chunk]
                batch.append({
                    "request": prefill_req,
                    "type": "prefill_chunk",
                    "tokens": chunk,
                    "chunk_id": prefill_req.current_chunk
                })

                prefill_req.current_chunk += 1

                # If all chunks processed, move to decode
                if prefill_req.current_chunk >= len(prefill_req.chunks):
                    self.prefill_queue.pop(0)
                    prefill_req.is_prefill = False
                    self.decode_queue.append(prefill_req)

        # Fill rest of batch with decode requests
        max_decode = 16  # Max decode requests per batch
        num_decode = min(len(self.decode_queue), max_decode)

        for i in range(num_decode):
            decode_req = self.decode_queue[i]
            batch.append({
                "request": decode_req,
                "type": "decode",
                "tokens": [decode_req.last_token]
            })

        return batch

    def simulate_scheduling(self, requests):
        """
        Simulate interleaved scheduling

        Args:
            requests: List of mock requests
        """
        # Add all requests
        for req in requests:
            self.add_request(req)

        print("Interleaved Prefill/Decode Scheduling:")
        print("=" * 70)

        iteration = 0
        while self.prefill_queue or self.decode_queue:
            iteration += 1

            batch = self.schedule_next_batch()

            if not batch:
                break

            # Print batch composition
            prefill_count = sum(1 for b in batch if b["type"] == "prefill_chunk")
            decode_count = sum(1 for b in batch if b["type"] == "decode")

            print(f"\nIteration {iteration}:")
            print(f"  Prefill chunks: {prefill_count}")
            print(f"  Decode requests: {decode_count}")

            # Show details
            for item in batch:
                req = item["request"]
                if item["type"] == "prefill_chunk":
                    print(f"    Prefill: Request {req.id}, chunk {item['chunk_id']+1}/{len(req.chunks)}")
                else:
                    print(f"    Decode: Request {req.id}")

            if iteration >= 10:  # Limit output
                print("\n  ...")
                break

        print(f"\nTotal iterations: {iteration}")

# Example usage
class MockRequest:
    _id_counter = 0

    def __init__(self, tokens, is_prefill=True):
        self.id = MockRequest._id_counter
        MockRequest._id_counter += 1

        self.tokens = tokens
        self.is_prefill = is_prefill
        self.last_token = tokens[-1] if tokens else 0
        self.chunks = []
        self.current_chunk = 0

def interleaved_scheduling_example():
    """Example of interleaved scheduling"""
    scheduler = InterleavedPrefillDecodeScheduler(chunk_size=512)

    # Create mix of requests
    requests = [
        MockRequest(tokens=list(range(2000)), is_prefill=True),  # Long prefill
        MockRequest(tokens=list(range(100)), is_prefill=True),   # Short prefill
        MockRequest(tokens=[42], is_prefill=False),              # Decode
        MockRequest(tokens=[43], is_prefill=False),              # Decode
        MockRequest(tokens=[44], is_prefill=False),              # Decode
    ]

    scheduler.simulate_scheduling(requests)

interleaved_scheduling_example()
```

---

## 3. Chunk Size Optimization

### 3.1 Optimal Chunk Size Selection

```python
class ChunkSizeOptimizer:
    """
    Determine optimal chunk size for given workload
    """

    def __init__(self):
        self.measurements = []

    def measure_chunk_performance(self, chunk_size, prompt_length):
        """
        Measure performance metrics for given chunk size

        Returns:
            dict with latency, throughput, fairness metrics
        """
        num_chunks = (prompt_length + chunk_size - 1) // chunk_size

        # Simulated metrics
        # Smaller chunks: better fairness, more overhead
        # Larger chunks: better throughput, worse fairness

        overhead_per_chunk = 0.002  # 2ms overhead per chunk
        processing_per_token = 0.00001  # 10μs per token

        total_overhead = num_chunks * overhead_per_chunk
        total_processing = prompt_length * processing_per_token
        total_time = total_overhead + total_processing

        # Fairness: smaller chunks = better (lower max blocking time)
        max_blocking_time = chunk_size * processing_per_token

        # Throughput: larger chunks = better (less overhead)
        throughput = prompt_length / total_time

        return {
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "total_time": total_time,
            "overhead_fraction": total_overhead / total_time,
            "max_blocking_time": max_blocking_time,
            "throughput": throughput
        }

    def find_optimal_chunk_size(self, prompt_length, max_blocking_time_ms=0.050):
        """
        Find optimal chunk size balancing throughput and fairness

        Args:
            prompt_length: Length of prompt to optimize for
            max_blocking_time_ms: Maximum acceptable blocking time (seconds)

        Returns:
            Optimal chunk size
        """
        chunk_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

        print(f"\nOptimizing chunk size for {prompt_length} token prompt:")
        print(f"Max blocking time constraint: {max_blocking_time_ms*1000:.1f} ms")
        print("=" * 80)
        print(f"{'Chunk Size':<12} {'Chunks':<8} {'Time (ms)':<12} {'Overhead':<12} {'Blocking (ms)':<15} {'Throughput':<12}")
        print("-" * 80)

        best_chunk_size = None
        best_score = -float('inf')

        for chunk_size in chunk_sizes:
            metrics = self.measure_chunk_performance(chunk_size, prompt_length)

            # Check constraint
            if metrics["max_blocking_time"] > max_blocking_time_ms:
                # Violates fairness constraint
                score = -1
            else:
                # Score: maximize throughput while meeting fairness constraint
                score = metrics["throughput"]

            print(f"{chunk_size:<12} {metrics['num_chunks']:<8} "
                  f"{metrics['total_time']*1000:<12.2f} "
                  f"{metrics['overhead_fraction']:<12.1%} "
                  f"{metrics['max_blocking_time']*1000:<15.2f} "
                  f"{metrics['throughput']:<12.0f} "
                  f"{'✓' if score >= 0 else '✗ (too slow)'}")

            if score > best_score:
                best_score = score
                best_chunk_size = chunk_size

        print("-" * 80)
        print(f"\nOptimal chunk size: {best_chunk_size}")

        return best_chunk_size

optimizer = ChunkSizeOptimizer()

# Optimize for different prompt lengths
for length in [1000, 4000, 8000]:
    optimal = optimizer.find_optimal_chunk_size(length, max_blocking_time_ms=0.030)
```

### 3.2 Adaptive Chunking

```python
class AdaptiveChunkedPrefill:
    """
    Adaptively adjust chunk size based on current system state
    """

    def __init__(self, base_chunk_size=512):
        self.base_chunk_size = base_chunk_size
        self.current_chunk_size = base_chunk_size

        # State tracking
        self.recent_latencies = []
        self.recent_utilizations = []

    def get_chunk_size(self, system_state):
        """
        Determine chunk size based on system state

        Args:
            system_state: Dict with current metrics

        Returns:
            Chunk size to use
        """
        num_decode_requests = system_state.get("num_decode_requests", 0)
        gpu_utilization = system_state.get("gpu_utilization", 0.5)
        avg_latency = system_state.get("avg_latency", 0.1)

        # Heuristics:
        # 1. Many decode requests → smaller chunks (better fairness)
        # 2. Low GPU utilization → larger chunks (better efficiency)
        # 3. High latency → smaller chunks (reduce blocking)

        chunk_size = self.base_chunk_size

        # Adjust based on decode queue length
        if num_decode_requests > 10:
            chunk_size = chunk_size // 2  # Smaller chunks for fairness
        elif num_decode_requests < 3:
            chunk_size = chunk_size * 2  # Larger chunks for efficiency

        # Adjust based on GPU utilization
        if gpu_utilization < 0.5:
            chunk_size = min(chunk_size * 2, 4096)  # Larger chunks to fill GPU
        elif gpu_utilization > 0.9:
            chunk_size = max(chunk_size // 2, 128)  # Smaller chunks to reduce contention

        # Clamp to reasonable range
        chunk_size = max(128, min(chunk_size, 4096))

        self.current_chunk_size = chunk_size
        return chunk_size

    def simulate_adaptive_chunking(self):
        """
        Simulate adaptive chunking under varying load
        """
        print("Adaptive Chunking Simulation:")
        print("=" * 70)

        scenarios = [
            {"name": "Low load", "num_decode_requests": 2, "gpu_utilization": 0.3, "avg_latency": 0.05},
            {"name": "Medium load", "num_decode_requests": 8, "gpu_utilization": 0.6, "avg_latency": 0.08},
            {"name": "High load", "num_decode_requests": 20, "gpu_utilization": 0.9, "avg_latency": 0.15},
            {"name": "Overloaded", "num_decode_requests": 50, "gpu_utilization": 0.95, "avg_latency": 0.30},
        ]

        for scenario in scenarios:
            chunk_size = self.get_chunk_size(scenario)

            print(f"\n{scenario['name']}:")
            print(f"  Decode requests: {scenario['num_decode_requests']}")
            print(f"  GPU utilization: {scenario['gpu_utilization']:.1%}")
            print(f"  Avg latency: {scenario['avg_latency']*1000:.1f} ms")
            print(f"  → Chunk size: {chunk_size}")

adaptive = AdaptiveChunkedPrefill(base_chunk_size=512)
adaptive.simulate_adaptive_chunking()
```

---

## 4. Production Deployment

### 4.1 Enabling Chunked Prefill in vLLM

```python
from vllm import LLM, SamplingParams

def deploy_chunked_prefill():
    """
    Deploy vLLM with chunked prefill optimization
    """
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",

        # Enable chunked prefill
        enable_chunked_prefill=True,

        # Chunk configuration
        max_num_batched_tokens=2048,  # Effective chunk size
        max_num_seqs=256,  # Allow batching decode with prefill chunks

        # Memory configuration
        gpu_memory_utilization=0.9,

        # Other optimizations
        enable_prefix_caching=True,
    )

    return llm

def production_chunked_prefill_example():
    """
    Production example with chunked prefill
    """
    llm = deploy_chunked_prefill()

    # Mix of long and short requests
    requests = [
        {
            "prompt": "Long document... " * 1000,  # ~8K tokens
            "max_tokens": 100
        },
        {
            "prompt": "Short question",  # ~3 tokens
            "max_tokens": 50
        },
        {
            "prompt": "Medium context " * 100,  # ~200 tokens
            "max_tokens": 80
        },
    ]

    sampling_params = SamplingParams(temperature=0.8)

    import time

    print("Chunked Prefill Production Example:")
    print("=" * 60)

    for i, req in enumerate(requests):
        start = time.time()
        output = llm.generate(req["prompt"], sampling_params)
        elapsed = time.time() - start

        print(f"\nRequest {i+1}:")
        print(f"  Prompt length: ~{len(req['prompt'].split())} words")
        print(f"  Latency: {elapsed:.2f}s")
```

### 4.2 Monitoring and Tuning

```python
class ChunkedPrefillMonitor:
    """
    Monitor chunked prefill performance
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            "total_requests": 0,
            "prefill_requests": 0,
            "decode_requests": 0,
            "total_chunks_processed": 0,
            "avg_chunks_per_request": [],
            "prefill_latencies": [],
            "decode_latencies": [],
            "blocking_times": [],
        }

    def record_prefill(self, num_chunks, latency, blocking_time):
        """Record prefill request metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["prefill_requests"] += 1
        self.metrics["total_chunks_processed"] += num_chunks
        self.metrics["avg_chunks_per_request"].append(num_chunks)
        self.metrics["prefill_latencies"].append(latency)
        self.metrics["blocking_times"].append(blocking_time)

    def record_decode(self, latency):
        """Record decode request metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["decode_requests"] += 1
        self.metrics["decode_latencies"].append(latency)

    def get_summary(self):
        """Get summary statistics"""
        import numpy as np

        if not self.metrics["total_requests"]:
            return {}

        summary = {
            "total_requests": self.metrics["total_requests"],
            "prefill_ratio": self.metrics["prefill_requests"] / self.metrics["total_requests"],
        }

        if self.metrics["avg_chunks_per_request"]:
            summary["avg_chunks_per_request"] = np.mean(self.metrics["avg_chunks_per_request"])

        if self.metrics["prefill_latencies"]:
            summary["avg_prefill_latency"] = np.mean(self.metrics["prefill_latencies"])
            summary["p95_prefill_latency"] = np.percentile(self.metrics["prefill_latencies"], 95)

        if self.metrics["decode_latencies"]:
            summary["avg_decode_latency"] = np.mean(self.metrics["decode_latencies"])
            summary["p95_decode_latency"] = np.percentile(self.metrics["decode_latencies"], 95)

        if self.metrics["blocking_times"]:
            summary["avg_blocking_time"] = np.mean(self.metrics["blocking_times"])
            summary["max_blocking_time"] = np.max(self.metrics["blocking_times"])

        return summary

    def print_report(self):
        """Print monitoring report"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("CHUNKED PREFILL MONITORING REPORT")
        print("=" * 60)

        print(f"\nRequests:")
        print(f"  Total: {summary['total_requests']}")
        print(f"  Prefill: {self.metrics['prefill_requests']}")
        print(f"  Decode: {self.metrics['decode_requests']}")

        if "avg_chunks_per_request" in summary:
            print(f"\nChunking:")
            print(f"  Avg chunks/request: {summary['avg_chunks_per_request']:.1f}")
            print(f"  Total chunks: {self.metrics['total_chunks_processed']}")

        if "avg_prefill_latency" in summary:
            print(f"\nPrefill Latency:")
            print(f"  Average: {summary['avg_prefill_latency']*1000:.2f} ms")
            print(f"  P95: {summary['p95_prefill_latency']*1000:.2f} ms")

        if "avg_decode_latency" in summary:
            print(f"\nDecode Latency:")
            print(f"  Average: {summary['avg_decode_latency']*1000:.2f} ms")
            print(f"  P95: {summary['p95_decode_latency']*1000:.2f} ms")

        if "avg_blocking_time" in summary:
            print(f"\nBlocking:")
            print(f"  Average: {summary['avg_blocking_time']*1000:.2f} ms")
            print(f"  Maximum: {summary['max_blocking_time']*1000:.2f} ms")

        print("=" * 60)

def monitoring_example():
    """Example of monitoring chunked prefill"""
    import random

    monitor = ChunkedPrefillMonitor()

    # Simulate requests
    for _ in range(100):
        if random.random() < 0.3:
            # Prefill request
            num_chunks = random.randint(1, 8)
            latency = random.uniform(0.05, 0.3)
            blocking = random.uniform(0.01, 0.05)
            monitor.record_prefill(num_chunks, latency, blocking)
        else:
            # Decode request
            latency = random.uniform(0.005, 0.02)
            monitor.record_decode(latency)

    monitor.print_report()

monitoring_example()
```

---

## 5. Advanced Techniques

### 5.1 Priority-Based Chunking

```python
class PriorityAwareChunking:
    """
    Adjust chunk sizes based on request priority
    """

    def __init__(self):
        self.priority_chunk_sizes = {
            "high": 256,    # Small chunks for low latency
            "medium": 512,  # Balanced
            "low": 1024,    # Large chunks for efficiency
        }

    def get_chunk_size(self, request):
        """Get chunk size based on request priority"""
        priority = request.get("priority", "medium")
        return self.priority_chunk_sizes.get(priority, 512)

    def process_with_priority(self, requests):
        """Process requests respecting priorities"""
        print("Priority-Based Chunking:")
        print("=" * 60)

        for req in requests:
            chunk_size = self.get_chunk_size(req)
            num_tokens = req.get("num_tokens", 1000)
            num_chunks = (num_tokens + chunk_size - 1) // chunk_size

            print(f"\nRequest (priority={req.get('priority', 'medium')}):")
            print(f"  Tokens: {num_tokens}")
            print(f"  Chunk size: {chunk_size}")
            print(f"  Num chunks: {num_chunks}")

priority_chunking = PriorityAwareChunking()

requests = [
    {"priority": "high", "num_tokens": 2000},
    {"priority": "medium", "num_tokens": 2000},
    {"priority": "low", "num_tokens": 2000},
]

priority_chunking.process_with_priority(requests)
```

---

## 6. Summary

### 6.1 Key Takeaways

1. **Chunked prefill prevents blocking:**
   - Break long prefills into chunks
   - Interleave with decode for fairness
   - Reduces latency variance

2. **Chunk size tradeoffs:**
   - Smaller chunks: better fairness, more overhead
   - Larger chunks: better throughput, worse fairness
   - Optimal size depends on workload

3. **Production recommendations:**
   - Enable for mixed workloads with long contexts
   - Start with 512-1024 token chunks
   - Monitor blocking times and adjust

### 6.2 Configuration Guide

```python
def get_chunked_prefill_config(workload_type):
    """
    Recommended configurations for different workloads
    """
    configs = {
        "long_context_qa": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 1024,
            "rationale": "Long documents need chunking, moderate chunks for balance"
        },

        "mixed_interactive": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 512,
            "rationale": "Many short requests need fairness, smaller chunks"
        },

        "batch_processing": {
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": 4096,
            "rationale": "No real-time requirements, maximize throughput"
        },
    }

    return configs.get(workload_type, configs["mixed_interactive"])
```

---

## 7. Exercises

**Exercise 8.6.1:** Implement chunked prefill processor with interleaving

**Exercise 8.6.2:** Optimize chunk size for your workload

**Exercise 8.6.3:** Measure blocking time reduction from chunking

**Exercise 8.6.4:** Implement adaptive chunk size selection

**Exercise 8.6.5:** Deploy and monitor in production

---

**Next:** [Module 8.7: Kubernetes Deployment](07_kubernetes_deployment.md)
