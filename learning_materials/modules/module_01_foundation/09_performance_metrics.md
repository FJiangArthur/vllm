# Module 1.9: Performance Metrics

## Learning Objectives

1. Understand key performance metrics
2. Measure throughput and latency
3. Calculate tokens per second
4. Monitor GPU utilization
5. Benchmark vLLM effectively
6. Interpret performance results

**Estimated Time:** 1.5 hours

---

## Key Metrics

### Throughput

Requests or tokens processed per second:

```python
# throughput.py
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompts = ["Test prompt"] * 100
params = SamplingParams(max_tokens=50)

start = time.time()
outputs = llm.generate(prompts, params)
duration = time.time() - start

# Calculate metrics
requests_per_sec = len(prompts) / duration
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tokens_per_sec = total_tokens / duration

print(f"Throughput: {requests_per_sec:.2f} requests/sec")
print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
```

### Latency

Time to complete a single request:

```python
# latency.py
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompts = ["Hello world"]
params = SamplingParams(max_tokens=100)

# Measure latency
start = time.time()
outputs = llm.generate(prompts, params)
latency = time.time() - start

print(f"Latency: {latency*1000:.2f} ms")
```

### Time to First Token (TTFT)

Time until first token is generated (important for streaming):

```python
# ttft.py
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio

async def measure_ttft():
    args = AsyncEngineArgs(model="facebook/opt-125m")
    engine = AsyncLLMEngine.from_engine_args(args)
    
    start = torch.cuda.Event(enable_timing=True)
    first_token = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    first_received = False
    async for output in engine.generate(
        "Hello",
        SamplingParams(max_tokens=100),
        "request-0"
    ):
        if not first_received:
            first_token.record()
            first_received = True
    
    torch.cuda.synchronize()
    ttft = start.elapsed_time(first_token)
    print(f"Time to First Token: {ttft:.2f} ms")

asyncio.run(measure_ttft())
```

### GPU Utilization

```python
# gpu_utilization.py
import torch
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# Monitor during inference
prompts = ["Test"] * 50
params = SamplingParams(max_tokens=100)

# Start inference in background
import threading

def run_inference():
    outputs = llm.generate(prompts, params)

thread = threading.Thread(target=run_inference)
thread.start()

# Monitor utilization
for i in range(10):
    time.sleep(0.5)
    util = torch.cuda.utilization()
    mem_used = torch.cuda.memory_allocated() / 1e9
    print(f"GPU Utilization: {util}%, Memory: {mem_used:.2f}GB")

thread.join()
```

---

## Comprehensive Benchmark

```python
# benchmark.py
"""
Comprehensive vLLM benchmark
"""

import time
import torch
from vllm import LLM, SamplingParams
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    requests: int
    total_tokens: int
    duration: float
    
    @property
    def throughput_rps(self) -> float:
        return self.requests / self.duration
    
    @property
    def throughput_tps(self) -> float:
        return self.total_tokens / self.duration
    
    @property
    def avg_latency(self) -> float:
        return (self.duration / self.requests) * 1000  # ms

def benchmark_vllm(
    model: str,
    num_prompts: int,
    max_tokens: int,
    batch_size: int = None
) -> BenchmarkResult:
    """Run benchmark and return results"""
    
    # Initialize model
    llm = LLM(model=model)
    
    # Prepare prompts
    prompts = [f"Prompt {i}: Tell me about AI" for i in range(num_prompts)]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    
    # Warmup
    print("Warming up...")
    llm.generate(prompts[:2], params)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark: {num_prompts} prompts, {max_tokens} max tokens")
    start = time.time()
    outputs = llm.generate(prompts, params)
    torch.cuda.synchronize()
    duration = time.time() - start
    
    # Calculate total tokens
    total_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    
    return BenchmarkResult(
        requests=num_prompts,
        total_tokens=total_tokens,
        duration=duration
    )

if __name__ == "__main__":
    # Run benchmark
    result = benchmark_vllm(
        model="facebook/opt-125m",
        num_prompts=100,
        max_tokens=50
    )
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Requests: {result.requests}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Throughput: {result.throughput_rps:.2f} requests/sec")
    print(f"Throughput: {result.throughput_tps:.2f} tokens/sec")
    print(f"Avg latency: {result.avg_latency:.2f}ms")
```

---

## Batch Size Impact

```python
# batch_size_sweep.py
"""
Test different batch sizes
"""

import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0.0, max_tokens=50)

batch_sizes = [1, 5, 10, 20, 50, 100]

print("Batch Size | Throughput (req/s) | Latency (ms)")
print("-" * 50)

for batch_size in batch_sizes:
    prompts = ["Test"] * batch_size
    
    start = time.time()
    outputs = llm.generate(prompts, params)
    duration = time.time() - start
    
    throughput = batch_size / duration
    latency = (duration / batch_size) * 1000
    
    print(f"{batch_size:10d} | {throughput:18.2f} | {latency:11.2f}")
```

---

## Memory Efficiency

```python
# memory_efficiency.py
"""
Measure memory efficiency
"""

import torch
from vllm import LLM, SamplingParams

def measure_memory_efficiency(model_name: str):
    # Measure baseline
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1e9
    
    # Load model
    llm = LLM(model=model_name, gpu_memory_utilization=0.9)
    model_memory = torch.cuda.memory_allocated() / 1e9 - baseline
    
    # Run inference
    prompts = ["Test"] * 100
    params = SamplingParams(max_tokens=50)
    outputs = llm.generate(prompts, params)
    
    total_memory = torch.cuda.memory_allocated() / 1e9
    kv_cache_memory = total_memory - model_memory - baseline
    
    print(f"Model: {model_name}")
    print(f"Model weights: {model_memory:.2f}GB")
    print(f"KV cache: {kv_cache_memory:.2f}GB")
    print(f"Total: {total_memory:.2f}GB")
    
    # Calculate efficiency
    max_batch_size = int(100 * (kv_cache_memory / (total_memory - model_memory)))
    print(f"Estimated max batch size: {max_batch_size}")

measure_memory_efficiency("facebook/opt-125m")
```

---

## Comparative Benchmarks

```python
# compare_configs.py
"""
Compare different vLLM configurations
"""

from vllm import LLM, SamplingParams
import time

configs = [
    {
        "name": "Default",
        "args": {}
    },
    {
        "name": "High Memory Util",
        "args": {"gpu_memory_utilization": 0.95}
    },
    {
        "name": "Eager Mode",
        "args": {"enforce_eager": True}
    },
    {
        "name": "Low Memory Util",
        "args": {"gpu_memory_utilization": 0.7}
    },
]

prompts = ["Test"] * 50
params = SamplingParams(temperature=0.0, max_tokens=50)

print("Configuration       | Throughput (req/s)")
print("-" * 50)

for config in configs:
    llm = LLM(model="facebook/opt-125m", **config["args"])
    
    start = time.time()
    outputs = llm.generate(prompts, params)
    duration = time.time() - start
    
    throughput = len(prompts) / duration
    print(f"{config['name']:18s} | {throughput:.2f}")
    
    del llm  # Free memory
```

---

## Metrics Collection

```python
# metrics_collector.py
"""
Collect and log metrics over time
"""

import time
import torch
from vllm import LLM, SamplingParams
from dataclasses import dataclass, asdict
import json

@dataclass
class InferenceMetrics:
    timestamp: float
    num_requests: int
    throughput_rps: float
    throughput_tps: float
    avg_latency_ms: float
    gpu_memory_gb: float
    gpu_utilization_pct: float

class MetricsCollector:
    def __init__(self):
        self.metrics = []
    
    def collect(self, result, start_time):
        metric = InferenceMetrics(
            timestamp=time.time(),
            num_requests=result.requests,
            throughput_rps=result.throughput_rps,
            throughput_tps=result.throughput_tps,
            avg_latency_ms=result.avg_latency,
            gpu_memory_gb=torch.cuda.memory_allocated() / 1e9,
            gpu_utilization_pct=torch.cuda.utilization()
        )
        self.metrics.append(metric)
    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
    
    def summary(self):
        if not self.metrics:
            return
        
        print("\n=== Metrics Summary ===")
        avg_throughput = sum(m.throughput_rps for m in self.metrics) / len(self.metrics)
        avg_latency = sum(m.avg_latency_ms for m in self.metrics) / len(self.metrics)
        print(f"Avg Throughput: {avg_throughput:.2f} req/s")
        print(f"Avg Latency: {avg_latency:.2f} ms")

# Usage
collector = MetricsCollector()
# ... run benchmarks, call collector.collect() ...
collector.summary()
collector.save("metrics.json")
```

---

## Practical Exercises

### Exercise 1.9.1: Basic Benchmarking
1. Run benchmark script
2. Measure throughput and latency
3. Document baseline performance

**Time:** 20 minutes

### Exercise 1.9.2: Batch Size Sweep
1. Test batch sizes 1, 10, 50, 100
2. Plot throughput vs batch size
3. Find optimal batch size

**Time:** 30 minutes

### Exercise 1.9.3: Configuration Comparison
1. Test different gpu_memory_utilization
2. Compare performance
3. Document tradeoffs

**Time:** 20 minutes

---

## Key Takeaways

1. **Throughput**: Requests/tokens per second
2. **Latency**: Time per request
3. **TTFT**: Important for streaming
4. **Batch size** affects throughput
5. **GPU utilization** indicates bottlenecks
6. **Monitor memory** for optimization

---

**Next:** 10_troubleshooting_guide.md
**Time:** 1.5 hours
**Source:** `/home/user/vllm-learn/vllm/engine/metrics.py`
