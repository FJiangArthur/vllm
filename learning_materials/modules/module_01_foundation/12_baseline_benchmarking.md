# Module 1.12: Baseline Benchmarking

## Learning Objectives

1. Establish performance baselines
2. Run standardized benchmarks
3. Compare different configurations
4. Document benchmark results
5. Identify optimization opportunities
6. Create benchmark reports

**Estimated Time:** 2 hours

---

## Benchmark Types

### 1. Latency Benchmark

Measure time per request:

```python
# latency_benchmark.py
import time
from vllm import LLM, SamplingParams

def benchmark_latency(model, num_runs=10):
    llm = LLM(model=model)
    params = SamplingParams(temperature=0.0, max_tokens=100)

    latencies = []

    # Warmup
    llm.generate(["warmup"], params)

    # Measure
    for i in range(num_runs):
        start = time.time()
        llm.generate(["Test prompt"], params)
        latency = time.time() - start
        latencies.append(latency * 1000)  # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"Model: {model}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"P50 Latency: {p50_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")

benchmark_latency("facebook/opt-125m")
```

### 2. Throughput Benchmark

Measure requests per second:

```python
# throughput_benchmark.py
import time
from vllm import LLM, SamplingParams

def benchmark_throughput(model, num_requests=100):
    llm = LLM(model=model)
    params = SamplingParams(temperature=0.0, max_tokens=50)

    prompts = [f"Request {i}" for i in range(num_requests)]

    # Warmup
    llm.generate(prompts[:5], params)

    # Measure
    start = time.time()
    outputs = llm.generate(prompts, params)
    duration = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    print(f"Model: {model}")
    print(f"Requests: {num_requests}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {num_requests / duration:.2f} req/s")
    print(f"Throughput: {total_tokens / duration:.2f} tokens/s")

benchmark_throughput("facebook/opt-125m")
```

### 3. Memory Benchmark

Measure memory usage:

```python
# memory_benchmark.py
import torch
from vllm import LLM, SamplingParams

def benchmark_memory(model):
    torch.cuda.empty_cache()

    # Baseline
    baseline = torch.cuda.memory_allocated() / 1e9

    # Load model
    llm = LLM(model=model, gpu_memory_utilization=0.9)
    model_mem = torch.cuda.memory_allocated() / 1e9 - baseline

    # Run inference
    prompts = ["Test"] * 100
    outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
    total_mem = torch.cuda.memory_allocated() / 1e9

    print(f"Model: {model}")
    print(f"Model Memory: {model_mem:.2f} GB")
    print(f"Total Memory: {total_mem:.2f} GB")
    print(f"Memory Utilization: {total_mem / (torch.cuda.get_device_properties(0).total_memory / 1e9) * 100:.1f}%")

benchmark_memory("facebook/opt-125m")
```

---

## Comprehensive Benchmark Suite

```python
# benchmark_suite.py
"""Complete benchmark suite for vLLM"""

import time
import torch
from vllm import LLM, SamplingParams
from dataclasses import dataclass
import json

@dataclass
class BenchmarkResults:
    model: str
    num_requests: int
    max_tokens: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    throughput_tps: float
    model_memory_gb: float
    total_memory_gb: float
    gpu_utilization_pct: float

def run_comprehensive_benchmark(
    model: str,
    num_requests: int = 100,
    max_tokens: int = 50,
    num_warmup: int = 5
) -> BenchmarkResults:
    """Run comprehensive benchmark"""

    # Initialize
    torch.cuda.empty_cache()
    baseline_mem = torch.cuda.memory_allocated() / 1e9

    llm = LLM(model=model, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    model_mem = torch.cuda.memory_allocated() / 1e9 - baseline_mem

    # Prepare prompts
    prompts = [f"Benchmark prompt {i}" for i in range(num_requests)]

    # Warmup
    print(f"Warming up with {num_warmup} requests...")
    llm.generate(prompts[:num_warmup], params)
    torch.cuda.synchronize()

    # Benchmark latency (single requests)
    print("Measuring latency...")
    latencies = []
    for i in range(min(20, num_requests)):
        start = time.time()
        llm.generate([prompts[i]], params)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)

    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]
    p99_latency = latencies[int(len(latencies) * 0.99)]

    # Benchmark throughput (batch)
    print(f"Measuring throughput with {num_requests} requests...")
    start = time.time()
    outputs = llm.generate(prompts, params)
    torch.cuda.synchronize()
    duration = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_rps = num_requests / duration
    throughput_tps = total_tokens / duration

    # Memory
    total_mem = torch.cuda.memory_allocated() / 1e9
    gpu_util = torch.cuda.utilization()

    return BenchmarkResults(
        model=model,
        num_requests=num_requests,
        max_tokens=max_tokens,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        throughput_rps=throughput_rps,
        throughput_tps=throughput_tps,
        model_memory_gb=model_mem,
        total_memory_gb=total_mem,
        gpu_utilization_pct=gpu_util
    )

def print_results(results: BenchmarkResults):
    """Print benchmark results"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {results.model}")
    print(f"Requests: {results.num_requests}")
    print(f"Max Tokens: {results.max_tokens}")
    print("\nLatency:")
    print(f"  Average: {results.avg_latency_ms:.2f} ms")
    print(f"  P50: {results.p50_latency_ms:.2f} ms")
    print(f"  P95: {results.p95_latency_ms:.2f} ms")
    print(f"  P99: {results.p99_latency_ms:.2f} ms")
    print("\nThroughput:")
    print(f"  {results.throughput_rps:.2f} requests/second")
    print(f"  {results.throughput_tps:.2f} tokens/second")
    print("\nMemory:")
    print(f"  Model: {results.model_memory_gb:.2f} GB")
    print(f"  Total: {results.total_memory_gb:.2f} GB")
    print(f"  GPU Utilization: {results.gpu_utilization_pct:.1f}%")
    print("=" * 60)

def save_results(results: BenchmarkResults, filename: str):
    """Save results to JSON"""
    with open(filename, 'w') as f:
        json.dump(results.__dict__, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Run benchmark
    results = run_comprehensive_benchmark(
        model="facebook/opt-125m",
        num_requests=100,
        max_tokens=50
    )

    # Print and save
    print_results(results)
    save_results(results, "benchmark_results.json")
```

---

## Model Comparison

```python
# compare_models.py
"""Compare multiple models"""

from benchmark_suite import run_comprehensive_benchmark, print_results

models = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
]

results = []

for model in models:
    print(f"\nBenchmarking {model}...")
    result = run_comprehensive_benchmark(
        model=model,
        num_requests=50,
        max_tokens=50
    )
    results.append(result)
    print_results(result)

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"{'Model':<25} {'Latency (ms)':<15} {'Throughput (req/s)':<20} {'Memory (GB)'}")
print("-" * 80)

for r in results:
    print(f"{r.model:<25} {r.avg_latency_ms:<15.2f} {r.throughput_rps:<20.2f} {r.total_memory_gb:.2f}")
```

---

## Configuration Sweep

```python
# config_sweep.py
"""Test different configurations"""

configs = [
    {"name": "Default", "gpu_mem_util": 0.9, "enforce_eager": False},
    {"name": "Low Memory", "gpu_mem_util": 0.7, "enforce_eager": False},
    {"name": "High Memory", "gpu_mem_util": 0.95, "enforce_eager": False},
    {"name": "Eager Mode", "gpu_mem_util": 0.9, "enforce_eager": True},
]

import time
from vllm import LLM, SamplingParams

model = "facebook/opt-125m"
prompts = ["Test"] * 50
params = SamplingParams(temperature=0.0, max_tokens=50)

print(f"{'Configuration':<20} {'Throughput (req/s)':<20} {'Latency (ms)'}")
print("-" * 60)

for config in configs:
    llm = LLM(
        model=model,
        gpu_memory_utilization=config["gpu_mem_util"],
        enforce_eager=config["enforce_eager"]
    )

    # Warmup
    llm.generate(prompts[:5], params)

    # Measure throughput
    start = time.time()
    outputs = llm.generate(prompts, params)
    duration = time.time() - start
    throughput = len(prompts) / duration

    # Measure latency
    start = time.time()
    llm.generate([prompts[0]], params)
    latency = (time.time() - start) * 1000

    print(f"{config['name']:<20} {throughput:<20.2f} {latency:.2f}")

    del llm  # Free memory
```

---

## Baseline Documentation

Create a baseline report:

```markdown
# vLLM Baseline Benchmark Report

**Date:** 2025-11-19
**GPU:** NVIDIA A100 40GB
**CUDA:** 12.1
**vLLM Version:** 0.2.7

## Test Configuration
- Model: facebook/opt-125m
- Batch Size: 100 requests
- Max Tokens: 50
- GPU Memory Utilization: 0.9

## Results

### Latency
- Average: 45.32 ms
- P50: 43.21 ms
- P95: 52.10 ms
- P99: 58.44 ms

### Throughput
- 156.78 requests/second
- 7,839 tokens/second

### Memory
- Model: 0.24 GB
- Total: 2.15 GB
- Utilization: 5.4%

## Conclusions
- Baseline performance established
- GPU underutilized (5.4%)
- Opportunity for optimization
```

---

## Practical Exercises

### Exercise 1.12.1: Run Baseline
1. Run comprehensive benchmark
2. Document results
3. Save JSON output

**Time:** 20 minutes

### Exercise 1.12.2: Compare Models
1. Benchmark 3 different model sizes
2. Compare performance
3. Create comparison table

**Time:** 30 minutes

### Exercise 1.12.3: Configuration Sweep
1. Test 5 different configurations
2. Find optimal settings
3. Document findings

**Time:** 30 minutes

---

## Key Takeaways

1. **Baseline benchmarks** establish performance expectations
2. **Comprehensive testing** covers latency, throughput, memory
3. **Document results** for future reference and comparison
4. **Compare configurations** to find optimal settings
5. **Regular benchmarking** tracks performance over time
6. **JSON export** enables analysis and visualization

---

## Module 1 Complete!

You've completed Module 1: Foundation & Setup!

**What you learned:**
1. Installation and setup
2. CUDA configuration
3. Environment management
4. Project structure
5. Basic inference API
6. Model loading
7. Sampling parameters
8. Logging and debugging
9. Performance metrics
10. Troubleshooting
11. Dependency management
12. Baseline benchmarking

**Next:** Module 2: Core Concepts
**Time:** 2 hours
**Source:** `/home/user/vllm-learn/benchmarks/`
