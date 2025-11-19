# Project 7.6: Comprehensive Benchmarking Suite

## Project Overview

**Duration:** 3-4 hours
**Difficulty:** ⭐⭐⭐⭐☆ (Advanced)
**Prerequisites:** Understanding of statistics, performance testing

### Objective

Create a flexible, reproducible benchmarking framework that simulates realistic workload patterns, measures key performance metrics with statistical rigor, and generates publication-quality reports for vLLM inference systems.

### Why This Matters

Rigorous benchmarking drives optimization decisions:
- **Validate optimizations**: Prove changes improve performance
- **Compare systems**: Objective comparison vs competitors
- **Capacity planning**: Understand system limits
- **Regression detection**: Catch performance degradations

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Workload Generator                             │
│  - Poisson arrivals                             │
│  - Bursty traffic                               │
│  - Uniform load                                 │
│  - Realistic traces                             │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Request Characteristics                         │
│  - Prompt length distribution                   │
│  - Output length distribution                   │
│  - Priority/SLA requirements                    │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Benchmark Runner                                │
│  - Execute workload                             │
│  - Collect metrics                              │
│  - Monitor system state                         │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Statistical Analysis                            │
│  - Latency percentiles (P50, P95, P99, P99.9)  │
│  - Throughput (tokens/s, requests/s)            │
│  - Resource utilization (GPU, memory)           │
│  - Confidence intervals                         │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Report Generator                                │
│  - Statistical tables                           │
│  - Performance charts                           │
│  - Comparative analysis                         │
│  - Recommendations                              │
└─────────────────────────────────────────────────┘
```

---

## Requirements

### FR1: Workload Patterns
- **Poisson Process**: Realistic arrival pattern (λ arrivals/sec)
- **Bursty Traffic**: Traffic spikes (beta distribution)
- **Uniform Load**: Constant arrival rate
- **Trace Replay**: Use real production traces
- **Configurable Parameters**: Arrival rate, duration, warmup period

### FR2: Request Characteristics
- **Prompt Length**: Normal distribution (μ, σ)
- **Output Length**: Configurable distribution
- **Batching**: Single vs batched requests
- **Priority Levels**: Support multi-priority workloads

### FR3: Metrics Collection
- **Latency**: End-to-end, TTFT, per-token
- **Throughput**: Tokens/s, requests/s
- **Resource Usage**: GPU util, memory, CPU
- **Queue Stats**: Queue length, wait time
- **Error Rates**: Failed requests, timeouts

### FR4: Statistical Rigor
- **Multiple Runs**: ≥3 runs per configuration
- **Confidence Intervals**: 95% CI for all metrics
- **Outlier Detection**: Identify and handle outliers
- **Hypothesis Testing**: Compare configurations statistically

---

## Implementation

### Step 1: Workload Generator (60 minutes)

```python
# src/workload_generator.py

import numpy as np
from typing import List, Iterator
from dataclasses import dataclass
from enum import Enum
import time

class ArrivalPattern(Enum):
    """Workload arrival patterns."""
    POISSON = "poisson"
    UNIFORM = "uniform"
    BURSTY = "bursty"
    TRACE = "trace"

@dataclass
class Request:
    """A benchmark request."""
    request_id: int
    arrival_time: float
    prompt_length: int
    expected_output_length: int
    priority: int = 0

class WorkloadGenerator:
    """
    Generate realistic workload patterns for benchmarking.
    """

    def __init__(
        self,
        pattern: ArrivalPattern,
        arrival_rate: float,  # requests per second
        duration: float,  # seconds
        prompt_length_dist: tuple = (100, 50),  # (mean, std)
        output_length_dist: tuple = (100, 30),
    ):
        self.pattern = pattern
        self.arrival_rate = arrival_rate
        self.duration = duration
        self.prompt_mean, self.prompt_std = prompt_length_dist
        self.output_mean, self.output_std = output_length_dist

    def generate(self) -> Iterator[Request]:
        """
        Generate requests according to pattern.

        Yields:
            Request objects with arrival times
        """
        if self.pattern == ArrivalPattern.POISSON:
            yield from self._generate_poisson()
        elif self.pattern == ArrivalPattern.UNIFORM:
            yield from self._generate_uniform()
        elif self.pattern == ArrivalPattern.BURSTY:
            yield from self._generate_bursty()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def _generate_poisson(self) -> Iterator[Request]:
        """Generate Poisson arrival process."""
        current_time = 0.0
        request_id = 0

        while current_time < self.duration:
            # Inter-arrival time: exponential distribution
            inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival

            if current_time >= self.duration:
                break

            # Create request
            yield self._create_request(request_id, current_time)
            request_id += 1

    def _generate_uniform(self) -> Iterator[Request]:
        """Generate uniform arrival pattern."""
        num_requests = int(self.arrival_rate * self.duration)
        inter_arrival = 1.0 / self.arrival_rate

        for i in range(num_requests):
            arrival_time = i * inter_arrival
            yield self._create_request(i, arrival_time)

    def _generate_bursty(self) -> Iterator[Request]:
        """
        Generate bursty traffic pattern.

        Bursts occur with beta distribution for variability.
        """
        current_time = 0.0
        request_id = 0

        while current_time < self.duration:
            # Burst or quiet period?
            is_burst = np.random.random() < 0.3  # 30% of time is burst

            if is_burst:
                # High rate during burst
                burst_rate = self.arrival_rate * 5
                burst_duration = 2.0  # 2 seconds
                num_requests = int(burst_rate * burst_duration)

                for _ in range(num_requests):
                    inter_arrival = np.random.exponential(1.0 / burst_rate)
                    current_time += inter_arrival

                    if current_time >= self.duration:
                        break

                    yield self._create_request(request_id, current_time)
                    request_id += 1
            else:
                # Quiet period: low rate
                quiet_duration = 5.0
                quiet_rate = self.arrival_rate * 0.2
                num_requests = int(quiet_rate * quiet_duration)

                for _ in range(num_requests):
                    inter_arrival = np.random.exponential(1.0 / quiet_rate)
                    current_time += inter_arrival

                    if current_time >= self.duration:
                        break

                    yield self._create_request(request_id, current_time)
                    request_id += 1

    def _create_request(self, request_id: int, arrival_time: float) -> Request:
        """Create a request with sampled characteristics."""
        # Sample prompt length (positive, truncated normal)
        prompt_length = max(10, int(np.random.normal(
            self.prompt_mean,
            self.prompt_std
        )))

        # Sample output length
        output_length = max(10, int(np.random.normal(
            self.output_mean,
            self.output_std
        )))

        return Request(
            request_id=request_id,
            arrival_time=arrival_time,
            prompt_length=prompt_length,
            expected_output_length=output_length,
        )
```

### Step 2: Benchmark Runner (60 minutes)

```python
# src/benchmark_runner.py

import asyncio
import time
from typing import List, Dict
from dataclasses import dataclass, field
import httpx
import numpy as np

@dataclass
class BenchmarkResult:
    """Results from a single request."""
    request_id: int
    arrival_time: float
    start_time: float
    end_time: float
    ttft: float  # time to first token
    tokens_generated: int
    success: bool
    error: str = ""

    @property
    def latency(self) -> float:
        """Total latency in seconds."""
        return self.end_time - self.start_time

    @property
    def wait_time(self) -> float:
        """Time waiting in queue."""
        return self.start_time - self.arrival_time

class BenchmarkRunner:
    """
    Execute benchmark workload and collect metrics.
    """

    def __init__(
        self,
        server_url: str,
        max_concurrent: int = 100,
        timeout: float = 30.0,
    ):
        self.server_url = server_url
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.results: List[BenchmarkResult] = []

    async def run_benchmark(
        self,
        workload: List[Request],
        warmup_requests: int = 10,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark with given workload.

        Args:
            workload: List of requests to execute
            warmup_requests: Number of warmup requests

        Returns:
            List of benchmark results
        """
        print(f"Running benchmark with {len(workload)} requests")

        # Warmup
        if warmup_requests > 0:
            print(f"Warming up with {warmup_requests} requests...")
            await self._run_warmup(warmup_requests)

        # Run benchmark
        self.results = []
        benchmark_start = time.time()

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create tasks for all requests
        tasks = [
            self._execute_request(req, semaphore, benchmark_start)
            for req in workload
        ]

        # Execute all requests
        await asyncio.gather(*tasks)

        print(f"Benchmark complete: {len(self.results)} results collected")
        return self.results

    async def _execute_request(
        self,
        request: Request,
        semaphore: asyncio.Semaphore,
        benchmark_start: float,
    ):
        """Execute a single request with timing."""
        # Wait until arrival time
        wait_time = request.arrival_time - (time.time() - benchmark_start)
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        actual_arrival = time.time()

        # Acquire semaphore (limit concurrency)
        async with semaphore:
            start_time = time.time()

            try:
                # Send request
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Generate prompt
                    prompt = self._generate_prompt(request.prompt_length)

                    # Make request
                    response = await client.post(
                        f"{self.server_url}/v1/completions",
                        json={
                            "prompt": prompt,
                            "max_tokens": request.expected_output_length,
                            "stream": False,
                        },
                    )

                    end_time = time.time()

                    if response.status_code == 200:
                        data = response.json()

                        result = BenchmarkResult(
                            request_id=request.request_id,
                            arrival_time=actual_arrival,
                            start_time=start_time,
                            end_time=end_time,
                            ttft=data.get('ttft', 0),  # if server provides it
                            tokens_generated=data.get('tokens_generated', 0),
                            success=True,
                        )
                    else:
                        result = BenchmarkResult(
                            request_id=request.request_id,
                            arrival_time=actual_arrival,
                            start_time=start_time,
                            end_time=end_time,
                            ttft=0,
                            tokens_generated=0,
                            success=False,
                            error=f"HTTP {response.status_code}",
                        )

            except Exception as e:
                result = BenchmarkResult(
                    request_id=request.request_id,
                    arrival_time=actual_arrival,
                    start_time=start_time,
                    end_time=time.time(),
                    ttft=0,
                    tokens_generated=0,
                    success=False,
                    error=str(e),
                )

            self.results.append(result)

    def _generate_prompt(self, length: int) -> str:
        """Generate prompt of specified token length."""
        # Approximate: 1 word ≈ 1.3 tokens
        num_words = int(length / 1.3)
        words = ["test"] * num_words
        return " ".join(words)

    async def _run_warmup(self, num_requests: int):
        """Run warmup requests."""
        warmup_requests = [
            Request(
                request_id=i,
                arrival_time=0,
                prompt_length=100,
                expected_output_length=50,
            )
            for i in range(num_requests)
        ]

        for req in warmup_requests:
            await self._execute_request(req, asyncio.Semaphore(1), time.time())
```

### Step 3: Statistical Analysis (45 minutes)

```python
# src/statistics.py

import numpy as np
from scipy import stats
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class StatisticalSummary:
    """Statistical summary of metrics."""
    mean: float
    median: float
    std: float
    p50: float
    p95: float
    p99: float
    p99_9: float
    min: float
    max: float
    confidence_interval_95: tuple  # (lower, upper)

class StatisticalAnalyzer:
    """Analyze benchmark results with statistical rigor."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.successful_results = [r for r in results if r.success]

    def analyze(self) -> Dict[str, StatisticalSummary]:
        """
        Perform comprehensive statistical analysis.

        Returns:
            Dict of metric name -> statistical summary
        """
        metrics = {}

        # Latency analysis
        latencies = [r.latency for r in self.successful_results]
        metrics['latency'] = self._compute_statistics(latencies)

        # TTFT analysis
        ttfts = [r.ttft for r in self.successful_results if r.ttft > 0]
        if ttfts:
            metrics['ttft'] = self._compute_statistics(ttfts)

        # Wait time analysis
        wait_times = [r.wait_time for r in self.successful_results]
        metrics['wait_time'] = self._compute_statistics(wait_times)

        # Throughput analysis
        total_time = max(r.end_time for r in self.results) - \
                     min(r.arrival_time for r in self.results)
        total_tokens = sum(r.tokens_generated for r in self.successful_results)
        metrics['throughput_tokens_per_sec'] = total_tokens / total_time

        # Success rate
        metrics['success_rate'] = len(self.successful_results) / len(self.results)

        return metrics

    def _compute_statistics(self, data: List[float]) -> StatisticalSummary:
        """Compute statistical summary for a metric."""
        if not data:
            return None

        data_array = np.array(data)

        # Basic statistics
        mean = np.mean(data_array)
        median = np.median(data_array)
        std = np.std(data_array)

        # Percentiles
        p50 = np.percentile(data_array, 50)
        p95 = np.percentile(data_array, 95)
        p99 = np.percentile(data_array, 99)
        p99_9 = np.percentile(data_array, 99.9)

        # Min/max
        min_val = np.min(data_array)
        max_val = np.max(data_array)

        # 95% confidence interval
        confidence_interval = stats.t.interval(
            0.95,
            len(data_array) - 1,
            loc=mean,
            scale=stats.sem(data_array)
        )

        return StatisticalSummary(
            mean=mean,
            median=median,
            std=std,
            p50=p50,
            p95=p95,
            p99=p99,
            p99_9=p99_9,
            min=min_val,
            max=max_val,
            confidence_interval_95=confidence_interval,
        )

    def compare_configurations(
        self,
        other: 'StatisticalAnalyzer',
        metric: str = 'latency',
    ) -> Dict:
        """
        Compare two configurations statistically.

        Returns hypothesis test results.
        """
        data1 = [r.latency for r in self.successful_results]
        data2 = [r.latency for r in other.successful_results]

        # T-test
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # Effect size (Cohen's d)
        cohens_d = (np.mean(data1) - np.mean(data2)) / \
                   np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
        }
```

### Step 4: Report Generation (45 minutes)

```python
# src/report_generator.py

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

class BenchmarkReportGenerator:
    """Generate comprehensive benchmark reports."""

    def __init__(self, results: Dict[str, List[BenchmarkResult]]):
        """
        Args:
            results: Dict mapping config name -> benchmark results
        """
        self.results = results

    def generate_report(self, output_dir: str):
        """Generate complete report."""
        print(f"Generating report in {output_dir}")

        # Summary table
        self._generate_summary_table(output_dir)

        # Latency distribution
        self._plot_latency_distribution(output_dir)

        # Percentile comparison
        self._plot_percentile_comparison(output_dir)

        # Throughput comparison
        self._plot_throughput_comparison(output_dir)

        # Time series
        self._plot_latency_over_time(output_dir)

        # Generate markdown
        self._generate_markdown(output_dir)

    def _generate_summary_table(self, output_dir):
        """Create summary statistics table."""
        data = []

        for config, results in self.results.items():
            analyzer = StatisticalAnalyzer(results)
            metrics = analyzer.analyze()

            row = {
                'Configuration': config,
                'Success Rate (%)': metrics['success_rate'] * 100,
                'Throughput (tok/s)': metrics['throughput_tokens_per_sec'],
                'P50 Latency (ms)': metrics['latency'].p50 * 1000,
                'P95 Latency (ms)': metrics['latency'].p95 * 1000,
                'P99 Latency (ms)': metrics['latency'].p99 * 1000,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/summary.csv", index=False)
        print(df.to_string(index=False))

    def _plot_latency_distribution(self, output_dir):
        """Plot latency distribution (histogram)."""
        fig, axes = plt.subplots(
            len(self.results), 1,
            figsize=(10, 4 * len(self.results))
        )

        if len(self.results) == 1:
            axes = [axes]

        for ax, (config, results) in zip(axes, self.results.items()):
            latencies = [r.latency * 1000 for r in results if r.success]

            ax.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{config} - Latency Distribution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png", dpi=300)
        plt.close()

    def _plot_percentile_comparison(self, output_dir):
        """Plot percentile comparison across configs."""
        fig, ax = plt.subplots(figsize=(10, 6))

        percentiles = [50, 75, 90, 95, 99, 99.9]
        width = 0.8 / len(self.results)

        for i, (config, results) in enumerate(self.results.items()):
            latencies = [r.latency * 1000 for r in results if r.success]
            values = [np.percentile(latencies, p) for p in percentiles]

            x = np.arange(len(percentiles))
            ax.bar(x + i * width, values, width, label=config)

        ax.set_xlabel('Percentile')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Percentiles Comparison')
        ax.set_xticks(np.arange(len(percentiles)) + width * (len(self.results) - 1) / 2)
        ax.set_xticklabels([f'P{p}' for p in percentiles])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/percentile_comparison.png", dpi=300)
        plt.close()
```

---

## Configuration

```yaml
# configs/benchmark_config.yaml

workloads:
  - name: "low_load"
    pattern: "poisson"
    arrival_rate: 1.0  # req/s
    duration: 300  # seconds

  - name: "medium_load"
    pattern: "poisson"
    arrival_rate: 10.0
    duration: 300

  - name: "high_load"
    pattern: "poisson"
    arrival_rate: 50.0
    duration: 300

  - name: "bursty"
    pattern: "bursty"
    arrival_rate: 10.0
    duration: 600

request_characteristics:
  prompt_length:
    mean: 512
    std: 128
  output_length:
    mean: 128
    std: 32

benchmark_settings:
  num_runs: 3
  warmup_requests: 20
  max_concurrent: 100
  timeout: 30.0

server:
  url: "http://localhost:8000"

output_dir: "benchmark_results/"
```

---

## Deliverables

1. **Benchmarking Suite** (workload generation, runner, analysis)
2. **Statistical Analysis** (rigorous metrics with CI)
3. **Report Generation** (tables, charts, markdown)
4. **Configuration System** (YAML-based)
5. **Documentation** (usage guide, methodology)

---

## Evaluation Rubric

**Total: 100 points | Passing: 70 points**

---

*Project 7.6: Benchmarking Suite | Duration: 3-4 hours*
