# Project 7.4: Real-Time Monitoring Dashboard

## Project Overview

**Duration:** 2-3 hours
**Difficulty:** ⭐⭐☆☆☆ (Beginner-Intermediate)
**Prerequisites:** Basic web development, metrics knowledge

### Objective

Create a comprehensive, real-time monitoring dashboard for vLLM inference that tracks key performance metrics, visualizes system health, and provides alerting capabilities. The dashboard will give operators immediate visibility into system performance and help identify issues before they impact users.

### Why This Matters

"You can't improve what you don't measure." Production ML systems require robust observability:
- **Detect issues early** (latency spikes, memory leaks, GPU failures)
- **Optimize performance** (identify bottlenecks, resource waste)
- **Track SLAs** (ensure latency/throughput targets met)
- **Debug problems** (correlate metrics with incidents)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   vLLM Service                            │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Instrumentation                                │     │
│  │  - Request counters                             │     │
│  │  - Latency histograms                           │     │
│  │  - GPU metrics                                  │     │
│  │  - KV cache stats                               │     │
│  └──────────────────┬──────────────────────────────┘     │
└─────────────────────┼────────────────────────────────────┘
                      │
                      │ (Prometheus scrape)
                      ▼
          ┌───────────────────────┐
          │   Prometheus Server   │
          │   (Time-series DB)    │
          └───────────┬───────────┘
                      │
                      │ (Query metrics)
                      ▼
          ┌───────────────────────┐
          │   Grafana Dashboard   │
          │   (Visualization)     │
          └───────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
     ┌────▼─────┐         ┌──────▼─────┐
     │ Alerting │         │   Users    │
     │ (Email/  │         │ (Web UI)   │
     │  Slack)  │         │            │
     └──────────┘         └────────────┘
```

---

## Requirements

### Core Metrics to Track

#### Request Metrics
- **Request rate** (requests/second)
- **Latency percentiles** (P50, P95, P99, P99.9)
- **Time-to-first-token** (TTFT)
- **Tokens per second** (throughput)
- **Request success rate** (%)
- **Error rate by type**

#### GPU Metrics
- **GPU utilization** (%)
- **GPU memory utilization** (%)
- **GPU temperature** (°C)
- **SM efficiency**
- **Memory bandwidth utilization**

#### vLLM-Specific Metrics
- **KV cache utilization** (%)
- **Number of running sequences**
- **Number of waiting sequences**
- **Number of swapped sequences**
- **Block table occupancy**
- **Scheduler iterations per second**

#### System Metrics
- **CPU utilization**
- **System memory usage**
- **Network I/O**
- **Disk I/O**

### Dashboard Features

#### FR1: Real-Time Updates
- Update metrics every 1-5 seconds
- Support auto-refresh
- Show "last updated" timestamp

#### FR2: Multiple Views
- **Overview**: High-level health snapshot
- **Performance**: Detailed latency/throughput analysis
- **Resources**: GPU/CPU/memory deep-dive
- **Errors**: Error tracking and debugging

#### FR3: Time Range Selection
- Last 5 minutes, 1 hour, 6 hours, 24 hours, 7 days
- Custom time range picker
- Time zone support

#### FR4: Alerting
- Define alert rules (e.g., P99 latency > 500ms)
- Multiple notification channels (email, Slack, PagerDuty)
- Alert history and acknowledgment

---

## Implementation Guide

### Step 1: Instrumentation (45 minutes)

```python
# monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Summary
from prometheus_client import start_http_server
import time

class VLLMMetrics:
    """
    Prometheus metrics for vLLM inference service.
    """

    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            'vllm_requests_total',
            'Total number of inference requests',
            ['status']  # success, error
        )

        self.request_latency = Histogram(
            'vllm_request_latency_seconds',
            'Request latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        self.ttft_latency = Histogram(
            'vllm_time_to_first_token_seconds',
            'Time to first token in seconds',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        )

        self.tokens_generated = Counter(
            'vllm_tokens_generated_total',
            'Total number of tokens generated'
        )

        self.throughput = Gauge(
            'vllm_throughput_tokens_per_second',
            'Current throughput in tokens/second'
        )

        # GPU metrics
        self.gpu_utilization = Gauge(
            'vllm_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )

        self.gpu_memory_used = Gauge(
            'vllm_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id']
        )

        self.gpu_memory_total = Gauge(
            'vllm_gpu_memory_total_bytes',
            'GPU memory total in bytes',
            ['gpu_id']
        )

        # vLLM-specific metrics
        self.kv_cache_utilization = Gauge(
            'vllm_kv_cache_utilization_percent',
            'KV cache utilization percentage'
        )

        self.num_running_sequences = Gauge(
            'vllm_num_running_sequences',
            'Number of currently running sequences'
        )

        self.num_waiting_sequences = Gauge(
            'vllm_num_waiting_sequences',
            'Number of waiting sequences'
        )

        self.num_swapped_sequences = Gauge(
            'vllm_num_swapped_sequences',
            'Number of swapped sequences'
        )

        self.scheduler_iterations = Counter(
            'vllm_scheduler_iterations_total',
            'Total scheduler iterations'
        )

    def record_request(self, latency: float, status: str, tokens: int, ttft: float):
        """Record a completed request."""
        self.request_counter.labels(status=status).inc()
        self.request_latency.observe(latency)
        self.ttft_latency.observe(ttft)
        self.tokens_generated.inc(tokens)

    def update_gpu_metrics(self, gpu_id: int):
        """Update GPU metrics using pynvml."""
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        # Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization.gpu)

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.gpu_memory_used.labels(gpu_id=str(gpu_id)).set(mem_info.used)
        self.gpu_memory_total.labels(gpu_id=str(gpu_id)).set(mem_info.total)

        pynvml.nvmlShutdown()

    def update_vllm_metrics(self, scheduler_state: dict):
        """Update vLLM-specific metrics."""
        self.num_running_sequences.set(scheduler_state.get('num_running', 0))
        self.num_waiting_sequences.set(scheduler_state.get('num_waiting', 0))
        self.num_swapped_sequences.set(scheduler_state.get('num_swapped', 0))
        self.kv_cache_utilization.set(scheduler_state.get('kv_cache_util', 0))

# Global metrics instance
metrics = VLLMMetrics()

# Start Prometheus server
def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
```

### Step 2: Integration with vLLM (30 minutes)

```python
# Add to vLLM server code

from monitoring.metrics import metrics
import time

class InstrumentedLLMEngine:
    """Wrapper around vLLM LLM engine with metrics."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompts, sampling_params):
        """Generate with metrics tracking."""
        start_time = time.time()

        try:
            # Track TTFT
            ttft_start = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            ttft = time.time() - ttft_start

            # Calculate metrics
            total_latency = time.time() - start_time
            total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)

            # Record metrics
            metrics.record_request(
                latency=total_latency,
                status='success',
                tokens=total_tokens,
                ttft=ttft
            )

            return outputs

        except Exception as e:
            metrics.record_request(
                latency=time.time() - start_time,
                status='error',
                tokens=0,
                ttft=0
            )
            raise

# Background thread to update GPU metrics
import threading

def gpu_metrics_updater(gpu_id: int, interval: int = 5):
    """Periodically update GPU metrics."""
    while True:
        metrics.update_gpu_metrics(gpu_id)
        time.sleep(interval)

# Start background thread
gpu_thread = threading.Thread(
    target=gpu_metrics_updater,
    args=(0, 5),
    daemon=True
)
gpu_thread.start()
```

### Step 3: Prometheus Configuration (15 minutes)

```yaml
# prometheus.yml

global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          instance: 'vllm-gpu-0'

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']  # node_exporter for system metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rule_files:
  - 'alerts.yml'
```

```yaml
# alerts.yml

groups:
  - name: vllm_alerts
    interval: 10s
    rules:
      # High latency alert
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(vllm_request_latency_seconds_bucket[5m])) > 2.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High P99 latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 2s)"

      # High error rate
      - alert: HighErrorRate
        expr: rate(vllm_requests_total{status="error"}[5m]) / rate(vllm_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # GPU memory exhaustion
      - alert: GPUMemoryHigh
        expr: (vllm_gpu_memory_used_bytes / vllm_gpu_memory_total_bytes) > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU {{ $labels.gpu_id }} memory usage is {{ $value | humanizePercentage }}"

      # Low throughput
      - alert: LowThroughput
        expr: vllm_throughput_tokens_per_second < 100
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Low throughput detected"
          description: "Throughput is {{ $value }} tokens/s"
```

### Step 4: Grafana Dashboards (60 minutes)

```json
{
  "dashboard": {
    "title": "vLLM Inference Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(vllm_requests_total[1m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(vllm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(vllm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(vllm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "vllm_gpu_utilization_percent",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      },
      {
        "title": "KV Cache Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "vllm_kv_cache_utilization_percent"
          }
        ],
        "options": {
          "thresholds": [
            { "value": 80, "color": "yellow" },
            { "value": 95, "color": "red" }
          ]
        }
      }
    ]
  }
}
```

---

## Example Dashboards

### Dashboard 1: System Overview

**Panels:**
1. **Request Rate** (graph) - requests/second over time
2. **Error Rate** (gauge) - current error percentage
3. **P99 Latency** (gauge) - current P99 latency
4. **GPU Utilization** (gauge) - average GPU utilization
5. **Active Requests** (stat) - current running sequences
6. **Throughput** (graph) - tokens/second over time

### Dashboard 2: Performance Deep Dive

**Panels:**
1. **Latency Heatmap** - distribution of latencies
2. **TTFT vs Total Latency** - scatter plot
3. **Throughput by Batch Size** - grouped bar chart
4. **Request Duration Breakdown** - stacked area chart (prefill vs decode)

### Dashboard 3: Resource Monitoring

**Panels:**
1. **GPU Memory Timeline** - memory usage over time
2. **GPU Temperature** - temperature monitoring
3. **CPU Usage** - system CPU utilization
4. **Network I/O** - bytes sent/received
5. **KV Cache Blocks** - block allocation/deallocation

### Dashboard 4: Error Analysis

**Panels:**
1. **Error Rate by Type** - pie chart
2. **Error Timeline** - errors over time
3. **Failed Request Details** - table with recent errors
4. **OOM Events** - out-of-memory incidents

---

## Testing

```python
# tests/test_metrics.py

import pytest
from monitoring.metrics import VLLMMetrics

def test_metrics_recording():
    """Test metrics are recorded correctly."""
    metrics = VLLMMetrics()

    # Record some requests
    metrics.record_request(latency=0.5, status='success', tokens=100, ttft=0.1)
    metrics.record_request(latency=1.0, status='success', tokens=200, ttft=0.2)
    metrics.record_request(latency=2.0, status='error', tokens=0, ttft=0.0)

    # Check counters
    assert metrics.request_counter.labels(status='success')._value._value == 2
    assert metrics.request_counter.labels(status='error')._value._value == 1
    assert metrics.tokens_generated._value._value == 300

def test_gpu_metrics_update():
    """Test GPU metrics update."""
    metrics = VLLMMetrics()
    metrics.update_gpu_metrics(gpu_id=0)

    # Should have set values (can't test exact values)
    # Just verify no exceptions
```

---

## Deliverables

1. **Metrics Implementation**
   - Prometheus instrumentation
   - vLLM integration
   - Background metric updaters

2. **Prometheus Setup**
   - Configuration file
   - Alert rules
   - Recording rules (optional)

3. **Grafana Dashboards**
   - 4+ comprehensive dashboards (JSON exports)
   - Panel descriptions
   - Variable templates

4. **Documentation**
   - Metrics glossary
   - Dashboard guide
   - Troubleshooting runbook

5. **Deployment**
   - Docker Compose for full stack
   - Setup scripts

---

## Evaluation Rubric

### Functionality (40 points)
- ✅ Metrics collection: 15 pts
- ✅ Dashboards functional: 15 pts
- ✅ Alerting works: 10 pts

### Usability (30 points)
- ✅ Dashboard clarity: 15 pts
- ✅ Metric coverage: 15 pts

### Documentation (20 points)
- ✅ Setup guide: 10 pts
- ✅ Metrics explained: 10 pts

### Deployment (10 points)
- ✅ Easy to deploy: 10 pts

**Total: 100 points | Passing: 70 points**

---

## Common Pitfalls

1. **Too Many Metrics**: Focus on actionable metrics
2. **Poor Naming**: Use consistent, descriptive metric names
3. **Wrong Metric Types**: Counter vs Gauge vs Histogram
4. **High Cardinality**: Avoid unbounded label values
5. **Missing Units**: Always specify units (seconds, bytes, etc.)

---

## Extensions

1. **Custom Visualizations**: Build React dashboard
2. **Anomaly Detection**: ML-based alert threshold tuning
3. **Cost Tracking**: Monitor GPU costs per request
4. **User Analytics**: Track per-user metrics
5. **Distributed Tracing**: Integrate Jaeger/Zipkin

---

**Build visibility into your system!**

*Project 7.4: Monitoring Dashboard | Duration: 2-3 hours*
