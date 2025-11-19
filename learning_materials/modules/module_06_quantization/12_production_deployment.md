# Production Deployment of Quantized LLMs

## Table of Contents
1. [Introduction](#introduction)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Architecture](#deployment-architecture)
4. [Configuration Management](#configuration-management)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Cost Optimization](#cost-optimization)

---

## Introduction

### Production Deployment Goals

Deploying quantized LLMs in production requires balancing multiple objectives:

```
Production Requirements:
├── Performance
│   ├── Low latency (P99 < target SLA)
│   ├── High throughput (tokens/second)
│   └── Consistent performance (low jitter)
├── Reliability
│   ├── High availability (99.9%+)
│   ├── Graceful degradation
│   └── Fault tolerance
├── Observability
│   ├── Comprehensive metrics
│   ├── Distributed tracing
│   └── Alerting
└── Cost Efficiency
    ├── Resource utilization (>80% GPU)
    ├── Autoscaling
    └── Cost per token optimization
```

---

## Pre-Deployment Checklist

### Comprehensive Production Checklist

```python
PRODUCTION_DEPLOYMENT_CHECKLIST = {
    "model_preparation": [
        "✓ Model quantized and validated",
        "✓ Accuracy benchmarks completed",
        "✓ Perplexity < threshold on test set",
        "✓ Downstream task performance verified",
        "✓ Model artifacts stored in versioned registry",
        "✓ Model size optimized for target GPU",
    ],

    "performance_validation": [
        "✓ Latency benchmarks on target hardware",
        "✓ Throughput tested with realistic load",
        "✓ Memory usage profiled",
        "✓ CUDA graph optimization enabled",
        "✓ Optimal batch size determined",
        "✓ Load testing completed",
    ],

    "infrastructure": [
        "✓ GPU infrastructure provisioned",
        "✓ Kubernetes cluster configured (if applicable)",
        "✓ Load balancer set up",
        "✓ Auto-scaling policies defined",
        "✓ Health checks implemented",
        "✓ Backup/failover strategy defined",
    ],

    "monitoring": [
        "✓ Metrics collection configured",
        "✓ Logging aggregation set up",
        "✓ Distributed tracing enabled",
        "✓ Alerting rules defined",
        "✓ Dashboards created",
        "✓ On-call procedures documented",
    ],

    "security": [
        "✓ API authentication implemented",
        "✓ Rate limiting configured",
        "✓ Input validation in place",
        "✓ Output filtering implemented",
        "✓ Secrets management configured",
        "✓ Security audit completed",
    ],

    "operations": [
        "✓ Deployment automation (CI/CD)",
        "✓ Rollback procedures tested",
        "✓ Runbooks created",
        "✓ Incident response plan",
        "✓ Capacity planning completed",
        "✓ Cost tracking enabled",
    ],
}


def verify_production_readiness():
    """Verify all checklist items are complete."""
    total_items = sum(len(items) for items in PRODUCTION_DEPLOYMENT_CHECKLIST.values())
    print(f"Production Deployment Checklist ({total_items} items)\n")

    for category, items in PRODUCTION_DEPLOYMENT_CHECKLIST.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for item in items:
            print(f"  {item}")

    print(f"\n{'='*60}")
    print("Review each item before deploying to production!")
    print(f"{'='*60}")
```

### Model Validation Script

```python
def validate_model_for_production(
    model_name,
    quantization_method,
    acceptance_criteria,
):
    """
    Comprehensive model validation before production deployment.

    Args:
        model_name: Model to validate
        quantization_method: Quantization type
        acceptance_criteria: Dict of thresholds

    Returns:
        Validation results and pass/fail status
    """
    from vllm import LLM, SamplingParams
    import time

    results = {}

    # 1. Load model
    print("1. Loading model...")
    try:
        llm = LLM(
            model=model_name,
            quantization=quantization_method,
            gpu_memory_utilization=0.9,
        )
        results["model_loading"] = "PASS"
    except Exception as e:
        results["model_loading"] = f"FAIL: {e}"
        return results, False

    # 2. Latency test
    print("2. Testing latency...")
    prompts = ["Test prompt for latency measurement"]
    params = SamplingParams(max_tokens=100, temperature=0.8)

    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        latency = (time.perf_counter() - start) * 1000

        latencies.append(latency)

    p99_latency = sorted(latencies)[98]  # 99th percentile

    if p99_latency <= acceptance_criteria.get("max_p99_latency_ms", float('inf')):
        results["latency"] = f"PASS (P99: {p99_latency:.2f}ms)"
    else:
        results["latency"] = f"FAIL (P99: {p99_latency:.2f}ms > {acceptance_criteria['max_p99_latency_ms']}ms)"

    # 3. Throughput test
    print("3. Testing throughput...")
    batch_prompts = ["Test prompt"] * 32

    start = time.perf_counter()
    outputs = llm.generate(batch_prompts, params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    if throughput >= acceptance_criteria.get("min_throughput_tok_s", 0):
        results["throughput"] = f"PASS ({throughput:.2f} tok/s)"
    else:
        results["throughput"] = f"FAIL ({throughput:.2f} tok/s < {acceptance_criteria['min_throughput_tok_s']})"

    # 4. Memory test
    print("4. Testing memory usage...")
    memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    if memory_gb <= acceptance_criteria.get("max_memory_gb", float('inf')):
        results["memory"] = f"PASS ({memory_gb:.2f} GB)"
    else:
        results["memory"] = f"FAIL ({memory_gb:.2f} GB > {acceptance_criteria['max_memory_gb']} GB)"

    # 5. Quality test (perplexity)
    print("5. Testing quality...")
    # ... (implement perplexity test)
    results["quality"] = "PASS (implement perplexity check)"

    # Overall pass/fail
    all_pass = all("PASS" in v for v in results.values())

    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for test, result in results.items():
        status = "✓" if "PASS" in result else "✗"
        print(f"{status} {test}: {result}")
    print("="*60)
    print(f"Overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("="*60)

    return results, all_pass


# Example usage
acceptance_criteria = {
    "max_p99_latency_ms": 100,
    "min_throughput_tok_s": 500,
    "max_memory_gb": 40,
    "max_perplexity": 6.0,
}

results, passed = validate_model_for_production(
    "TheBloke/Llama-2-7B-AWQ",
    "awq",
    acceptance_criteria,
)
```

---

## Deployment Architecture

### Single GPU Deployment

```python
# Docker Compose for single GPU deployment

docker_compose_single_gpu = """
version: '3.8'

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    container_name: vllm-llama2-7b-awq
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    command:
      - --model
      - TheBloke/Llama-2-7B-AWQ
      - --quantization
      - awq
      - --dtype
      - half
      - --max-model-len
      - "2048"
      - --gpu-memory-utilization
      - "0.9"
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
"""

# Kubernetes deployment for single GPU
kubernetes_single_gpu = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama2-7b-awq
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama2
  template:
    metadata:
      labels:
        app: vllm-llama2
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - vllm
          - serve
          - TheBloke/Llama-2-7B-AWQ
          - --quantization=awq
          - --dtype=half
          - --max-model-len=2048
          - --gpu-memory-utilization=0.9
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 48Gi
          requests:
            nvidia.com/gpu: 1
            memory: 40Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-llama2
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

### Multi-GPU Deployment with Tensor Parallelism

```yaml
# Kubernetes deployment with tensor parallelism
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama2-70b-tp4
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama2-70b
  template:
    metadata:
      labels:
        app: vllm-llama2-70b
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
          - vllm
          - serve
          - TheBloke/Llama-2-70B-AWQ
          - --quantization=awq
          - --tensor-parallel-size=4
          - --dtype=half
          - --max-model-len=4096
          - --gpu-memory-utilization=0.95
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 4  # 4 GPUs for TP
            memory: 320Gi
          requests:
            nvidia.com/gpu: 4
            memory: 256Gi
        env:
        - name: NCCL_DEBUG
          value: "INFO"
```

### Horizontal Scaling (Multiple Replicas)

```python
# Kubernetes Horizontal Pod Autoscaler
hpa_config = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama2-7b-awq
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: vllm_num_requests_running
      target:
        type: AverageValue
        averageValue: "32"  # Scale when avg running requests > 32
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5min before scale down
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Pods
        value: 2
        periodSeconds: 15
"""
```

---

## Configuration Management

### Environment-Specific Configurations

```python
# Production configuration template

PRODUCTION_CONFIGS = {
    "development": {
        "model": "TheBloke/Llama-2-7B-AWQ",
        "quantization": "awq",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 1024,
        "max_num_seqs": 16,
        "enable_prefix_caching": False,
        "log_level": "DEBUG",
    },

    "staging": {
        "model": "TheBloke/Llama-2-7B-AWQ",
        "quantization": "awq",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
        "max_num_seqs": 64,
        "enable_prefix_caching": True,
        "log_level": "INFO",
    },

    "production": {
        "model": "TheBloke/Llama-2-7B-AWQ",
        "quantization": "awq",
        "gpu_memory_utilization": 0.95,
        "max_model_len": 2048,
        "max_num_seqs": 128,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "log_level": "WARNING",
        "disable_log_stats": False,
    },
}


def get_config(environment="production"):
    """Get environment-specific configuration."""
    if environment not in PRODUCTION_CONFIGS:
        raise ValueError(f"Unknown environment: {environment}")

    return PRODUCTION_CONFIGS[environment]


# Load configuration
config = get_config("production")

# Initialize vLLM with config
from vllm import LLM

llm = LLM(**config)
```

---

## Monitoring and Observability

### Prometheus Metrics

```python
# Export vLLM metrics to Prometheus

from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time

# Define metrics
requests_total = Counter('vllm_requests_total', 'Total number of requests')
requests_in_progress = Gauge('vllm_requests_in_progress', 'Requests currently being processed')
request_duration = Histogram('vllm_request_duration_seconds', 'Request duration in seconds')
tokens_generated = Counter('vllm_tokens_generated_total', 'Total tokens generated')
gpu_memory_used = Gauge('vllm_gpu_memory_bytes', 'GPU memory used')


class MonitoredVLLM:
    """vLLM wrapper with Prometheus metrics."""

    def __init__(self, llm):
        self.llm = llm

        # Start Prometheus metrics server
        start_http_server(9090)  # Metrics on port 9090

    def generate(self, prompts, params):
        """Generate with metrics collection."""
        requests_total.inc(len(prompts))
        requests_in_progress.inc(len(prompts))

        start = time.perf_counter()

        try:
            outputs = self.llm.generate(prompts, params)

            # Record metrics
            duration = time.perf_counter() - start
            request_duration.observe(duration)

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            tokens_generated.inc(total_tokens)

            # GPU memory
            memory = torch.cuda.max_memory_allocated()
            gpu_memory_used.set(memory)

            return outputs

        finally:
            requests_in_progress.dec(len(prompts))


# Usage
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")
monitored_llm = MonitoredVLLM(llm)

# Prometheus will scrape metrics from http://localhost:9090/metrics
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "vLLM Production Monitoring",
    "panels": [
      {
        "title": "Requests Per Second",
        "targets": [
          {
            "expr": "rate(vllm_requests_total[1m])"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Throughput (tokens/s)",
        "targets": [
          {
            "expr": "rate(vllm_tokens_generated_total[1m])"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "vllm_gpu_memory_bytes / 1024 / 1024 / 1024"
          }
        ]
      }
    ]
  }
}
```

### Distributed Tracing

```python
# OpenTelemetry tracing for vLLM

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)


class TracedVLLM:
    """vLLM with distributed tracing."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompts, params):
        """Generate with tracing."""
        with tracer.start_as_current_span("vllm.generate") as span:
            span.set_attribute("num_prompts", len(prompts))
            span.set_attribute("max_tokens", params.max_tokens)

            # Prefill phase
            with tracer.start_as_current_span("vllm.prefill"):
                outputs = self.llm.generate(prompts, params)

            # Record output
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            span.set_attribute("total_tokens_generated", total_tokens)

            return outputs
```

---

## Performance Optimization

### Production Performance Tuning

```python
def optimize_for_production(model_name, target_metric="balanced"):
    """
    Generate optimized configuration for production deployment.

    Args:
        model_name: Model to deploy
        target_metric: "latency", "throughput", or "balanced"

    Returns:
        Optimized configuration
    """
    configs = {
        "latency": {
            # Minimize latency
            "max_num_seqs": 4,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.85,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": True,
        },

        "throughput": {
            # Maximize throughput
            "max_num_seqs": 256,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.95,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
        },

        "balanced": {
            # Balance latency and throughput
            "max_num_seqs": 64,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
        },
    }

    base_config = {
        "model": model_name,
        "quantization": "awq",
        "dtype": "half",
    }

    base_config.update(configs[target_metric])

    return base_config
```

---

## Troubleshooting

### Common Production Issues

```python
TROUBLESHOOTING_GUIDE = {
    "Out of Memory (OOM)": {
        "symptoms": [
            "CUDA out of memory error",
            "Model fails to load",
            "Random crashes under load",
        ],
        "solutions": [
            "Reduce gpu_memory_utilization (try 0.8)",
            "Decrease max_num_seqs",
            "Reduce max_model_len",
            "Enable swap_space",
            "Use smaller quantization (INT4 instead of INT8)",
        ],
    },

    "High Latency": {
        "symptoms": [
            "P99 latency > SLA",
            "Slow response times",
            "Timeouts",
        ],
        "solutions": [
            "Reduce batch size (max_num_seqs)",
            "Enable CUDA graphs (enforce_eager=False)",
            "Check for CPU bottlenecks",
            "Optimize request routing",
            "Add more replicas",
        ],
    },

    "Low Throughput": {
        "symptoms": [
            "Tokens/second below expected",
            "Low GPU utilization",
        ],
        "solutions": [
            "Increase batch size (max_num_seqs)",
            "Enable prefix caching",
            "Use continuous batching",
            "Check for I/O bottlenecks",
            "Profile with Nsight Systems",
        ],
    },

    "Accuracy Degradation": {
        "symptoms": [
            "Poor quality outputs",
            "Higher perplexity than expected",
        ],
        "solutions": [
            "Verify quantization was done correctly",
            "Check calibration data quality",
            "Try different quantization method (AWQ vs GPTQ)",
            "Validate against baseline",
        ],
    },
}
```

---

## Cost Optimization

### Cloud Cost Analysis

```python
def calculate_deployment_cost(
    gpu_type,
    num_gpus,
    hourly_rate,
    requests_per_day,
    avg_tokens_per_request,
):
    """
    Calculate monthly deployment cost.

    Args:
        gpu_type: GPU type (e.g., "A100-40GB")
        num_gpus: Number of GPUs
        hourly_rate: Cost per GPU hour
        requests_per_day: Daily request volume
        avg_tokens_per_request: Average tokens per request

    Returns:
        Cost analysis
    """
    # Monthly GPU cost
    hours_per_month = 24 * 30
    monthly_gpu_cost = num_gpus * hourly_rate * hours_per_month

    # Total tokens per month
    total_requests_month = requests_per_day * 30
    total_tokens_month = total_requests_month * avg_tokens_per_request

    # Cost per token
    cost_per_token = monthly_gpu_cost / total_tokens_month
    cost_per_1m_tokens = cost_per_token * 1_000_000

    print(f"Cost Analysis for {gpu_type}:")
    print(f"  GPUs:                {num_gpus}")
    print(f"  Hourly rate:         ${hourly_rate:.2f}")
    print(f"  Monthly GPU cost:    ${monthly_gpu_cost:.2f}")
    print(f"  Requests/day:        {requests_per_day:,}")
    print(f"  Total tokens/month:  {total_tokens_month:,}")
    print(f"  Cost per 1M tokens:  ${cost_per_1m_tokens:.4f}")

    return {
        "monthly_cost": monthly_gpu_cost,
        "cost_per_1m_tokens": cost_per_1m_tokens,
    }


# Compare quantization methods
print("="*60)
print("Cost Comparison: FP16 vs INT4 Quantization")
print("="*60)

# FP16: Requires 4x A100-80GB
fp16_cost = calculate_deployment_cost(
    gpu_type="A100-80GB",
    num_gpus=4,
    hourly_rate=4.0,
    requests_per_day=100_000,
    avg_tokens_per_request=200,
)

print()

# INT4 (AWQ): Requires 1x A100-80GB
int4_cost = calculate_deployment_cost(
    gpu_type="A100-80GB",
    num_gpus=1,
    hourly_rate=4.0,
    requests_per_day=100_000,
    avg_tokens_per_request=200,
)

print()
print(f"Monthly Savings: ${fp16_cost['monthly_cost'] - int4_cost['monthly_cost']:.2f}")
print(f"Cost Reduction: {(1 - int4_cost['monthly_cost'] / fp16_cost['monthly_cost']) * 100:.1f}%")
```

---

## Summary

### Production Deployment Checklist (Final)

```
✓ Model validated and benchmarked
✓ Infrastructure provisioned
✓ Configuration optimized
✓ Monitoring and alerting configured
✓ Load testing completed
✓ Security measures implemented
✓ Disaster recovery plan in place
✓ Documentation complete
✓ Team trained on operations
✓ Cost tracking enabled
```

### Key Success Metrics

**Performance:**
- P99 latency < 100ms
- Throughput > 500 tokens/s
- GPU utilization > 80%

**Reliability:**
- Uptime > 99.9%
- Error rate < 0.1%
- Successful rollbacks tested

**Cost:**
- Cost per 1M tokens optimized
- Resource utilization > 80%
- Autoscaling working correctly

---

## References

1. vLLM Documentation: https://docs.vllm.ai
2. Kubernetes Best Practices for ML: https://kubernetes.io/docs/concepts/
3. Prometheus Monitoring: https://prometheus.io/docs/
4. OpenTelemetry Tracing: https://opentelemetry.io/docs/

---

**Module 6.12 Complete** • **Module 6: Quantization & Optimization COMPLETE!**

---

## Module 6 Summary

Congratulations! You've completed Module 6: Quantization & Optimization.

**You've learned:**
- Quantization fundamentals and techniques
- INT8, INT4, and FP8 quantization
- GPTQ, AWQ, and SmoothQuant algorithms
- Quantized kernel development and optimization
- Accuracy evaluation methodologies
- Performance tuning strategies
- Batch size optimization
- CUDA graph optimization
- Production deployment best practices

**Key Achievements:**
- Understand trade-offs between different quantization methods
- Able to quantize models for production deployment
- Can optimize inference performance systematically
- Ready to deploy quantized LLMs at scale

**Next Steps:**
- Practice quantizing models end-to-end
- Deploy a quantized model to production
- Contribute quantization improvements to vLLM
- Explore cutting-edge quantization research

---

**End of Module 6**
