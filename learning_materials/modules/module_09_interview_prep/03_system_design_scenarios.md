# System Design Scenarios - Interview Questions

## Introduction

This document presents 10 comprehensive system design scenarios commonly asked in senior ML infrastructure engineering interviews. Each scenario includes requirements gathering, solution architecture, trade-off analysis, and evaluation criteria.

**Target Roles**: Senior ML Engineer, Staff Engineer, System Architect
**Duration per Question**: 45-60 minutes
**Format**: Whiteboard or collaborative design tool

---

## Scenario 1: Design High-Throughput LLM Serving System

### Problem Statement

"Design a production system to serve a 70B parameter LLM with the following requirements:
- 10,000 requests per second peak load
- P95 latency < 500ms
- Support variable input/output lengths (10-2048 tokens)
- 99.9% uptime SLA
- Cost-efficient deployment"

### Requirements Gathering (5 minutes)

**Clarifying Questions**:

**Q: What's the traffic pattern?**
A: Assume diurnal pattern - 10K RPS peak, 3K RPS average, 500 RPS off-peak

**Q: What's the acceptable degradation under overload?**
A: Return 429 (rate limit) rather than serving with high latency. Priority queue for premium users.

**Q: Geographic distribution?**
A: Initially single region (US-West), plan for multi-region

**Q: Model updates frequency?**
A: Weekly model updates, zero-downtime deployment required

**Q: Budget constraints?**
A: ~$50K/month target (100 GPU-hours/day @ ~$2/hr)

### High-Level Architecture (10 minutes)

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer (Layer 7)              │
│              (NGINX/HAProxy with health checks)          │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ API Gateway  │ │ API Gateway  │ │ API Gateway  │
│   Pod 1      │ │   Pod 2      │ │   Pod N      │
│ (Stateless)  │ │ (Stateless)  │ │ (Stateless)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
            ┌───────────▼───────────┐
            │  Request Queue        │
            │  (Redis/Kafka)        │
            │  - Priority queues    │
            │  - Rate limiting      │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ vLLM Engine  │ │ vLLM Engine  │ │ vLLM Engine  │
│   (8x A100)  │ │   (8x A100)  │ │   (8x A100)  │
│ Tensor Par=8 │ │ Tensor Par=8 │ │ Tensor Par=8 │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
            ┌───────────▼────────────┐
            │  Monitoring & Metrics  │
            │  (Prometheus/Grafana)  │
            │  - Latency, throughput │
            │  - GPU utilization     │
            │  - Queue depth         │
            └────────────────────────┘
```

### Detailed Design (20 minutes)

#### 1. Request Handling Flow

**API Gateway (Stateless)**:
```python
@app.post("/v1/completions")
async def generate(request: CompletionRequest):
    # 1. Validate request
    validate_input(request)

    # 2. Authentication & rate limiting
    user = authenticate(request.api_key)
    if not rate_limiter.allow(user):
        return 429, "Rate limit exceeded"

    # 3. Priority assignment
    priority = get_priority(user.tier)  # premium, standard, free

    # 4. Enqueue with timeout
    request_id = str(uuid.uuid4())
    await queue.enqueue(
        request_id=request_id,
        prompt=request.prompt,
        params=request.params,
        priority=priority,
        timeout=30  # Max queueing time
    )

    # 5. Wait for result or timeout
    result = await wait_for_completion(request_id, timeout=35)

    # 6. Return response
    return format_response(result)
```

**Request Queue Design**:
```
Redis Sorted Sets (one per priority level):
- Key: "queue:priority:{level}"
- Score: Timestamp (for FCFS within priority)
- Value: Serialized request

Three priority levels:
1. Premium (SLA: P95 < 300ms)
2. Standard (SLA: P95 < 500ms)
3. Free (Best effort, may reject under load)

Dequeue strategy:
1. Check premium queue first
2. If empty, check standard
3. If empty, check free
4. Aging mechanism to prevent starvation
```

**vLLM Engine Configuration**:
```python
llm = LLM(
    model="llama-70b",
    tensor_parallel_size=8,  # 8 GPUs per instance
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.90,  # Leave 10% margin
    max_num_batched_tokens=16384,  # Batch size optimization
    max_num_seqs=256,  # Max concurrent sequences
    enable_prefix_caching=True,  # For common prompts
)

# Continuous batching automatically enabled
# PagedAttention block_size=16 (default)
```

#### 2. Capacity Planning

**Single 8xA100 Instance**:
```
Throughput estimation:
- Llama-70B with TP=8: ~30-40 tokens/sec per request
- Average output length: 150 tokens → ~4 sec per request
- With continuous batching, batch_size ~100-150
- Effective throughput: ~1000-1200 requests/minute = ~20 RPS

Memory:
- Model: ~140GB (70B * 2 bytes for FP16)
- KV cache per request: ~500MB (for 2048 tokens)
- Total requests in memory: (320GB - 140GB) / 500MB ≈ 360 requests
- But realistically, continuous batching keeps ~100-150 active

Cost:
- 8xA100 instance: ~$25/hour (on-demand AWS)
- $600/day, $18K/month per instance
```

**Cluster Sizing for 10K RPS**:
```
Peak load: 10,000 RPS
Capacity per instance: ~20 RPS
Instances needed: 10,000 / 20 = 500 instances

With safety margin (80% target utilization):
500 / 0.8 = 625 instances

Cost: 625 * $25/hour = $15,625/hour (Way over budget!)

Optimization needed!
```

#### 3. Cost Optimization Strategies

**Strategy 1: Quantization**
```
Use INT8 quantization (GPTQ or AWQ):
- Memory: 140GB → 70GB (2x reduction)
- Speed: 1.5-2x faster inference
- Quality: <1% accuracy degradation

New capacity:
- ~40 RPS per instance (2x improvement)
- Instances needed: 10K / 40 = 250 instances
- Cost: 250 * $25 = $6,250/hour ($150K/month)

Still over budget, need more optimization
```

**Strategy 2: Autoscaling + Spot Instances**
```
Traffic pattern optimization:
- Peak (10K RPS): 8 hours/day
- Medium (5K RPS): 8 hours/day
- Low (1K RPS): 8 hours/day

Scaling strategy:
- Minimum instances (always-on, on-demand): 25 (for 1K RPS)
  Cost: 25 * $25 * 24 = $15K/month

- Auto-scale instances (on-demand): 0-100 (scale with load)
  Average: ~50 instances * 16 hours = 800 instance-hours/day
  Cost: 800 * $25 = $20K/month

- Spot instances (80% cheaper): 100 instances during peak
  Cost: 100 * $5 * 8 hours * 30 days = $120K → $24K/month

Total: $15K + $20K + $24K = $59K/month (closer to budget!)
```

**Strategy 3: Request Batching & Caching**
```
Observation: Many requests share prefixes (e.g., system prompts)

Prefix caching benefits:
- 30-40% of requests share common prefixes
- Memory savings: 30% fewer blocks needed
- Latency improvement: Skip prefill for cached prefix

New capacity:
- ~50 RPS per instance (25% improvement from caching)
- Instances needed: 10K / 50 = 200 instances

Revised cost with all optimizations:
- Base instances: 20 * $25 * 24 = $12K/month
- Auto-scale on-demand: 30 * $25 * 16 * 30 = $360K → $12K/month
- Spot instances: 80 * $5 * 8 * 30 = $96K
Total: ~$50K/month ✓ (Meets budget!)
```

#### 4. High Availability Design

**Multi-Instance Redundancy**:
```
Load balancer health checks:
- HTTP GET /health every 10s
- Timeout: 2s
- Unhealthy threshold: 3 consecutive failures
- Remove from pool automatically

Rolling updates:
1. Deploy new version to 10% of instances
2. Monitor error rates, latency
3. If healthy, continue rolling to 100%
4. If issues, automatic rollback

Zero-downtime: Always maintain >50% healthy instances
```

**Data Durability**:
```
Request queue (Redis):
- Redis Cluster with replication
- 3 replicas, automatic failover
- AOF (Append-Only File) persistence
- RTO: <1 minute

Request results cache (Redis):
- TTL: 5 minutes
- If worker dies mid-request, client can retry
- Idempotency keys prevent duplicate work
```

**Monitoring & Alerting**:
```
Critical metrics:
1. P95/P99 latency (alert if >SLA)
2. Error rate (alert if >0.1%)
3. GPU utilization (alert if <70% or >95%)
4. Queue depth (alert if >1000)
5. Instance health (alert if <80% healthy)

Incident response:
- Automatic scaling on high queue depth
- Circuit breaker for failing workers
- Fallback to cached responses if available
```

#### 5. Scaling Strategies

**Horizontal Scaling (Preferred)**:
```
Advantages:
+ Cost-effective (can use spot instances)
+ High availability (multiple replicas)
+ Gradual scaling
+ Geographic distribution possible

Challenges:
- Load balancing complexity
- Request routing overhead
- Consistency across instances

Implementation:
- Kubernetes StatefulSets for GPU pods
- Horizontal Pod Autoscaler based on queue depth
- Anti-affinity rules for fault isolation
```

**Vertical Scaling (Limited Use)**:
```
Use larger instances (e.g., 8xH100 vs 8xA100):
+ 2-3x faster per instance
+ Fewer instances to manage

Challenges:
- Higher cost per instance
- Less fault tolerance
- Limited by largest available GPU instance

When to use: Very low latency requirements
```

### Trade-offs and Alternatives (10 minutes)

#### Trade-off 1: Latency vs Throughput
```
High throughput (large batches):
+ More requests per GPU
+ Better cost efficiency
- Higher queueing time
- P99 latency suffers

Low latency (small batches):
+ Fast response times
+ Better user experience
- Lower GPU utilization
- Higher cost per request

Solution: Multi-tier service levels
- Premium: Small dedicated batches, low latency
- Standard: Balanced batches
- Free: Large batches, best effort
```

#### Trade-off 2: Quantization vs Quality
```
INT8 quantization:
+ 2x memory, 1.5-2x speed
- ~1% accuracy drop
- Requires calibration

FP16:
+ Better quality
- 2x memory usage
- Slower

Decision: Use INT8 for most requests, FP16 for premium tier
```

#### Trade-off 3: On-Demand vs Spot Instances
```
On-demand:
+ Guaranteed availability
+ No interruptions
- 4-5x more expensive

Spot:
+ 80% cost savings
- Can be interrupted
- Not always available

Solution: Hybrid
- On-demand for baseline capacity
- Spot for peak scaling
- Graceful handling of spot interruptions
```

### Extension Questions

**Q: How would you handle multi-region deployment?**
A: "Add routing layer:
```
Global Load Balancer (Cloud CDN):
- Route to nearest region
- Failover to other regions if down

Per-region architecture:
- Independent deployment of full stack
- Regional request queues
- Model sync via S3/GCS
- Monitoring aggregated globally

Challenges:
- Model consistency across regions
- Request routing for long-running requests
- Cost of running multiple regions
```"

**Q: How would you implement A/B testing for model versions?**
A: "Traffic splitting at API gateway:
```python
model_version = assign_experiment_group(user_id)
# 90% → model_v1
# 10% → model_v2

request.route_to = f"queue:model:{model_version}"
```

Separate vLLM instances for each model version
Monitor metrics per version
Graduate winning model to 100%"

**Q: How would you detect and handle abusive users?**
A: "Multi-layer approach:
1. **Rate limiting**: Standard tier limits
2. **Anomaly detection**: Unusual patterns (high request rate, very long outputs)
3. **Content filtering**: Detect prompt injection, jailbreak attempts
4. **Progressive penalties**: Warnings → temporary ban → permanent ban
5. **Challenge-response**: CAPTCHA for suspicious patterns"

### Evaluation Criteria

**Excellent (5/5)**:
- Complete architecture with all components
- Detailed capacity planning with numbers
- Cost analysis and optimization
- High availability design
- Handles follow-up questions
- Discusses trade-offs
- Practical, deployable solution

**Good (4/5)**:
- Solid architecture
- Reasonable capacity estimates
- Considers HA and scaling
- Some cost analysis

**Adequate (3/5)**:
- Basic architecture
- Missing some components
- Limited capacity planning

**Poor (1-2/5)**:
- Incomplete or flawed design
- No capacity planning
- Doesn't address requirements

---

## Scenario 2: Design Multi-Tenant LLM Platform

### Problem Statement

"Design a platform that allows 100+ different organizations to serve their own fine-tuned models. Requirements:
- Resource isolation between tenants
- Cost attribution per tenant
- Support 1K-70B parameter models
- Fair resource sharing
- Security isolation"

### Solution Highlights

**Key Decisions**:
1. **Namespace isolation**: Kubernetes namespaces per tenant
2. **Resource quotas**: GPU hours, memory limits per tenant
3. **Model storage**: S3 with IAM policies
4. **Billing**: Track GPU-time per tenant, usage-based pricing
5. **Scheduling**: Fair-share scheduler with weights

**Architecture**:
- Shared control plane
- Per-tenant worker pools (optional)
- Centralized monitoring
- Multi-tenant vLLM with request tagging

---

## Scenarios 3-10: Additional Design Problems

*For brevity, providing titles and key aspects. Each would be expanded similarly.*

---

## Scenario 3: Cost-Optimized Inference for Batch Workloads

**Focus**: Offline batch processing, maximize cost efficiency
**Key decisions**: Spot instances, large batches, no SLA

---

## Scenario 4: Ultra-Low Latency Serving (<100ms P95)

**Focus**: Trading cost for speed
**Key decisions**: Dedicated GPUs, FP8, speculative decoding

---

## Scenario 5: Multi-Model Serving (100+ Models)

**Focus**: Resource sharing, model loading/unloading
**Key decisions**: Model cache, cold start optimization

---

## Scenario 6: Streaming Response System

**Focus**: Token-by-token streaming
**Key decisions**: WebSockets/SSE, backpressure handling

---

## Scenario 7: Fault-Tolerant Distributed Training + Serving

**Focus**: Training during low-traffic, serving during high
**Key decisions**: GPU time-sharing, checkpointing

---

## Scenario 8: Compliance & Security (HIPAA, SOC2)

**Focus**: Data privacy, audit logs, encryption
**Key decisions**: Private endpoints, data residency

---

## Scenario 9: Edge Deployment (On-Premise)

**Focus**: Limited resources, no cloud
**Key decisions**: Smaller models, CPU fallback

---

## Scenario 10: Research Platform (Experimentation)

**Focus**: Flexibility, versioning, reproducibility
**Key decisions**: Experiment tracking, model registry

---

## General System Design Framework

### 1. Requirements (5 min)
- Functional requirements
- Non-functional (scale, latency, availability)
- Constraints (cost, timeline)

### 2. High-Level Design (10 min)
- Main components
- Data flow
- APIs

### 3. Detailed Design (20 min)
- Component details
- Algorithms
- Data models

### 4. Scaling & Optimization (10 min)
- Bottlenecks
- Scaling strategies
- Trade-offs

### 5. Trade-offs (5 min)
- Discuss alternatives
- Justify decisions

---

*Last Updated: 2025-11-19*
*Module 9: System Design Scenarios*
