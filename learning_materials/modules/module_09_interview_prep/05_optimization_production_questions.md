# Optimization & Production Interview Questions

## Introduction

13 questions covering quantization, production deployment, monitoring, and operational excellence for LLM serving systems.

**Target Roles**: Production ML Engineer, MLOps Engineer, SRE

---

## Question 1: Compare Quantization Techniques (GPTQ, AWQ, INT8)

### Detailed Answer

**INT8 Post-Training Quantization**:
```
Method: Simple linear quantization
- Calibrate with small dataset (~1000 samples)
- Compute scale = (max - min) / 255
- Quantize: q = round(x / scale)
- Dequantize: x_approx = q * scale

Pros:
+ Simple, fast calibration
+ Wide hardware support
+ Minimal quality loss (<1%)

Cons:
- Not optimal for outliers
- Requires calibration data

Performance: 2x memory, 1.5-2x speed
Accuracy: ~99% of FP16
```

**GPTQ (Generative Pre-trained Transformer Quantization)**:
```
Method: Hessian-based quantization
- Analyzes weight importance (2nd derivatives)
- Quantizes less important weights more aggressively
- Iterative optimization

Pros:
+ Better accuracy than simple INT8
+ 3-4 bit quantization possible
+ No activation quantization needed (weight-only)

Cons:
- Slower calibration (hours for 70B model)
- More complex implementation

Performance: 4x memory savings (4-bit), 2-3x speed
Accuracy: ~98-99% of FP16 at 4-bit
```

**AWQ (Activation-aware Weight Quantization)**:
```
Method: Protect salient weights
- Identify weights with high activation magnitudes
- Keep important weights at higher precision
- Quantize others aggressively

Pros:
+ Best accuracy/compression trade-off
+ Faster calibration than GPTQ
+ 4-bit with minimal loss

Cons:
- Needs activation statistics
- Custom kernels for inference

Performance: 4x memory, 2-3x speed
Accuracy: ~99% of FP16 at 4-bit (better than GPTQ)
```

**Comparison Table**:

| Method | Bits | Memory | Speed | Quality | Calibration Time |
|--------|------|--------|-------|---------|-----------------|
| FP16 | 16 | 1x | 1x | 100% | - |
| INT8 | 8 | 2x | 1.5-2x | 99% | Minutes |
| GPTQ | 4 | 4x | 2-3x | 98% | Hours |
| AWQ | 4 | 4x | 2-3x | 99% | ~1 hour |
| FP8 (H100) | 8 | 2x | 2-3x | ~100% | - |

**Decision Framework**:
```
Use INT8 when:
- Have calibration data
- Need simplicity
- 2x savings sufficient

Use GPTQ when:
- Need max compression
- Can afford calibration time
- Weight-only quantization OK

Use AWQ when:
- Need best quality at 4-bit
- Have activation data
- Want faster calibration than GPTQ

Use FP8 when:
- Have H100 GPUs
- Want native HW support
- Need minimal quality loss
```

---

## Question 2: Production Deployment Best Practices

### Condensed Answer

**Key Practices**:
1. **Health checks**: Liveness (is running?) + Readiness (can serve?)
2. **Graceful shutdown**: Finish in-flight requests, drain queue
3. **Rolling updates**: Deploy gradually, monitor, rollback if needed
4. **Resource limits**: Set CPU/memory/GPU limits in Kubernetes
5. **Observability**: Logs, metrics, traces (OpenTelemetry)
6. **Chaos testing**: Simulate failures, validate recovery
7. **Documentation**: Runbooks for common issues

---

## Question 3: Key Monitoring Metrics

### Condensed Answer

**Latency Metrics**:
- Time-to-first-token (TTFT): P50, P95, P99
- Time-per-output-token (TPOT): P50, P95, P99
- End-to-end latency: Request to completion

**Throughput Metrics**:
- Requests per second (RPS)
- Tokens per second (TPS): Input + output
- Effective batch size

**Resource Metrics**:
- GPU utilization: Compute + memory
- GPU memory: Used, cached, reserved
- CPU, network I/O

**Business Metrics**:
- Error rate, timeout rate
- Cost per request, cost per token
- User satisfaction (derived from latency)

---

## Question 4: How to Handle Traffic Spikes?

### Condensed Answer

**Strategies**:
1. **Auto-scaling**: Horizontal pod autoscaler based on queue depth
2. **Rate limiting**: Per-user, global limits
3. **Priority queues**: Premium users served first
4. **Caching**: Cache responses for common queries
5. **Graceful degradation**: Return cached/simplified responses
6. **Circuit breaker**: Reject requests if system unhealthy

---

## Question 5: Cost Optimization Strategies

### Condensed Answer

**Techniques**:
1. **Quantization**: 2-4x cost reduction
2. **Spot instances**: 80% savings vs on-demand
3. **Right-sizing**: Match instance to workload
4. **Prefix caching**: Reduce duplicate computation
5. **Batch optimization**: Larger batches = better GPU utilization
6. **Auto-scaling**: Scale down during low traffic
7. **Multi-tenancy**: Share GPUs across workloads

---

## Question 6: Continuous Batching Parameter Tuning

### Condensed Answer

**Key Parameters**:
```python
max_num_seqs = 256  # Max concurrent sequences
max_num_batched_tokens = 16384  # Max tokens per iteration
max_model_len = 2048  # Max sequence length
gpu_memory_utilization = 0.90  # Memory headroom
```

**Tuning Process**:
1. Start with defaults
2. Increase `max_num_seqs` until OOM
3. Tune `max_num_batched_tokens` for latency target
4. Adjust `gpu_memory_utilization` for stability
5. Benchmark and iterate

---

## Question 7: Model Loading Strategies

### Condensed Answer

**Cold Start Optimization**:
1. **Model caching**: Pre-load on pod startup
2. **Shared storage**: NFS/S3 with local cache
3. **Lazy loading**: Load layers as needed
4. **Keep-alive**: Don't unload frequently-used models

---

## Question 8: Graceful Degradation Patterns

### Condensed Answer

**Strategies**:
1. **Fallback models**: Smaller model if primary unavailable
2. **Cached responses**: Serve cached for common queries
3. **Partial responses**: Return what's generated so far
4. **Queue shedding**: Reject low-priority requests
5. **Timeout handling**: Return error vs hanging

---

## Question 9: Autoscaling Strategies

### Condensed Answer

**Metrics for Scaling**:
- Queue depth: Scale up if >threshold
- GPU utilization: Scale if >85% for 5 min
- Latency: Scale if P95 > SLA

**Policies**:
- Scale up: Aggressive (2x pods)
- Scale down: Conservative (10% at a time)
- Cool-down: 5 min between scaling events

---

## Question 10: Common Production Issues

### Condensed Answer

**Issues & Solutions**:
1. **OOM**: Reduce batch size, enable swapping
2. **Slow requests**: Timeout, kill zombie requests
3. **GPU hangs**: Health check, restart pod
4. **High latency**: Check queue depth, scale up
5. **Model quality**: Version pinning, gradual rollout

---

## Question 11: Deployment Patterns

### Condensed Answer

**Blue-Green Deployment**:
- Run two versions simultaneously
- Switch traffic with zero downtime
- Instant rollback if issues

**Canary Deployment**:
- 5% traffic to new version
- Monitor metrics
- Gradually increase to 100%

**Shadow Deployment**:
- Send requests to both versions
- Compare outputs
- No user impact

---

## Question 12: Security Best Practices

### Condensed Answer

**Key Measures**:
1. **Authentication**: API keys, OAuth
2. **Rate limiting**: Prevent abuse
3. **Input validation**: Sanitize prompts
4. **Content filtering**: Block harmful content
5. **Audit logging**: Track all requests
6. **Encryption**: TLS for transport, at-rest

---

## Question 13: Incident Response

### Condensed Answer

**Process**:
1. **Detection**: Monitoring alerts
2. **Triage**: Assess severity, impact
3. **Mitigation**: Quick fix (rollback, scale)
4. **Resolution**: Root cause fix
5. **Postmortem**: Document, prevent recurrence

---

*Last Updated: 2025-11-19*
*Module 9: Optimization & Production Questions*
