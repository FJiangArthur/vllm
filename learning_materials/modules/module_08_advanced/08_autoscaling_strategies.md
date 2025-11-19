# Module 8.8: Autoscaling Strategies for vLLM

## Overview

Autoscaling is critical for production LLM deployments to handle variable load while optimizing costs. This module covers horizontal and vertical scaling strategies, custom metrics, and advanced autoscaling policies for vLLM on Kubernetes.

**Topics Covered:**
- Horizontal Pod Autoscaling (HPA) configuration
- Custom metrics for LLM workloads
- Queue-based autoscaling
- Predictive scaling algorithms
- Cost-aware scaling policies
- GPU resource management

**Estimated Time:** 3-4 hours

---

## 1. Kubernetes Horizontal Pod Autoscaler (HPA)

### 1.1 Basic HPA Configuration
- CPU/Memory-based scaling
- Target utilization settings
- Scale-up and scale-down policies
- Stabilization windows

### 1.2 Custom Metrics for LLM Serving
- Requests per second (RPS)
- Queue depth
- P95/P99 latency
- GPU utilization
- Token throughput

## 2. Queue-Based Autoscaling

### 2.1 KEDA Integration
- ScaledObject configuration
- Prometheus metrics integration
- Queue length thresholds
- Custom scaling logic

### 2.2 Request Queue Metrics
- Pending requests monitoring
- Time-in-queue tracking
- SLA violation detection
- Adaptive thresholds

## 3. GPU-Aware Autoscaling

### 3.1 GPU Utilization Monitoring
- NVIDIA DCGM metrics
- Memory utilization tracking
- Multi-GPU balancing
- Node-level vs pod-level metrics

### 3.2 GPU Scheduling Strategies
- Bin packing
- Spread across nodes
- GPU sharing considerations
- MIG (Multi-Instance GPU) support

## 4. Predictive Scaling

### 4.1 Time-Series Forecasting
- Historical traffic patterns
- Seasonal trends
- Special event handling
- ML-based prediction models

### 4.2 Proactive Scale-Out
- Pre-warming instances
- Scheduled scaling
- Traffic prediction algorithms
- Confidence intervals

## 5. Cost-Aware Scaling

### 5.1 Cost Optimization Strategies
- Spot instance integration
- Reserved capacity management
- Scale-to-zero for idle periods
- Regional failover

### 5.2 SLA vs Cost Tradeoffs
- Latency budgets
- Throughput guarantees
- Cost per request optimization
- Quality of service tiers

## 6. Implementation Examples

### Example 1: Basic HPA with Custom Metrics
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: gpu-utilization
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: vllm_request_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

### Example 2: KEDA ScaledObject
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaledobject
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 1
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_queue_depth
      threshold: '15'
      query: sum(rate(vllm_request_queue_size[1m]))
```

## 7. Monitoring and Observability

### 7.1 Key Metrics to Track
- Scale events (up/down)
- Time to scale
- Resource utilization before/after
- Cost per scale event
- SLA violations during scaling

### 7.2 Alerting Rules
- Rapid scaling detection
- Scaling failure alerts
- Resource exhaustion warnings
- Cost threshold breaches

## 8. Best Practices

1. **Start Conservative**: Begin with higher thresholds and adjust
2. **Monitor Costs**: Track cost per request during scaling
3. **Test Scale Events**: Simulate traffic spikes regularly
4. **Set Appropriate Limits**: Prevent runaway scaling
5. **Use Stabilization Windows**: Avoid flapping
6. **Combine Strategies**: Use both proactive and reactive scaling

## 9. Advanced Topics

- Multi-cluster scaling
- Cross-region failover
- Serverless function integration
- Hybrid cloud scaling
- Carbon-aware scaling

## Exercises

1. Configure HPA with custom metrics
2. Implement queue-based autoscaling
3. Benchmark scaling performance
4. Optimize costs while maintaining SLAs
5. Design predictive scaling system

---

**Next:** [Module 8.9: Distributed Tracing](09_distributed_tracing.md)
