# Project 7.7: Production Deployment

## Project Overview

**Duration:** 3-4 hours
**Difficulty:** ⭐⭐⭐⭐☆ (Advanced)
**Prerequisites:** Docker/Kubernetes knowledge, DevOps experience

### Objective

Deploy vLLM in a production-ready configuration using Kubernetes with autoscaling, health checks, logging, distributed tracing, and zero-downtime deployments. Create comprehensive runbooks for operations.

### Why This Matters

Code that works locally but fails in production has no value. This project covers:
- **Containerization**: Reproducible deployments
- **Orchestration**: Automated scaling and recovery
- **Observability**: Logging, metrics, tracing
- **Operations**: Runbooks for common scenarios
- **Reliability**: High availability and fault tolerance

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   External Load Balancer                │
│                  (AWS ALB / GCP LB)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │    Kubernetes Cluster   │
         │  ┌──────────────────┐   │
         │  │  Ingress NGINX   │   │
         │  └────────┬─────────┘   │
         │           │             │
         │  ┌────────▼─────────┐   │
         │  │  vLLM Service    │   │
         │  │  (LoadBalancer)  │   │
         │  └────────┬─────────┘   │
         │           │             │
         │  ┌────────▼─────────────────────┐
         │  │  vLLM Deployment             │
         │  │  ┌──────┐ ┌──────┐ ┌──────┐ │
         │  │  │ Pod1 │ │ Pod2 │ │ Pod3 │ │
         │  │  │GPU 0 │ │GPU 1 │ │GPU 2 │ │
         │  │  └──────┘ └──────┘ └──────┘ │
         │  └──────────────────────────────┘
         │           │             │
         │  ┌────────▼─────────┐   │
         │  │  HPA (Autoscaler)│   │
         │  └──────────────────┘   │
         │           │             │
         │  ┌────────▼─────────────┐
         │  │  Prometheus + Grafana│
         │  │  (Monitoring)        │
         │  └──────────────────────┘
         │           │             │
         │  ┌────────▼─────────────┐
         │  │  Loki (Logs)         │
         │  └──────────────────────┘
         │           │             │
         │  ┌────────▼─────────────┐
         │  │  Jaeger (Tracing)    │
         │  └──────────────────────┘
         └─────────────────────────┘
```

---

## Requirements

### FR1: Containerization
- Multi-stage Dockerfile for vLLM
- Optimized image size (<5GB)
- GPU support (CUDA base image)
- Model caching strategy

### FR2: Kubernetes Deployment
- Deployment with GPU allocation
- ConfigMap for configuration
- Secrets for API keys
- Resource limits and requests

### FR3: Health & Readiness
- Health check endpoint
- Readiness probe (model loaded)
- Liveness probe (service responsive)
- Graceful shutdown

### FR4: Autoscaling
- Horizontal Pod Autoscaler (HPA)
- Custom metrics (queue length, GPU util)
- Scale-up/down policies
- Min/max replicas

### FR5: Observability
- Structured logging (JSON)
- Prometheus metrics export
- Distributed tracing (Jaeger)
- Log aggregation (Loki/ELK)

### FR6: Deployment Strategy
- Rolling updates
- Zero-downtime deployments
- Canary deployments (optional)
- Rollback capability

---

## Implementation

### Step 1: Dockerfile (30 minutes)

```dockerfile
# Dockerfile

# Multi-stage build for optimization
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir vllm

# Production stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python runtime
RUN apt-get update && apt-get install -y \
    python3.10 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 vllm
USER vllm
WORKDIR /app

# Copy application code
COPY --chown=vllm:vllm src/ ./src/
COPY --chown=vllm:vllm config/ ./config/

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Entry point
CMD ["python3", "src/server.py"]
```

```dockerfile
# .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
*.md
tests/
docs/
.venv/
*.log
```

### Step 2: Kubernetes Manifests (60 minutes)

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vllm-prod
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
  namespace: vllm-prod
data:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_model_len: "4096"
  gpu_memory_utilization: "0.9"
  max_num_seqs: "256"
  log_level: "INFO"
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: vllm-prod
  labels:
    app: vllm
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero-downtime
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      # GPU node selector
      nodeSelector:
        nvidia.com/gpu: "true"

      # Init container to download model
      initContainers:
      - name: model-downloader
        image: vllm-server:latest
        command:
        - python3
        - -c
        - |
          from transformers import AutoTokenizer, AutoModelForCausalLM
          import os
          model_name = os.environ.get('MODEL_NAME')
          AutoTokenizer.from_pretrained(model_name)
          # Model will be cached
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: model_name
        - name: HF_HOME
          value: /model-cache
        volumeMounts:
        - name: model-cache
          mountPath: /model-cache

      containers:
      - name: vllm-server
        image: vllm-server:latest
        imagePullPolicy: IfNotPresent

        # Resource allocation
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1

        # Environment variables
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: model_name
        - name: MAX_MODEL_LEN
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: max_model_len
        - name: GPU_MEMORY_UTILIZATION
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: gpu_memory_utilization
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: log_level
        - name: HF_HOME
          value: /model-cache
        - name: CUDA_VISIBLE_DEVICES
          value: "0"

        # Ports
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        # Probes
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        # Lifecycle hooks
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - sleep 10  # Give time for connections to drain

        # Volume mounts
        volumeMounts:
        - name: model-cache
          mountPath: /model-cache
        - name: tmp
          mountPath: /tmp

      # Volumes
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: tmp
        emptyDir: {}

      # Termination grace period
      terminationGracePeriodSeconds: 30
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: vllm-prod
  labels:
    app: vllm
spec:
  type: LoadBalancer
  selector:
    app: vllm
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  sessionAffinity: None
```

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: vllm-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # Custom metric: request queue length
  - type: Pods
    pods:
      metric:
        name: vllm_num_waiting_sequences
      target:
        type: AverageValue
        averageValue: "10"

  # Scale-up/down behavior
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scale down
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: vllm-prod
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

### Step 3: Application Code (Health Checks) (30 minutes)

```python
# src/server.py

from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
import logging
import sys

# Configure logging (JSON format for production)
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}',
    stream=sys.stdout
)

app = FastAPI()

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Health check state
class HealthStatus:
    def __init__(self):
        self.model_loaded = False
        self.is_healthy = False

health_status = HealthStatus()

@app.on_event("startup")
async def startup_event():
    """Initialize vLLM on startup."""
    logging.info("Starting vLLM server")

    try:
        # Load model
        # global llm
        # llm = LLM(...)
        health_status.model_loaded = True
        health_status.is_healthy = True
        logging.info("vLLM initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize vLLM: {e}")
        health_status.is_healthy = False

@app.get("/health")
async def health():
    """
    Liveness probe: Is the service running?
    """
    if health_status.is_healthy:
        return {"status": "healthy"}
    else:
        return Response(
            content='{"status": "unhealthy"}',
            status_code=503,
            media_type="application/json"
        )

@app.get("/health/ready")
async def readiness():
    """
    Readiness probe: Is the service ready to serve traffic?
    """
    if health_status.model_loaded:
        return {"status": "ready"}
    else:
        return Response(
            content='{"status": "not ready"}',
            status_code=503,
            media_type="application/json"
        )

@app.post("/v1/completions")
async def completions(request: dict):
    """Inference endpoint."""
    # Implementation here
    pass
```

### Step 4: Deployment Scripts (30 minutes)

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

NAMESPACE="vllm-prod"
IMAGE_TAG=${1:-latest}

echo "Deploying vLLM to production"
echo "Namespace: $NAMESPACE"
echo "Image tag: $IMAGE_TAG"

# Build Docker image
echo "Building Docker image..."
docker build -t vllm-server:$IMAGE_TAG .

# Push to registry
echo "Pushing to registry..."
docker tag vllm-server:$IMAGE_TAG your-registry.com/vllm-server:$IMAGE_TAG
docker push your-registry.com/vllm-server:$IMAGE_TAG

# Update Kubernetes deployment
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for rollout
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/vllm-server -n $NAMESPACE --timeout=10m

echo "Deployment complete!"
kubectl get pods -n $NAMESPACE
```

```bash
#!/bin/bash
# scripts/rollback.sh

set -e

NAMESPACE="vllm-prod"

echo "Rolling back vLLM deployment"

kubectl rollout undo deployment/vllm-server -n $NAMESPACE

kubectl rollout status deployment/vllm-server -n $NAMESPACE --timeout=5m

echo "Rollback complete!"
```

### Step 5: Runbooks (45 minutes)

```markdown
# operations/runbook.md

# vLLM Production Runbook

## Common Operations

### Deploy New Version

```bash
./scripts/deploy.sh v1.2.3
```

Monitor deployment:
```bash
kubectl get pods -n vllm-prod -w
kubectl logs -f deployment/vllm-server -n vllm-prod
```

### Rollback Deployment

If issues detected:
```bash
./scripts/rollback.sh
```

### Scale Manually

```bash
kubectl scale deployment/vllm-server --replicas=5 -n vllm-prod
```

### Check Logs

```bash
# All pods
kubectl logs -l app=vllm -n vllm-prod --tail=100

# Specific pod
kubectl logs <pod-name> -n vllm-prod -f

# Previous instance (if crashed)
kubectl logs <pod-name> -n vllm-prod --previous
```

### Check Metrics

```bash
# Pod metrics
kubectl top pods -n vllm-prod

# Node metrics
kubectl top nodes

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open http://localhost:3000
```

## Troubleshooting

### Pod Not Starting

**Symptoms:** Pods stuck in `Pending` or `CrashLoopBackOff`

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n vllm-prod
kubectl logs <pod-name> -n vllm-prod
```

**Common Causes:**
1. **No GPU available**: Check node GPU allocation
2. **OOM**: Increase memory limits
3. **Image pull error**: Check image name and registry access
4. **Model download failure**: Check HF token in secrets

### High Latency

**Symptoms:** P99 latency > SLA

**Diagnosis:**
```bash
# Check current metrics
curl http://vllm-service/metrics | grep latency

# Check GPU utilization
kubectl exec -it <pod-name> -n vllm-prod -- nvidia-smi
```

**Actions:**
1. Scale up: `kubectl scale deployment/vllm-server --replicas=<n>`
2. Check for stragglers in logs
3. Review recent config changes

### Service Unavailable

**Symptoms:** 503 errors

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n vllm-prod

# Check service endpoints
kubectl get endpoints vllm-service -n vllm-prod

# Check health
kubectl exec -it <pod-name> -n vllm-prod -- \
  curl http://localhost:8000/health
```

**Actions:**
1. Check if pods are ready
2. Verify health checks passing
3. Check resource limits (CPU/memory/GPU)

### Out of Memory

**Symptoms:** Pods killed with OOMKilled status

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n vllm-prod | grep -A 5 "Last State"
```

**Actions:**
1. Increase memory limits in deployment
2. Reduce `gpu_memory_utilization` in config
3. Reduce `max_num_seqs`

## Monitoring Alerts

### Critical Alerts

**Alert: High Error Rate**
- **Condition:** Error rate > 5%
- **Action:** Check logs, consider rollback

**Alert: Pod Crash Loop**
- **Condition:** Pod restarting > 3 times
- **Action:** Check logs, investigate root cause

**Alert: High Memory Usage**
- **Condition:** Memory > 90%
- **Action:** Scale horizontally or increase limits

### Warning Alerts

**Alert: High Latency**
- **Condition:** P99 > 2s
- **Action:** Consider scaling up

**Alert: Low GPU Utilization**
- **Condition:** GPU < 50%
- **Action:** Optimize batch size or scale down

## Backup and Recovery

### Backup Model Cache

```bash
kubectl exec -it <pod-name> -n vllm-prod -- \
  tar czf /tmp/model-cache.tar.gz /model-cache

kubectl cp vllm-prod/<pod-name>:/tmp/model-cache.tar.gz ./model-cache-backup.tar.gz
```

### Restore from Backup

```bash
kubectl cp ./model-cache-backup.tar.gz vllm-prod/<pod-name>:/tmp/

kubectl exec -it <pod-name> -n vllm-prod -- \
  tar xzf /tmp/model-cache.tar.gz -C /
```
```

---

## Deliverables

1. **Docker Configuration** (Dockerfile, .dockerignore)
2. **Kubernetes Manifests** (deployment, service, HPA, etc.)
3. **Application Code** (health checks, graceful shutdown)
4. **Deployment Scripts** (deploy, rollback, scale)
5. **Runbooks** (operations guide, troubleshooting)
6. **Monitoring Setup** (Prometheus, Grafana dashboards)
7. **Documentation** (architecture, deployment guide)

---

## Evaluation Rubric

### Functionality (40 points)
- ✅ Successful deployment: 15 pts
- ✅ Health checks working: 10 pts
- ✅ Autoscaling functional: 10 pts
- ✅ Zero-downtime updates: 5 pts

### Observability (20 points)
- ✅ Logging configured: 7 pts
- ✅ Metrics exposed: 7 pts
- ✅ Tracing (optional): 6 pts

### Operations (20 points)
- ✅ Runbooks complete: 10 pts
- ✅ Deployment automation: 10 pts

### Documentation (20 points)
- ✅ Setup guide: 10 pts
- ✅ Architecture diagram: 10 pts

**Total: 100 points | Passing: 70 points**

---

## Common Pitfalls

1. **GPU Node Selection**: Ensure GPU nodes available
2. **Resource Limits**: Set appropriate limits
3. **Graceful Shutdown**: Implement properly
4. **Model Caching**: Use PVC for model cache
5. **Health Checks**: Configure probes correctly

---

**Deploy with confidence!**

*Project 7.7: Production Deployment | Duration: 3-4 hours*
