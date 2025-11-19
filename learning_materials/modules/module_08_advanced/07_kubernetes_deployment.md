# Module 8.7: Kubernetes Deployment for vLLM

## Overview

This module covers production deployment of vLLM on Kubernetes, including configuration, scaling, resource management, and operational best practices. Learn how to deploy vLLM at scale with high availability and efficient resource utilization.

**Learning Objectives:**
- Deploy vLLM on Kubernetes with proper resource configuration
- Implement autoscaling for LLM serving workloads
- Configure GPU scheduling and resource quotas
- Set up health checks, monitoring, and logging
- Implement rolling updates and zero-downtime deployments

**Prerequisites:**
- Kubernetes fundamentals (pods, services, deployments)
- Understanding of vLLM architecture
- Docker and containerization basics
- Production deployment experience

**Estimated Time:** 4-5 hours

---

## 1. Container Image Preparation

### 1.1 Dockerfile for vLLM

```dockerfile
# Dockerfile for vLLM production deployment
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM
RUN pip3 install --no-cache-dir \
    vllm==0.2.7 \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk

# Create app directory
WORKDIR /app

# Copy application code
COPY server.py /app/
COPY config/ /app/config/

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_USE_MODELSCOPE=False

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python3", "server.py"]
