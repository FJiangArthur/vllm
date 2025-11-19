# Project 7.3: Multi-GPU Inference Service

## Project Overview

**Duration:** 3-4 hours
**Difficulty:** ⭐⭐⭐⭐☆ (Advanced)
**Prerequisites:** Modules 4, 5 (System Components, Distributed Inference)

### Objective

Design and implement a production-ready, distributed inference service that spans multiple GPUs with intelligent load balancing, fault tolerance, and dynamic resource allocation. Your service will handle request routing, GPU health monitoring, and graceful failure recovery.

### Why This Matters

Large-scale ML serving requires horizontal scaling across multiple GPUs. A robust multi-GPU service:
- **Increases throughput** linearly with GPU count
- **Provides high availability** (no single point of failure)
- **Optimizes resource utilization** (balance load across GPUs)
- **Handles failures gracefully** (continue serving despite GPU failures)

### Real-World Context

This project mirrors production systems at:
- **OpenAI**: Serving GPT models across hundreds of GPUs
- **Anthropic**: Load balancing Claude requests across clusters
- **NVIDIA**: Multi-tenant GPU sharing in cloud infrastructure

---

## Learning Objectives

1. ✅ Implement distributed system design patterns (load balancing, service discovery)
2. ✅ Build fault-tolerant systems with health checking and failover
3. ✅ Optimize GPU resource utilization across multiple devices
4. ✅ Design observable systems with comprehensive monitoring
5. ✅ Handle production scenarios (rolling updates, scaling, failures)

---

## Architecture Design

### High-Level Architecture

```
                         ┌─────────────────┐
                         │  Load Balancer  │
                         │   (nginx/HAProxy)│
                         └────────┬────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
         ┌───────▼───────┐ ┌─────▼──────┐ ┌──────▼──────┐
         │ Service Router │ │   Router   │ │   Router    │
         │   (Instance 1) │ │ (Instance 2)│ │(Instance 3) │
         └───────┬────────┘ └─────┬──────┘ └──────┬──────┘
                 │                │                │
         ┌───────┴────────────────┴────────────────┴──────┐
         │           GPU Worker Registry                   │
         │          (health status, capacity)              │
         └────────┬──────────┬──────────┬──────────────────┘
                  │          │          │
         ┌────────▼──┐  ┌────▼─────┐  ┌▼───────────┐
         │GPU Worker 0│  │ Worker 1 │  │ Worker 2   │
         │  (vLLM)    │  │  (vLLM)  │  │   (vLLM)   │
         │CUDA:0 A100 │  │ CUDA:1   │  │  CUDA:2    │
         └────────────┘  └──────────┘  └────────────┘
                  │          │          │
         ┌────────▼──────────▼──────────▼────────┐
         │      Prometheus Metrics                │
         │   + Grafana Dashboards                 │
         └────────────────────────────────────────┘
```

### Key Components

#### 1. Service Router
- Receives inference requests
- Routes to available GPU workers
- Implements load balancing strategies
- Handles worker failures

#### 2. GPU Worker
- Wraps vLLM inference engine
- Exposes HTTP/gRPC API
- Reports health and capacity
- Processes inference requests

#### 3. Worker Registry
- Tracks available workers
- Maintains health status
- Stores capacity metrics
- Enables service discovery

#### 4. Health Monitor
- Periodically checks worker health
- Detects GPU failures (OOM, crashes)
- Triggers worker restart/removal
- Updates registry

---

## Requirements Specification

### Functional Requirements

#### FR1: Multi-GPU Support
- Support 2-8 GPUs on single or multiple nodes
- Dynamic GPU worker registration
- Automatic worker discovery

#### FR2: Load Balancing
- Implement at least 2 strategies:
  - **Round-robin**: Equal distribution
  - **Least-loaded**: Route to GPU with lowest queue length
  - **Weighted**: Consider GPU capacity/speed
- Configurable strategy selection

#### FR3: Fault Tolerance
- Detect GPU failures within 5 seconds
- Automatically remove failed workers from rotation
- Retry failed requests on healthy workers
- Support graceful worker shutdown

#### FR4: Monitoring & Observability
- Expose Prometheus metrics:
  - Requests per GPU
  - Latency percentiles per GPU
  - GPU utilization
  - Queue lengths
  - Error rates
- Health check endpoint

#### FR5: Rolling Updates
- Support zero-downtime deployment
- Gradual traffic shifting to new workers
- Rollback capability

### Non-Functional Requirements

#### NFR1: Performance
- Routing overhead < 5ms per request
- Support > 1000 requests/second aggregate throughput
- Achieve > 85% GPU utilization under load

#### NFR2: Availability
- Service uptime > 99.9% (excluding scheduled maintenance)
- Survive single GPU failure without service disruption
- Restart failed workers automatically

#### NFR3: Scalability
- Linear throughput scaling with GPU count (>90% efficiency)
- Support dynamic worker addition/removal
- Handle traffic spikes gracefully

---

## Implementation Guide

### Step 1: Project Structure (15 minutes)

```bash
mkdir -p multi_gpu_service/{router,worker,registry,monitoring}
cd multi_gpu_service

# Directory structure
# router/       - Service router implementation
# worker/       - GPU worker wrapper
# registry/     - Worker registry and health checks
# monitoring/   - Prometheus metrics and dashboards
# config/       - Configuration files
# tests/        - Unit and integration tests
# scripts/      - Deployment scripts
```

### Step 2: GPU Worker Implementation (60 minutes)

```python
# worker/gpu_worker.py

import asyncio
import logging
from typing import Optional, Dict
from dataclasses import dataclass
import torch
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    """Inference request format."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    """Inference response format."""
    request_id: str
    generated_text: str
    tokens_generated: int
    latency_ms: float

@dataclass
class WorkerConfig:
    """GPU worker configuration."""
    worker_id: str
    gpu_id: int
    model_name: str
    max_num_seqs: int = 256
    max_model_len: int = 4096
    port: int = 8000
    registry_url: str = "http://localhost:5000"

class GPUWorker:
    """
    GPU worker that wraps vLLM for multi-GPU serving.

    Responsibilities:
    - Load model on specific GPU
    - Process inference requests
    - Report health and capacity to registry
    - Expose metrics
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.app = FastAPI(title=f"GPU Worker {config.worker_id}")

        # Initialize vLLM engine
        self.llm = self._initialize_vllm()

        # Metrics tracking
        self.total_requests = 0
        self.active_requests = 0
        self.failed_requests = 0

        # Register routes
        self._setup_routes()

    def _initialize_vllm(self) -> LLM:
        """Initialize vLLM engine on specified GPU."""
        logger.info(f"Initializing vLLM on GPU {self.config.gpu_id}")

        # Set CUDA device
        torch.cuda.set_device(self.config.gpu_id)

        # Create LLM instance
        llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
        )

        logger.info(f"vLLM initialized successfully on GPU {self.config.gpu_id}")
        return llm

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.post("/v1/completions", response_model=InferenceResponse)
        async def generate(request: InferenceRequest):
            """Handle inference request."""
            import time
            start_time = time.time()

            self.active_requests += 1
            self.total_requests += 1

            try:
                # Prepare sampling parameters
                sampling_params = SamplingParams(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

                # Generate
                outputs = self.llm.generate([request.prompt], sampling_params)

                # Extract result
                generated_text = outputs[0].outputs[0].text
                tokens_generated = len(outputs[0].outputs[0].token_ids)

                latency_ms = (time.time() - start_time) * 1000

                return InferenceResponse(
                    request_id=request.request_id or f"{self.config.worker_id}-{self.total_requests}",
                    generated_text=generated_text,
                    tokens_generated=tokens_generated,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                self.failed_requests += 1
                logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            finally:
                self.active_requests -= 1

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            gpu_available = torch.cuda.is_available()
            gpu_memory = torch.cuda.memory_allocated(self.config.gpu_id)

            return {
                "status": "healthy" if gpu_available else "unhealthy",
                "worker_id": self.config.worker_id,
                "gpu_id": self.config.gpu_id,
                "gpu_memory_mb": gpu_memory / (1024 ** 2),
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
            }

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return {
                "worker_total_requests": self.total_requests,
                "worker_active_requests": self.active_requests,
                "worker_failed_requests": self.failed_requests,
                "worker_gpu_memory_allocated": torch.cuda.memory_allocated(self.config.gpu_id),
                "worker_gpu_memory_reserved": torch.cuda.memory_reserved(self.config.gpu_id),
            }

        @self.app.get("/capacity")
        async def capacity():
            """Report current capacity."""
            max_capacity = self.config.max_num_seqs
            current_load = self.active_requests

            return {
                "worker_id": self.config.worker_id,
                "max_capacity": max_capacity,
                "current_load": current_load,
                "available_capacity": max_capacity - current_load,
                "utilization": current_load / max_capacity if max_capacity > 0 else 0,
            }

    def run(self):
        """Start the worker server."""
        logger.info(f"Starting GPU worker {self.config.worker_id} on port {self.config.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.config.port)

# Entry point
if __name__ == "__main__":
    import sys

    config = WorkerConfig(
        worker_id=sys.argv[1],
        gpu_id=int(sys.argv[2]),
        model_name=sys.argv[3],
        port=int(sys.argv[4]),
    )

    worker = GPUWorker(config)
    worker.run()
```

### Step 3: Worker Registry (45 minutes)

```python
# registry/worker_registry.py

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    """Information about a GPU worker."""
    worker_id: str
    url: str
    gpu_id: int
    status: str = "healthy"  # healthy, unhealthy, draining
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_requests: int = 0
    total_requests: int = 0
    available_capacity: int = 256

class WorkerRegistry:
    """
    Central registry for GPU workers.

    Responsibilities:
    - Track available workers
    - Monitor worker health
    - Provide worker discovery
    - Remove unhealthy workers
    """

    def __init__(self, health_check_interval: int = 5):
        self.workers: Dict[str, WorkerInfo] = {}
        self.health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None

    async def register_worker(self, worker_info: WorkerInfo) -> bool:
        """Register a new worker."""
        logger.info(f"Registering worker {worker_info.worker_id}")
        self.workers[worker_info.worker_id] = worker_info
        return True

    async def deregister_worker(self, worker_id: str) -> bool:
        """Remove worker from registry."""
        if worker_id in self.workers:
            logger.info(f"Deregistering worker {worker_id}")
            del self.workers[worker_id]
            return True
        return False

    def get_healthy_workers(self) -> List[WorkerInfo]:
        """Get list of healthy workers."""
        return [
            worker for worker in self.workers.values()
            if worker.status == "healthy"
        ]

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get specific worker by ID."""
        return self.workers.get(worker_id)

    async def check_worker_health(self, worker: WorkerInfo) -> bool:
        """
        Check health of a single worker.

        Returns True if healthy, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{worker.url}/health")

                if response.status_code == 200:
                    health_data = response.json()

                    # Update worker info
                    worker.last_heartbeat = datetime.now()
                    worker.active_requests = health_data.get("active_requests", 0)
                    worker.status = health_data.get("status", "healthy")

                    # Get capacity info
                    capacity_response = await client.get(f"{worker.url}/capacity")
                    if capacity_response.status_code == 200:
                        capacity_data = capacity_response.json()
                        worker.available_capacity = capacity_data.get("available_capacity", 0)

                    return worker.status == "healthy"
                else:
                    worker.status = "unhealthy"
                    return False

        except Exception as e:
            logger.warning(f"Health check failed for {worker.worker_id}: {e}")
            worker.status = "unhealthy"
            return False

    async def run_health_checks(self):
        """Continuously monitor worker health."""
        logger.info("Starting health check loop")

        while True:
            await asyncio.sleep(self.health_check_interval)

            # Check all workers
            for worker in list(self.workers.values()):
                is_healthy = await self.check_worker_health(worker)

                if not is_healthy:
                    # Check if worker has been unhealthy for too long
                    if datetime.now() - worker.last_heartbeat > timedelta(seconds=30):
                        logger.error(f"Worker {worker.worker_id} unhealthy for 30s, removing")
                        await self.deregister_worker(worker.worker_id)

    def start_health_checks(self):
        """Start background health checking."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self.run_health_checks())
```

### Step 4: Service Router with Load Balancing (60 minutes)

```python
# router/service_router.py

import asyncio
import logging
from typing import List, Optional
from enum import Enum
import random
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from registry.worker_registry import WorkerRegistry, WorkerInfo

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    RANDOM = "random"

class ServiceRouter:
    """
    Routes requests to GPU workers with load balancing.

    Supports multiple load balancing strategies and fault tolerance.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
    ):
        self.registry = registry
        self.strategy = strategy
        self.round_robin_index = 0

        self.app = FastAPI(title="Multi-GPU Service Router")
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.post("/v1/completions")
        async def route_request(request: dict):
            """Route inference request to a worker."""
            # Select worker based on strategy
            worker = self._select_worker()

            if worker is None:
                raise HTTPException(
                    status_code=503,
                    detail="No healthy workers available"
                )

            # Forward request to worker
            try:
                return await self._forward_request(worker, request)
            except Exception as e:
                logger.error(f"Request failed on {worker.worker_id}: {e}")

                # Retry on different worker
                retry_worker = self._select_worker(exclude=[worker.worker_id])
                if retry_worker:
                    logger.info(f"Retrying on {retry_worker.worker_id}")
                    return await self._forward_request(retry_worker, request)
                else:
                    raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/workers")
        async def list_workers():
            """List all registered workers."""
            return {
                "workers": [
                    {
                        "worker_id": w.worker_id,
                        "status": w.status,
                        "active_requests": w.active_requests,
                        "available_capacity": w.available_capacity,
                    }
                    for w in self.registry.workers.values()
                ]
            }

        @self.app.get("/health")
        async def health():
            """Router health check."""
            healthy_workers = self.registry.get_healthy_workers()
            return {
                "status": "healthy" if healthy_workers else "unhealthy",
                "total_workers": len(self.registry.workers),
                "healthy_workers": len(healthy_workers),
            }

    def _select_worker(self, exclude: List[str] = None) -> Optional[WorkerInfo]:
        """
        Select a worker based on load balancing strategy.

        Args:
            exclude: List of worker IDs to exclude

        Returns:
            Selected worker, or None if no workers available
        """
        exclude = exclude or []
        healthy_workers = [
            w for w in self.registry.get_healthy_workers()
            if w.worker_id not in exclude
        ]

        if not healthy_workers:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            worker = healthy_workers[self.round_robin_index % len(healthy_workers)]
            self.round_robin_index += 1
            return worker

        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select worker with most available capacity
            return min(healthy_workers, key=lambda w: w.active_requests)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            # Weight by available capacity
            total_capacity = sum(w.available_capacity for w in healthy_workers)
            if total_capacity == 0:
                return random.choice(healthy_workers)

            weights = [w.available_capacity / total_capacity for w in healthy_workers]
            return random.choices(healthy_workers, weights=weights)[0]

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_workers)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _forward_request(self, worker: WorkerInfo, request: dict) -> dict:
        """Forward request to worker and return response."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{worker.url}/v1/completions",
                json=request,
            )

            if response.status_code != 200:
                raise Exception(f"Worker returned status {response.status_code}")

            return response.json()

    def run(self, port: int = 8080):
        """Start the router service."""
        import uvicorn
        logger.info(f"Starting service router on port {port}")
        uvicorn.run(self.app, host="0.0.0.0", port=port)
```

### Step 5: Deployment Scripts (30 minutes)

```bash
#!/bin/bash
# scripts/start_multi_gpu_service.sh

set -e

MODEL_NAME="facebook/opt-1.3b"
NUM_GPUS=4
WORKER_BASE_PORT=8000
ROUTER_PORT=8080

echo "Starting Multi-GPU Inference Service"
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS"

# Start worker registry
echo "Starting worker registry..."
python -m registry.server &
REGISTRY_PID=$!
sleep 2

# Start GPU workers
echo "Starting GPU workers..."
for ((i=0; i<NUM_GPUS; i++)); do
    WORKER_PORT=$((WORKER_BASE_PORT + i))
    echo "  Starting worker $i on GPU $i (port $WORKER_PORT)"

    CUDA_VISIBLE_DEVICES=$i python -m worker.gpu_worker \
        "worker-$i" \
        $i \
        "$MODEL_NAME" \
        $WORKER_PORT &

    WORKER_PIDS[$i]=$!
    sleep 5  # Give worker time to initialize
done

# Start router
echo "Starting service router on port $ROUTER_PORT..."
python -m router.service_router --port $ROUTER_PORT &
ROUTER_PID=$!

echo ""
echo "Multi-GPU Service Started!"
echo "Router endpoint: http://localhost:$ROUTER_PORT"
echo "Worker endpoints: http://localhost:$WORKER_BASE_PORT - http://localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1))"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "echo 'Stopping services...'; kill $REGISTRY_PID $ROUTER_PID ${WORKER_PIDS[@]}; exit" SIGINT
wait
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_router.py

import pytest
from router.service_router import ServiceRouter, LoadBalancingStrategy
from registry.worker_registry import WorkerRegistry, WorkerInfo

@pytest.fixture
def registry():
    """Create registry with mock workers."""
    reg = WorkerRegistry()
    return reg

@pytest.fixture
async def router(registry):
    """Create router with registry."""
    router = ServiceRouter(registry, LoadBalancingStrategy.ROUND_ROBIN)
    return router

@pytest.mark.asyncio
async def test_round_robin_selection(registry, router):
    """Test round-robin load balancing."""
    # Add workers
    for i in range(3):
        await registry.register_worker(WorkerInfo(
            worker_id=f"worker-{i}",
            url=f"http://localhost:800{i}",
            gpu_id=i,
        ))

    # Select workers multiple times
    selected = [router._select_worker().worker_id for _ in range(6)]

    # Should cycle through workers
    assert selected == ["worker-0", "worker-1", "worker-2"] * 2

@pytest.mark.asyncio
async def test_least_loaded_selection(registry):
    """Test least-loaded selection."""
    router = ServiceRouter(registry, LoadBalancingStrategy.LEAST_LOADED)

    # Add workers with different loads
    await registry.register_worker(WorkerInfo(
        worker_id="worker-0", url="http://localhost:8000",
        gpu_id=0, active_requests=10
    ))
    await registry.register_worker(WorkerInfo(
        worker_id="worker-1", url="http://localhost:8001",
        gpu_id=1, active_requests=2
    ))

    # Should select least loaded
    selected = router._select_worker()
    assert selected.worker_id == "worker-1"
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
import httpx

@pytest.mark.asyncio
async def test_end_to_end_inference():
    """Test complete inference flow through router."""
    router_url = "http://localhost:8080"

    request = {
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{router_url}/v1/completions",
            json=request,
            timeout=30.0,
        )

    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert len(data["generated_text"]) > 0

@pytest.mark.asyncio
async def test_fault_tolerance():
    """Test that service handles worker failure."""
    # TODO: Implement by simulating worker crash
    pass
```

---

## Deliverables

1. **Multi-GPU Service Implementation**
   - GPU worker wrapper
   - Worker registry
   - Service router with load balancing
   - Health checking system

2. **Deployment Scripts**
   - Start/stop scripts
   - Configuration management
   - Docker Compose (optional)

3. **Monitoring Setup**
   - Prometheus metrics
   - Grafana dashboards

4. **Documentation**
   - Architecture diagram
   - API documentation
   - Deployment guide
   - Troubleshooting guide

5. **Test Suite**
   - Unit tests
   - Integration tests
   - Load tests

---

## Evaluation Rubric

### Functionality (40 points)
- ✅ Multi-GPU support working: 10 pts
- ✅ Load balancing strategies: 10 pts
- ✅ Fault tolerance: 10 pts
- ✅ Health monitoring: 10 pts

### Performance (20 points)
- ✅ >85% GPU utilization: 10 pts
- ✅ <5ms routing overhead: 10 pts

### Code Quality (15 points)
- ✅ Clean architecture: 5 pts
- ✅ Error handling: 5 pts
- ✅ Documentation: 5 pts

### Testing (15 points)
- ✅ Unit tests: 5 pts
- ✅ Integration tests: 5 pts
- ✅ Load tests: 5 pts

### Operations (10 points)
- ✅ Deployment automation: 5 pts
- ✅ Monitoring setup: 5 pts

**Total: 100 points | Passing: 70 points**

---

## Common Pitfalls

1. **Race Conditions**: Use proper locking in registry
2. **Request Timeouts**: Set appropriate timeout values
3. **Memory Leaks**: Clean up after failed requests
4. **Health Check Storms**: Rate limit health checks
5. **Split Brain**: Ensure consistent worker state

---

**Time to build a production-grade distributed system!**

*Project 7.3: Multi-GPU Service | Duration: 3-4 hours*
