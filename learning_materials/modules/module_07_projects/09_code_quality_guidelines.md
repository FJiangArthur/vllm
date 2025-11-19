# Code Quality Guidelines

## Introduction

High-quality code is:
- **Correct**: Works as intended
- **Readable**: Easy to understand
- **Maintainable**: Easy to modify
- **Testable**: Easy to verify
- **Performant**: Meets efficiency requirements

This guide provides concrete standards for Module 7 projects.

---

## Python Code Standards

### 1. Style Guide (PEP 8)

Follow [PEP 8](https://pep8.org/) with these key points:

#### Naming Conventions

```python
# Good: Clear, descriptive names
class RequestScheduler:
    def schedule_batch(self, pending_requests: List[Request]) -> SchedulingDecision:
        max_batch_size = self.config.max_batch_size
        selected_requests = []
        # ...

# Bad: Unclear abbreviations
class ReqSched:
    def sched_bat(self, pr: List) -> SD:
        mbs = self.cfg.mbs
        sr = []
        # ...
```

**Rules:**
- Classes: `PascalCase` (e.g., `GPUWorker`, `MetricsCollector`)
- Functions/methods: `snake_case` (e.g., `calculate_latency`, `send_request`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- Private members: `_leading_underscore` (e.g., `_internal_state`)

#### Line Length

```python
# Good: Line < 100 characters
result = self.model.generate(
    prompts=prompts,
    max_tokens=max_tokens,
    temperature=temperature,
)

# Bad: Line too long
result = self.model.generate(prompts=prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty)
```

**Limit:** 100 characters per line (99 preferred)

#### Imports

```python
# Good: Organized imports
# Standard library
import asyncio
import logging
from typing import List, Dict, Optional

# Third-party
import numpy as np
import torch
from fastapi import FastAPI

# Local
from .scheduler import PriorityScheduler
from .metrics import MetricsCollector

# Bad: Disorganized, wildcard imports
from typing import *
import torch
from .scheduler import *
import numpy as np
import asyncio
```

**Rules:**
- Group imports: standard library, third-party, local
- Alphabetize within groups
- No wildcard imports (`from module import *`)
- Use explicit imports for clarity

### 2. Type Hints

Use type hints throughout for clarity and error detection.

#### Function Signatures

```python
# Excellent: Complete type hints with docstring
def schedule_requests(
    self,
    pending_requests: List[Request],
    available_memory: int,
    max_batch_size: int = 32,
) -> SchedulingDecision:
    """
    Schedule requests based on priority and resources.

    Args:
        pending_requests: List of requests to consider
        available_memory: Available GPU memory in bytes
        max_batch_size: Maximum number of requests per batch

    Returns:
        SchedulingDecision with selected requests and metadata

    Raises:
        ValueError: If max_batch_size < 1
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    # ...

# Good: Type hints present
def schedule_requests(
    pending_requests: List[Request],
    available_memory: int,
) -> SchedulingDecision:
    # ...

# Bad: No type hints
def schedule_requests(pending_requests, available_memory):
    # ...
```

#### Data Classes

```python
from dataclasses import dataclass
from typing import Optional

# Excellent: Type-safe data class
@dataclass
class Request:
    """Represents an inference request."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    arrival_time: float = 0.0
    priority: Optional[int] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")

# Good: Basic dataclass
@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int

# Bad: Plain dictionary
request = {
    'request_id': '123',
    'prompt': 'Hello',
    'max_tokens': 100,
}
```

#### Type Checking

```bash
# Run mypy for type checking
mypy src/ --strict
```

Fix all mypy errors before submission.

### 3. Documentation

#### Module-Level Docstrings

```python
"""
GPU Worker for vLLM Multi-GPU Service.

This module implements a GPU worker that wraps vLLM for distributed
inference serving. It handles model loading, request processing,
and health reporting.

Key Classes:
    - GPUWorker: Main worker implementation
    - WorkerConfig: Configuration dataclass

Example:
    >>> config = WorkerConfig(
    ...     worker_id="worker-0",
    ...     gpu_id=0,
    ...     model_name="meta-llama/Llama-2-7b-hf"
    ... )
    >>> worker = GPUWorker(config)
    >>> worker.run()
"""
```

#### Function Docstrings

```python
def calculate_percentile_latency(
    latencies: List[float],
    percentile: float,
) -> float:
    """
    Calculate percentile latency from a list of measurements.

    Uses linear interpolation between closest ranks (NumPy method).

    Args:
        latencies: List of latency measurements in seconds
        percentile: Percentile to calculate (0-100)

    Returns:
        Latency value at the specified percentile in seconds

    Raises:
        ValueError: If latencies is empty or percentile not in [0, 100]

    Examples:
        >>> latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> calculate_percentile_latency(latencies, 50)
        0.3
        >>> calculate_percentile_latency(latencies, 95)
        0.49
    """
    if not latencies:
        raise ValueError("latencies cannot be empty")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")

    return np.percentile(latencies, percentile)
```

**Docstring Format:** Google style (Args, Returns, Raises, Examples)

#### Inline Comments

```python
# Good: Explain "why", not "what"
# Use exponential backoff to avoid overwhelming the server during retries
for attempt in range(max_retries):
    try:
        return self._send_request(request)
    except RequestException:
        wait_time = (2 ** attempt) * base_delay
        time.sleep(wait_time)

# Bad: State the obvious
# Increment i by 1
i += 1

# Loop through requests
for request in requests:
    # Process request
    process(request)
```

**Guidelines:**
- Comment complex algorithms
- Explain non-obvious decisions
- Don't comment obvious code
- Use TODO/FIXME/NOTE markers

```python
# TODO: Add support for custom sampling parameters
# FIXME: Handle race condition when queue is empty
# NOTE: This assumes requests are already sorted by priority
```

### 4. Error Handling

#### Specific Exceptions

```python
# Good: Specific exception with helpful message
class InsufficientMemoryError(Exception):
    """Raised when GPU memory is insufficient for request."""
    pass

def allocate_memory(required_bytes: int) -> None:
    available = get_available_memory()
    if required_bytes > available:
        raise InsufficientMemoryError(
            f"Required {required_bytes} bytes but only "
            f"{available} bytes available"
        )

# Bad: Generic exception
def allocate_memory(required_bytes):
    if required_bytes > get_available_memory():
        raise Exception("Not enough memory")
```

#### Error Context

```python
# Good: Preserve error context
try:
    result = self.model.generate(prompt)
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"OOM during generation for request {request_id}")
    raise InsufficientMemoryError(
        f"GPU OOM for request {request_id}"
    ) from e

# Bad: Swallow errors
try:
    result = self.model.generate(prompt)
except:
    pass  # Silent failure
```

#### Validation

```python
# Good: Validate inputs early
def schedule_batch(
    self,
    requests: List[Request],
    max_batch_size: int,
) -> List[Request]:
    """Schedule a batch of requests."""
    # Validate inputs
    if not requests:
        raise ValueError("requests cannot be empty")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if len(requests) < max_batch_size:
        logger.warning(
            f"Only {len(requests)} requests but max_batch_size "
            f"is {max_batch_size}"
        )

    # Proceed with validated inputs
    # ...
```

### 5. Logging

#### Structured Logging

```python
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Good: Structured logs with context
logger.info(
    "Request scheduled",
    extra={
        "request_id": request.request_id,
        "priority": request.priority,
        "wait_time_ms": wait_time * 1000,
    }
)

# For production: JSON logging
import json

def log_json(level: str, message: str, **kwargs):
    log_entry = {
        "timestamp": time.time(),
        "level": level,
        "message": message,
        **kwargs
    }
    print(json.dumps(log_entry))

log_json("INFO", "Request scheduled",
         request_id="123", priority=2)
```

#### Log Levels

```python
# DEBUG: Detailed diagnostic info
logger.debug(f"GPU memory: {gpu_mem_mb} MB")

# INFO: General informational messages
logger.info("Model loaded successfully")

# WARNING: Something unexpected but handled
logger.warning(f"Queue length {queue_len} exceeds threshold")

# ERROR: Serious problem, operation failed
logger.error(f"Failed to process request {req_id}: {error}")

# CRITICAL: System failure
logger.critical("GPU device not found, shutting down")
```

**Rule:** Use appropriate log levels. Production systems should run at INFO level, with DEBUG available for troubleshooting.

### 6. Testing Best Practices

#### Test Organization

```python
# tests/test_scheduler.py

import pytest
from src.scheduler import PriorityScheduler, SchedulerConfig

class TestPriorityScheduler:
    """Test suite for PriorityScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler instance for testing."""
        config = SchedulerConfig(max_batch_size=10)
        return PriorityScheduler(config)

    def test_empty_queue_returns_empty_batch(self, scheduler):
        """Test that scheduling with no requests returns empty batch."""
        decision = scheduler.schedule(available_memory=1000)
        assert len(decision.selected_requests) == 0

    def test_single_request_scheduled(self, scheduler):
        """Test that a single request is scheduled successfully."""
        scheduler.add_request("req1", priority=1, num_tokens=100)
        decision = scheduler.schedule(available_memory=1000)

        assert len(decision.selected_requests) == 1
        assert decision.selected_requests[0] == "req1"

    @pytest.mark.parametrize("num_requests,expected_batches", [
        (5, 1),
        (15, 2),
        (25, 3),
    ])
    def test_batching_with_varying_load(
        self, scheduler, num_requests, expected_batches
    ):
        """Test batching behavior with different request counts."""
        for i in range(num_requests):
            scheduler.add_request(f"req{i}", priority=1)

        batches = 0
        while scheduler.queue_manager.total_pending() > 0:
            scheduler.schedule(available_memory=10000)
            batches += 1

        assert batches == expected_batches
```

#### Test Coverage

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

**Target:** >85% code coverage, 100% for critical paths

#### Test Types

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Validate performance requirements
4. **Edge Case Tests**: Test boundary conditions

```python
# Edge case testing
def test_scheduler_with_zero_memory():
    """Test behavior when no memory available."""
    scheduler = PriorityScheduler(config)
    scheduler.add_request("req1")

    decision = scheduler.schedule(available_memory=0)
    assert len(decision.selected_requests) == 0

def test_scheduler_with_exact_memory_fit():
    """Test when request exactly fits available memory."""
    scheduler = PriorityScheduler(config)
    scheduler.add_request("req1", num_tokens=1000)

    decision = scheduler.schedule(available_memory=1000)
    assert len(decision.selected_requests) == 1
```

---

## CUDA/C++ Code Standards

### 1. Kernel Documentation

```cuda
/**
 * @brief Optimized softmax kernel for row-wise computation.
 *
 * Computes softmax across rows of a 2D matrix using warp-level
 * primitives for efficient parallel reduction.
 *
 * @param[in]  input   Input matrix (batch_size x seq_len)
 * @param[out] output  Output matrix (same shape as input)
 * @param[in]  batch_size  Number of rows
 * @param[in]  seq_len     Number of columns
 *
 * @note Each block processes one row
 * @note Uses shared memory for intermediate results
 */
__global__ void optimized_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Implementation
}
```

### 2. Error Checking

```cuda
// Good: Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Use in code
CUDA_CHECK(cudaMalloc(&d_input, size));
CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

// Bad: No error checking
cudaMalloc(&d_input, size);
cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
```

### 3. Code Organization

```cuda
// Constants at top
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Helper device functions
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel implementation
__global__ void my_kernel(...) {
    // Clear variable names
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;

    // Shared memory
    __shared__ float shared_data[BLOCK_SIZE];

    // Algorithm steps with comments
    // ...
}
```

---

## Git Best Practices

### 1. Commit Messages

```bash
# Good: Clear, descriptive commit messages
git commit -m "Add priority-based aging to prevent starvation

- Implement aging mechanism that promotes requests after threshold
- Add configurable aging parameters to SchedulerConfig
- Update tests to verify no starvation under load
- Fixes #42"

# Bad: Vague or uninformative
git commit -m "fix stuff"
git commit -m "updates"
```

**Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, refactor, test, perf

### 2. Branch Strategy

```bash
# Create feature branch
git checkout -b feature/priority-scheduling

# Work on feature, commit often
git commit -m "feat: add Priority enum and data structures"
git commit -m "feat: implement priority queue manager"

# When done, merge to main
git checkout main
git merge feature/priority-scheduling
```

### 3. .gitignore

```bash
# .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/

# IDEs
.vscode/
.idea/
*.swp

# Testing
.pytest_cache/
.coverage
htmlcov/

# CUDA
*.o
*.so

# Logs
*.log

# Model caches
model_cache/
*.bin

# Secrets (IMPORTANT!)
.env
config/secrets.yaml
*.pem
*.key
```

---

## Code Review Checklist

Before submitting code for review:

### ✅ Functionality
- [ ] Code works as intended
- [ ] All requirements met
- [ ] Edge cases handled

### ✅ Quality
- [ ] Follows style guide (PEP 8)
- [ ] Type hints throughout
- [ ] Well-documented
- [ ] No TODOs or FIXMEs (unless tracked)

### ✅ Testing
- [ ] Tests written and passing
- [ ] Coverage >85%
- [ ] Performance tests included

### ✅ Security
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] No SQL injection risks (if applicable)

### ✅ Performance
- [ ] No obvious inefficiencies
- [ ] Profiled if performance-critical

### ✅ Git
- [ ] Clean commit history
- [ ] Meaningful commit messages
- [ ] .gitignore configured

---

## Tools

### Automated Formatting

```bash
# Black: Opinionated code formatter
black src/ tests/

# isort: Sort imports
isort src/ tests/

# Configuration in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']

[tool.isort]
profile = "black"
line_length = 100
```

### Linting

```bash
# flake8: Style checking
flake8 src/ --max-line-length=100

# pylint: Comprehensive linting
pylint src/

# mypy: Type checking
mypy src/ --strict
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        args: ['--strict']
```

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Summary

High-quality code is not just about making it work—it's about making it maintainable, testable, and professional. Following these guidelines will:

1. **Make your code easier to understand** (for yourself and others)
2. **Reduce bugs** (through typing, testing, validation)
3. **Enable collaboration** (through clear documentation)
4. **Demonstrate professionalism** (important for interviews!)

**Remember:** Code is read far more often than it's written. Invest in clarity!

---

*Module 7: Code Quality Guidelines*
*Last Updated: 2025-11-19*
