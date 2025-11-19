# Module 1.4: vLLM Project Structure Tour

## Learning Objectives

By the end of this lesson, you will be able to:

1. Navigate the vLLM codebase confidently
2. Understand the purpose of each major directory
3. Identify key source files for different functionality
4. Trace code flow from API to GPU execution
5. Locate relevant code when debugging or implementing features
6. Understand the architectural organization

**Estimated Time:** 2 hours

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Top-Level Structure](#top-level-structure)
3. [Core Directories Deep Dive](#core-directories-deep-dive)
4. [Key Source Files](#key-source-files)
5. [Code Flow: Request to Response](#code-flow-request-to-response)
6. [Finding Code for Specific Features](#finding-code-for-specific-features)
7. [Build System Files](#build-system-files)
8. [Testing and Benchmarking](#testing-and-benchmarking)
9. [Practical Exercises](#practical-exercises)
10. [Key Takeaways](#key-takeaways)

---

## Repository Overview

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    vLLM Architecture Layers                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Entrypoints (API Servers)                                   │
│  └─ OpenAI-compatible API, CLI, Anthropic API               │
│                          │                                    │
│                          ▼                                    │
│  Engine (Orchestration)                                      │
│  └─ LLMEngine, AsyncLLMEngine                               │
│                          │                                    │
│                          ▼                                    │
│  Scheduler & Block Manager                                   │
│  └─ Request scheduling, KV cache management                 │
│                          │                                    │
│                          ▼                                    │
│  Model Executor (GPU Workers)                                │
│  └─ Model loading, forward pass, sampling                   │
│                          │                                    │
│                          ▼                                    │
│  CUDA Kernels (csrc/)                                        │
│  └─ Attention, cache ops, quantization                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Top-Level Structure

### Directory Layout

```bash
# View actual structure
cd /home/user/vllm-learn
tree -L 1 -d

# Output:
vllm-learn/
├── .github/              # CI/CD workflows, issue templates
├── benchmarks/           # End-to-end benchmarking scripts
├── cmake/                # CMake modules and configuration
├── csrc/                 # C++ and CUDA source code
├── docs/                 # Documentation (Sphinx)
├── examples/             # Example scripts
├── tests/                # Unit and integration tests
├── vllm/                 # Main Python package
└── ...
```

### Critical Files at Root

```bash
# Build and dependency files
setup.py              # Python package setup and build orchestration
CMakeLists.txt        # CMake build configuration for C++/CUDA
pyproject.toml        # Project metadata and build config
requirements*.txt     # Dependency specifications

# Configuration
.clang-format         # C++ code formatting
.pre-commit-config.yaml  # Pre-commit hooks
.readthedocs.yaml     # Documentation build config

# Documentation
README.md             # Project overview
CONTRIBUTING.md       # Contribution guidelines
LICENSE               # Apache 2.0 license
```

**Reference File Locations:**
- Setup script: `/home/user/vllm-learn/setup.py`
- CMake config: `/home/user/vllm-learn/CMakeLists.txt`
- Requirements: `/home/user/vllm-learn/requirements-common.txt`

---

## Core Directories Deep Dive

### 1. `vllm/` - Main Python Package

**Location:** `/home/user/vllm-learn/vllm/`

This is the heart of vLLM. Let's explore subdirectories:

```
vllm/
├── __init__.py                  # Package initialization, exports LLM, SamplingParams
├── attention/                   # Attention implementations
│   ├── backends/                # Different attention backends
│   │   ├── flash_attn.py       # FlashAttention backend
│   │   ├── xformers.py         # xFormers backend
│   │   └── torch_sdpa.py       # PyTorch SDPA backend
│   ├── layers/                  # Attention layer implementations
│   └── ops/                     # Attention CUDA operations
│
├── config/                      # Configuration classes
│   ├── model_config.py         # Model configuration
│   ├── cache_config.py         # KV cache configuration
│   ├── parallel_config.py      # Distributed parallelism config
│   └── scheduler_config.py     # Scheduler configuration
│
├── engine/                      # Core inference engine
│   ├── llm_engine.py           # Main LLMEngine class
│   ├── async_llm_engine.py     # Async version for servers
│   ├── arg_utils.py            # Argument parsing utilities
│   └── output_processor.py     # Output processing
│
├── entrypoints/                 # API servers and entry points
│   ├── openai/                 # OpenAI-compatible API
│   │   ├── api_server.py       # FastAPI server
│   │   └── protocol.py         # Request/response schemas
│   ├── anthropic/              # Anthropic-compatible API
│   └── cli/                    # Command-line interface
│
├── model_executor/              # Model execution on GPU
│   ├── model_loader/           # Model loading and initialization
│   ├── models/                 # Model implementations (Llama, GPT, etc.)
│   │   ├── llama.py           # Llama model
│   │   ├── gpt2.py            # GPT-2 model
│   │   └── ...                # 50+ model architectures
│   ├── layers/                 # Neural network layers
│   └── parallel_utils/         # Tensor/pipeline parallelism
│
├── distributed/                 # Distributed execution
│   ├── parallel_state.py       # Parallel group management
│   ├── communication_ops.py    # Collective operations (all-reduce, etc.)
│   └── device_communicators/   # NCCL, GLOO communicators
│
└── ... (more directories below)
```

### 2. `csrc/` - C++ and CUDA Code

**Location:** `/home/user/vllm-learn/csrc/`

Low-level performance-critical operations:

```
csrc/
├── attention/                   # Attention CUDA kernels
│   ├── attention_kernels.cu    # PagedAttention implementation
│   └── attention_generic.cuh   # Generic attention templates
│
├── cache_kernels.cu            # KV cache operations
│   └── reshape_and_cache, copy_blocks, gather_cached_kv
│
├── quantization/               # Quantization kernels
│   ├── awq/                   # AWQ (Activation-aware Weight Quantization)
│   ├── gptq/                  # GPTQ kernels
│   └── squeezellm/            # SqueezeLLM kernels
│
├── cuda_utils/                 # CUDA utilities
│   └── dispatch_utils.h       # Kernel dispatch helpers
│
├── pos_encoding_kernels.cu    # Positional encoding (RoPE)
├── activation_kernels.cu      # Activation functions (GELU, SiLU)
├── layernorm_kernels.cu       # Layer normalization
└── ops.h                      # Operation declarations
```

**Key CUDA Files:**
- PagedAttention: `/home/user/vllm-learn/csrc/attention/attention_kernels.cu`
- Cache operations: `/home/user/vllm-learn/csrc/cache_kernels.cu`
- Custom ops: `/home/user/vllm-learn/vllm/_custom_ops.py` (Python bindings)

### 3. `vllm/engine/` - Inference Engine

**Location:** `/home/user/vllm-learn/vllm/engine/`

The orchestration layer:

```python
# Key files:

llm_engine.py (550+ lines)
  ├── class LLMEngine
  │   ├── __init__()            # Initialize engine with configs
  │   ├── add_request()         # Add new request to queue
  │   ├── step()                # Execute one iteration
  │   └── _process_outputs()    # Process model outputs
  │
  └── Coordinates: scheduler, workers, block manager

async_llm_engine.py (450+ lines)
  └── class AsyncLLMEngine      # Async wrapper for API servers

arg_utils.py (600+ lines)
  └── class EngineArgs           # Parse and validate arguments
```

**File Reference:**
- Main engine: `/home/user/vllm-learn/vllm/engine/llm_engine.py`
- Async engine: `/home/user/vllm-learn/vllm/engine/async_llm_engine.py`

### 4. `vllm/entrypoints/` - API Servers

**Location:** `/home/user/vllm-learn/vllm/entrypoints/`

How users interact with vLLM:

```python
# Main entry points:

1. Python API (vllm/__init__.py)
   from vllm import LLM, SamplingParams
   llm = LLM(model="...")
   outputs = llm.generate(prompts, sampling_params)

2. OpenAI-compatible API (entrypoints/openai/api_server.py)
   python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf
   # Provides: /v1/completions, /v1/chat/completions, /v1/models

3. Anthropic-compatible API (entrypoints/anthropic/api_server.py)
   # Claude-style messages API

4. CLI (entrypoints/cli/cli.py)
   vllm serve meta-llama/Llama-2-7b-hf
```

**File Reference:**
- OpenAI server: `/home/user/vllm-learn/vllm/entrypoints/openai/api_server.py`
- Protocol definitions: `/home/user/vllm-learn/vllm/entrypoints/openai/protocol.py`

### 5. `vllm/model_executor/` - GPU Workers

**Location:** `/home/user/vllm-learn/vllm/model_executor/`

Executes models on GPU:

```
model_executor/
├── model_loader/
│   ├── loader.py              # Model loading from HuggingFace
│   ├── weight_utils.py        # Weight initialization and conversion
│   └── ...
│
├── models/                    # 50+ model implementations
│   ├── llama.py              # LlamaForCausalLM
│   ├── gpt2.py               # GPT2LMHeadModel
│   ├── mixtral.py            # Mixtral MoE
│   └── ...
│
├── layers/
│   ├── attention.py          # Attention layer wrapper
│   ├── linear.py             # Linear layers with parallelism
│   ├── activation.py         # Activation functions
│   ├── layernorm.py          # Layer normalization
│   ├── vocab_parallel_embedding.py  # Embedding with tensor parallelism
│   └── sampler.py            # Token sampling
│
└── parallel_utils/
    ├── parallel_state.py     # Manage tensor/pipeline parallel groups
    └── communication_ops.py  # Distributed communication primitives
```

**Key Model File Example:**
```python
# /home/user/vllm-learn/vllm/model_executor/models/llama.py

class LlamaForCausalLM(nn.Module):
    def __init__(self, ...):
        # Initialize model layers
        self.model = LlamaModel(...)
        self.lm_head = ParallelLMHead(...)

    def forward(self, ...):
        # Forward pass through model
        hidden_states = self.model(...)
        return self.lm_head(hidden_states)
```

### 6. Core Scheduling Components

```
vllm/
├── core/                        # Core scheduling logic
│   ├── scheduler.py            # Main scheduler (FCFS, priority)
│   ├── block_manager_v1.py     # Block manager v1
│   ├── block_manager_v2.py     # Block manager v2 (prefix caching)
│   └── policy.py               # Scheduling policies
│
└── sequence.py                 # Sequence and SequenceGroup classes
```

**Scheduler Flow:**
```python
# /home/user/vllm-learn/vllm/core/scheduler.py

class Scheduler:
    def schedule(self):
        # 1. Select sequences to run
        # 2. Allocate GPU blocks
        # 3. Handle preemption if needed
        # 4. Return scheduling metadata
        ...
```

### 7. Attention Backends

**Location:** `/home/user/vllm-learn/vllm/attention/`

```
attention/
├── backends/
│   ├── abstract.py            # Abstract backend interface
│   ├── flash_attn.py          # FlashAttention (default, fastest)
│   ├── xformers.py            # xFormers memory-efficient attention
│   └── torch_sdpa.py          # PyTorch scaled_dot_product_attention
│
├── layers/
│   └── attention.py           # Main Attention layer
│
└── ops/                       # Operation wrappers
    └── paged_attn.py          # PagedAttention operation
```

**Attention Backend Selection:**
```python
# Configured via environment variable:
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # (default)
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
```

---

## Key Source Files

### Essential Files to Understand

```python
# 1. Main API entry point
"/home/user/vllm-learn/vllm/__init__.py"
  └─ Exports: LLM, SamplingParams, EngineArgs, AsyncLLMEngine

# 2. Core inference loop
"/home/user/vllm-learn/vllm/engine/llm_engine.py"
  └─ class LLMEngine.step() - Execute one iteration

# 3. Request scheduling
"/home/user/vllm-learn/vllm/core/scheduler.py"
  └─ class Scheduler.schedule() - Schedule sequences

# 4. KV cache management
"/home/user/vllm-learn/vllm/core/block_manager_v2.py"
  └─ class BlockSpaceManagerV2 - Manage GPU blocks

# 5. Attention implementation
"/home/user/vllm-learn/vllm/attention/backends/flash_attn.py"
  └─ FlashAttention backend

# 6. Model execution
"/home/user/vllm-learn/vllm/worker/worker.py"
  └─ class Worker.execute_model() - Run forward pass

# 7. Sampling
"/home/user/vllm-learn/vllm/model_executor/layers/sampler.py"
  └─ class Sampler.forward() - Sample next tokens

# 8. Custom CUDA operations
"/home/user/vllm-learn/vllm/_custom_ops.py"
  └─ Python bindings for C++/CUDA kernels
```

### Configuration Files

```python
# All configuration classes
"/home/user/vllm-learn/vllm/config/"
  ├── model_config.py          # Model-specific config
  ├── cache_config.py          # KV cache size and type
  ├── parallel_config.py       # Tensor/pipeline parallelism
  ├── scheduler_config.py      # Scheduler settings
  └── device_config.py         # Device (GPU/CPU) config
```

---

## Code Flow: Request to Response

### Complete Flow Diagram

```
User Request
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ 1. Entrypoint (openai/api_server.py)                   │
│    - Receive HTTP request                              │
│    - Parse prompt, sampling_params                     │
│    - Convert to internal format                        │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 2. AsyncLLMEngine (engine/async_llm_engine.py)         │
│    - Enqueue request                                   │
│    - Generate request_id                               │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 3. LLMEngine (engine/llm_engine.py)                    │
│    - add_request(): Add to request queue              │
│    - step(): Execute one iteration                     │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 4. Scheduler (core/scheduler.py)                       │
│    - Select sequences (prefill vs decode)              │
│    - Check if GPU blocks available                     │
│    - Handle preemption if memory pressure              │
│    - Return SchedulerOutputs                           │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 5. BlockManager (core/block_manager_v2.py)             │
│    - Allocate GPU blocks for KV cache                  │
│    - Manage block tables                               │
│    - Handle swapping (GPU ↔ CPU)                       │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 6. Worker (worker/worker.py)                           │
│    - execute_model(): Run forward pass                 │
│    - Prepare input tensors                             │
│    - Call model_runner                                 │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 7. ModelRunner (worker/model_runner.py)                │
│    - Batch inputs                                      │
│    - Prepare attention metadata                        │
│    - Call model.forward()                              │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 8. Model (model_executor/models/llama.py)              │
│    - Transformer forward pass                          │
│    - Embedding → Decoder Layers → LM Head             │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 9. Attention (attention/backends/flash_attn.py)        │
│    - PagedAttention (C++/CUDA kernel)                  │
│    - Access KV cache via block tables                  │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 10. Sampler (model_executor/layers/sampler.py)         │
│     - Sample next token from logits                    │
│     - Apply temperature, top_p, top_k                  │
│     - Return sampled token IDs                         │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ 11. Back to LLMEngine                                  │
│     - Update sequences with new tokens                 │
│     - Check stopping criteria (EOS, max_tokens)        │
│     - Return outputs to user                           │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
        User Response
```

### Tracing a Request in Code

```python
# Start: User makes API call
# File: vllm/entrypoints/openai/api_server.py
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    # Add to engine
    results_generator = engine.generate(
        request.prompt,
        sampling_params=SamplingParams(...)
    )
    # Stream results back
    async for output in results_generator:
        yield output

# Step 1: Add request to engine
# File: vllm/engine/async_llm_engine.py
async def generate(...):
    request_id = random_uuid()
    await self.add_request(request_id, prompt, sampling_params)
    # Yield outputs as they're generated
    async for output in self._process_request(request_id):
        yield output

# Step 2: Engine processes request
# File: vllm/engine/llm_engine.py
def step(self):
    # Get next batch from scheduler
    scheduler_outputs = self.scheduler.schedule()

    # Execute on workers
    output = self._run_workers("execute_model", scheduler_outputs)

    # Process outputs (sampling, stopping)
    return self._process_model_outputs(output)

# Step 3: Worker executes model
# File: vllm/worker/worker.py
def execute_model(self, scheduler_outputs):
    return self.model_runner.execute_model(scheduler_outputs)

# Step 4: Model forward pass
# File: vllm/model_executor/models/llama.py
def forward(self, ...):
    hidden_states = self.model(input_ids, positions, kv_caches, ...)
    return self.lm_head(hidden_states)

# Step 5: Attention computation
# File: vllm/attention/backends/flash_attn.py
def forward(self, query, key_cache, value_cache, ...):
    # Call PagedAttention CUDA kernel
    output = PagedAttention.forward(query, key_cache, value_cache, ...)
    return output
```

---

## Finding Code for Specific Features

### Common Search Patterns

```bash
# 1. Find where a specific API parameter is used
cd /home/user/vllm-learn
grep -r "temperature" vllm/ --include="*.py" | head -20

# 2. Find CUDA kernel implementations
find csrc/ -name "*.cu" -o -name "*.cuh"

# 3. Find all model implementations
ls vllm/model_executor/models/*.py

# 4. Find configuration options
grep -r "class.*Config" vllm/config/

# 5. Find where environment variables are read
grep -r "os.environ" vllm/ --include="*.py"

# 6. Find scheduling logic
grep -r "class.*Scheduler" vllm/ --include="*.py"
```

### Feature → File Mapping

```
Feature                    → Primary File(s)
─────────────────────────────────────────────────────────────
PagedAttention             → csrc/attention/attention_kernels.cu
                          → vllm/attention/ops/paged_attn.py

Continuous Batching        → vllm/core/scheduler.py
                          → vllm/engine/llm_engine.py

Prefix Caching             → vllm/core/block_manager_v2.py

Tensor Parallelism         → vllm/model_executor/parallel_utils/
                          → vllm/distributed/

Quantization (AWQ/GPTQ)    → csrc/quantization/
                          → vllm/model_executor/layers/quantization/

Model Loading              → vllm/model_executor/model_loader/loader.py

Token Sampling             → vllm/model_executor/layers/sampler.py

OpenAI API                 → vllm/entrypoints/openai/

Metrics & Monitoring       → vllm/engine/metrics.py
```

---

## Build System Files

### CMake Structure

```cmake
# /home/user/vllm-learn/CMakeLists.txt
cmake_minimum_required(VERSION 3.21)
project(vllm LANGUAGES CXX CUDA)

# Find dependencies
find_package(CUDA REQUIRED)
find_package(Python REQUIRED)

# Add subdirectories
add_subdirectory(csrc)

# Define targets
add_library(vllm_ops SHARED
    csrc/attention/attention_kernels.cu
    csrc/cache_kernels.cu
    # ...
)

# Link libraries
target_link_libraries(vllm_ops PUBLIC CUDA::cudart)
```

### Python Setup

```python
# /home/user/vllm-learn/setup.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='vllm._C',
        sources=[
            'csrc/attention/attention_kernels.cu',
            'csrc/cache_kernels.cu',
            # ...
        ],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
    )
]

setup(
    name='vllm',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    # ...
)
```

---

## Testing and Benchmarking

### Test Structure

```
tests/
├── conftest.py                 # Pytest configuration
├── core/                       # Core component tests
│   ├── test_scheduler.py
│   └── test_block_manager.py
├── engine/                     # Engine tests
│   └── test_llm_engine.py
├── models/                     # Model tests
│   └── test_llama.py
├── kernels/                    # CUDA kernel tests
│   └── test_attention.py
└── entrypoints/                # API tests
    └── test_openai_server.py
```

### Benchmark Structure

```
benchmarks/
├── benchmark_latency.py        # Measure latency
├── benchmark_throughput.py     # Measure throughput
├── benchmark_serving.py        # Serving benchmark
└── kernels/                    # Kernel benchmarks
    ├── benchmark_attention.py
    └── benchmark_paged_attention.py
```

---

## Practical Exercises

### Exercise 1.4.1: Code Navigation

**Objective:** Navigate the codebase to find specific functionality.

**Tasks:**
1. Find where `max_tokens` parameter is validated
2. Locate the PagedAttention CUDA kernel source
3. Find all model implementations that support LoRA
4. Identify where GPU memory is allocated

**Time:** 30 minutes

### Exercise 1.4.2: Trace Request Flow

**Objective:** Follow a request through the entire system.

**Tasks:**
1. Add print statements at each layer
2. Run a simple inference
3. Observe the call stack
4. Document the flow

**Time:** 45 minutes

### Exercise 1.4.3: Modify and Test

**Objective:** Make a small modification and verify it works.

**Tasks:**
1. Add a custom log message in `LLMEngine.step()`
2. Rebuild if necessary
3. Run inference and verify your message appears
4. Restore original code

**Time:** 20 minutes

---

## Key Takeaways

1. **vLLM is organized into clear layers**: Entrypoints → Engine → Scheduler → Workers → Kernels
2. **Key directories**:
   - `vllm/engine/`: Orchestration
   - `vllm/core/`: Scheduling and memory management
   - `vllm/model_executor/`: Model execution
   - `csrc/`: CUDA kernels
   - `vllm/entrypoints/`: API servers
3. **Request flow**: API → Engine → Scheduler → Worker → Model → Attention → Sampler
4. **Most modifications happen in** `vllm/` Python code, not `csrc/` unless optimizing kernels
5. **Understanding the structure helps with**:
   - Debugging issues
   - Adding features
   - Performance optimization
   - Reading documentation

---

## Quick Reference: File Locations

```bash
# Core entry point
/home/user/vllm-learn/vllm/__init__.py

# Main engine
/home/user/vllm-learn/vllm/engine/llm_engine.py

# Scheduler
/home/user/vllm-learn/vllm/core/scheduler.py

# Block manager
/home/user/vllm-learn/vllm/core/block_manager_v2.py

# Worker
/home/user/vllm-learn/vllm/worker/worker.py

# Attention kernels
/home/user/vllm-learn/csrc/attention/attention_kernels.cu

# Model implementations
/home/user/vllm-learn/vllm/model_executor/models/

# OpenAI API server
/home/user/vllm-learn/vllm/entrypoints/openai/api_server.py

# Configuration
/home/user/vllm-learn/vllm/config/

# Build files
/home/user/vllm-learn/setup.py
/home/user/vllm-learn/CMakeLists.txt
```

---

## Next Steps

In the next lesson, **05_basic_inference_api.md**, we'll cover:
- Using the LLM class for inference
- Understanding SamplingParams
- Running your first inference workload
- Measuring basic performance

---

**Estimated Completion Time:** 2 hours
**Difficulty:** Intermediate
**Prerequisites:** Python programming, basic understanding of ML frameworks
