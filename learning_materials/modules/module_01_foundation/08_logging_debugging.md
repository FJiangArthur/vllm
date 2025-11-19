# Module 1.8: Logging and Debugging

## Learning Objectives

1. Configure vLLM logging effectively
2. Use debugging tools and techniques
3. Trace request flow through system
4. Diagnose common issues
5. Profile GPU usage
6. Monitor memory consumption

**Estimated Time:** 1.5 hours

---

## vLLM Logging

### Log Levels

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

from vllm import LLM

llm = LLM(model="facebook/opt-125m")
# INFO: Initializing an LLM engine with config: ...
# INFO: # GPU blocks: 1234, # CPU blocks: 567
```

**Levels:**
- DEBUG: Detailed information
- INFO: General information (default)
- WARNING: Warnings
- ERROR: Errors only

### Environment Variable

```bash
# Set via environment
export VLLM_LOGGING_LEVEL=DEBUG

python your_script.py
```

### Custom Logger

```python
# custom_logging.py
import logging

# Configure vLLM logger
vllm_logger = logging.getLogger("vllm")
vllm_logger.setLevel(logging.DEBUG)

# Add custom handler
handler = logging.FileHandler("vllm.log")
handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
vllm_logger.addHandler(handler)

from vllm import LLM
llm = LLM(model="facebook/opt-125m")
```

---

## Debugging Tools

### Enable Eager Mode

Disable CUDA graphs for easier debugging:

```python
llm = LLM(
    model="facebook/opt-125m",
    enforce_eager=True  # Easier debugging
)
```

### Trace Function Calls

```bash
export VLLM_TRACE_FUNCTION=1

python your_script.py
```

### CUDA Launch Blocking

For debugging CUDA errors:

```bash
export CUDA_LAUNCH_BLOCKING=1

python your_script.py
# CUDA operations become synchronous
# Easier to pinpoint exact error location
```

---

## Request Tracing

### Add Custom Logging

```python
# trace_request.py
import logging
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.DEBUG)

# Patch to add custom logging
import vllm.engine.llm_engine
original_step = vllm.engine.llm_engine.LLMEngine.step

def logged_step(self):
    logging.info("=== Engine step starting ===")
    result = original_step(self)
    logging.info(f"=== Engine step completed, outputs: {len(result)} ===")
    return result

vllm.engine.llm_engine.LLMEngine.step = logged_step

# Now run inference
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
```

### Inspect Scheduler Decisions

```python
# scheduler_trace.py
from vllm import LLM
import vllm.core.scheduler

# Add logging to scheduler
original_schedule = vllm.core.scheduler.Scheduler.schedule

def logged_schedule(self):
    import logging
    result = original_schedule(self)
    logging.info(f"Scheduled {len(result.scheduled_seq_groups)} sequence groups")
    return result

vllm.core.scheduler.Scheduler.schedule = logged_schedule

llm = LLM(model="facebook/opt-125m")
llm.generate(["Test"], vllm.SamplingParams(max_tokens=10))
```

---

## GPU Profiling

### Using nvidia-smi

```bash
# Monitor GPU in real-time
nvidia-smi dmon -s pucvmet

# While running vLLM
python your_inference.py &
watch -n 1 nvidia-smi
```

### PyTorch Profiler

```python
# profile_gpu.py
import torch
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    outputs = llm.generate(
        ["Hello world"] * 10,
        SamplingParams(max_tokens=50)
    )

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
# View trace.json in chrome://tracing
```

### CUDA Events for Timing

```python
# cuda_timing.py
import torch
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=100))
end.record()

torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end):.2f} ms")
```

---

## Memory Debugging

### Track Memory Usage

```python
# memory_tracking.py
import torch
from vllm import LLM, SamplingParams

def print_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

print("Before loading:")
print_memory()

llm = LLM(model="facebook/opt-125m")

print("\nAfter loading:")
print_memory()

outputs = llm.generate(["Test"] * 100, SamplingParams(max_tokens=50))

print("\nAfter inference:")
print_memory()

# Get detailed memory summary
print("\nDetailed memory summary:")
print(torch.cuda.memory_summary())
```

### Detect Memory Leaks

```python
# memory_leak_detection.py
import torch
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# Run multiple iterations
for i in range(10):
    outputs = llm.generate(["Test"], SamplingParams(max_tokens=10))
    
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Iteration {i}: {allocated:.2f}GB")
    
    # Memory should stabilize after first few iterations
    # If it keeps growing, there's likely a leak
```

---

## Common Debugging Scenarios

### Issue 1: OOM Errors

```python
# Debug OOM with memory tracking
import torch
from vllm import LLM, SamplingParams

try:
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        gpu_memory_utilization=0.95  # Too high?
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM during loading. Solutions:")
        print("1. Reduce gpu_memory_utilization")
        print("2. Use smaller model")
        print("3. Enable tensor parallelism")
        print("4. Increase swap_space")
        
        # Check available memory
        free, total = torch.cuda.mem_get_info()
        print(f"\nGPU Memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")
```

### Issue 2: Slow Inference

```python
# Profile to find bottleneck
import time
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompts = ["Test"] * 100
params = SamplingParams(max_tokens=50)

# Measure components
start = time.time()
outputs = llm.generate(prompts, params)
total_time = time.time() - start

print(f"Total time: {total_time:.2f}s")
print(f"Throughput: {len(prompts)/total_time:.2f} req/s")

# Check if GPU is bottleneck
import torch
print(f"GPU Utilization: {torch.cuda.utilization()}%")
```

### Issue 3: Incorrect Output

```python
# Debug with logprobs
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

params = SamplingParams(
    temperature=0.0,  # Deterministic
    max_tokens=20,
    logprobs=5  # Inspect probabilities
)

outputs = llm.generate(["The capital of France is"], params)

print("Generated:", outputs[0].outputs[0].text)
print("\nToken probabilities:")
for token_id, logprobs in zip(
    outputs[0].outputs[0].token_ids,
    outputs[0].outputs[0].logprobs
):
    print(f"Token {token_id}: {logprobs}")
```

---

## Logging Best Practices

### Production Logging

```python
# production_logging.py
import logging
import sys

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vllm_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce vLLM verbosity in production
logging.getLogger("vllm").setLevel(logging.WARNING)

from vllm import LLM
llm = LLM(model="facebook/opt-125m")
```

### Development Logging

```python
# development_logging.py
import logging

# Verbose logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

from vllm import LLM
llm = LLM(model="facebook/opt-125m")
```

---

## Practical Exercises

### Exercise 1.8.1: Configure Logging
1. Set up DEBUG level logging
2. Run inference
3. Analyze log output

**Time:** 15 minutes

### Exercise 1.8.2: Profile GPU Usage
1. Use nvidia-smi during inference
2. Use PyTorch profiler
3. Identify bottlenecks

**Time:** 30 minutes

### Exercise 1.8.3: Debug OOM
1. Intentionally cause OOM
2. Use memory tracking
3. Fix by adjusting parameters

**Time:** 20 minutes

---

## Key Takeaways

1. **Logging** helps understand vLLM behavior
2. **Eager mode** simplifies debugging
3. **GPU profiling** identifies bottlenecks
4. **Memory tracking** prevents OOM
5. **Logprobs** debug generation issues
6. **Structured logging** for production

---

**Next:** 09_performance_metrics.md
**Time:** 1.5 hours
**Source:** `/home/user/vllm-learn/vllm/logger.py`
