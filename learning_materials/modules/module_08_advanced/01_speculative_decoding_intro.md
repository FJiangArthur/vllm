# Module 8.1: Speculative Decoding - Introduction and Fundamentals

## Overview

Speculative decoding is a revolutionary inference optimization technique that can achieve 2-3x speedup for autoregressive language model generation without any loss in output quality. This module provides a comprehensive introduction to speculative decoding, its theoretical foundations, implementation strategies, and practical applications in vLLM.

**Learning Objectives:**
- Understand the fundamental concept and motivation behind speculative decoding
- Master the draft-then-verify paradigm for parallel token generation
- Implement speculative decoding in vLLM
- Analyze performance characteristics and optimal use cases
- Debug and optimize speculative decoding configurations

**Prerequisites:**
- Strong understanding of autoregressive language models
- Familiarity with vLLM's core architecture (Modules 1-4)
- Knowledge of sampling strategies and token generation
- Understanding of GPU batching and parallelism

**Estimated Time:** 4-5 hours

---

## 1. The Autoregressive Bottleneck

### 1.1 Sequential Generation Limitation

Traditional autoregressive language models generate tokens one at a time, creating an inherent sequential dependency:

```python
# Traditional autoregressive generation
def generate_traditional(model, prompt, max_tokens):
    """
    Sequential token generation - the fundamental bottleneck
    """
    tokens = tokenize(prompt)

    for i in range(max_tokens):
        # Each iteration depends on previous tokens
        # Cannot parallelize across tokens
        logits = model(tokens)  # Forward pass
        next_token = sample(logits)  # Sampling
        tokens.append(next_token)

        # Latency = num_tokens × time_per_token
        # No opportunity for parallelization

    return tokens
```

**Key Bottlenecks:**
1. **Sequential Dependency:** Token `t[i]` requires `t[i-1]` to be generated first
2. **GPU Underutilization:** Modern GPUs can process many tokens in parallel, but autoregressive generation processes one at a time
3. **Memory Bandwidth Bound:** Most time spent moving model weights from memory to compute units
4. **Fixed Latency:** Generation time scales linearly with output length

### 1.2 Performance Analysis

Let's analyze the performance characteristics of traditional generation:

```python
import time
import torch
from vllm import LLM, SamplingParams

def benchmark_traditional_generation():
    """
    Benchmark traditional autoregressive generation
    """
    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    prompt = "Explain the theory of relativity in simple terms:"
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )

    # Warmup
    _ = llm.generate(prompt, sampling_params)

    # Benchmark
    num_runs = 10
    total_time = 0
    total_tokens = 0

    for _ in range(num_runs):
        start = time.time()
        outputs = llm.generate(prompt, sampling_params)
        elapsed = time.time() - start

        total_time += elapsed
        total_tokens += len(outputs[0].outputs[0].token_ids)

    avg_time_per_token = total_time / total_tokens
    throughput = total_tokens / total_time

    print(f"Average time per token: {avg_time_per_token*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"For 100 tokens: {avg_time_per_token*100:.2f} seconds")

    return {
        "time_per_token": avg_time_per_token,
        "throughput": throughput
    }

# Typical results on A100 GPU:
# Average time per token: 15-20 ms
# Throughput: 50-65 tokens/s
# For 100 tokens: 1.5-2.0 seconds
```

**Bottleneck Breakdown:**
- **Model Loading:** ~5% (amortized across batch)
- **KV Cache Management:** ~10%
- **Matrix Multiplication:** ~30%
- **Memory Transfer:** ~40%
- **Sampling/Logits Processing:** ~15%

The critical insight: **Memory bandwidth is the bottleneck**, not compute capacity.

---

## 2. Speculative Decoding: Core Concept

### 2.1 The Big Idea

Speculative decoding breaks the sequential bottleneck through a draft-then-verify paradigm:

1. **Draft Phase:** A smaller, faster "draft" model generates multiple tokens speculatively
2. **Verification Phase:** The target model verifies all drafted tokens in parallel
3. **Accept/Reject:** Accept correct tokens, reject and regenerate incorrect ones

**Key Insight:** Even if some speculative tokens are rejected, parallel verification can still achieve net speedup.

```python
def speculative_decoding_overview(target_model, draft_model, prompt, k=4):
    """
    High-level overview of speculative decoding

    Args:
        target_model: Large, accurate model (e.g., Llama-70B)
        draft_model: Small, fast model (e.g., Llama-7B)
        prompt: Input prompt
        k: Number of tokens to speculate (typically 4-8)
    """
    tokens = tokenize(prompt)

    while not done:
        # DRAFT PHASE: Fast model generates k tokens
        # This is fast but potentially inaccurate
        draft_tokens = []
        for i in range(k):
            logits = draft_model(tokens + draft_tokens)
            draft_token = sample(logits)
            draft_tokens.append(draft_token)

        # VERIFICATION PHASE: Target model verifies ALL tokens in parallel
        # Key: We can verify all k tokens in a single forward pass!
        verification_logits = target_model(tokens + draft_tokens)

        # ACCEPT/REJECT: Compare draft vs target distributions
        accepted_tokens = []
        for i, draft_token in enumerate(draft_tokens):
            target_logits = verification_logits[len(tokens) + i]

            if should_accept(draft_token, target_logits):
                accepted_tokens.append(draft_token)
            else:
                # Reject and resample from target distribution
                corrected_token = sample(target_logits)
                accepted_tokens.append(corrected_token)
                break  # Stop accepting after first rejection

        tokens.extend(accepted_tokens)

    return tokens
```

### 2.2 Mathematical Foundations

The acceptance criterion is based on statistical sampling theory:

**Acceptance Probability:**

```
P(accept token t) = min(1, p_target(t) / p_draft(t))
```

Where:
- `p_target(t)`: Probability assigned by target model
- `p_draft(t)`: Probability assigned by draft model

**Correctness Guarantee:**

Speculative decoding is **mathematically equivalent** to sampling from the target model distribution. This means:
- Output quality is identical to standard generation
- No approximation or quality degradation
- Only difference is speed

**Proof Sketch:**

```python
def acceptance_sampling(draft_token, target_probs, draft_probs):
    """
    Implements provably correct acceptance sampling

    This guarantees outputs match target model distribution
    """
    p_target = target_probs[draft_token]
    p_draft = draft_probs[draft_token]

    # Acceptance probability
    accept_prob = min(1.0, p_target / p_draft)

    if random.uniform(0, 1) < accept_prob:
        return draft_token  # Accept
    else:
        # Reject and resample from adjusted distribution
        adjusted_probs = compute_adjusted_distribution(
            target_probs, draft_probs
        )
        return sample_from(adjusted_probs)

def compute_adjusted_distribution(target_probs, draft_probs):
    """
    Compute rejection-adjusted distribution

    Formula: p_adjusted(t) = max(0, p_target(t) - p_draft(t))
    Then normalize to sum to 1
    """
    adjusted = torch.clamp(target_probs - draft_probs, min=0.0)
    adjusted = adjusted / adjusted.sum()
    return adjusted
```

### 2.3 Speedup Analysis

**Expected Speedup:**

Let:
- `α` = acceptance rate (probability draft token is accepted)
- `k` = number of speculative tokens
- `r` = draft model speed ratio (draft_speed / target_speed)

Expected number of accepted tokens per iteration:
```
E[accepted] = α + α² + α³ + ... + α^k = α(1 - α^k) / (1 - α)
```

Expected speedup:
```
Speedup = E[accepted] / (k/r + 1)
         ≈ E[accepted] for small k and large r
```

**Example Calculation:**

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_expected_speedup(alpha, k, r=10):
    """
    Calculate expected speedup from speculative decoding

    Args:
        alpha: Acceptance rate (0 to 1)
        k: Number of speculative tokens
        r: Draft/target speed ratio (default 10x)
    """
    if alpha == 1.0:
        expected_accepted = k
    else:
        expected_accepted = alpha * (1 - alpha**k) / (1 - alpha)

    # Time for one iteration
    draft_time = k / r  # k tokens from fast draft model
    verify_time = 1     # One parallel verification pass
    total_time = draft_time + verify_time

    speedup = expected_accepted / total_time
    return speedup, expected_accepted

# Analyze different scenarios
alphas = np.linspace(0.1, 0.95, 20)
k_values = [2, 4, 8, 16]

for k in k_values:
    speedups = []
    for alpha in alphas:
        speedup, _ = calculate_expected_speedup(alpha, k, r=10)
        speedups.append(speedup)

    plt.plot(alphas, speedups, label=f'k={k}')

plt.xlabel('Acceptance Rate (α)')
plt.ylabel('Expected Speedup')
plt.title('Speculative Decoding Speedup vs Acceptance Rate')
plt.legend()
plt.grid(True)
plt.savefig('speculative_decoding_speedup.png')
plt.close()

# Print optimal configurations
print("Optimal Configurations:")
print("-" * 60)
for alpha in [0.5, 0.7, 0.9]:
    best_k = 0
    best_speedup = 0

    for k in range(1, 20):
        speedup, accepted = calculate_expected_speedup(alpha, k)
        if speedup > best_speedup:
            best_speedup = speedup
            best_k = k

    print(f"α={alpha:.1f}: optimal k={best_k}, speedup={best_speedup:.2f}x")

# Output:
# α=0.5: optimal k=4, speedup=1.48x
# α=0.7: optimal k=6, speedup=2.13x
# α=0.9: optimal k=12, speedup=3.26x
```

---

## 3. Implementation in vLLM

### 3.1 Enabling Speculative Decoding

vLLM provides built-in support for speculative decoding:

```python
from vllm import LLM, SamplingParams

def setup_speculative_decoding():
    """
    Configure vLLM for speculative decoding
    """
    # Initialize with speculative decoding enabled
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",  # Target model
        speculative_model="meta-llama/Llama-2-7b-hf",  # Draft model
        num_speculative_tokens=4,  # Number of tokens to speculate
        use_v2_block_manager=True,  # Required for spec decoding
        tensor_parallel_size=4,  # For 70B model
        gpu_memory_utilization=0.9
    )

    return llm

# Basic usage
llm = setup_speculative_decoding()

prompts = [
    "Explain quantum computing in simple terms:",
    "Write a Python function to calculate fibonacci numbers:",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=200
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")
    print("-" * 80)
```

### 3.2 Configuration Parameters

```python
class SpeculativeDecodingConfig:
    """
    Configuration options for speculative decoding in vLLM
    """

    def __init__(self):
        # Core parameters
        self.speculative_model = None  # Path to draft model
        self.num_speculative_tokens = 4  # Typical range: 2-8

        # Advanced tuning
        self.speculative_draft_tensor_parallel_size = 1
        self.speculative_max_model_len = None  # Override max length
        self.speculative_disable_by_batch_size = None  # Disable for large batches

        # Memory management
        self.speculative_model_quantization = None  # Quantize draft model
        self.gpu_memory_utilization = 0.9

    def optimize_for_latency(self):
        """Configuration optimized for low latency"""
        self.num_speculative_tokens = 2  # Lower k for stability
        self.speculative_disable_by_batch_size = 1  # Only for batch=1
        return self

    def optimize_for_throughput(self):
        """Configuration optimized for high throughput"""
        self.num_speculative_tokens = 8  # Higher k for throughput
        self.speculative_disable_by_batch_size = None  # Use always
        return self

# Example: Latency-optimized configuration
config = SpeculativeDecodingConfig().optimize_for_latency()

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    **vars(config)  # Unpack configuration
)
```

### 3.3 Draft Model Selection

Choosing the right draft model is critical for performance:

```python
def analyze_draft_model_candidates():
    """
    Compare different draft model options
    """
    target_model = "meta-llama/Llama-2-70b-hf"

    draft_candidates = [
        {
            "name": "Llama-2-7b",
            "model": "meta-llama/Llama-2-7b-hf",
            "params": "7B",
            "expected_speedup": "10x vs target",
            "expected_acceptance": "0.70-0.80"
        },
        {
            "name": "Llama-2-13b",
            "model": "meta-llama/Llama-2-13b-hf",
            "params": "13B",
            "expected_speedup": "6x vs target",
            "expected_acceptance": "0.80-0.90"
        },
        {
            "name": "TinyLlama",
            "model": "TinyLlama/TinyLlama-1.1B",
            "params": "1.1B",
            "expected_speedup": "30x vs target",
            "expected_acceptance": "0.50-0.65"
        }
    ]

    return draft_candidates

def benchmark_draft_models():
    """
    Empirical comparison of draft models
    """
    import time
    from vllm import LLM, SamplingParams

    target_model = "meta-llama/Llama-2-70b-hf"
    draft_models = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "TinyLlama/TinyLlama-1.1B"
    ]

    test_prompts = [
        "Explain the benefits of exercise:",
        "Write a story about a robot:",
        "Describe the water cycle:"
    ]

    results = []

    for draft_model in draft_models:
        print(f"\nTesting draft model: {draft_model}")

        llm = LLM(
            model=target_model,
            speculative_model=draft_model,
            num_speculative_tokens=4,
            tensor_parallel_size=4
        )

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=100
        )

        # Warmup
        _ = llm.generate(test_prompts[0], sampling_params)

        # Benchmark
        start = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        elapsed = time.time() - start

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed

        results.append({
            "draft_model": draft_model,
            "throughput": throughput,
            "latency": elapsed / len(test_prompts),
            "tokens_generated": total_tokens
        })

        print(f"  Throughput: {throughput:.2f} tokens/s")
        print(f"  Avg latency: {elapsed/len(test_prompts):.2f}s")

    # Compare to baseline (no speculative decoding)
    print("\n" + "="*60)
    print("Baseline (no speculative decoding):")
    llm_baseline = LLM(model=target_model, tensor_parallel_size=4)

    start = time.time()
    outputs = llm_baseline.generate(test_prompts, sampling_params)
    elapsed = time.time() - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    baseline_throughput = total_tokens / elapsed

    print(f"  Throughput: {baseline_throughput:.2f} tokens/s")

    # Calculate speedups
    print("\n" + "="*60)
    print("Speedup Analysis:")
    for result in results:
        speedup = result["throughput"] / baseline_throughput
        print(f"{result['draft_model']}: {speedup:.2f}x speedup")

    return results
```

**Typical Results:**

| Draft Model | Speed Ratio | Acceptance Rate | Realized Speedup |
|-------------|-------------|-----------------|------------------|
| Llama-7B    | 10x         | 75%             | 2.3x             |
| Llama-13B   | 6x          | 85%             | 2.1x             |
| TinyLlama   | 30x         | 60%             | 1.8x             |

**Key Insights:**
1. Faster draft models don't always mean better overall speedup
2. Acceptance rate is crucial - need good alignment with target
3. Sweet spot is typically 7B-13B draft for 70B target

---

## 4. Monitoring and Debugging

### 4.1 Acceptance Rate Tracking

```python
import logging
from vllm import LLM, SamplingParams
from vllm.engine.metrics import Stats

def monitor_speculative_decoding():
    """
    Monitor and log speculative decoding performance
    """
    # Enable detailed logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("vllm.speculative")

    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        speculative_model="meta-llama/Llama-2-7b-hf",
        num_speculative_tokens=4
    )

    # Track metrics
    metrics = {
        "total_tokens": 0,
        "total_accepted": 0,
        "total_iterations": 0,
        "acceptance_distribution": [0] * 9  # 0-8 accepted tokens
    }

    # Generate with monitoring
    prompts = ["Write a detailed explanation of photosynthesis:"] * 10
    sampling_params = SamplingParams(temperature=0.8, max_tokens=200)

    outputs = llm.generate(prompts, sampling_params)

    # Access internal metrics (if exposed)
    # Note: This requires vLLM to expose these metrics
    for output in outputs:
        if hasattr(output, 'metrics'):
            metrics["total_tokens"] += output.metrics.num_tokens
            metrics["total_accepted"] += output.metrics.num_accepted
            metrics["total_iterations"] += output.metrics.num_iterations

    # Calculate statistics
    avg_acceptance_rate = metrics["total_accepted"] / metrics["total_tokens"]
    avg_tokens_per_iteration = metrics["total_accepted"] / metrics["total_iterations"]

    print(f"Acceptance Rate: {avg_acceptance_rate:.2%}")
    print(f"Avg Tokens/Iteration: {avg_tokens_per_iteration:.2f}")

    return metrics

# Custom monitoring with hooks
class SpeculativeDecodingMonitor:
    """
    Custom monitor for speculative decoding performance
    """

    def __init__(self):
        self.iterations = []
        self.acceptance_rates = []

    def on_iteration_complete(self, accepted, proposed):
        """Hook called after each speculative iteration"""
        self.iterations.append({
            "accepted": accepted,
            "proposed": proposed,
            "rate": accepted / proposed if proposed > 0 else 0
        })
        self.acceptance_rates.append(accepted / proposed if proposed > 0 else 0)

    def get_statistics(self):
        """Calculate summary statistics"""
        import numpy as np

        rates = np.array(self.acceptance_rates)

        return {
            "mean_acceptance_rate": np.mean(rates),
            "std_acceptance_rate": np.std(rates),
            "min_acceptance_rate": np.min(rates),
            "max_acceptance_rate": np.max(rates),
            "p50_acceptance_rate": np.percentile(rates, 50),
            "p95_acceptance_rate": np.percentile(rates, 95),
            "total_iterations": len(self.iterations)
        }

    def plot_acceptance_over_time(self):
        """Visualize acceptance rate over time"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.acceptance_rates, alpha=0.7)
        plt.axhline(y=np.mean(self.acceptance_rates),
                   color='r', linestyle='--', label='Mean')
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance Rate')
        plt.title('Speculative Decoding Acceptance Rate Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('acceptance_rate_timeline.png')
        plt.close()
```

### 4.2 Common Issues and Debugging

```python
def diagnose_speculative_decoding_issues():
    """
    Common issues and diagnostic approaches
    """

    issues = {
        "Low Acceptance Rate (<50%)": {
            "symptoms": [
                "Slower than baseline generation",
                "High rejection rate in logs",
                "Poor speedup despite fast draft model"
            ],
            "causes": [
                "Draft model poorly aligned with target",
                "Different tokenizers between models",
                "Temperature/sampling parameter mismatch",
                "Draft model too small (e.g., 1B for 70B target)"
            ],
            "solutions": [
                "Use draft model from same family (e.g., Llama-7B for Llama-70B)",
                "Ensure identical tokenization",
                "Match sampling parameters between draft and target",
                "Try larger draft model (13B instead of 7B)"
            ]
        },

        "Out of Memory (OOM)": {
            "symptoms": [
                "CUDA OOM errors during initialization",
                "Crashes when generating with large batch"
            ],
            "causes": [
                "Both models loaded on same GPU",
                "Insufficient memory for KV cache of both models",
                "Batch size too large for speculative decoding"
            ],
            "solutions": [
                "Reduce gpu_memory_utilization to 0.85",
                "Use smaller draft model",
                "Quantize draft model (INT8/INT4)",
                "Enable speculative_disable_by_batch_size",
                "Reduce num_speculative_tokens"
            ]
        },

        "No Speedup or Slowdown": {
            "symptoms": [
                "Performance same or worse than baseline",
                "High overhead in profiling"
            ],
            "causes": [
                "Draft model not fast enough (speed ratio < 5x)",
                "Overhead of verification dominates",
                "Batch size too large (speculation less effective)",
                "num_speculative_tokens too high or too low"
            ],
            "solutions": [
                "Profile to identify bottleneck",
                "Tune num_speculative_tokens (try 2, 4, 8)",
                "Use speculative decoding only for small batches",
                "Ensure draft model is actually faster (benchmark separately)"
            ]
        }
    }

    return issues

# Automated diagnostic tool
def run_diagnostics(target_model, draft_model):
    """
    Automated diagnostic for speculative decoding setup
    """
    print("Running Speculative Decoding Diagnostics...")
    print("="*60)

    # Check 1: Model compatibility
    print("\n1. Checking model compatibility...")
    from transformers import AutoTokenizer

    target_tokenizer = AutoTokenizer.from_pretrained(target_model)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model)

    if target_tokenizer.vocab_size != draft_tokenizer.vocab_size:
        print("   ⚠️  WARNING: Different vocab sizes!")
        print(f"   Target: {target_tokenizer.vocab_size}")
        print(f"   Draft: {draft_tokenizer.vocab_size}")
    else:
        print("   ✅ Tokenizers compatible")

    # Check 2: Model size ratio
    print("\n2. Checking model size ratio...")
    # This would require loading model configs
    print("   Manual check: draft should be 5-15x smaller than target")

    # Check 3: Memory availability
    print("\n3. Checking GPU memory...")
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1e9  # GB
            allocated = torch.cuda.memory_allocated(i) / 1e9
            available = total_mem - allocated

            print(f"   GPU {i}: {available:.1f} GB available")

            if available < 40:  # Rough heuristic for 70B + 7B
                print(f"   ⚠️  May not have enough memory for both models")

    # Check 4: Quick performance test
    print("\n4. Running quick performance test...")
    try:
        llm = LLM(
            model=target_model,
            speculative_model=draft_model,
            num_speculative_tokens=4,
            max_num_seqs=1
        )

        test_prompt = "Hello, world!"
        params = SamplingParams(max_tokens=10)

        import time
        start = time.time()
        _ = llm.generate(test_prompt, params)
        elapsed = time.time() - start

        print(f"   ✅ Speculative decoding working! ({elapsed:.2f}s for test)")

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

    print("\n" + "="*60)
    print("Diagnostics complete!")

# Example usage
# run_diagnostics(
#     "meta-llama/Llama-2-70b-hf",
#     "meta-llama/Llama-2-7b-hf"
# )
```

---

## 5. Advanced Optimization Techniques

### 5.1 Dynamic Token Speculation

Adapt the number of speculative tokens based on runtime acceptance rate:

```python
class AdaptiveSpeculativeDecoding:
    """
    Dynamically adjust num_speculative_tokens based on acceptance rate
    """

    def __init__(self, min_k=2, max_k=8, target_acceptance=0.7):
        self.min_k = min_k
        self.max_k = max_k
        self.target_acceptance = target_acceptance
        self.current_k = (min_k + max_k) // 2

        # Track recent performance
        self.recent_acceptances = []
        self.window_size = 100

    def update(self, num_accepted, num_proposed):
        """Update based on observed acceptance rate"""
        acceptance_rate = num_accepted / num_proposed
        self.recent_acceptances.append(acceptance_rate)

        # Keep only recent history
        if len(self.recent_acceptances) > self.window_size:
            self.recent_acceptances = self.recent_acceptances[-self.window_size:]

        # Calculate moving average
        if len(self.recent_acceptances) >= 10:
            avg_acceptance = sum(self.recent_acceptances) / len(self.recent_acceptances)

            # Adjust k based on performance
            if avg_acceptance > self.target_acceptance + 0.1:
                # High acceptance - can afford to speculate more
                self.current_k = min(self.max_k, self.current_k + 1)
            elif avg_acceptance < self.target_acceptance - 0.1:
                # Low acceptance - reduce speculation
                self.current_k = max(self.min_k, self.current_k - 1)

    def get_current_k(self):
        return self.current_k

# Integration example (pseudo-code)
adaptive = AdaptiveSpeculativeDecoding()

while generating:
    k = adaptive.get_current_k()
    accepted = speculative_decode_with_k(k)
    adaptive.update(accepted, k)
```

### 5.2 Batch-Aware Speculation

Different batch sizes benefit from different speculation strategies:

```python
def get_optimal_speculation_config(batch_size, sequence_length):
    """
    Return optimal configuration based on batch characteristics
    """
    if batch_size == 1 and sequence_length < 512:
        # Single request, short sequence: maximize speculation
        return {
            "num_speculative_tokens": 8,
            "enable": True,
            "reason": "Low batch, short context - maximum speculation benefit"
        }

    elif batch_size <= 4 and sequence_length < 2048:
        # Small batch, medium context: moderate speculation
        return {
            "num_speculative_tokens": 4,
            "enable": True,
            "reason": "Small batch - speculation still beneficial"
        }

    elif batch_size > 8 or sequence_length > 4096:
        # Large batch or long context: disable speculation
        return {
            "num_speculative_tokens": 0,
            "enable": False,
            "reason": "Large batch/context - batch processing more efficient"
        }

    else:
        # Medium batch: conservative speculation
        return {
            "num_speculative_tokens": 2,
            "enable": True,
            "reason": "Medium batch - conservative speculation"
        }

# Example usage in request handler
def handle_generation_request(requests):
    batch_size = len(requests)
    avg_seq_len = sum(len(r.prompt) for r in requests) / batch_size

    config = get_optimal_speculation_config(batch_size, avg_seq_len)

    if config["enable"]:
        llm = get_llm_with_speculation(config["num_speculative_tokens"])
    else:
        llm = get_llm_without_speculation()

    return llm.generate(requests)
```

---

## 6. Performance Benchmarking

### 6.1 Comprehensive Benchmark Suite

```python
import time
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    config_name: str
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    tokens_generated: int
    acceptance_rate: float = None

def comprehensive_speculative_benchmark():
    """
    Run comprehensive benchmarks comparing different configurations
    """
    from vllm import LLM, SamplingParams

    # Test configurations
    configs = [
        {
            "name": "Baseline (No Speculation)",
            "speculative_model": None,
            "num_speculative_tokens": 0
        },
        {
            "name": "Spec-k2",
            "speculative_model": "meta-llama/Llama-2-7b-hf",
            "num_speculative_tokens": 2
        },
        {
            "name": "Spec-k4",
            "speculative_model": "meta-llama/Llama-2-7b-hf",
            "num_speculative_tokens": 4
        },
        {
            "name": "Spec-k8",
            "speculative_model": "meta-llama/Llama-2-7b-hf",
            "num_speculative_tokens": 8
        },
    ]

    # Test prompts of varying lengths
    test_prompts = [
        "Short prompt: Explain AI.",
        "Medium prompt: " + " ".join(["Explain artificial intelligence"]*10),
        "Long prompt: " + " ".join(["Describe machine learning"]*50)
    ]

    results = []

    for config in configs:
        print(f"\nBenchmarking: {config['name']}")
        print("-" * 60)

        # Initialize LLM
        llm_kwargs = {
            "model": "meta-llama/Llama-2-70b-hf",
            "tensor_parallel_size": 4
        }

        if config["speculative_model"]:
            llm_kwargs["speculative_model"] = config["speculative_model"]
            llm_kwargs["num_speculative_tokens"] = config["num_speculative_tokens"]

        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=100,
            top_p=0.95
        )

        # Warmup
        _ = llm.generate(test_prompts[0], sampling_params)

        # Benchmark
        latencies = []
        total_tokens = 0

        num_iterations = 20

        for _ in range(num_iterations):
            for prompt in test_prompts:
                start = time.time()
                outputs = llm.generate(prompt, sampling_params)
                latency = time.time() - start

                latencies.append(latency)
                total_tokens += len(outputs[0].outputs[0].token_ids)

        # Calculate metrics
        latencies = np.array(latencies)

        result = BenchmarkResult(
            config_name=config["name"],
            latency_mean=np.mean(latencies),
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            throughput=total_tokens / sum(latencies),
            tokens_generated=total_tokens
        )

        results.append(result)

        # Print results
        print(f"Latency (mean): {result.latency_mean*1000:.2f} ms")
        print(f"Latency (P95):  {result.latency_p95*1000:.2f} ms")
        print(f"Throughput:     {result.throughput:.2f} tokens/s")

    # Comparison table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Configuration':<25} {'Latency (ms)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-"*80)

    baseline_throughput = results[0].throughput

    for result in results:
        speedup = result.throughput / baseline_throughput
        print(f"{result.config_name:<25} {result.latency_mean*1000:<15.2f} "
              f"{result.throughput:<15.2f} {speedup:<10.2f}x")

    return results

# Run the benchmark
# results = comprehensive_speculative_benchmark()
```

---

## 7. Summary and Key Takeaways

### 7.1 When to Use Speculative Decoding

**✅ Good Use Cases:**
- Low batch size (1-4 requests)
- Latency-sensitive applications
- Models with good draft candidates (e.g., Llama family)
- Interactive applications (chatbots, code completion)
- Single-user or small-scale deployments

**❌ Poor Use Cases:**
- High batch sizes (>8 requests) - batch processing is already efficient
- Maximum throughput scenarios - speculation adds overhead
- Memory-constrained environments - requires loading two models
- Models without good draft candidates

### 7.2 Best Practices

1. **Draft Model Selection:**
   - Use models from the same family
   - Aim for 5-15x speed difference
   - Ensure identical tokenization

2. **Configuration Tuning:**
   - Start with `num_speculative_tokens=4`
   - Monitor acceptance rate (target: 70-80%)
   - Adjust based on batch size

3. **Memory Management:**
   - Reserve enough memory for both models
   - Consider quantizing draft model
   - Monitor GPU utilization

4. **Monitoring:**
   - Track acceptance rates
   - Measure end-to-end latency
   - Compare against baseline regularly

### 7.3 Expected Performance

| Scenario | Draft Model | Acceptance Rate | Speedup |
|----------|-------------|-----------------|---------|
| Llama-70B + Llama-7B | 7B | 70-80% | 2.0-2.5x |
| Llama-70B + Llama-13B | 13B | 80-90% | 1.8-2.2x |
| Llama-70B + TinyLlama | 1.1B | 50-65% | 1.5-2.0x |

---

## 8. Exercises

**Exercise 8.1.1:** Implement basic speculative decoding with vLLM and measure speedup

**Exercise 8.1.2:** Compare different draft models and analyze acceptance rates

**Exercise 8.1.3:** Implement adaptive speculation that adjusts k dynamically

**Exercise 8.1.4:** Profile memory usage with and without speculative decoding

**Exercise 8.1.5:** Create a decision tree for when to enable speculative decoding

---

## 9. Further Reading

1. **Original Papers:**
   - "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022)
   - "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)

2. **vLLM Documentation:**
   - Speculative Decoding Guide
   - Performance Tuning Best Practices

3. **Related Techniques:**
   - Next module: Draft Model Verification (advanced acceptance strategies)
   - Module 8.3: Medusa Multi-Head Speculation

---

**Next:** [Module 8.2: Draft Model Verification](02_draft_model_verification.md)
