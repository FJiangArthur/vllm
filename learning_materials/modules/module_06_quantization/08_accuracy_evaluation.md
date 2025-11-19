# Accuracy Evaluation for Quantized LLMs

## Table of Contents
1. [Introduction](#introduction)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Benchmark Datasets](#benchmark-datasets)
4. [Evaluation Framework](#evaluation-framework)
5. [Layer-wise Analysis](#layer-wise-analysis)
6. [Production Monitoring](#production-monitoring)

---

## Introduction

### Why Accuracy Evaluation Matters

Quantization inevitably introduces some accuracy degradation. Proper evaluation ensures:

1. **Acceptable Quality**: Quantized model meets requirements
2. **Method Comparison**: Choose best quantization technique
3. **Regression Detection**: Catch accuracy drops early
4. **Production Monitoring**: Track real-world performance

### Evaluation Dimensions

```
Accuracy Evaluation:
├── Intrinsic Metrics
│   ├── Perplexity
│   └── Token-level accuracy
├── Extrinsic Metrics
│   ├── Downstream task performance
│   └── Human evaluation
└── Operational Metrics
    ├── Latency/throughput
    └── Resource utilization
```

---

## Evaluation Metrics

### 1. Perplexity

**Definition**: Inverse probability of test set, measuring how well model predicts text.

```
PPL = exp(-1/N * Σ log P(x_i | x_<i))
```

Lower is better (perfectly predicts → PPL = 1).

**Implementation:**

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def compute_perplexity(model, tokenizer, dataset_name="wikitext", split="test"):
    """
    Compute perplexity on a dataset.

    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset_name: Dataset to use
        split: Dataset split

    Returns:
        Perplexity score
    """
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_column = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
        dataset = dataset.take(1000)
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for example in dataset:
            text = example[text_column]

            if not text.strip():
                continue

            # Tokenize
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = encodings.input_ids.to(model.device)

            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Accumulate
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()


# Example usage
def evaluate_quantized_perplexity():
    """Compare FP16 vs quantized perplexity."""
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # FP16 baseline
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ppl_fp16 = compute_perplexity(model_fp16, tokenizer)

    # Quantized model
    model_quant = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-GPTQ",
        device_map="auto"
    )
    ppl_quant = compute_perplexity(model_quant, tokenizer)

    # Results
    print(f"FP16 Perplexity:      {ppl_fp16:.4f}")
    print(f"Quantized Perplexity: {ppl_quant:.4f}")
    print(f"Degradation:          {((ppl_quant - ppl_fp16) / ppl_fp16 * 100):.2f}%")

    return ppl_fp16, ppl_quant
```

### 2. Downstream Task Evaluation

**Common Benchmarks:**

- **HellaSwag**: Commonsense reasoning
- **MMLU**: Multitask language understanding
- **TruthfulQA**: Factual accuracy
- **HumanEval**: Code generation
- **GSM8K**: Mathematical reasoning

**Implementation using lm-evaluation-harness:**

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

def evaluate_downstream_tasks(model_path, tasks=["hellaswag", "arc_easy", "truthfulqa"]):
    """
    Evaluate model on downstream tasks.

    Args:
        model_path: Path to model
        tasks: List of tasks to evaluate

    Returns:
        Results dictionary
    """
    # Create model wrapper
    lm = HFLM(pretrained=model_path, device="cuda")

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=8,
    )

    return results["results"]


# Compare quantization methods
def compare_quantization_accuracy():
    """Compare accuracy across quantization methods."""
    models = {
        "FP16": "meta-llama/Llama-2-7b-hf",
        "GPTQ-4bit": "TheBloke/Llama-2-7B-GPTQ",
        "AWQ-4bit": "TheBloke/Llama-2-7B-AWQ",
        "INT8": "meta-llama/Llama-2-7b-hf",  # With INT8 quantization
    }

    tasks = ["hellaswag", "arc_easy", "truthfulqa_mc"]

    results = {}
    for name, model_path in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_downstream_tasks(model_path, tasks)

    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Method':<15} {'HellaSwag':<12} {'ARC-Easy':<12} {'TruthfulQA':<12}")
    print("="*80)

    for name, result in results.items():
        hellaswag = result.get("hellaswag", {}).get("acc_norm", 0) * 100
        arc = result.get("arc_easy", {}).get("acc_norm", 0) * 100
        truthful = result.get("truthfulqa_mc", {}).get("mc2", 0) * 100

        print(f"{name:<15} {hellaswag:<12.2f} {arc:<12.2f} {truthful:<12.2f}")

    return results


# Example output:
# ================================================================================
# Method          HellaSwag    ARC-Easy     TruthfulQA
# ================================================================================
# FP16            75.82        74.32        38.45
# GPTQ-4bit       75.34        73.89        37.92
# AWQ-4bit        75.61        74.15        38.21
# INT8            75.79        74.28        38.39
```

### 3. Token-Level Accuracy

```python
def compute_token_accuracy(model_fp16, model_quant, tokenizer, test_data, top_k=1):
    """
    Compute token-level prediction accuracy.

    Args:
        model_fp16: FP16 baseline model
        model_quant: Quantized model
        tokenizer: Tokenizer
        test_data: Test dataset
        top_k: Consider top-k predictions

    Returns:
        Accuracy metrics
    """
    model_fp16.eval()
    model_quant.eval()

    total_tokens = 0
    exact_matches = 0
    top_k_matches = 0

    with torch.no_grad():
        for example in test_data:
            # Tokenize
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True)
            input_ids = inputs.input_ids.to(model_fp16.device)

            # Get predictions from both models
            outputs_fp16 = model_fp16(input_ids)
            outputs_quant = model_quant(input_ids)

            logits_fp16 = outputs_fp16.logits
            logits_quant = outputs_quant.logits

            # Compare predictions
            pred_fp16 = logits_fp16.argmax(dim=-1)
            pred_quant = logits_quant.argmax(dim=-1)

            # Exact match
            exact_matches += (pred_fp16 == pred_quant).sum().item()

            # Top-k match
            topk_fp16 = logits_fp16.topk(top_k, dim=-1).indices
            topk_quant = logits_quant.topk(top_k, dim=-1).indices

            for i in range(topk_quant.size(0)):
                for j in range(topk_quant.size(1)):
                    if pred_quant[i, j] in topk_fp16[i, j]:
                        top_k_matches += 1

            total_tokens += pred_fp16.numel()

    return {
        "exact_match_rate": exact_matches / total_tokens,
        "top_k_match_rate": top_k_matches / total_tokens,
        "total_tokens": total_tokens,
    }
```

### 4. Logit Similarity

```python
def compute_logit_similarity(model_fp16, model_quant, tokenizer, test_data):
    """
    Measure similarity between FP16 and quantized logits.

    Args:
        model_fp16: FP16 model
        model_quant: Quantized model
        tokenizer: Tokenizer
        test_data: Test data

    Returns:
        Similarity metrics
    """
    import torch.nn.functional as F

    cosine_sims = []
    kl_divs = []
    mses = []

    model_fp16.eval()
    model_quant.eval()

    with torch.no_grad():
        for example in test_data:
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True)
            input_ids = inputs.input_ids.to(model_fp16.device)

            # Get logits
            logits_fp16 = model_fp16(input_ids).logits
            logits_quant = model_quant(input_ids).logits

            # Flatten
            logits_fp16_flat = logits_fp16.reshape(-1, logits_fp16.size(-1))
            logits_quant_flat = logits_quant.reshape(-1, logits_quant.size(-1))

            # Cosine similarity
            cos_sim = F.cosine_similarity(logits_fp16_flat, logits_quant_flat, dim=1).mean()
            cosine_sims.append(cos_sim.item())

            # KL divergence
            log_probs_fp16 = F.log_softmax(logits_fp16_flat, dim=1)
            log_probs_quant = F.log_softmax(logits_quant_flat, dim=1)
            kl_div = F.kl_div(log_probs_quant, log_probs_fp16, reduction='batchmean', log_target=True)
            kl_divs.append(kl_div.item())

            # MSE
            mse = F.mse_loss(logits_fp16_flat, logits_quant_flat)
            mses.append(mse.item())

    return {
        "cosine_similarity": sum(cosine_sims) / len(cosine_sims),
        "kl_divergence": sum(kl_divs) / len(kl_divs),
        "mse": sum(mses) / len(mses),
    }
```

---

## Benchmark Datasets

### Recommended Datasets

```python
EVALUATION_DATASETS = {
    # Language modeling
    "wikitext": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "metric": "perplexity",
    },

    # Common sense reasoning
    "hellaswag": {
        "tasks": ["hellaswag"],
        "metric": "acc_norm",
    },

    # Knowledge
    "mmlu": {
        "tasks": ["mmlu"],
        "metric": "acc",
    },

    # Truthfulness
    "truthfulqa": {
        "tasks": ["truthfulqa_mc"],
        "metric": "mc2",
    },

    # Math
    "gsm8k": {
        "tasks": ["gsm8k"],
        "metric": "acc",
    },

    # Code
    "humaneval": {
        "tasks": ["humaneval"],
        "metric": "pass@1",
    },
}


def run_full_evaluation(model, tokenizer):
    """Run comprehensive evaluation suite."""
    results = {}

    # Perplexity
    print("Computing perplexity...")
    results["perplexity"] = compute_perplexity(model, tokenizer)

    # Downstream tasks
    print("Evaluating downstream tasks...")
    tasks = ["hellaswag", "arc_easy", "mmlu", "truthfulqa_mc"]
    downstream = evaluate_downstream_tasks(model, tasks)
    results["downstream"] = downstream

    return results
```

---

## Evaluation Framework

### Complete Evaluation Pipeline

```python
class QuantizationEvaluator:
    """
    Comprehensive quantization evaluation framework.
    """

    def __init__(
        self,
        model_fp16,
        model_quant,
        tokenizer,
        device="cuda",
    ):
        self.model_fp16 = model_fp16.to(device)
        self.model_quant = model_quant.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_all(self):
        """Run all evaluations."""
        results = {}

        # 1. Perplexity
        print("1. Computing perplexity...")
        results["perplexity"] = {
            "fp16": compute_perplexity(self.model_fp16, self.tokenizer),
            "quant": compute_perplexity(self.model_quant, self.tokenizer),
        }
        results["perplexity"]["degradation"] = (
            (results["perplexity"]["quant"] - results["perplexity"]["fp16"])
            / results["perplexity"]["fp16"] * 100
        )

        # 2. Downstream tasks
        print("2. Evaluating downstream tasks...")
        tasks = ["hellaswag", "arc_easy"]
        results["downstream"] = {
            "fp16": evaluate_downstream_tasks(self.model_fp16, tasks),
            "quant": evaluate_downstream_tasks(self.model_quant, tasks),
        }

        # 3. Logit similarity
        print("3. Computing logit similarity...")
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_data = test_data.select(range(100))  # Small subset
        results["logit_similarity"] = compute_logit_similarity(
            self.model_fp16,
            self.model_quant,
            self.tokenizer,
            test_data,
        )

        # 4. Generate report
        self.print_report(results)

        return results

    def print_report(self, results):
        """Print evaluation report."""
        print("\n" + "="*80)
        print("QUANTIZATION EVALUATION REPORT")
        print("="*80)

        # Perplexity
        print("\nPerplexity:")
        print(f"  FP16:        {results['perplexity']['fp16']:.4f}")
        print(f"  Quantized:   {results['perplexity']['quant']:.4f}")
        print(f"  Degradation: {results['perplexity']['degradation']:.2f}%")

        # Downstream
        print("\nDownstream Tasks:")
        for task in results["downstream"]["fp16"].keys():
            fp16_acc = results["downstream"]["fp16"][task].get("acc_norm", 0) * 100
            quant_acc = results["downstream"]["quant"][task].get("acc_norm", 0) * 100
            print(f"  {task}:")
            print(f"    FP16:      {fp16_acc:.2f}%")
            print(f"    Quantized: {quant_acc:.2f}%")
            print(f"    Delta:     {quant_acc - fp16_acc:.2f}%")

        # Logit similarity
        print("\nLogit Similarity:")
        print(f"  Cosine Sim: {results['logit_similarity']['cosine_similarity']:.4f}")
        print(f"  KL Div:     {results['logit_similarity']['kl_divergence']:.4f}")
        print(f"  MSE:        {results['logit_similarity']['mse']:.4f}")

        print("\n" + "="*80)


# Example usage
def evaluate_quantization_methods():
    """Compare multiple quantization methods."""
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load models
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    models_quant = {
        "GPTQ-4bit": "TheBloke/Llama-2-7B-GPTQ",
        "AWQ-4bit": "TheBloke/Llama-2-7B-AWQ",
    }

    results_all = {}

    for method, model_path in models_quant.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {method}")
        print(f"{'='*80}")

        model_quant = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )

        evaluator = QuantizationEvaluator(model_fp16, model_quant, tokenizer)
        results_all[method] = evaluator.evaluate_all()

    return results_all
```

---

## Layer-wise Analysis

### Identifying Sensitive Layers

```python
def analyze_layer_sensitivity(model_fp16, model_quant, test_data):
    """
    Identify which layers are most affected by quantization.

    Args:
        model_fp16: FP16 model
        model_quant: Quantized model
        test_data: Test dataset

    Returns:
        Per-layer sensitivity metrics
    """
    layer_errors = {}

    # Hook to capture layer outputs
    fp16_activations = {}
    quant_activations = {}

    def make_hook(storage, name):
        def hook(module, input, output):
            storage[name] = output.detach()
        return hook

    # Register hooks
    hooks_fp16 = []
    hooks_quant = []

    for name, module in model_fp16.named_modules():
        if "layer" in name and "mlp" in name:
            hooks_fp16.append(module.register_forward_hook(make_hook(fp16_activations, name)))

    for name, module in model_quant.named_modules():
        if "layer" in name and "mlp" in name:
            hooks_quant.append(module.register_forward_hook(make_hook(quant_activations, name)))

    # Run inference
    model_fp16.eval()
    model_quant.eval()

    with torch.no_grad():
        for example in test_data.select(range(10)):
            inputs = tokenizer(example["text"], return_tensors="pt")
            _ = model_fp16(**inputs)
            _ = model_quant(**inputs)

    # Compute layer-wise errors
    for name in fp16_activations.keys():
        fp16_act = fp16_activations[name]
        quant_act = quant_activations.get(name)

        if quant_act is not None:
            mse = F.mse_loss(fp16_act, quant_act).item()
            cosine = F.cosine_similarity(
                fp16_act.flatten(),
                quant_act.flatten(),
                dim=0
            ).item()

            layer_errors[name] = {
                "mse": mse,
                "cosine_similarity": cosine,
            }

    # Remove hooks
    for hook in hooks_fp16 + hooks_quant:
        hook.remove()

    # Print most sensitive layers
    print("\nMost Sensitive Layers (by MSE):")
    sorted_layers = sorted(layer_errors.items(), key=lambda x: x[1]["mse"], reverse=True)
    for name, metrics in sorted_layers[:10]:
        print(f"  {name}: MSE={metrics['mse']:.6f}, CosSim={metrics['cosine_similarity']:.4f}")

    return layer_errors
```

---

## Production Monitoring

### Real-time Quality Metrics

```python
class QuantizationMonitor:
    """
    Monitor quantization quality in production.
    """

    def __init__(self, model, reference_stats=None):
        self.model = model
        self.reference_stats = reference_stats or {}
        self.metrics = []

    def log_inference(self, input_ids, outputs, ground_truth=None):
        """Log metrics for one inference."""
        metrics = {}

        # Token-level entropy
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        metrics["entropy"] = entropy

        # Max probability (confidence)
        max_prob = probs.max(dim=-1)[0].mean().item()
        metrics["confidence"] = max_prob

        # Perplexity (if ground truth available)
        if ground_truth is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                ground_truth.view(-1)
            )
            metrics["perplexity"] = torch.exp(loss).item()

        self.metrics.append(metrics)

        # Check for anomalies
        self.check_anomalies(metrics)

        return metrics

    def check_anomalies(self, metrics):
        """Check if metrics are within expected ranges."""
        if self.reference_stats:
            # Check entropy
            ref_entropy = self.reference_stats.get("entropy_mean", 0)
            ref_entropy_std = self.reference_stats.get("entropy_std", float("inf"))

            if abs(metrics["entropy"] - ref_entropy) > 3 * ref_entropy_std:
                print(f"⚠ WARNING: Entropy anomaly detected: {metrics['entropy']:.4f}")

            # Check confidence
            if metrics["confidence"] < 0.1:
                print(f"⚠ WARNING: Low confidence: {metrics['confidence']:.4f}")

    def get_summary(self):
        """Get summary statistics."""
        if not self.metrics:
            return {}

        import numpy as np

        entropies = [m["entropy"] for m in self.metrics]
        confidences = [m["confidence"] for m in self.metrics]

        return {
            "entropy_mean": np.mean(entropies),
            "entropy_std": np.std(entropies),
            "confidence_mean": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "num_samples": len(self.metrics),
        }
```

---

## Summary

### Key Takeaways

1. **Multiple Metrics**: Use perplexity, downstream tasks, and logit similarity
2. **Comprehensive Evaluation**: Test on diverse benchmarks
3. **Layer Analysis**: Identify sensitive layers
4. **Production Monitoring**: Track metrics in deployment
5. **Regression Testing**: Continuous evaluation

### Evaluation Checklist

```
✓ Perplexity on WikiText-2
✓ HellaSwag (common sense)
✓ MMLU (knowledge)
✓ TruthfulQA (factuality)
✓ Task-specific benchmarks
✓ Logit similarity analysis
✓ Layer-wise sensitivity
✓ Token-level accuracy
✓ Production metrics monitoring
```

---

## References

1. lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
2. "Holistic Evaluation of Language Models" (HELM)
3. MMLU Benchmark Paper
4. vLLM Documentation: Quantization Quality

---

**Module 6.8 Complete** • Next: [Performance Tuning](09_performance_tuning.md)
