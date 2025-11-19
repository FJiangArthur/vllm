# Project 7.5: End-to-End Quantization Pipeline

## Project Overview

**Duration:** 3-4 hours
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)
**Prerequisites:** Module 6 (Quantization & Optimization)

### Objective

Build an automated pipeline that quantizes models using multiple techniques (GPTQ, AWQ, INT8), evaluates accuracy on benchmark datasets, performs performance benchmarking, and generates comprehensive comparison reports.

### Why This Matters

Quantization is critical for cost-effective deployment:
- **Reduce memory**: 4x smaller models (FP16→INT4)
- **Increase throughput**: Faster inference with quantized kernels
- **Lower costs**: Serve more users per GPU

But quantization introduces accuracy/performance trade-offs that must be evaluated systematically.

---

## Pipeline Architecture

```
┌──────────────┐
│  Base Model  │
│  (FP16/BF16) │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Quantization Stage                    │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐│
│  │  GPTQ    │ │   AWQ    │ │  INT8   ││
│  │INT4/INT8 │ │   INT4   │ │ Smooth  ││
│  └────┬─────┘ └────┬─────┘ └────┬────┘│
└───────┼────────────┼───────────  ─┼────┘
        │            │              │
        └────────────┼──────────────┘
                     │
       ┌─────────────┴──────────────┐
       │                            │
  ┌────▼─────────┐         ┌────────▼────────┐
  │  Accuracy    │         │  Performance    │
  │  Evaluation  │         │  Benchmarking   │
  │              │         │                 │
  │ - Perplexity │         │ - Latency       │
  │ - MMLU       │         │ - Throughput    │
  │ - HellaSwag  │         │ - Memory        │
  └──────┬───────┘         └────────┬────────┘
         │                          │
         └────────┬─────────────────┘
                  │
         ┌────────▼─────────┐
         │ Report Generator │
         │  - Comparisons   │
         │  - Charts        │
         │  - Recommendations│
         └──────────────────┘
```

---

## Requirements

### FR1: Quantization Methods
- **GPTQ**: 4-bit and 8-bit variants
- **AWQ**: 4-bit with activation-aware quantization
- **INT8**: Smoothquant or similar
- Support for different group sizes (32, 64, 128)

### FR2: Accuracy Evaluation
- **Perplexity** on WikiText-2, C4
- **Question Answering** (MMLU, TruthfulQA)
- **Common Sense** (HellaSwag, PIQA)
- **Accuracy degradation** vs baseline

### FR3: Performance Benchmarking
- **Latency**: P50, P95, P99 (various batch sizes)
- **Throughput**: Tokens/second
- **Memory**: GPU memory footprint
- **Speedup**: vs FP16 baseline

### FR4: Reporting
- **Comparison tables**: All methods side-by-side
- **Visualization**: Charts for accuracy/speed trade-offs
- **Recommendations**: Best method for use case
- **Export formats**: Markdown, HTML, PDF

---

## Implementation Guide

### Step 1: Project Structure (15 minutes)

```bash
quantization_pipeline/
├── src/
│   ├── quantizers/
│   │   ├── gptq_quantizer.py
│   │   ├── awq_quantizer.py
│   │   └── int8_quantizer.py
│   ├── evaluators/
│   │   ├── perplexity.py
│   │   ├── qa_eval.py
│   │   └── benchmark.py
│   ├── reporters/
│   │   └── report_generator.py
│   └── pipeline.py
├── configs/
│   └── pipeline_config.yaml
├── tests/
├── requirements.txt
└── README.md
```

### Step 2: Quantization Implementations (60 minutes)

```python
# src/quantizers/gptq_quantizer.py

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class GPTQQuantizer:
    """GPTQ quantization implementation."""

    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
    ):
        self.model_name = model_name
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act

    def quantize(
        self,
        calibration_dataset: list,
        output_dir: str,
    ) -> str:
        """
        Quantize model using GPTQ.

        Args:
            calibration_dataset: List of calibration examples
            output_dir: Where to save quantized model

        Returns:
            Path to quantized model
        """
        print(f"Quantizing {self.model_name} with GPTQ (bits={self.bits})")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Prepare quantization config
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            damp_percent=0.01,
        )

        # Load model
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_name,
            quantize_config=quantize_config,
        )

        # Prepare calibration data
        calibration_data = self._prepare_calibration_data(
            calibration_dataset,
            tokenizer,
        )

        # Quantize
        model.quantize(calibration_data)

        # Save
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Saved GPTQ model to {output_dir}")
        return output_dir

    def _prepare_calibration_data(self, dataset, tokenizer):
        """Tokenize calibration dataset."""
        examples = []
        for text in dataset[:128]:  # Use 128 samples
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            )
            examples.append(tokens)
        return examples


# src/quantizers/awq_quantizer.py

from awq import AutoAWQForCausalLM

class AWQQuantizer:
    """AWQ quantization implementation."""

    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        group_size: int = 128,
    ):
        self.model_name = model_name
        self.bits = bits
        self.group_size = group_size

    def quantize(
        self,
        calibration_dataset: list,
        output_dir: str,
    ) -> str:
        """Quantize model using AWQ."""
        print(f"Quantizing {self.model_name} with AWQ (bits={self.bits})")

        # Load model
        model = AutoAWQForCausalLM.from_pretrained(self.model_name)

        # Quantize
        model.quantize(
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            quant_config={
                "zero_point": True,
                "q_group_size": self.group_size,
                "w_bit": self.bits,
            },
            calib_data=calibration_dataset[:128],
        )

        # Save
        model.save_quantized(output_dir)
        print(f"Saved AWQ model to {output_dir}")
        return output_dir
```

### Step 3: Accuracy Evaluation (45 minutes)

```python
# src/evaluators/perplexity.py

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

class PerplexityEvaluator:
    """Evaluate model perplexity on datasets."""

    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def evaluate(self, dataset_name: str = "wikitext-2") -> float:
        """
        Calculate perplexity on dataset.

        Args:
            dataset_name: 'wikitext-2' or 'c4'

        Returns:
            Perplexity score
        """
        # Load dataset
        if dataset_name == "wikitext-2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text_column = "text"
        elif dataset_name == "c4":
            dataset = load_dataset("c4", "en", split="validation", streaming=True)
            dataset = list(dataset.take(1000))
            text_column = "text"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Calculate perplexity
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for example in dataset:
                text = example[text_column]
                if len(text.strip()) == 0:
                    continue

                # Tokenize
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=True,
                ).to(self.model.device)

                # Forward pass
                outputs = self.model(**encodings, labels=encodings["input_ids"])

                # Accumulate loss
                total_loss += outputs.loss.item() * encodings["input_ids"].numel()
                total_tokens += encodings["input_ids"].numel()

        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"Perplexity on {dataset_name}: {perplexity:.2f}")
        return perplexity


# src/evaluators/qa_eval.py

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

class QAEvaluator:
    """Evaluate on question-answering benchmarks."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def evaluate(self, tasks: list = None) -> dict:
        """
        Evaluate on QA tasks.

        Args:
            tasks: List of tasks (e.g., ['mmlu', 'truthfulqa', 'hellaswag'])

        Returns:
            Dict of task: score
        """
        if tasks is None:
            tasks = ['mmlu', 'hellaswag', 'piqa']

        print(f"Evaluating on tasks: {tasks}")

        # Create model wrapper
        lm = HFLM(pretrained=self.model_path)

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=5,
            batch_size=8,
        )

        # Extract scores
        scores = {}
        for task in tasks:
            if task in results['results']:
                # Get accuracy metric
                task_results = results['results'][task]
                if 'acc' in task_results:
                    scores[task] = task_results['acc']
                elif 'acc_norm' in task_results:
                    scores[task] = task_results['acc_norm']

        print(f"QA Evaluation Results: {scores}")
        return scores
```

### Step 4: Performance Benchmarking (45 minutes)

```python
# src/evaluators/benchmark.py

import torch
import time
from vllm import LLM, SamplingParams
from typing import List, Tuple

class PerformanceBenchmark:
    """Benchmark model performance."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def benchmark(
        self,
        batch_sizes: List[int] = None,
        num_iterations: int = 100,
    ) -> dict:
        """
        Run performance benchmarks.

        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per config

        Returns:
            Dict with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        # Load model with vLLM
        llm = LLM(model=self.model_path, gpu_memory_utilization=0.9)

        results = {
            'batch_sizes': batch_sizes,
            'latency_p50': [],
            'latency_p95': [],
            'latency_p99': [],
            'throughput': [],
            'memory_mb': [],
        }

        # Test prompts
        test_prompt = "Once upon a time in a galaxy far, far away"

        for batch_size in batch_sizes:
            print(f"Benchmarking batch_size={batch_size}")

            prompts = [test_prompt] * batch_size
            sampling_params = SamplingParams(
                max_tokens=100,
                temperature=0.0,
            )

            # Warmup
            for _ in range(5):
                llm.generate(prompts, sampling_params)

            # Benchmark
            latencies = []
            for _ in range(num_iterations):
                start = time.time()
                outputs = llm.generate(prompts, sampling_params)
                latency = time.time() - start
                latencies.append(latency)

            # Calculate metrics
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]

            # Throughput (tokens/second)
            total_tokens = batch_size * 100  # max_tokens
            throughput = total_tokens / p50

            # Memory usage
            memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            results['latency_p50'].append(p50)
            results['latency_p95'].append(p95)
            results['latency_p99'].append(p99)
            results['throughput'].append(throughput)
            results['memory_mb'].append(memory_mb)

        return results
```

### Step 5: Report Generation (60 minutes)

```python
# src/reporters/report_generator.py

import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

class ReportGenerator:
    """Generate comparison reports."""

    def __init__(self, results: Dict[str, dict]):
        """
        Args:
            results: Dict mapping method name to results
                {
                    'baseline_fp16': {...},
                    'gptq_4bit': {...},
                    'awq_4bit': {...},
                    'int8': {...},
                }
        """
        self.results = results

    def generate_report(self, output_dir: str):
        """Generate comprehensive report."""
        print(f"Generating report in {output_dir}")

        # Create comparison tables
        self._generate_accuracy_table(output_dir)
        self._generate_performance_table(output_dir)

        # Create visualizations
        self._plot_accuracy_vs_speedup(output_dir)
        self._plot_memory_comparison(output_dir)
        self._plot_latency_percentiles(output_dir)

        # Generate markdown report
        self._generate_markdown_report(output_dir)

        print(f"Report generated: {output_dir}/report.md")

    def _generate_accuracy_table(self, output_dir):
        """Create accuracy comparison table."""
        data = []
        for method, results in self.results.items():
            row = {'Method': method}
            row['Perplexity (WikiText-2)'] = results.get('perplexity_wikitext', 'N/A')
            row['MMLU'] = results.get('mmlu_score', 'N/A')
            row['HellaSwag'] = results.get('hellaswag_score', 'N/A')
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/accuracy_comparison.csv", index=False)
        return df

    def _generate_performance_table(self, output_dir):
        """Create performance comparison table."""
        data = []
        for method, results in self.results.items():
            row = {'Method': method}
            row['P99 Latency (ms)'] = results.get('latency_p99', 'N/A')
            row['Throughput (tok/s)'] = results.get('throughput', 'N/A')
            row['Memory (MB)'] = results.get('memory_mb', 'N/A')
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/performance_comparison.csv", index=False)
        return df

    def _plot_accuracy_vs_speedup(self, output_dir):
        """Plot accuracy vs speedup scatter."""
        fig, ax = plt.subplots(figsize=(10, 6))

        baseline_latency = self.results['baseline_fp16']['latency_p99']
        baseline_perplexity = self.results['baseline_fp16']['perplexity_wikitext']

        for method, results in self.results.items():
            speedup = baseline_latency / results['latency_p99']
            perplexity_delta = (results['perplexity_wikitext'] - baseline_perplexity) / baseline_perplexity

            ax.scatter(speedup, perplexity_delta, s=100, label=method)

        ax.set_xlabel('Speedup vs FP16')
        ax.set_ylabel('Perplexity Degradation (%)')
        ax.set_title('Accuracy vs Performance Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(f"{output_dir}/accuracy_vs_speedup.png", dpi=300)
        plt.close()

    def _generate_markdown_report(self, output_dir):
        """Generate markdown report."""
        report = []
        report.append("# Quantization Pipeline Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")

        report.append("## Executive Summary\n")
        # Add summary of best methods

        report.append("## Accuracy Comparison\n")
        df_acc = pd.read_csv(f"{output_dir}/accuracy_comparison.csv")
        report.append(df_acc.to_markdown(index=False))

        report.append("\n## Performance Comparison\n")
        df_perf = pd.read_csv(f"{output_dir}/performance_comparison.csv")
        report.append(df_perf.to_markdown(index=False))

        report.append("\n## Recommendations\n")
        # Add recommendations based on results

        with open(f"{output_dir}/report.md", 'w') as f:
            f.write('\n'.join(report))
```

### Step 6: Pipeline Orchestration (30 minutes)

```python
# src/pipeline.py

import yaml
from pathlib import Path
from quantizers.gptq_quantizer import GPTQQuantizer
from quantizers.awq_quantizer import AWQQuantizer
from evaluators.perplexity import PerplexityEvaluator
from evaluators.qa_eval import QAEvaluator
from evaluators.benchmark import PerformanceBenchmark
from reporters.report_generator import ReportGenerator

class QuantizationPipeline:
    """End-to-end quantization pipeline."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def run(self):
        """Run complete pipeline."""
        print("Starting Quantization Pipeline")

        model_name = self.config['model_name']
        calibration_data = self._load_calibration_data()
        results = {}

        # 1. Baseline (FP16)
        print("\n=== Evaluating Baseline (FP16) ===")
        results['baseline_fp16'] = self._evaluate_model(model_name)

        # 2. GPTQ Quantization
        if 'gptq' in self.config['methods']:
            print("\n=== GPTQ Quantization ===")
            gptq = GPTQQuantizer(model_name, bits=4)
            gptq_path = gptq.quantize(calibration_data, "outputs/gptq_4bit")
            results['gptq_4bit'] = self._evaluate_model(gptq_path)

        # 3. AWQ Quantization
        if 'awq' in self.config['methods']:
            print("\n=== AWQ Quantization ===")
            awq = AWQQuantizer(model_name, bits=4)
            awq_path = awq.quantize(calibration_data, "outputs/awq_4bit")
            results['awq_4bit'] = self._evaluate_model(awq_path)

        # 4. Generate Report
        print("\n=== Generating Report ===")
        reporter = ReportGenerator(results)
        reporter.generate_report("outputs/report")

        print("\nPipeline Complete!")
        print(f"Report available at: outputs/report/report.md")

    def _evaluate_model(self, model_path: str) -> dict:
        """Run all evaluations on a model."""
        results = {}

        # Perplexity
        if self.config['eval_perplexity']:
            ppl_eval = PerplexityEvaluator(model_path)
            results['perplexity_wikitext'] = ppl_eval.evaluate('wikitext-2')

        # QA tasks
        if self.config['eval_qa']:
            qa_eval = QAEvaluator(model_path)
            qa_results = qa_eval.evaluate(['mmlu', 'hellaswag'])
            results.update(qa_results)

        # Performance
        if self.config['eval_performance']:
            benchmark = PerformanceBenchmark(model_path)
            perf_results = benchmark.benchmark()
            results.update(perf_results)

        return results

    def _load_calibration_data(self) -> list:
        """Load calibration dataset."""
        # Load from config
        return ["sample text"] * 128  # Placeholder
```

---

## Configuration

```yaml
# configs/pipeline_config.yaml

model_name: "meta-llama/Llama-2-7b-hf"

methods:
  - gptq
  - awq
  - int8

quantization_config:
  gptq:
    bits: [4, 8]
    group_size: 128
  awq:
    bits: 4
    group_size: 128

eval_perplexity: true
eval_qa: true
eval_performance: true

qa_tasks:
  - mmlu
  - hellaswag
  - piqa

output_dir: "outputs/"
```

---

## Deliverables

1. **Pipeline Implementation** (all quantization methods)
2. **Evaluation Scripts** (accuracy + performance)
3. **Report Generator** (tables + charts)
4. **Configuration System**
5. **Documentation** (usage guide)

---

## Evaluation Rubric

**Total: 100 points | Passing: 70 points**

---

*Project 7.5: Quantization Pipeline | Duration: 3-4 hours*
