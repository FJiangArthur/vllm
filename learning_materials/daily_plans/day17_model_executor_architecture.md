# Day 17: Model Executor Architecture

> **Goal**: Understand model loading, weight management, and execution pipeline in vLLM
> **Time**: 4-6 hours
> **Prerequisites**: Days 15-16 complete, PyTorch basics
> **Deliverables**: Model execution flowchart, weight loading analyzer, forward pass trace

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): Model Executor Architecture

**9:00-9:30** - Model Executor Overview
**9:30-10:30** - Model Loading & Weight Management
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Model Registry & Architectures

### Afternoon Session (2-3 hours): Forward Pass Pipeline

**14:00-15:00** - Forward Pass Execution Flow
**15:00-16:00** - Input Preparation & Batching
**16:00-16:30** - Break
**16:30-17:30** - Hands-On: Trace Model Execution

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Review execution flow, document findings, preview Day 18

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain the ModelExecutor's role in vLLM
- [ ] Trace model loading from HuggingFace to GPU
- [ ] Understand weight quantization and sharding
- [ ] Follow the forward pass from inputs to logits
- [ ] Identify optimization opportunities in execution

---

## 📂 Morning: Model Executor Architecture (9:00-12:00)

### Task 1: Model Executor Overview (30 min)

**File**: `vllm/model_executor/model_loader.py`

**ModelExecutor Responsibilities**:

```python
class ModelExecutor:
    """
    Responsible for:
    1. Loading model weights from disk/HuggingFace
    2. Applying quantization/compression
    3. Distributing weights across GPUs (if multi-GPU)
    4. Executing forward pass
    5. Managing GPU memory for weights
    """
```

**Architecture Overview**:

```
Model Execution Pipeline:
┌─────────────────────────────────────────────────┐
│ 1. Model Loading                                 │
│    - Download from HuggingFace                   │
│    - Load weights to CPU/GPU                     │
│    - Apply quantization                          │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ 2. Model Initialization                          │
│    - Create model architecture                   │
│    - Initialize layers                           │
│    - Set up attention backend                    │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ 3. Forward Pass Execution (per step)            │
│    - Prepare inputs (tokens, positions)          │
│    - Execute model layers                        │
│    - Get output logits                           │
└─────────────────────────────────────────────────┘
```

**Key Classes**:

```python
# vllm/model_executor/model_loader.py

class ModelLoader:
    """Handles model loading from various sources."""

    def load_model(self, model_config) -> nn.Module:
        """
        Load model based on config.

        Process:
        1. Determine model class (Llama, GPT, etc.)
        2. Load weights from HF or local path
        3. Apply quantization if specified
        4. Move to target device(s)
        """
        pass

# vllm/model_executor/model_runner.py

class ModelRunner:
    """Executes model forward pass."""

    def execute_model(
        self,
        seq_group_metadata_list,
        kv_caches,
    ):
        """
        Execute one step of model inference.

        Inputs:
        - seq_group_metadata_list: Request metadata
        - kv_caches: PagedAttention KV cache

        Returns:
        - logits: Model outputs for sampling
        """
        pass
```

### Task 2: Model Loading & Weight Management (60 min)

**File**: `vllm/model_executor/model_loader.py`

**Loading Process**:

```python
def get_model(model_config, device_config, **kwargs):
    """
    Main entry point for model loading.
    """

    # Step 1: Determine model architecture
    model_class = _get_model_architecture(model_config.model)
    # Example: "meta-llama/Llama-2-7b-hf" → LlamaForCausalLM

    # Step 2: Load configuration
    hf_config = AutoConfig.from_pretrained(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Step 3: Initialize model
    with torch.device("cuda"):
        model = model_class(
            hf_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    # Step 4: Load weights
    if model_config.quantization:
        # Load quantized weights
        load_quantized_weights(model, model_config)
    else:
        # Load full precision weights
        load_weights(model, model_config)

    return model.eval()
```

**Weight Loading Strategies**:

```python
# Strategy 1: Standard PyTorch loading
def load_weights_standard(model, model_path):
    """Load weights from checkpoint."""

    state_dict = torch.load(
        f"{model_path}/pytorch_model.bin",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.to("cuda")


# Strategy 2: Memory-efficient loading (safetensors)
def load_weights_safetensors(model, model_path):
    """Load using safetensors (faster, safer)."""

    from safetensors.torch import load_file

    # Load directly to GPU
    state_dict = load_file(
        f"{model_path}/model.safetensors",
        device="cuda:0"
    )

    model.load_state_dict(state_dict, strict=True)


# Strategy 3: Sharded loading (for large models)
def load_weights_sharded(model, model_path):
    """Load weights in shards (for > GPU memory models)."""

    # Load index
    with open(f"{model_path}/pytorch_model.bin.index.json") as f:
        index = json.load(f)

    # Load each shard
    for shard_file in index["weight_map"].values():
        shard_path = f"{model_path}/{shard_file}"

        # Load shard to CPU
        shard = torch.load(shard_path, map_location="cpu")

        # Transfer relevant weights to GPU
        for param_name, param in shard.items():
            if param_name in model.state_dict():
                model.state_dict()[param_name].copy_(param)

        del shard  # Free CPU memory
```

**Quantization Integration**:

```python
# vllm/model_executor/layers/quantization/

def apply_quantization(model, quant_config):
    """
    Apply quantization to model weights.

    Supported methods:
    - AWQ (Activation-aware Weight Quantization)
    - GPTQ (Generative Pre-trained Transformer Quantization)
    - SqueezeLLM
    - FP8
    """

    if quant_config.method == "awq":
        # AWQ: INT4 weights, FP16 activations
        from vllm.model_executor.layers.quantization.awq import (
            AWQLinearMethod
        )

        # Replace linear layers with quantized versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quant_linear = AWQLinearMethod.create_weights(
                    module,
                    quant_config=quant_config,
                )
                # Replace module
                setattr(model, name, quant_linear)

    elif quant_config.method == "gptq":
        # GPTQ: INT4/INT8 weights
        # Similar replacement strategy
        pass

    return model
```

**Memory Footprint Analysis**:

```python
#!/usr/bin/env python3
"""
Analyze model memory footprint.
"""

def calculate_model_memory(model_config):
    """
    Calculate memory required for model weights.
    """

    # Get model architecture info
    config = AutoConfig.from_pretrained(model_config.model)

    # Parameters
    num_params = (
        config.hidden_size * config.vocab_size +  # Embeddings
        config.num_hidden_layers * (
            # Self-attention
            4 * config.hidden_size * config.hidden_size +
            # MLP
            2 * config.hidden_size * config.intermediate_size
        ) +
        config.hidden_size * config.vocab_size  # LM head
    )

    # Bytes per parameter
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1,
        'int4': 0.5,
    }

    # Calculate for different precisions
    print(f"\nModel: {model_config.model}")
    print(f"Parameters: {num_params / 1e9:.2f}B")
    print(f"\nMemory footprint:")

    for dtype, bytes_val in bytes_per_param.items():
        memory_gb = (num_params * bytes_val) / (1024 ** 3)
        print(f"  {dtype}: {memory_gb:.2f} GB")

# Example
# Llama-2-7B:
#   Parameters: 7B
#   FP16: 14 GB
#   INT8: 7 GB
#   INT4: 3.5 GB
```

### Task 3: Model Registry & Architectures (60 min)

**File**: `vllm/model_executor/models/`

**Supported Architectures**:

```bash
# List supported models
ls vllm/model_executor/models/

# Output:
# llama.py          - Llama/Llama-2/CodeLlama
# gpt2.py           - GPT-2
# gpt_neox.py       - GPT-NeoX, Pythia
# opt.py            - OPT
# bloom.py          - BLOOM
# mistral.py        - Mistral
# mixtral.py        - Mixtral (MoE)
# qwen.py           - Qwen
# ...and more
```

**Model Architecture Pattern**:

```python
# vllm/model_executor/models/llama.py

class LlamaForCausalLM(nn.Module):
    """
    Llama model implementation for vLLM.

    Modifications from HuggingFace:
    1. Uses PagedAttention instead of standard attention
    2. Optimized for batched inference
    3. Supports quantization
    4. Custom sampling
    """

    def __init__(self, config, cache_config=None, quant_config=None):
        super().__init__()

        self.config = config

        # Embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (output projection)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ):
        """
        Forward pass.

        Args:
            input_ids: [num_tokens] - flattened token IDs
            positions: [num_tokens] - position in sequence
            kv_caches: List of KV caches for each layer
            input_metadata: Metadata for attention (block tables, etc.)
        """

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_caches[i],
                input_metadata=input_metadata,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return logits
```

**Layer Implementation**:

```python
class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer layer."""

    def __init__(self, config, cache_config, quant_config):
        super().__init__()

        # Self-attention
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,  # GQA
            cache_config=cache_config,
        )

        # MLP
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, positions, hidden_states, kv_cache, input_metadata):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

---

## 🔬 Afternoon: Forward Pass Pipeline (14:00-17:30)

### Task 4: Forward Pass Execution (60 min)

**File**: `vllm/model_executor/model_runner.py`

**Execute Model Flow**:

```python
class ModelRunner:
    """Runs model forward pass."""

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[KVCache],
    ):
        """
        Execute one step of inference.

        Process:
        1. Prepare inputs from seq_group_metadata
        2. Run model forward
        3. Return logits for sampling
        """

        # 1. Prepare inputs
        input_tokens, input_positions, input_metadata = (
            self._prepare_inputs(seq_group_metadata_list)
        )

        # 2. Execute model
        with torch.inference_mode():
            hidden_states = self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=kv_caches,
                input_metadata=input_metadata,
            )

        # 3. Extract logits
        logits = self._get_logits(hidden_states, seq_group_metadata_list)

        return logits
```

**Input Preparation**:

```python
def _prepare_inputs(self, seq_group_metadata_list):
    """
    Prepare batched inputs for model.

    Converts:
    - List of SequenceGroupMetadata → Flat tensors
    """

    # Collect all tokens
    input_tokens = []
    input_positions = []
    slot_mapping = []
    context_lens = []

    for seq_group_metadata in seq_group_metadata_list:
        for seq_id, seq_data in seq_group_metadata.seq_data.items():

            if seq_group_metadata.is_prompt:
                # Prefill: all prompt tokens
                tokens = seq_data.get_token_ids()
                positions = list(range(len(tokens)))

            else:
                # Decode: last token only
                tokens = [seq_data.get_last_token_id()]
                positions = [seq_data.get_len() - 1]

            input_tokens.extend(tokens)
            input_positions.extend(positions)

    # Convert to tensors
    input_tokens = torch.tensor(input_tokens, dtype=torch.long, device="cuda")
    input_positions = torch.tensor(input_positions, dtype=torch.long, device="cuda")

    # Build input metadata (for attention)
    input_metadata = InputMetadata(
        seq_group_metadata_list=seq_group_metadata_list,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        # ... more metadata
    )

    return input_tokens, input_positions, input_metadata
```

**Example Batch**:

```
Input batch with 2 prefill + 2 decode:

Seq A (prefill): "Hello world" → [15496, 296]
Seq B (decode): Generated 50 tokens, generating 51st
Seq C (prefill): "The capital of" → [464, 3139, 273]
Seq D (decode): Generated 120 tokens, generating 121st

Prepared inputs:
input_tokens:    [15496, 296, <token_50>, 464, 3139, 273, <token_120>]
                  └─ A ──┘  └─── B ────┘ └──── C ────┘  └──── D ─────┘

input_positions: [0,     1,   50,         0,    1,     2,    120      ]
                  └─ A ──┘  └─── B ────┘ └──── C ────┘  └──── D ─────┘

Total tokens in batch: 7
```

### Task 5: Input Batching Details (60 min)

**Block Tables for Attention**:

```python
def _prepare_block_tables(self, seq_group_metadata_list):
    """
    Prepare block tables for PagedAttention.

    Each sequence has a block table mapping logical → physical blocks.
    """

    block_tables = []
    max_blocks = 0

    for seq_group_metadata in seq_group_metadata_list:
        for seq_id in seq_group_metadata.seq_data.keys():
            # Get block table from block manager
            block_table = seq_group_metadata.block_tables[seq_id]
            block_tables.append(block_table)

            max_blocks = max(max_blocks, len(block_table))

    # Pad to same length (for batching)
    padded_block_tables = []
    for block_table in block_tables:
        padded = block_table + [-1] * (max_blocks - len(block_table))
        padded_block_tables.append(padded)

    # Convert to tensor
    block_tables_tensor = torch.tensor(
        padded_block_tables,
        dtype=torch.int32,
        device="cuda"
    )

    return block_tables_tensor

# Example output:
# block_tables_tensor = [
#     [5, 7, 9, -1],    # Seq A: 3 blocks
#     [2, 4, 12, 15],   # Seq B: 4 blocks
#     [8, 10, -1, -1],  # Seq C: 2 blocks
#     [1, 3, 6, 11],    # Seq D: 4 blocks
# ]
```

**Slot Mapping for KV Cache Writes**:

```python
def _prepare_slot_mapping(self, seq_group_metadata_list):
    """
    Prepare slot mapping for writing KV cache.

    For each token being processed, determine where to write its K/V.
    """

    slot_mapping = []

    for seq_group_metadata in seq_group_metadata_list:
        block_table = seq_group_metadata.block_tables[seq_id]

        if seq_group_metadata.is_prompt:
            # Prefill: map all prompt tokens to slots
            num_tokens = len(tokens)

            for i in range(num_tokens):
                # Logical slot: position i
                logical_slot = i

                # Physical slot: block_table[block_idx] * block_size + offset
                block_idx = logical_slot // self.block_size
                block_offset = logical_slot % self.block_size

                physical_block = block_table[block_idx]
                physical_slot = physical_block * self.block_size + block_offset

                slot_mapping.append(physical_slot)

        else:
            # Decode: map only last token
            seq_len = seq_data.get_len()
            logical_slot = seq_len - 1

            block_idx = logical_slot // self.block_size
            block_offset = logical_slot % self.block_size

            physical_block = block_table[block_idx]
            physical_slot = physical_block * self.block_size + block_offset

            slot_mapping.append(physical_slot)

    return torch.tensor(slot_mapping, dtype=torch.long, device="cuda")

# Example:
# Seq A: 2 tokens → slot_mapping = [slot_for_token_0, slot_for_token_1]
# Seq B: decode → slot_mapping = [slot_for_token_50]
```

### Task 6: Trace Model Execution (60 min)

**Exercise: End-to-End Execution Trace**

```python
#!/usr/bin/env python3
"""
Trace model execution with detailed logging.
"""

import torch
from vllm import LLM, SamplingParams

class ExecutionTracer:
    """Trace model execution."""

    def __init__(self):
        self.logs = []

    def trace_forward_pass(self):
        """Trace a forward pass."""

        print("="*60)
        print("TRACING MODEL EXECUTION")
        print("="*60)

        # Create model
        llm = LLM(model="facebook/opt-125m", max_model_len=256)

        # Hook into model layers
        def forward_hook(module, input, output):
            self.logs.append({
                'module': module.__class__.__name__,
                'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                'output_shape': output.shape if isinstance(output, torch.Tensor) else None,
            })

        # Register hooks
        for name, module in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_modules():
            module.register_forward_hook(forward_hook)

        # Run inference
        prompt = "Hello, my name is"
        outputs = llm.generate([prompt], SamplingParams(max_tokens=5))

        # Analyze logs
        print(f"\n{'='*60}")
        print(f"Forward Pass Trace")
        print(f"{'='*60}")

        for i, log in enumerate(self.logs[:20]):  # First 20 modules
            print(f"{i+1}. {log['module']}")
            print(f"   Input:  {log['input_shape']}")
            print(f"   Output: {log['output_shape']}")

tracer = ExecutionTracer()
tracer.trace_forward_pass()
```

**Weight Inspection Tool**:

```python
def inspect_model_weights(model_name: str):
    """Inspect model weight structure."""

    from transformers import AutoModelForCausalLM

    print(f"\nInspecting: {model_name}")
    print("="*60)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Analyze parameters
    total_params = 0
    layer_params = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Group by layer type
        layer_type = name.split('.')[0]
        layer_params[layer_type] = layer_params.get(layer_type, 0) + num_params

        # Print first few
        if len(layer_params) <= 5:
            print(f"{name:50s} {param.shape} ({num_params:,})")

    print(f"\n{'='*60}")
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"\nBy layer type:")
    for layer_type, count in sorted(layer_params.items()):
        pct = count / total_params * 100
        print(f"  {layer_type:30s} {count:15,} ({pct:5.1f}%)")

# Example usage
inspect_model_weights("facebook/opt-125m")
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Model Executor Architecture**
- Loading models from HuggingFace
- Weight quantization strategies
- Model registry and architectures

✅ **Forward Pass Pipeline**
- Input preparation and batching
- Block tables and slot mapping
- Execution flow through layers

✅ **Implementation Details**
- vLLM model modifications
- PagedAttention integration
- Memory management

### Knowledge Check Questions

**Q1**: What are the main differences between vLLM's model implementations and HuggingFace?
<details>
<summary>Answer</summary>
1. **Attention**: PagedAttention instead of standard attention
2. **Batching**: Optimized for batched inference with variable lengths
3. **KV Cache**: Block-based storage with indirection
4. **Quantization**: Built-in support for AWQ, GPTQ, etc.
5. **Input format**: Flattened tokens vs. padded sequences
</details>

**Q2**: How does quantization reduce memory footprint?
<details>
<summary>Answer</summary>
Reduces bits per weight parameter:
- **FP16**: 2 bytes → 14GB for 7B params
- **INT8**: 1 byte → 7GB (50% reduction)
- **INT4**: 0.5 bytes → 3.5GB (75% reduction)

Trade-off: Lower precision = smaller memory but potential accuracy loss
Techniques like AWQ/GPTQ minimize accuracy impact
</details>

**Q3**: What is the purpose of slot_mapping in model execution?
<details>
<summary>Answer</summary>
Maps each token being processed to its physical KV cache location.

For token at position `i` in sequence with block_table `[b0, b1, ...]`:
- Block index: `i // block_size`
- Block offset: `i % block_size`
- Physical slot: `block_table[block_idx] * block_size + offset`

Used by attention kernels to read/write KV cache correctly.
</details>

---

## 🚀 Preview: Day 18

Tomorrow you'll explore:
- **GPU Execution Pipeline**: CUDA kernel launches and synchronization
- **Attention Execution**: How PagedAttention runs on GPU
- **Memory Operations**: Copies, swaps, and cache updates
- **Performance Profiling**: Analyzing GPU execution

**Preparation**:
- Review CUDA execution model
- Understand kernel launch parameters
- Prepare profiling tools (Nsight)

---

## 📚 Additional Resources

**Key Files**:
- [ ] `vllm/model_executor/model_loader.py`
- [ ] `vllm/model_executor/model_runner.py`
- [ ] `vllm/model_executor/models/llama.py`
- [ ] `vllm/model_executor/layers/attention.py`

**Concepts**:
- [ ] PyTorch model loading
- [ ] Quantization techniques (AWQ, GPTQ)
- [ ] Tensor parallelism basics

---

**Completed: ___/___/___**
**Time spent: _____ hours**
**Confidence level (1-10): _____**
