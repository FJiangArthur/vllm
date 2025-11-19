# Megatron-Style Model Parallelism

## Table of Contents
1. [Introduction to Megatron-LM](#introduction)
2. [Megatron Architecture Overview](#architecture-overview)
3. [Transformer Layer Parallelization](#transformer-parallelization)
4. [Attention Mechanism Partitioning](#attention-partitioning)
5. [MLP Layer Partitioning](#mlp-partitioning)
6. [Embedding and Output Layers](#embedding-output)
7. [vLLM's Megatron Implementation](#vllm-implementation)
8. [Weight Initialization and Loading](#weight-loading)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

---

## Introduction to Megatron-LM {#introduction}

Megatron-LM, developed by NVIDIA, is a framework for efficient large-scale language model training and inference using model parallelism. vLLM adopts Megatron's parallelization strategies for distributed inference.

### What is Megatron-Style Parallelism?

Megatron-style parallelism refers to a specific way of partitioning transformer models:
- **Intra-layer parallelism**: Split individual layers across GPUs
- **Minimal communication**: Only 2 all-reduce operations per transformer layer
- **Mathematically equivalent**: Produces same results as single-GPU execution
- **Memory efficient**: Reduces activation memory through partitioning

### Key Innovations

1. **Column and Row Parallel Linear Layers**
   - Strategic placement to minimize communication
   - Fused operations to reduce synchronization points

2. **Parallel Self-Attention**
   - Split attention heads across GPUs
   - Partition QKV projections efficiently

3. **Parallel MLP**
   - Split intermediate activations
   - Minimize memory footprint

4. **Efficient Communication Patterns**
   - Overlap communication with computation
   - Use high-bandwidth interconnects (NVLink, InfiniBand)

### Historical Context

**Original Megatron Papers:**
1. **Megatron-LM (2019)**: Introduced efficient model parallelism for BERT and GPT
2. **Megatron-LM v2 (2021)**: Added pipeline parallelism and 3D parallelism
3. **Megatron-LM v3 (2022)**: Sequence parallelism and further optimizations

**Model Sizes Enabled:**
- GPT-3 175B (2020)
- Megatron-Turing NLG 530B (2021)
- Current: 1T+ parameter models

---

## Megatron Architecture Overview {#architecture-overview}

### Parallelization Principles

Megatron follows these core principles:

```
1. Minimize Communication:
   - Only f (forward) and g (backward) all-reduce per layer
   - No additional synchronization in forward pass

2. Balance Computation:
   - Equal work distribution across GPUs
   - Same FLOPS per GPU

3. Partition Activations:
   - Reduce activation memory
   - Enable larger batch sizes

4. Preserve Semantics:
   - Identical outputs to single-GPU
   - No approximations or lossy operations
```

### Communication Patterns

```python
# Megatron transformer layer communication pattern

class MegatronTransformerLayer:
    """
    Single transformer layer with Megatron-style parallelism.

    Communication:
    - f: forward all-reduce after attention
    - g: forward all-reduce after MLP
    Total: 2 all-reduces per layer
    """

    def __init__(self, config):
        # Column parallel: Q, K, V projections
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            gather_output=False  # Keep partitioned
        )

        # Row parallel: attention output projection
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            input_is_parallel=True  # Already partitioned
        )

        # Column parallel: MLP up projection
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            gather_output=False
        )

        # Row parallel: MLP down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            input_is_parallel=True
        )

    def forward(self, hidden_states):
        # === ATTENTION ===
        # Input: [batch, seq, hidden] - replicated on all GPUs

        # Column parallel QKV: no communication
        qkv = self.qkv_proj(hidden_states)
        # Output: [batch, seq, 3*hidden/N] - partitioned

        # Self-attention computation (local to each GPU)
        attn_output = self.self_attention(qkv)
        # Output: [batch, seq, hidden/N] - partitioned

        # Row parallel output projection: f all-reduce
        attn_output = self.o_proj(attn_output)  # <-- ALL-REDUCE 1
        # Output: [batch, seq, hidden] - replicated

        # Residual connection
        hidden_states = hidden_states + attn_output

        # === MLP ===
        # Input: [batch, seq, hidden] - replicated

        # Column parallel gate/up: no communication
        intermediate = self.gate_up_proj(hidden_states)
        # Output: [batch, seq, intermediate/N] - partitioned

        # Activation (local)
        intermediate = activation(intermediate)

        # Row parallel down projection: g all-reduce
        mlp_output = self.down_proj(intermediate)  # <-- ALL-REDUCE 2
        # Output: [batch, seq, hidden] - replicated

        # Residual connection
        hidden_states = hidden_states + mlp_output

        return hidden_states
```

### Memory Savings

```python
def calculate_memory_savings(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    tensor_parallel_size: int
):
    """
    Calculate memory savings from Megatron-style parallelism.
    """

    # Model parameters (weights)
    params_per_layer = (
        4 * hidden_size * hidden_size +  # QKV + O projections
        8 * hidden_size * hidden_size     # MLP (assumes 4x intermediate)
    )
    total_params = params_per_layer * num_layers

    # FP16: 2 bytes per parameter
    param_memory_single_gpu = total_params * 2
    param_memory_per_gpu = param_memory_single_gpu / tensor_parallel_size

    # Activations (intermediate tensors)
    # Key insight: Megatron keeps activations partitioned
    activation_per_layer = (
        batch_size * seq_length * hidden_size * 2 +  # Input/output
        batch_size * seq_length * (hidden_size / tensor_parallel_size) * 2  # Partitioned intermediate
    )
    total_activations = activation_per_layer * num_layers

    # Gradient memory (same as activations in backward pass)
    gradient_memory = total_activations

    # Optimizer states (Adam: 2x parameters for momentum and variance)
    optimizer_memory = param_memory_per_gpu * 2

    total_memory_per_gpu = (
        param_memory_per_gpu +
        total_activations +
        gradient_memory +
        optimizer_memory
    )

    return {
        'param_memory_gb': param_memory_per_gpu / 1e9,
        'activation_memory_gb': total_activations / 1e9,
        'total_memory_gb': total_memory_per_gpu / 1e9,
        'savings_ratio': param_memory_single_gpu / param_memory_per_gpu
    }


# Example: Llama-70B with 4-way TP
memory = calculate_memory_savings(
    batch_size=32,
    seq_length=2048,
    hidden_size=8192,
    num_layers=80,
    tensor_parallel_size=4
)

print(f"Memory per GPU: {memory['total_memory_gb']:.2f} GB")
print(f"Savings ratio: {memory['savings_ratio']:.1f}x")
```

---

## Transformer Layer Parallelization {#transformer-parallelization}

### Complete Transformer Layer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)


class MegatronSelfAttention(nn.Module):
    """
    Multi-head self-attention with Megatron-style tensor parallelism.

    Splits attention heads across tensor parallel ranks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.total_num_heads = num_attention_heads

        # Grouped-query attention support
        self.num_kv_heads = num_kv_heads or num_attention_heads

        # Tensor parallel configuration
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Heads per GPU
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size

        # GQA: KV heads may be fewer
        assert self.num_kv_heads % self.tp_size == 0
        self.num_kv_heads_per_gpu = self.num_kv_heads // self.tp_size

        self.head_dim = hidden_size // self.total_num_heads

        # Column parallel QKV projection
        # Shape: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim / tp_size]
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.num_heads + 2 * self.num_kv_heads_per_gpu) * self.head_dim,
            bias=False,
            gather_output=False,
        )

        # Row parallel output projection
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # QKV projection (column parallel, no communication)
        qkv, _ = self.qkv_proj(hidden_states)

        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads_per_gpu * self.head_dim

        q = qkv[:, :q_size]
        k = qkv[:, q_size:q_size + kv_size]
        v = qkv[:, q_size + kv_size:]

        # Reshape for attention
        # Q: [batch * seq, num_heads, head_dim]
        # K, V: [batch * seq, num_kv_heads_per_gpu, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads_per_gpu, self.head_dim)
        v = v.view(-1, self.num_kv_heads_per_gpu, self.head_dim)

        # Apply RoPE (rotary position embeddings)
        q, k = self.rotary_emb(positions, q, k)

        # Update KV cache
        k, v = self.kv_cache.update(k, v, attn_metadata)

        # Grouped-query attention
        # Repeat KV heads if necessary
        if self.num_heads != self.num_kv_heads_per_gpu:
            k = k.repeat_interleave(
                self.num_heads // self.num_kv_heads_per_gpu,
                dim=1
            )
            v = v.repeat_interleave(
                self.num_heads // self.num_kv_heads_per_gpu,
                dim=1
            )

        # Attention computation (local to this GPU)
        # Uses PagedAttention or FlashAttention
        attn_output = self.attn(q, k, v, attn_metadata)

        # Reshape back
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)

        # Output projection (row parallel, includes all-reduce)
        output, _ = self.o_proj(attn_output)

        return output


class MegatronMLP(nn.Module):
    """
    MLP with Megatron-style tensor parallelism.

    Uses SwiGLU activation (common in modern LLMs).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn: str = "silu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Column parallel: splits intermediate dimension
        # SwiGLU needs 2 projections (gate and up)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
        )

        # Row parallel: combines outputs
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        # Activation function
        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate and up projection (column parallel, no communication)
        gate_up, _ = self.gate_up_proj(x)

        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)

        # SwiGLU: gate activation * up
        intermediate = self.act_fn(gate) * up

        # Down projection (row parallel, includes all-reduce)
        output, _ = self.down_proj(intermediate)

        return output


class MegatronTransformerLayer(nn.Module):
    """
    Complete transformer layer with Megatron-style parallelism.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size

        # Pre-attention layer norm
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )

        # Self-attention
        self.self_attn = MegatronSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", None),
        )

        # Pre-MLP layer norm
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )

        # MLP
        self.mlp = MegatronMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            act_fn=config.hidden_act,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # === Self-Attention Block ===
        residual = hidden_states

        # Layer norm (local operation)
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention (includes 1 all-reduce in o_proj)
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            kv_cache,
            attn_metadata,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # === MLP Block ===
        residual = hidden_states

        # Layer norm (local operation)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP (includes 1 all-reduce in down_proj)
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states
```

---

## Attention Mechanism Partitioning {#attention-partitioning}

### Head-Parallel Attention

```python
def visualize_attention_parallelism():
    """
    Visualize how attention heads are split across GPUs.
    """

    # Example: 32 attention heads, 4 GPUs
    total_heads = 32
    tp_size = 4
    heads_per_gpu = total_heads // tp_size  # 8

    print("Attention Head Distribution:")
    print("=" * 60)

    for gpu_id in range(tp_size):
        start_head = gpu_id * heads_per_gpu
        end_head = (gpu_id + 1) * heads_per_gpu

        print(f"GPU {gpu_id}: Heads {start_head:2d}-{end_head-1:2d} "
              f"({heads_per_gpu} heads)")

    print("\nMemory per GPU:")
    hidden_size = 8192
    head_dim = hidden_size // total_heads  # 256

    # QKV weights per GPU
    qkv_weights_per_gpu = (
        hidden_size *  # Input dimension (not split)
        heads_per_gpu * head_dim * 3  # Q, K, V for local heads
    )

    # Output projection weights per GPU
    o_weights_per_gpu = (
        heads_per_gpu * head_dim *  # Input dimension (split)
        hidden_size  # Output dimension (not split)
    )

    total_weights = (qkv_weights_per_gpu + o_weights_per_gpu) * 2 / 1e9  # FP16

    print(f"Attention weights per GPU: {total_weights:.2f} GB")


# Example output:
# GPU 0: Heads  0- 7 (8 heads)
# GPU 1: Heads  8-15 (8 heads)
# GPU 2: Heads 16-23 (8 heads)
# GPU 3: Heads 24-31 (8 heads)
```

### Grouped Query Attention (GQA) Support

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with Megatron-style parallelism.

    GQA uses fewer KV heads than Q heads, reducing KV cache memory.
    Example: Llama-70B uses 64 Q heads, 8 KV heads (8x reduction).
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()

        # Q heads split across GPUs
        assert num_q_heads % self.tp_size == 0
        self.num_q_heads_per_gpu = num_q_heads // self.tp_size

        # KV heads split across GPUs
        assert num_kv_heads % self.tp_size == 0
        self.num_kv_heads_per_gpu = num_kv_heads // self.tp_size

        # Heads per KV head
        self.num_q_per_kv = num_q_heads // num_kv_heads

        self.head_dim = hidden_size // num_q_heads

        # QKV projection with different sizes
        q_size = self.num_q_heads_per_gpu * self.head_dim
        kv_size = self.num_kv_heads_per_gpu * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            q_size + 2 * kv_size,  # Q + K + V
            bias=False,
        )

    def forward(self, x):
        qkv = self.qkv_proj(x)

        # Split Q, K, V
        q_size = self.num_q_heads_per_gpu * self.head_dim
        kv_size = self.num_kv_heads_per_gpu * self.head_dim

        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]

        # Reshape
        q = q.view(*q.shape[:-1], self.num_q_heads_per_gpu, self.head_dim)
        k = k.view(*k.shape[:-1], self.num_kv_heads_per_gpu, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_kv_heads_per_gpu, self.head_dim)

        # Expand KV to match Q heads
        # Each KV head is used by num_q_per_kv Q heads
        k = k.repeat_interleave(self.num_q_per_kv, dim=-2)
        v = v.repeat_interleave(self.num_q_per_kv, dim=-2)

        # Now k, v have same number of heads as q
        # Standard attention computation
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        return out
```

---

## MLP Layer Partitioning {#mlp-partitioning}

### SwiGLU Activation with TP

```python
class MegatronSwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit used in modern LLMs.

    Formula: SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    where Swish(x) = x * sigmoid(x)

    With tensor parallelism:
    - W and V are column-partitioned
    - Each GPU computes partial intermediate activations
    - Down projection combines with row-parallelism
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        tp_size = get_tensor_model_parallel_world_size()
        self.intermediate_size_per_gpu = intermediate_size // tp_size

        # Column parallel: gate and up projections
        # Combined into single matrix for efficiency
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,  # Both gate and up
            bias=False,
            gather_output=False,
        )

        # Row parallel: down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [batch, seq, hidden_size]

        Returns:
            Output tensor [batch, seq, hidden_size]
        """
        # Gate and up projection (no communication)
        # Output: [batch, seq, 2 * intermediate_size / tp_size]
        gate_up = self.gate_up_proj(x)

        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)

        # SwiGLU activation
        # Swish(gate) = gate * sigmoid(gate)
        intermediate = F.silu(gate) * up

        # Down projection (all-reduce communication)
        # Output: [batch, seq, hidden_size]
        output = self.down_proj(intermediate)

        return output


def compare_mlp_variants():
    """
    Compare different MLP activation functions with TP.
    """

    hidden = 8192
    intermediate = 28672  # 3.5x expansion (common in LLMs)
    tp_size = 4

    print("MLP Variants with Tensor Parallelism:")
    print("=" * 60)

    variants = {
        "GELU": {
            "projections": 2,  # up, down
            "activations": 1,  # gelu
            "intermediate_per_gpu": intermediate // tp_size,
        },
        "SwiGLU": {
            "projections": 3,  # gate, up, down
            "activations": 2,  # silu, multiply
            "intermediate_per_gpu": intermediate // tp_size,
        },
        "GeGLU": {
            "projections": 3,  # gate, up, down
            "activations": 2,  # gelu, multiply
            "intermediate_per_gpu": intermediate // tp_size,
        },
    }

    for name, config in variants.items():
        # Compute parameters per GPU
        if config["projections"] == 2:
            params = (
                hidden * config["intermediate_per_gpu"] +  # up
                config["intermediate_per_gpu"] * hidden     # down
            )
        else:  # 3 projections (gate and up combined)
            params = (
                hidden * config["intermediate_per_gpu"] * 2 +  # gate + up
                config["intermediate_per_gpu"] * hidden         # down
            )

        params_gb = params * 2 / 1e9  # FP16

        print(f"\n{name}:")
        print(f"  Projections: {config['projections']}")
        print(f"  Intermediate per GPU: {config['intermediate_per_gpu']:,}")
        print(f"  Params per GPU: {params_gb:.2f} GB")
        print(f"  All-reduces per forward: 1")
```

---

## Embedding and Output Layers {#embedding-output}

### Parallel Embedding Layers

```python
class ParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.

    Splits vocabulary across tensor parallel ranks.
    Used for input embeddings in large models.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Tensor parallel config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Split vocabulary
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        # Embedding weight (partitioned vocabulary)
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with vocabulary parallelism.

        Each GPU handles a slice of the vocabulary.
        Tokens outside this GPU's range are zeroed out.
        Final result is sum across all GPUs (all-reduce).
        """
        # Create mask for tokens in this partition's vocabulary range
        mask = (
            (input_ids >= self.vocab_start_index) &
            (input_ids < self.vocab_end_index)
        )

        # Adjust indices to local range
        local_input_ids = input_ids - self.vocab_start_index
        local_input_ids = torch.where(
            mask,
            local_input_ids,
            torch.zeros_like(local_input_ids)
        )

        # Lookup embeddings (only for local vocabulary)
        output = F.embedding(
            local_input_ids,
            self.weight,
            padding_idx=self.padding_idx,
        )

        # Zero out embeddings for tokens not in this partition
        mask = mask.unsqueeze(-1).expand_as(output)
        output = torch.where(mask, output, torch.zeros_like(output))

        # All-reduce to combine results from all partitions
        output = tensor_model_parallel_all_reduce(output)

        return output


class ParallelLMHead(nn.Module):
    """
    Language model head with vocabulary parallelism.

    Computes logits over partitioned vocabulary.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Use column parallelism (split vocabulary dimension)
        self.lm_head = ColumnParallelLinear(
            hidden_size,
            vocab_size,
            bias=bias,
            gather_output=True,  # Gather full vocabulary for sampling
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits over vocabulary.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            logits: [batch, seq, vocab_size]
        """
        # Column parallel projection with gathering
        logits = self.lm_head(hidden_states)

        return logits


# Alternative: Keep logits partitioned for efficient sampling
class ParallelLMHeadWithPartitionedLogits(nn.Module):
    """
    LM head that keeps logits partitioned until sampling.

    More memory efficient for large vocabularies.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()

        self.lm_head = ColumnParallelLinear(
            hidden_size,
            vocab_size,
            bias=False,
            gather_output=False,  # Keep partitioned
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute partial logits (no communication)
        partial_logits = self.lm_head(hidden_states)

        return partial_logits

    def sample(self, logits: torch.Tensor, temperature: float = 1.0):
        """
        Sample from partitioned logits.

        Each GPU computes softmax over its vocabulary partition,
        then collaborates to sample globally.
        """
        # Local softmax over partition
        local_probs = F.softmax(logits / temperature, dim=-1)

        # Gather probabilities from all partitions
        all_probs = tensor_model_parallel_all_gather(local_probs)

        # Sample from full distribution
        sampled_tokens = torch.multinomial(all_probs, num_samples=1)

        return sampled_tokens
```

---

## vLLM's Megatron Implementation {#vllm-implementation}

### Model Architecture Registration

```python
# From vllm/model_executor/models/llama.py

class LlamaForCausalLM(nn.Module):
    """
    Llama model with Megatron-style tensor parallelism.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layer (may use vocabulary parallelism for large vocabs)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # LM head (vocabulary parallelism)
        self.lm_head = ParallelLMHead(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        # Embedding lookup
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor:
        # LM head projection
        logits = self.lm_head(hidden_states)

        return logits
```

### Checkpoint Loading

```python
def load_megatron_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    tensor_parallel_size: int,
):
    """
    Load Megatron-style checkpoint with proper weight distribution.

    Megatron checkpoints store weights in partitioned format.
    Each TP rank loads its corresponding partition.
    """
    tp_rank = get_tensor_model_parallel_rank()

    # Load checkpoint for this TP rank
    checkpoint_file = f"{checkpoint_path}/mp_rank_{tp_rank:02d}/model.pt"
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    # Load state dict
    model.load_state_dict(checkpoint['model'], strict=True)

    print(f"Loaded checkpoint for TP rank {tp_rank}/{tensor_parallel_size}")


def convert_hf_to_megatron(
    hf_model_path: str,
    output_path: str,
    tensor_parallel_size: int,
):
    """
    Convert HuggingFace checkpoint to Megatron format.

    Splits weights appropriately for tensor parallelism.
    """
    from transformers import AutoModelForCausalLM

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    hf_state_dict = hf_model.state_dict()

    # Create partitioned checkpoints
    for tp_rank in range(tensor_parallel_size):
        megatron_state_dict = {}

        for name, param in hf_state_dict.items():
            if 'qkv_proj' in name or 'gate_up_proj' in name:
                # Column parallel: split along output dimension
                split_size = param.shape[0] // tensor_parallel_size
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                megatron_state_dict[name] = param[start_idx:end_idx].clone()

            elif 'o_proj' in name or 'down_proj' in name:
                # Row parallel: split along input dimension
                split_size = param.shape[1] // tensor_parallel_size
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                megatron_state_dict[name] = param[:, start_idx:end_idx].clone()

            else:
                # Replicated parameters (layer norms, etc.)
                megatron_state_dict[name] = param.clone()

        # Save partition
        output_file = f"{output_path}/mp_rank_{tp_rank:02d}/model.pt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save({'model': megatron_state_dict}, output_file)

        print(f"Saved partition {tp_rank}/{tensor_parallel_size}")
```

---

## Weight Initialization and Loading {#weight-loading}

### Parallel Weight Initialization

```python
def initialize_megatron_weights(
    model: nn.Module,
    init_method: str = 'normal',
    init_std: float = 0.02,
):
    """
    Initialize weights for Megatron model.

    Important: Must maintain same initialization across all TP ranks
    for mathematically equivalent results.
    """

    # Set same random seed across all TP ranks
    # This ensures replicated weights are identical
    base_seed = 12345
    torch.manual_seed(base_seed)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name:
                if init_method == 'normal':
                    # Normal initialization
                    nn.init.normal_(param, mean=0.0, std=init_std)
                elif init_method == 'xavier':
                    # Xavier initialization
                    nn.init.xavier_uniform_(param)

                # For partitioned parameters, adjust std
                if isinstance(param, nn.Parameter) and hasattr(param, 'partition_dim'):
                    param.data *= math.sqrt(get_tensor_model_parallel_world_size())

            elif 'bias' in name:
                # Zero initialize biases
                nn.init.zeros_(param)
```

---

## Performance Optimization {#performance-optimization}

### Fused Operations

```python
class FusedMegatronMLP(nn.Module):
    """
    Fused MLP for better performance.

    Combines gate and up projections into single GEMM.
    Reduces kernel launch overhead.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        # Single fused projection (2x intermediate for gate and up)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single GEMM for both gate and up (more efficient)
        gate_up = self.gate_up_proj(x)

        # Split and apply activation in single fused kernel
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up

        # Down projection
        return self.down_proj(intermediate)


# Benchmark fused vs unfused
def benchmark_mlp_fusion():
    hidden = 8192
    intermediate = 28672
    batch_size = 32
    seq_len = 2048

    x = torch.randn(batch_size, seq_len, hidden, device='cuda')

    # Unfused version
    gate_proj = ColumnParallelLinear(hidden, intermediate)
    up_proj = ColumnParallelLinear(hidden, intermediate)
    down_proj = RowParallelLinear(intermediate, hidden)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        gate = gate_proj(x)
        up = up_proj(x)
        intermediate = F.silu(gate) * up
        out = down_proj(intermediate)
    torch.cuda.synchronize()
    unfused_time = time.time() - start

    # Fused version
    fused_mlp = FusedMegatronMLP(hidden, intermediate)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = fused_mlp(x)
    torch.cuda.synchronize()
    fused_time = time.time() - start

    print(f"Unfused: {unfused_time:.3f}s")
    print(f"Fused: {fused_time:.3f}s")
    print(f"Speedup: {unfused_time / fused_time:.2f}x")
```

---

## Best Practices {#best-practices}

### 1. Validate Numerical Equivalence

```python
def validate_megatron_equivalence():
    """
    Verify that Megatron TP produces same results as single-GPU.
    """
    import torch.distributed as dist

    hidden_size = 4096
    batch_size = 4
    seq_len = 128

    # Single-GPU reference
    if dist.get_rank() == 0:
        reference_model = TransformerLayer(hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        with torch.no_grad():
            reference_output = reference_model(x)
    else:
        x = None
        reference_output = None

    # Broadcast input to all ranks
    x = broadcast_tensor(x, src=0)

    # Megatron model
    megatron_model = MegatronTransformerLayer(hidden_size)
    with torch.no_grad():
        megatron_output = megatron_model(x)

    # Compare
    if dist.get_rank() == 0:
        max_diff = torch.max(torch.abs(reference_output - megatron_output))
        print(f"Max difference: {max_diff.item():.2e}")
        assert max_diff < 1e-5, "Outputs don't match!"
```

### 2. Optimize for Your Hardware

```python
def configure_for_hardware(num_gpus: int, gpu_type: str):
    """
    Configure TP settings based on hardware.
    """
    configs = {
        # NVLink configurations (high bandwidth)
        ('A100', 8): {
            'tp_size': 8,
            'use_nvlink': True,
            'nccl_algo': 'Ring',  # Ring works well with NVLink
        },
        ('A100', 4): {
            'tp_size': 4,
            'use_nvlink': True,
            'nccl_algo': 'Tree',
        },

        # PCIe configurations (lower bandwidth)
        ('A6000', 4): {
            'tp_size': 2,  # Use less TP due to slower interconnect
            'use_nvlink': False,
            'nccl_algo': 'Tree',
        },
    }

    config = configs.get((gpu_type, num_gpus))

    if config['use_nvlink']:
        os.environ['NCCL_P2P_LEVEL'] = 'NVL'

    os.environ['NCCL_ALGO'] = config['nccl_algo']

    return config
```

### 3. Monitor and Debug

```python
def monitor_megatron_health():
    """
    Monitor Megatron distributed inference health.
    """
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()

    # Check GPU memory balance
    memory = torch.cuda.max_memory_allocated() / 1e9

    # Gather from all ranks
    all_memory = [torch.tensor(0.0) for _ in range(world_size)]
    dist.all_gather(all_memory, torch.tensor(memory))

    if rank == 0:
        avg_memory = sum(all_memory) / world_size
        max_imbalance = max(abs(m - avg_memory) for m in all_memory)

        print(f"Average GPU memory: {avg_memory:.2f} GB")
        print(f"Max imbalance: {max_imbalance:.2f} GB")

        if max_imbalance > 5.0:  # >5GB imbalance
            print("WARNING: Significant memory imbalance detected!")
```

---

## Summary

Megatron-style model parallelism provides efficient distributed inference:

**Key Principles:**
1. Minimal communication (2 all-reduces per transformer layer)
2. Balanced computation across GPUs
3. Mathematically equivalent to single-GPU execution
4. Efficient memory utilization

**Implementation Highlights:**
- Column and row parallel linear layers
- Head-parallel attention mechanism
- Fused operations for efficiency
- Support for modern architectures (GQA, SwiGLU)

**Best Practices:**
- Validate numerical equivalence
- Configure for hardware characteristics
- Monitor memory balance and performance
- Use appropriate fusion and optimization techniques

Continue to the next module to learn about pipeline parallelism and combining TP with PP for even larger models.
