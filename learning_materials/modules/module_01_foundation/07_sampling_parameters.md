# Module 1.7: Sampling Parameters Deep Dive

## Learning Objectives

1. Understand all sampling parameters
2. Choose appropriate settings for use cases
3. Control randomness and creativity
4. Implement advanced sampling strategies
5. Debug unexpected generation behavior

**Estimated Time:** 1.5 hours

---

## Core Parameters

### Temperature

Controls randomness in token selection:

```python
from vllm import SamplingParams

# Greedy (deterministic)
greedy = SamplingParams(temperature=0.0, max_tokens=100)

# Conservative
conservative = SamplingParams(temperature=0.3, max_tokens=100)

# Balanced
balanced = SamplingParams(temperature=0.7, max_tokens=100)

# Creative
creative = SamplingParams(temperature=1.5, max_tokens=100)
```

**Temperature Guide:**
- 0.0: Deterministic (factual QA, code)
- 0.1-0.3: Focused (summaries)
- 0.4-0.7: Balanced (conversation)
- 0.8-1.2: Creative (writing)
- >1.3: Very random (experimental)

---

## Top-k and Top-p

### Top-k Sampling

Sample from top-k highest probability tokens:

```python
params = SamplingParams(
    temperature=1.0,
    top_k=50,  # Consider top 50 tokens
    max_tokens=100
)
```

### Top-p (Nucleus) Sampling

Sample from smallest set with cumulative probability >= p:

```python
params = SamplingParams(
    temperature=0.8,
    top_p=0.95,  # Nucleus sampling
    max_tokens=100
)
```

### Combined

```python
# Recommended: Use both together
params = SamplingParams(
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    max_tokens=100
)
```

---

## Penalties

### Presence Penalty

Penalizes tokens that appeared before:

```python
params = SamplingParams(
    presence_penalty=0.5,  # 0.0 to 2.0
    max_tokens=100
)
# Encourages diverse vocabulary
```

### Frequency Penalty

Penalizes based on frequency:

```python
params = SamplingParams(
    frequency_penalty=0.5,  # 0.0 to 2.0
    max_tokens=100
)
# Reduces repetitive tokens proportionally
```

### Repetition Penalty

Alternative penalty mechanism:

```python
params = SamplingParams(
    repetition_penalty=1.2,  # >1.0 penalizes
    max_tokens=100
)
# Multiplicative penalty on repeated tokens
```

---

## Beam Search

Explores multiple generation paths:

```python
params = SamplingParams(
    use_beam_search=True,
    best_of=5,  # Beam width
    temperature=0.0,  # Must be 0 for beam search
    length_penalty=1.0,  # >1 favors longer
    max_tokens=100
)
```

**Use Cases:**
- Translation
- Summarization
- When quality > diversity

---

## Stopping Criteria

### Stop Sequences

```python
params = SamplingParams(
    stop=["\n\n", "END", "</s>"],
    include_stop_str_in_output=False,
    max_tokens=200
)
```

### Min/Max Tokens

```python
params = SamplingParams(
    min_tokens=50,
    max_tokens=200
)
```

---

## Logprobs

Return log probabilities:

```python
params = SamplingParams(
    logprobs=5,  # Top-5 logprobs per token
    prompt_logprobs=1,  # Also for prompt
    max_tokens=50
)

outputs = llm.generate(["Hello"], params)

# Access logprobs
for token_id, logprobs_dict in zip(
    outputs[0].outputs[0].token_ids,
    outputs[0].outputs[0].logprobs
):
    print(f"Token {token_id}: {logprobs_dict}")
```

---

## Recommended Presets

### Factual/Deterministic

```python
factual = SamplingParams(
    temperature=0.0,
    max_tokens=100
)
```

### Balanced Conversation

```python
conversation = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    presence_penalty=0.6,
    frequency_penalty=0.3,
    max_tokens=150
)
```

### Creative Writing

```python
creative = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    presence_penalty=0.4,
    max_tokens=500
)
```

### Code Generation

```python
code = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    stop=["\n\n", "```"],
    max_tokens=200
)
```

### Translation

```python
translation = SamplingParams(
    use_beam_search=True,
    best_of=4,
    temperature=0.0,
    length_penalty=1.0,
    max_tokens=200
)
```

---

## Practical Exercises

### Exercise 1.7.1: Temperature Exploration
1. Generate with T=0.0, 0.5, 1.0, 1.5
2. Compare outputs
3. Document differences

**Time:** 20 minutes

### Exercise 1.7.2: Penalty Effects
1. Generate without penalties
2. Apply presence_penalty=0.6
3. Apply frequency_penalty=0.6
4. Compare repetition

**Time:** 20 minutes

### Exercise 1.7.3: Custom Preset
1. Design preset for use case
2. Test and iterate
3. Document choices

**Time:** 30 minutes

---

## Key Takeaways

1. **Temperature** controls randomness
2. **Top-p** adapts to confidence
3. **Penalties** reduce repetition
4. **Beam search** for quality
5. **Combine parameters** optimally
6. **Test iteratively**

---

**Next:** 08_logging_debugging.md
**Time:** 1.5 hours
**Source:** `/home/user/vllm-learn/vllm/sampling_params.py`
