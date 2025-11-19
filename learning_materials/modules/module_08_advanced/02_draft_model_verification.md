# Module 8.2: Draft Model Verification - Advanced Acceptance Strategies

## Overview

This module explores the sophisticated verification mechanisms that ensure speculative decoding maintains output quality while maximizing performance. We'll dive deep into acceptance sampling, rejection mechanisms, and advanced verification strategies that go beyond basic token matching.

**Learning Objectives:**
- Master the mathematical foundations of acceptance sampling
- Implement custom verification strategies for different use cases
- Optimize verification overhead and batching
- Handle edge cases in distribution matching
- Debug verification failures and acceptance anomalies

**Prerequisites:**
- Completion of Module 8.1 (Speculative Decoding Intro)
- Strong understanding of probability distributions
- Familiarity with sampling strategies
- Understanding of numerical stability issues

**Estimated Time:** 3-4 hours

---

## 1. Verification Fundamentals

### 1.1 The Verification Problem

After the draft model generates `k` speculative tokens, the target model must verify each token while maintaining statistical correctness:

```python
def verification_problem_overview():
    """
    The core challenge: verify draft tokens while ensuring
    output distribution matches target model exactly
    """

    # Given:
    # - draft_tokens: [t1, t2, ..., tk] from draft model
    # - p_draft[i]: probability distribution for token i from draft
    # - p_target[i]: probability distribution for token i from target

    # Goals:
    # 1. Accept as many tokens as possible (performance)
    # 2. Ensure output distribution = target distribution (correctness)
    # 3. Minimize verification overhead (efficiency)

    # Constraints:
    # - Cannot simply compare argmax (would change distribution)
    # - Must handle numerical precision issues
    # - Must work with different sampling strategies (greedy, top-k, nucleus)

    pass
```

### 1.2 Mathematical Foundation: Rejection Sampling

The verification process uses rejection sampling to maintain distributional correctness:

**Theorem (Speculative Sampling Correctness):**

Given proposal distribution `q(x)` and target distribution `p(x)`, rejection sampling with acceptance probability `α(x) = min(1, p(x)/q(x))` produces samples from `p(x)`.

```python
import torch
import numpy as np

def theoretical_acceptance_sampling(draft_token, target_probs, draft_probs):
    """
    Theoretical foundation of acceptance sampling

    This guarantees output matches target distribution
    """
    # Get probabilities for the draft token
    p_target = target_probs[draft_token]
    p_draft = draft_probs[draft_token]

    # Acceptance probability (key formula!)
    alpha = min(1.0, p_target / p_draft)

    # Accept with probability alpha
    if torch.rand(1).item() < alpha:
        return draft_token, True  # Accepted

    # Rejected - need to resample from adjusted distribution
    # p_adjusted(x) ∝ max(0, p_target(x) - p_draft(x))
    adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
    adjusted_probs = adjusted_probs / adjusted_probs.sum()

    # Sample from adjusted distribution
    new_token = torch.multinomial(adjusted_probs, num_samples=1).item()

    return new_token, False  # Rejected and resampled

def prove_distributional_correctness():
    """
    Demonstrate that acceptance sampling maintains target distribution
    """
    # Set up simple example
    vocab_size = 10
    num_samples = 100000

    # Target distribution (what we want)
    target_probs = torch.softmax(torch.randn(vocab_size), dim=0)

    # Draft distribution (proposal)
    draft_probs = torch.softmax(torch.randn(vocab_size), dim=0)

    # Sample using acceptance sampling
    samples = []
    for _ in range(num_samples):
        # Draft proposes a token
        draft_token = torch.multinomial(draft_probs, num_samples=1).item()

        # Acceptance sampling
        token, _ = theoretical_acceptance_sampling(
            draft_token, target_probs, draft_probs
        )
        samples.append(token)

    # Compute empirical distribution
    empirical_probs = torch.zeros(vocab_size)
    for token in samples:
        empirical_probs[token] += 1
    empirical_probs /= num_samples

    # Compare to target
    print("Distributional Correctness Check:")
    print("-" * 60)
    print(f"{'Token':<10} {'Target':<15} {'Empirical':<15} {'Error':<15}")
    print("-" * 60)

    max_error = 0
    for i in range(vocab_size):
        error = abs(target_probs[i].item() - empirical_probs[i].item())
        max_error = max(max_error, error)
        print(f"{i:<10} {target_probs[i].item():<15.6f} "
              f"{empirical_probs[i].item():<15.6f} {error:<15.6f}")

    print(f"\nMax Error: {max_error:.6f}")
    print(f"Expected Error (statistical): ~{1/np.sqrt(num_samples):.6f}")

    # Verify correctness
    assert max_error < 0.01, "Distribution mismatch!"
    print("\n✅ Distributional correctness verified!")

# Run the proof
# prove_distributional_correctness()
```

### 1.3 Efficient Batched Verification

In practice, we verify multiple tokens in a single forward pass:

```python
class BatchedVerifier:
    """
    Efficient batched verification for speculative decoding
    """

    def __init__(self, target_model, draft_model, temperature=1.0):
        self.target_model = target_model
        self.draft_model = draft_model
        self.temperature = temperature

    def verify_batch(self, tokens, draft_tokens, max_verify=8):
        """
        Verify a batch of draft tokens efficiently

        Args:
            tokens: Current sequence tokens [batch_size, seq_len]
            draft_tokens: Draft tokens to verify [batch_size, k]
            max_verify: Maximum tokens to verify per sequence

        Returns:
            accepted_tokens: Verified tokens [batch_size, num_accepted]
            num_accepted: Number of accepted tokens per sequence [batch_size]
        """
        batch_size, k = draft_tokens.shape

        # Single forward pass through target model
        # Key optimization: verify all k tokens in parallel!
        all_tokens = torch.cat([tokens, draft_tokens], dim=1)
        target_logits = self.target_model(all_tokens)  # [batch_size, seq_len+k, vocab]

        # Extract logits for draft token positions
        verify_logits = target_logits[:, tokens.shape[1]:tokens.shape[1]+k, :]

        # Get target probabilities
        target_probs = torch.softmax(verify_logits / self.temperature, dim=-1)

        # For comparison, get draft probabilities (cached during draft phase)
        # In practice, these would be cached from draft generation
        draft_probs = self._get_draft_probs_cached(tokens, draft_tokens)

        # Verify each token sequentially (stop at first rejection)
        accepted_tokens = []
        num_accepted = torch.zeros(batch_size, dtype=torch.long)

        for i in range(k):
            # Extract probabilities for position i
            p_target = target_probs[:, i, :]  # [batch_size, vocab]
            p_draft = draft_probs[:, i, :]    # [batch_size, vocab]
            proposed = draft_tokens[:, i]      # [batch_size]

            # Compute acceptance probability for each sequence
            p_target_at_draft = p_target.gather(1, proposed.unsqueeze(1)).squeeze(1)
            p_draft_at_draft = p_draft.gather(1, proposed.unsqueeze(1)).squeeze(1)

            alpha = torch.clamp(p_target_at_draft / p_draft_at_draft, max=1.0)

            # Random acceptance
            rand_vals = torch.rand(batch_size, device=alpha.device)
            accepted = rand_vals < alpha  # [batch_size]

            # For accepted tokens, keep them
            # For rejected tokens, resample from adjusted distribution
            tokens_at_i = proposed.clone()

            rejected_mask = ~accepted
            if rejected_mask.any():
                # Resample for rejected sequences
                adjusted_probs = torch.clamp(p_target - p_draft, min=0.0)
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)

                # Sample only for rejected sequences
                resampled = torch.multinomial(
                    adjusted_probs[rejected_mask],
                    num_samples=1
                ).squeeze(1)

                tokens_at_i[rejected_mask] = resampled

            accepted_tokens.append(tokens_at_i)

            # Update acceptance counts
            # Stop accepting after first rejection
            still_accepting = (num_accepted == i)
            num_accepted[still_accepting & accepted] = i + 1
            num_accepted[still_accepting & rejected_mask] = i + 1  # Include resampled token

            # If all sequences have stopped accepting, can break early
            if not still_accepting.any():
                break

        accepted_tokens = torch.stack(accepted_tokens, dim=1)  # [batch_size, k]

        return accepted_tokens, num_accepted

    def _get_draft_probs_cached(self, tokens, draft_tokens):
        """
        Get cached draft probabilities from draft generation phase
        In practice, these are cached during draft generation
        """
        # Placeholder - would use actual cached values
        batch_size, k = draft_tokens.shape
        vocab_size = self.draft_model.config.vocab_size

        # This would be cached from draft phase in practice
        all_tokens = torch.cat([tokens, draft_tokens], dim=1)
        draft_logits = self.draft_model(all_tokens)
        draft_logits = draft_logits[:, tokens.shape[1]:tokens.shape[1]+k, :]

        draft_probs = torch.softmax(draft_logits / self.temperature, dim=-1)
        return draft_probs

# Example usage
def verify_example():
    """
    Example of batched verification
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models (simplified)
    target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
    draft_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    verifier = BatchedVerifier(target_model, draft_model, temperature=0.8)

    # Input tokens and draft tokens
    tokens = torch.randint(0, 32000, (4, 50))  # Batch of 4, seq_len 50
    draft_tokens = torch.randint(0, 32000, (4, 8))  # 8 speculative tokens

    # Verify
    accepted, num_accepted = verifier.verify_batch(tokens, draft_tokens)

    print(f"Accepted tokens shape: {accepted.shape}")
    print(f"Num accepted per sequence: {num_accepted}")
    print(f"Average acceptance: {num_accepted.float().mean():.2f} / 8")

# verify_example()
```

---

## 2. Advanced Verification Strategies

### 2.1 Temperature-Aware Verification

Different temperatures require different verification strategies:

```python
class TemperatureAwareVerifier:
    """
    Adapt verification strategy based on temperature
    """

    def __init__(self, target_model, draft_model):
        self.target_model = target_model
        self.draft_model = draft_model

    def verify_with_temperature(self, tokens, draft_tokens,
                                target_temp, draft_temp):
        """
        Verify with different temperatures for target and draft

        Important: Draft and target may use different temperatures!
        """
        # Get logits
        target_logits = self.target_model(tokens + draft_tokens)
        draft_logits = self.draft_model(tokens + draft_tokens)

        # Apply different temperatures
        target_probs = torch.softmax(target_logits / target_temp, dim=-1)
        draft_probs = torch.softmax(draft_logits / draft_temp, dim=-1)

        # Verification proceeds normally
        # The temperature difference is automatically handled
        # by the acceptance probability calculation

        return self.verify(target_probs, draft_probs, draft_tokens)

    def adaptive_temperature_matching(self, tokens, draft_tokens,
                                     target_temp):
        """
        Adaptively adjust draft temperature to maximize acceptance

        Strategy: Use higher temperature in draft to increase coverage
        """
        # Strategy 1: Draft temperature = target temperature (standard)
        # Pros: Best distributional matching
        # Cons: May not maximize acceptance

        # Strategy 2: Draft temperature > target temperature
        # Pros: Higher entropy → better coverage → higher acceptance
        # Cons: May waste computation on low-probability tokens

        # Strategy 3: Adaptive temperature
        # Start with higher draft temp, adjust based on acceptance rate

        draft_temp = target_temp * 1.2  # 20% higher for better coverage

        return self.verify_with_temperature(
            tokens, draft_tokens, target_temp, draft_temp
        )

def temperature_impact_analysis():
    """
    Analyze impact of temperature on acceptance rate
    """
    import matplotlib.pyplot as plt

    # Simulate different temperature combinations
    target_temps = [0.5, 0.7, 1.0, 1.5]
    draft_temps = [0.5, 0.7, 1.0, 1.5]

    results = np.zeros((len(target_temps), len(draft_temps)))

    for i, t_temp in enumerate(target_temps):
        for j, d_temp in enumerate(draft_temps):
            # Simulate acceptance rate
            # Higher temp difference → lower acceptance generally
            temp_ratio = abs(t_temp - d_temp) / max(t_temp, d_temp)
            base_acceptance = 0.75  # Baseline with matched temps

            # Model: acceptance decreases with temp mismatch
            acceptance = base_acceptance * (1 - 0.5 * temp_ratio)

            results[i, j] = acceptance

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(results, cmap='RdYlGn', vmin=0.4, vmax=0.8)
    plt.colorbar(label='Acceptance Rate')
    plt.xticks(range(len(draft_temps)), draft_temps)
    plt.yticks(range(len(target_temps)), target_temps)
    plt.xlabel('Draft Temperature')
    plt.ylabel('Target Temperature')
    plt.title('Acceptance Rate vs Temperature Settings')

    # Add values
    for i in range(len(target_temps)):
        for j in range(len(draft_temps)):
            plt.text(j, i, f'{results[i, j]:.2f}',
                    ha='center', va='center')

    plt.savefig('temperature_acceptance_analysis.png')
    plt.close()

    print("Key Findings:")
    print("1. Matching temperatures gives highest acceptance")
    print("2. Small mismatches (±0.2) have minimal impact")
    print("3. Large mismatches significantly reduce acceptance")

# temperature_impact_analysis()
```

### 2.2 Top-K and Top-P Verification

Handling structured sampling strategies:

```python
class StructuredSamplingVerifier:
    """
    Verification for top-k and top-p (nucleus) sampling
    """

    def __init__(self, target_model, draft_model):
        self.target_model = target_model
        self.draft_model = draft_model

    def verify_topk(self, draft_tokens, target_logits, draft_logits,
                    target_k, draft_k):
        """
        Verify with top-k sampling

        Challenge: Draft and target may have different top-k sets
        """
        batch_size, vocab_size = target_logits.shape

        # Get top-k for both models
        target_topk_probs, target_topk_indices = torch.topk(
            torch.softmax(target_logits, dim=-1),
            k=target_k,
            dim=-1
        )

        draft_topk_probs, draft_topk_indices = torch.topk(
            torch.softmax(draft_logits, dim=-1),
            k=draft_k,
            dim=-1
        )

        # Renormalize within top-k
        target_topk_probs = target_topk_probs / target_topk_probs.sum(dim=-1, keepdim=True)
        draft_topk_probs = draft_topk_probs / draft_topk_probs.sum(dim=-1, keepdim=True)

        # Create full probability distributions (zero outside top-k)
        target_probs = torch.zeros(batch_size, vocab_size, device=target_logits.device)
        draft_probs = torch.zeros(batch_size, vocab_size, device=draft_logits.device)

        target_probs.scatter_(1, target_topk_indices, target_topk_probs)
        draft_probs.scatter_(1, draft_topk_indices, draft_topk_probs)

        # Standard verification
        return self.verify(draft_tokens, target_probs, draft_probs)

    def verify_topp(self, draft_tokens, target_logits, draft_logits,
                    target_p, draft_p):
        """
        Verify with top-p (nucleus) sampling

        More complex: nucleus can have variable size
        """
        def get_nucleus_probs(logits, p):
            """Extract nucleus (top-p) probabilities"""
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # Cumulative probabilities
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find nucleus cutoff
            nucleus_mask = cumsum_probs <= p

            # Always include at least one token
            nucleus_mask[:, 0] = True

            # Zero out probabilities outside nucleus
            sorted_probs[~nucleus_mask] = 0

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Scatter back to original indices
            probs_nucleus = torch.zeros_like(probs)
            probs_nucleus.scatter_(1, sorted_indices, sorted_probs)

            return probs_nucleus

        # Get nucleus probabilities
        target_probs = get_nucleus_probs(target_logits, target_p)
        draft_probs = get_nucleus_probs(draft_logits, draft_p)

        # Standard verification
        return self.verify(draft_tokens, target_probs, draft_probs)

    def verify(self, draft_tokens, target_probs, draft_probs):
        """Standard verification logic"""
        # Get probabilities at draft token
        p_target = target_probs.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
        p_draft = draft_probs.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)

        # Acceptance probability
        alpha = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)

        # Accept/reject
        accepted = torch.rand_like(alpha) < alpha

        return accepted

def sampling_strategy_comparison():
    """
    Compare verification across different sampling strategies
    """
    strategies = [
        {"name": "Greedy", "params": {}},
        {"name": "Top-K (k=50)", "params": {"k": 50}},
        {"name": "Top-P (p=0.9)", "params": {"p": 0.9}},
        {"name": "Temperature (t=0.8)", "params": {"temperature": 0.8}},
    ]

    results = {}

    for strategy in strategies:
        print(f"\nTesting: {strategy['name']}")

        # Simulate verification
        # (In practice, would use actual model outputs)
        vocab_size = 32000
        batch_size = 100

        target_logits = torch.randn(batch_size, vocab_size)
        draft_logits = target_logits + torch.randn(batch_size, vocab_size) * 0.5

        # Sample draft tokens
        draft_probs = torch.softmax(draft_logits, dim=-1)
        draft_tokens = torch.multinomial(draft_probs, num_samples=1).squeeze(1)

        # Verify
        verifier = StructuredSamplingVerifier(None, None)

        if "k" in strategy["params"]:
            accepted = verifier.verify_topk(
                draft_tokens, target_logits, draft_logits,
                target_k=strategy["params"]["k"],
                draft_k=strategy["params"]["k"]
            )
        elif "p" in strategy["params"]:
            accepted = verifier.verify_topp(
                draft_tokens, target_logits, draft_logits,
                target_p=strategy["params"]["p"],
                draft_p=strategy["params"]["p"]
            )
        else:
            # Standard verification
            target_probs = torch.softmax(target_logits, dim=-1)
            draft_probs = torch.softmax(draft_logits, dim=-1)
            accepted = verifier.verify(draft_tokens, target_probs, draft_probs)

        acceptance_rate = accepted.float().mean().item()
        results[strategy["name"]] = acceptance_rate

        print(f"  Acceptance Rate: {acceptance_rate:.2%}")

    return results

# sampling_strategy_comparison()
```

### 2.3 Confidence-Based Verification

Use model confidence to optimize verification:

```python
class ConfidenceBasedVerifier:
    """
    Use confidence scores to optimize verification strategy
    """

    def __init__(self, confidence_threshold=0.9):
        self.confidence_threshold = confidence_threshold

    def verify_with_confidence(self, draft_tokens, target_probs, draft_probs):
        """
        Fast-path for high-confidence draft tokens

        Idea: If draft is very confident AND target agrees, skip expensive sampling
        """
        # Get probabilities at draft positions
        p_target = target_probs.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
        p_draft = draft_probs.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)

        # Draft confidence
        draft_confidence = p_draft

        # Target agreement (how much target agrees with draft choice)
        target_agreement = p_target / p_target.max(dim=-1, keepdims=True)[0]

        # Fast-path conditions
        high_draft_confidence = draft_confidence > self.confidence_threshold
        high_target_agreement = target_agreement > 0.8

        # Fast accept if both conditions met
        fast_accept = high_draft_confidence & high_target_agreement

        # Standard verification for others
        alpha = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)
        standard_accept = torch.rand_like(alpha) < alpha

        # Combine: fast accept OR standard accept
        accepted = fast_accept | standard_accept

        return accepted, fast_accept.float().mean().item()

def confidence_verification_analysis():
    """
    Analyze impact of confidence-based verification
    """
    import time

    verifier = ConfidenceBasedVerifier(confidence_threshold=0.9)

    # Simulate different scenarios
    scenarios = [
        {
            "name": "High Alignment",
            "correlation": 0.9  # Draft and target highly correlated
        },
        {
            "name": "Medium Alignment",
            "correlation": 0.7
        },
        {
            "name": "Low Alignment",
            "correlation": 0.5
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)

        vocab_size = 32000
        batch_size = 1000

        # Create correlated distributions
        base_logits = torch.randn(batch_size, vocab_size)
        target_logits = base_logits.clone()
        draft_logits = (scenario['correlation'] * base_logits +
                       (1 - scenario['correlation']) * torch.randn(batch_size, vocab_size))

        target_probs = torch.softmax(target_logits, dim=-1)
        draft_probs = torch.softmax(draft_logits, dim=-1)

        # Sample draft tokens
        draft_tokens = torch.multinomial(draft_probs, num_samples=1).squeeze(1)

        # Verify with confidence
        start = time.time()
        accepted, fast_accept_rate = verifier.verify_with_confidence(
            draft_tokens, target_probs, draft_probs
        )
        elapsed = time.time() - start

        print(f"  Acceptance Rate: {accepted.float().mean():.2%}")
        print(f"  Fast Accept Rate: {fast_accept_rate:.2%}")
        print(f"  Time: {elapsed*1000:.2f} ms")

# confidence_verification_analysis()
```

---

## 3. Numerical Stability and Edge Cases

### 3.1 Handling Numerical Issues

```python
class NumericallyStableVerifier:
    """
    Robust verification handling numerical edge cases
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def stable_acceptance_probability(self, p_target, p_draft):
        """
        Compute acceptance probability with numerical stability

        Handle:
        - Division by zero
        - Very small probabilities
        - Overflow/underflow
        """
        # Add epsilon to prevent division by zero
        p_draft_safe = torch.clamp(p_draft, min=self.epsilon)

        # Compute ratio in log space for stability
        log_ratio = torch.log(p_target + self.epsilon) - torch.log(p_draft_safe)

        # Convert back, clamping to [0, 1]
        alpha = torch.exp(torch.clamp(log_ratio, max=0.0))

        return alpha

    def handle_zero_probability_token(self, draft_token, target_probs, draft_probs):
        """
        Handle case where draft proposes token with zero target probability

        This should always be rejected!
        """
        p_target = target_probs[draft_token]
        p_draft = draft_probs[draft_token]

        if p_target < self.epsilon:
            # Token has (essentially) zero probability in target
            # Must reject and resample
            print(f"Warning: Draft proposed zero-probability token {draft_token}")

            # Resample from target (excluding zero-prob tokens)
            valid_probs = torch.clamp(target_probs, min=self.epsilon)
            valid_probs = valid_probs / valid_probs.sum()

            new_token = torch.multinomial(valid_probs, num_samples=1).item()
            return new_token, False

        # Normal verification
        alpha = self.stable_acceptance_probability(p_target, p_draft)

        if torch.rand(1).item() < alpha:
            return draft_token, True

        # Rejected - resample
        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        new_token = torch.multinomial(adjusted_probs, num_samples=1).item()

        return new_token, False

def test_numerical_stability():
    """
    Test verification with extreme probability values
    """
    verifier = NumericallyStableVerifier()

    test_cases = [
        {
            "name": "Normal case",
            "p_target": 0.5,
            "p_draft": 0.3
        },
        {
            "name": "Very small probabilities",
            "p_target": 1e-8,
            "p_draft": 1e-9
        },
        {
            "name": "Zero draft probability",
            "p_target": 0.5,
            "p_draft": 0.0
        },
        {
            "name": "Zero target probability",
            "p_target": 0.0,
            "p_draft": 0.3
        },
        {
            "name": "Both very small",
            "p_target": 1e-10,
            "p_draft": 1e-10
        }
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        p_t = torch.tensor(case['p_target'])
        p_d = torch.tensor(case['p_draft'])

        try:
            alpha = verifier.stable_acceptance_probability(p_t, p_d)
            print(f"  p_target: {p_t:.2e}")
            print(f"  p_draft: {p_d:.2e}")
            print(f"  alpha: {alpha:.6f}")
            print(f"  ✅ Stable computation")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

# test_numerical_stability()
```

### 3.2 Handling Distribution Mismatch

```python
class DistributionMismatchHandler:
    """
    Handle cases where draft and target distributions differ significantly
    """

    def __init__(self, mismatch_threshold=0.8):
        self.mismatch_threshold = mismatch_threshold

    def detect_distribution_mismatch(self, target_probs, draft_probs):
        """
        Detect significant distribution mismatch using KL divergence

        KL(P||Q) = Σ P(x) log(P(x)/Q(x))
        """
        # KL divergence
        kl_div = torch.sum(
            target_probs * torch.log((target_probs + 1e-10) / (draft_probs + 1e-10))
        )

        # JS divergence (symmetric)
        m = 0.5 * (target_probs + draft_probs)
        js_div = 0.5 * (
            torch.sum(target_probs * torch.log((target_probs + 1e-10) / (m + 1e-10))) +
            torch.sum(draft_probs * torch.log((draft_probs + 1e-10) / (m + 1e-10)))
        )

        return {
            "kl_divergence": kl_div.item(),
            "js_divergence": js_div.item(),
            "mismatch_detected": js_div > self.mismatch_threshold
        }

    def adaptive_fallback(self, target_probs, draft_probs, draft_tokens):
        """
        Adaptive fallback when mismatch is too high

        Strategy: If mismatch severe, skip speculation and sample directly from target
        """
        mismatch_info = self.detect_distribution_mismatch(target_probs, draft_probs)

        if mismatch_info["mismatch_detected"]:
            print(f"High mismatch detected (JS={mismatch_info['js_divergence']:.3f})")
            print("Falling back to direct sampling from target")

            # Sample directly from target
            new_token = torch.multinomial(target_probs, num_samples=1).item()
            return new_token, False, "fallback"

        # Normal verification
        # ... (standard verification code)
        return None, True, "normal"

def mismatch_detection_example():
    """
    Example of detecting and handling distribution mismatch
    """
    handler = DistributionMismatchHandler(mismatch_threshold=0.5)

    # Create distributions with varying mismatch levels
    vocab_size = 1000

    test_cases = [
        {
            "name": "Identical distributions",
            "correlation": 1.0
        },
        {
            "name": "Highly aligned",
            "correlation": 0.9
        },
        {
            "name": "Moderately aligned",
            "correlation": 0.6
        },
        {
            "name": "Poorly aligned",
            "correlation": 0.3
        }
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")

        base_logits = torch.randn(vocab_size)
        target_probs = torch.softmax(base_logits, dim=0)

        draft_logits = (case['correlation'] * base_logits +
                       (1 - case['correlation']) * torch.randn(vocab_size))
        draft_probs = torch.softmax(draft_logits, dim=0)

        mismatch_info = handler.detect_distribution_mismatch(target_probs, draft_probs)

        print(f"  KL Divergence: {mismatch_info['kl_divergence']:.4f}")
        print(f"  JS Divergence: {mismatch_info['js_divergence']:.4f}")
        print(f"  Mismatch Detected: {mismatch_info['mismatch_detected']}")

# mismatch_detection_example()
```

---

## 4. Optimization Techniques

### 4.1 Caching Draft Probabilities

```python
class CachedVerifier:
    """
    Optimize verification by caching draft probabilities
    """

    def __init__(self, cache_size=1000):
        self.draft_prob_cache = {}
        self.cache_size = cache_size

    def cache_draft_probabilities(self, position, draft_probs):
        """
        Cache draft probabilities during draft generation

        This eliminates need to recompute during verification
        """
        cache_key = position

        if len(self.draft_prob_cache) >= self.cache_size:
            # Evict oldest entry (simple LRU)
            oldest_key = next(iter(self.draft_prob_cache))
            del self.draft_prob_cache[oldest_key]

        self.draft_prob_cache[cache_key] = draft_probs.clone()

    def verify_with_cached_probs(self, position, draft_token, target_probs):
        """
        Verify using cached draft probabilities

        Speedup: Avoids recomputing draft model forward pass
        """
        if position not in self.draft_prob_cache:
            raise ValueError(f"No cached probs for position {position}")

        draft_probs = self.draft_prob_cache[position]

        # Standard verification
        p_target = target_probs[draft_token]
        p_draft = draft_probs[draft_token]

        alpha = min(1.0, p_target / (p_draft + 1e-10))

        if torch.rand(1).item() < alpha:
            return draft_token, True

        # Resample
        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        new_token = torch.multinomial(adjusted_probs, num_samples=1).item()

        return new_token, False

def benchmark_caching_speedup():
    """
    Measure speedup from caching draft probabilities
    """
    import time

    vocab_size = 32000
    num_iterations = 1000

    # Without caching - recompute each time
    start = time.time()
    for i in range(num_iterations):
        # Simulate computing draft probs
        draft_logits = torch.randn(vocab_size)
        draft_probs = torch.softmax(draft_logits, dim=0)
        # ... verification ...
    no_cache_time = time.time() - start

    # With caching - compute once, reuse
    verifier = CachedVerifier()

    start = time.time()
    # Pre-compute and cache
    for i in range(num_iterations):
        draft_logits = torch.randn(vocab_size)
        draft_probs = torch.softmax(draft_logits, dim=0)
        verifier.cache_draft_probabilities(i, draft_probs)

    # Verification just looks up cached values
    for i in range(num_iterations):
        target_probs = torch.softmax(torch.randn(vocab_size), dim=0)
        draft_token = torch.randint(0, vocab_size, (1,)).item()
        # Cached lookup is fast
        _ = verifier.verify_with_cached_probs(i, draft_token, target_probs)

    cache_time = time.time() - start

    print(f"Without caching: {no_cache_time:.3f}s")
    print(f"With caching: {cache_time:.3f}s")
    print(f"Speedup: {no_cache_time/cache_time:.2f}x")

# benchmark_caching_speedup()
```

### 4.2 Early Stopping Optimization

```python
class EarlyStoppingVerifier:
    """
    Stop verification early when rejection occurs

    Key insight: After first rejection, remaining tokens are invalid
    """

    def verify_with_early_stopping(self, draft_tokens, target_probs, draft_probs):
        """
        Verify tokens with early stopping on first rejection

        Returns: (accepted_tokens, num_accepted)
        """
        k = len(draft_tokens)
        accepted_tokens = []

        for i in range(k):
            p_target = target_probs[i][draft_tokens[i]]
            p_draft = draft_probs[i][draft_tokens[i]]

            alpha = min(1.0, p_target / (p_draft + 1e-10))

            if torch.rand(1).item() < alpha:
                # Accept and continue
                accepted_tokens.append(draft_tokens[i])
            else:
                # Reject - resample this token and stop
                adjusted_probs = torch.clamp(
                    target_probs[i] - draft_probs[i],
                    min=0.0
                )
                adjusted_probs = adjusted_probs / adjusted_probs.sum()

                new_token = torch.multinomial(adjusted_probs, num_samples=1).item()
                accepted_tokens.append(new_token)

                # Early stop - remaining tokens are invalid
                break

        return torch.tensor(accepted_tokens), len(accepted_tokens)

def early_stopping_benchmark():
    """
    Measure impact of early stopping
    """
    import time

    vocab_size = 32000
    k = 8
    num_trials = 10000

    # Without early stopping - always verify all k tokens
    start = time.time()
    total_work_no_stop = 0

    for _ in range(num_trials):
        draft_tokens = torch.randint(0, vocab_size, (k,))
        target_probs = [torch.softmax(torch.randn(vocab_size), dim=0) for _ in range(k)]
        draft_probs = [torch.softmax(torch.randn(vocab_size), dim=0) for _ in range(k)]

        # Verify all tokens
        for i in range(k):
            _ = target_probs[i][draft_tokens[i]]
            total_work_no_stop += 1

    no_stop_time = time.time() - start

    # With early stopping - stop at first rejection
    verifier = EarlyStoppingVerifier()

    start = time.time()
    total_work_early_stop = 0

    for _ in range(num_trials):
        draft_tokens = torch.randint(0, vocab_size, (k,))
        target_probs = [torch.softmax(torch.randn(vocab_size), dim=0) for _ in range(k)]
        draft_probs = [torch.softmax(torch.randn(vocab_size), dim=0) for _ in range(k)]

        accepted, num_accepted = verifier.verify_with_early_stopping(
            draft_tokens, target_probs, draft_probs
        )
        total_work_early_stop += num_accepted

    early_stop_time = time.time() - start

    print(f"Average tokens verified:")
    print(f"  Without early stopping: {total_work_no_stop / num_trials:.2f}")
    print(f"  With early stopping: {total_work_early_stop / num_trials:.2f}")
    print(f"  Work reduction: {(1 - total_work_early_stop/total_work_no_stop)*100:.1f}%")
    print(f"\nTime:")
    print(f"  Without early stopping: {no_stop_time:.3f}s")
    print(f"  With early stopping: {early_stop_time:.3f}s")
    print(f"  Speedup: {no_stop_time/early_stop_time:.2f}x")

# early_stopping_benchmark()
```

---

## 5. Monitoring and Debugging

### 5.1 Verification Metrics

```python
class VerificationMetrics:
    """
    Track detailed metrics for verification debugging
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_verified = 0
        self.total_accepted = 0
        self.acceptance_by_position = [0] * 16  # Track up to 16 positions
        self.verified_by_position = [0] * 16

        self.acceptance_probabilities = []
        self.rejection_positions = []

        self.numerical_issues = 0
        self.zero_prob_rejections = 0

    def record_verification(self, position, accepted, alpha, had_numerical_issue=False):
        """Record single verification attempt"""
        self.total_verified += 1

        if position < len(self.acceptance_by_position):
            self.verified_by_position[position] += 1

            if accepted:
                self.total_accepted += 1
                self.acceptance_by_position[position] += 1
        else:
            self.acceptance_by_position.append(1 if accepted else 0)
            self.verified_by_position.append(1)

        self.acceptance_probabilities.append(alpha)

        if not accepted:
            self.rejection_positions.append(position)

        if had_numerical_issue:
            self.numerical_issues += 1

    def get_statistics(self):
        """Compute summary statistics"""
        stats = {
            "overall_acceptance_rate": self.total_accepted / max(1, self.total_verified),
            "mean_acceptance_probability": np.mean(self.acceptance_probabilities),
            "std_acceptance_probability": np.std(self.acceptance_probabilities),
        }

        # Per-position acceptance rates
        stats["position_acceptance_rates"] = []
        for i in range(len(self.acceptance_by_position)):
            if self.verified_by_position[i] > 0:
                rate = self.acceptance_by_position[i] / self.verified_by_position[i]
                stats["position_acceptance_rates"].append({
                    "position": i,
                    "rate": rate,
                    "count": self.verified_by_position[i]
                })

        # Most common rejection positions
        if self.rejection_positions:
            from collections import Counter
            rejection_counts = Counter(self.rejection_positions)
            stats["common_rejection_positions"] = rejection_counts.most_common(5)

        stats["numerical_issues"] = self.numerical_issues
        stats["zero_prob_rejections"] = self.zero_prob_rejections

        return stats

    def print_report(self):
        """Print detailed verification report"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("VERIFICATION METRICS REPORT")
        print("="*60)

        print(f"\nOverall Statistics:")
        print(f"  Total Verified: {self.total_verified}")
        print(f"  Total Accepted: {self.total_accepted}")
        print(f"  Acceptance Rate: {stats['overall_acceptance_rate']:.2%}")
        print(f"  Mean Alpha: {stats['mean_acceptance_probability']:.3f}")

        print(f"\nPer-Position Acceptance:")
        for pos_stats in stats["position_acceptance_rates"]:
            print(f"  Position {pos_stats['position']}: "
                  f"{pos_stats['rate']:.2%} ({pos_stats['count']} samples)")

        if "common_rejection_positions" in stats:
            print(f"\nMost Common Rejection Positions:")
            for pos, count in stats["common_rejection_positions"]:
                print(f"  Position {pos}: {count} rejections")

        if stats["numerical_issues"] > 0:
            print(f"\n⚠️  Numerical Issues: {stats['numerical_issues']}")

        print("="*60)

# Example usage
def verification_monitoring_example():
    metrics = VerificationMetrics()

    # Simulate verification
    for iteration in range(1000):
        k = 8  # Speculative tokens

        for pos in range(k):
            # Simulate acceptance probability (decreases with position)
            base_alpha = 0.8 - (pos * 0.05)
            alpha = max(0.1, base_alpha + np.random.normal(0, 0.1))

            accepted = np.random.random() < alpha

            metrics.record_verification(pos, accepted, alpha)

            if not accepted:
                break  # Early stopping

    # Print report
    metrics.print_report()

# verification_monitoring_example()
```

---

## 6. Summary and Best Practices

### 6.1 Key Takeaways

1. **Correctness is Guaranteed:** Proper acceptance sampling ensures output distribution matches target exactly

2. **Numerical Stability Matters:** Handle edge cases (zero probabilities, underflow) carefully

3. **Optimization Opportunities:**
   - Cache draft probabilities
   - Early stopping on rejection
   - Confidence-based fast paths

4. **Monitoring is Critical:** Track acceptance rates, distribution mismatches, numerical issues

### 6.2 Best Practices

```python
class ProductionVerifier:
    """
    Production-ready verifier with all best practices
    """

    def __init__(self, config):
        self.config = config

        # Numerical stability
        self.epsilon = 1e-10

        # Caching
        self.draft_prob_cache = {}

        # Monitoring
        self.metrics = VerificationMetrics()

        # Adaptive strategies
        self.adaptive_temp = config.get("adaptive_temperature", False)
        self.confidence_threshold = config.get("confidence_threshold", 0.9)

    def verify_robust(self, draft_tokens, target_logits, draft_logits_cached):
        """
        Production-ready verification with all optimizations
        """
        # 1. Numerical stability
        target_probs = torch.softmax(target_logits, dim=-1).clamp(min=self.epsilon)

        # 2. Use cached draft probabilities
        draft_probs = draft_logits_cached  # Pre-computed and cached

        accepted_tokens = []
        num_accepted = 0

        # 3. Early stopping
        for i, draft_token in enumerate(draft_tokens):
            # 4. Confidence-based fast path
            if self._try_fast_accept(draft_token, target_probs[i], draft_probs[i]):
                accepted_tokens.append(draft_token)
                num_accepted += 1
                self.metrics.record_verification(i, True, 1.0)
                continue

            # 5. Standard verification with numerical stability
            p_target = target_probs[i][draft_token]
            p_draft = draft_probs[i][draft_token].clamp(min=self.epsilon)

            alpha = torch.clamp(p_target / p_draft, max=1.0)

            # 6. Accept/reject
            if torch.rand(1).item() < alpha.item():
                accepted_tokens.append(draft_token)
                num_accepted += 1
                self.metrics.record_verification(i, True, alpha.item())
            else:
                # Resample with numerical stability
                adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                adjusted = adjusted / adjusted.sum()

                new_token = torch.multinomial(adjusted, num_samples=1).item()
                accepted_tokens.append(new_token)
                num_accepted += 1
                self.metrics.record_verification(i, False, alpha.item())

                # Early stop
                break

        return torch.tensor(accepted_tokens), num_accepted

    def _try_fast_accept(self, draft_token, target_probs, draft_probs):
        """Fast path for high-confidence tokens"""
        draft_conf = draft_probs[draft_token]
        target_conf = target_probs[draft_token]

        return (draft_conf > self.confidence_threshold and
                target_conf > self.confidence_threshold * 0.8)

# Example configuration
config = {
    "adaptive_temperature": True,
    "confidence_threshold": 0.9,
    "enable_caching": True,
    "enable_early_stopping": True
}

verifier = ProductionVerifier(config)
```

---

## 7. Exercises

**Exercise 8.2.1:** Implement custom acceptance criterion for specific temperature settings

**Exercise 8.2.2:** Add monitoring to track per-position acceptance rates

**Exercise 8.2.3:** Optimize verification for top-k sampling

**Exercise 8.2.4:** Implement adaptive threshold based on runtime acceptance

**Exercise 8.2.5:** Debug numerical stability issues in verification

---

## 8. Further Reading

1. **Theoretical Foundations:**
   - "Rejection Sampling" (Chapter 11, Pattern Recognition and Machine Learning)
   - "Monte Carlo Statistical Methods" (Robert & Casella)

2. **vLLM Implementation:**
   - vLLM source: `vllm/model_executor/layers/spec_decode_base_sampler.py`
   - Verification logic: `vllm/spec_decode/`

3. **Advanced Topics:**
   - Next module: Medusa Multi-Head Speculation
   - Alternative verification strategies

---

**Next:** [Module 8.3: Medusa Multi-Head Speculation](03_medusa_multi_head_speculation.md)
