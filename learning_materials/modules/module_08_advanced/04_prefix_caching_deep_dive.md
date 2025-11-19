# Module 8.4: Prefix Caching Deep Dive

## Overview

Prefix caching is a powerful optimization that dramatically improves performance when multiple requests share common prefixes. This module explores automatic prefix detection, caching strategies, and the RadixAttention algorithm that makes efficient prefix sharing possible in vLLM.

**Learning Objectives:**
- Understand prefix caching fundamentals and benefits
- Master automatic prefix detection algorithms
- Implement efficient cache eviction policies
- Optimize prefix matching for maximum hit rates
- Deploy prefix caching in production systems

**Prerequisites:**
- Understanding of KV cache in transformers
- Familiarity with vLLM's memory management (Module 2)
- Knowledge of caching algorithms (LRU, LFU)
- Experience with production deployment

**Estimated Time:** 3-4 hours

---

## 1. Prefix Caching Fundamentals

### 1.1 The Prefix Problem

```python
"""
Problem: Repeated computation for shared prefixes

Example: Multiple questions about the same document

Request 1: "Given the following document: [10K tokens]... What is the main idea?"
Request 2: "Given the following document: [10K tokens]... Who is the author?"
Request 3: "Given the following document: [10K tokens]... When was it written?"

Without caching: Compute attention for [10K tokens] three times
With caching: Compute once, reuse for all three requests
"""

def measure_prefix_waste_without_caching():
    """
    Measure wasted computation without prefix caching
    """
    import time
    from vllm import LLM, SamplingParams

    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    # Common prefix (simulating long document)
    common_prefix = "Please analyze the following document: " + "word " * 5000

    # Different questions about the document
    questions = [
        "What is the main theme?",
        "Who are the key figures mentioned?",
        "What is the historical context?",
        "What are the implications?"
    ]

    prompts = [common_prefix + q for q in questions]

    sampling_params = SamplingParams(max_tokens=50, temperature=0)

    # Without prefix caching
    print("Without Prefix Caching:")
    start = time.time()

    for prompt in prompts:
        _ = llm.generate(prompt, sampling_params)

    no_cache_time = time.time() - start
    print(f"  Total time: {no_cache_time:.2f}s")
    print(f"  Time per request: {no_cache_time / len(prompts):.2f}s")

    # With prefix caching (if available)
    llm_cached = LLM(
        model="meta-llama/Llama-2-7b-hf",
        enable_prefix_caching=True
    )

    print("\nWith Prefix Caching:")
    start = time.time()

    for prompt in prompts:
        _ = llm_cached.generate(prompt, sampling_params)

    cache_time = time.time() - start
    print(f"  Total time: {cache_time:.2f}s")
    print(f"  Time per request: {cache_time / len(prompts):.2f}s")

    # Savings
    print(f"\nSpeedup: {no_cache_time / cache_time:.2f}x")
    print(f"Time saved: {no_cache_time - cache_time:.2f}s ({(no_cache_time - cache_time) / no_cache_time * 100:.1f}%)")

# Typical results:
# Without Prefix Caching: 12.5s total (3.1s per request)
# With Prefix Caching: 3.8s total (0.95s per request)
# Speedup: 3.3x, Time saved: 8.7s (69.6%)
```

### 1.2 KV Cache Sharing

The key insight: The KV cache can be shared across requests with common prefixes.

```python
class KVCacheExplainer:
    """
    Explain how KV cache sharing works
    """

    def __init__(self):
        pass

    def explain_kv_cache_structure(self):
        """
        KV cache structure in transformers
        """
        explanation = """
        KV Cache for a single sequence:

        For each layer l and position i:
        - K[l, i] = Key vector at layer l, position i
        - V[l, i] = Value vector at layer l, position i

        Shape: [num_layers, sequence_length, hidden_dim]

        Example for Llama-7B:
        - 32 layers
        - 4096 hidden dim
        - For 1000 token sequence: 32 × 1000 × 4096 × 2 bytes = ~256 MB

        Key insight: If two sequences share first N tokens,
        they have IDENTICAL KV cache for positions 0..N-1
        """
        return explanation

    def demonstrate_sharing(self):
        """
        Demonstrate KV cache sharing
        """
        import torch

        # Simulate KV cache computation
        def compute_kv_cache(tokens, num_layers=4, hidden_dim=128):
            """Simplified KV cache computation"""
            seq_len = len(tokens)
            # Keys and Values for all layers
            K = torch.randn(num_layers, seq_len, hidden_dim)
            V = torch.randn(num_layers, seq_len, hidden_dim)
            return K, V

        # Common prefix
        prefix = [1, 2, 3, 4, 5]

        # Three sequences with shared prefix
        seq1 = prefix + [10, 11]
        seq2 = prefix + [20, 21]
        seq3 = prefix + [30, 31]

        # Without sharing: compute KV cache for each sequence
        print("Without Sharing:")
        K1, V1 = compute_kv_cache(seq1)
        K2, V2 = compute_kv_cache(seq2)
        K3, V3 = compute_kv_cache(seq3)

        print(f"  Seq1 KV cache: {K1.shape}")
        print(f"  Seq2 KV cache: {K2.shape}")
        print(f"  Seq3 KV cache: {K3.shape}")
        print(f"  Total memory: {(K1.numel() + V1.numel() + K2.numel() + V2.numel() + K3.numel() + V3.numel()) * 4 / 1024:.2f} KB")

        # With sharing: compute prefix once, reuse
        print("\nWith Sharing:")
        K_prefix, V_prefix = compute_kv_cache(prefix)

        # Only compute unique suffixes
        K1_suffix, V1_suffix = compute_kv_cache([10, 11])
        K2_suffix, V2_suffix = compute_kv_cache([20, 21])
        K3_suffix, V3_suffix = compute_kv_cache([30, 31])

        print(f"  Shared prefix KV cache: {K_prefix.shape}")
        print(f"  Seq1 suffix KV cache: {K1_suffix.shape}")
        print(f"  Seq2 suffix KV cache: {K2_suffix.shape}")
        print(f"  Seq3 suffix KV cache: {K3_suffix.shape}")

        total_shared_memory = (
            K_prefix.numel() + V_prefix.numel() +
            K1_suffix.numel() + V1_suffix.numel() +
            K2_suffix.numel() + V2_suffix.numel() +
            K3_suffix.numel() + V3_suffix.numel()
        ) * 4 / 1024

        print(f"  Total memory: {total_shared_memory:.2f} KB")

        # Savings
        no_sharing_mem = (K1.numel() + V1.numel() + K2.numel() + V2.numel() + K3.numel() + V3.numel()) * 4 / 1024
        savings = (no_sharing_mem - total_shared_memory) / no_sharing_mem * 100

        print(f"\nMemory saved: {savings:.1f}%")

explainer = KVCacheExplainer()
print(explainer.explain_kv_cache_structure())
explainer.demonstrate_sharing()
```

---

## 2. Automatic Prefix Detection

### 2.1 Longest Common Prefix (LCP) Algorithm

```python
class PrefixDetector:
    """
    Detect common prefixes between sequences
    """

    def __init__(self):
        self.cache = {}  # Store known prefixes

    def find_longest_common_prefix(self, seq1, seq2):
        """
        Find longest common prefix between two sequences

        Args:
            seq1, seq2: Lists of tokens

        Returns:
            Length of longest common prefix
        """
        lcp_length = 0

        for i in range(min(len(seq1), len(seq2))):
            if seq1[i] == seq2[i]:
                lcp_length += 1
            else:
                break

        return lcp_length

    def find_best_prefix_match(self, new_sequence, candidate_sequences):
        """
        Find which candidate sequence shares longest prefix with new sequence

        Args:
            new_sequence: New sequence to process
            candidate_sequences: List of previously processed sequences

        Returns:
            (best_match_idx, lcp_length)
        """
        best_match_idx = -1
        best_lcp_length = 0

        for idx, candidate in enumerate(candidate_sequences):
            lcp_length = self.find_longest_common_prefix(new_sequence, candidate)

            if lcp_length > best_lcp_length:
                best_lcp_length = lcp_length
                best_match_idx = idx

        return best_match_idx, best_lcp_length

def benchmark_prefix_detection():
    """
    Benchmark prefix detection performance
    """
    import random
    import time

    detector = PrefixDetector()

    # Generate test sequences with shared prefixes
    base_prefix = list(range(1000))  # Common prefix of 1000 tokens

    sequences = []
    for i in range(100):
        # Each sequence shares prefix but has unique suffix
        suffix = [random.randint(10000, 20000) for _ in range(100)]
        sequences.append(base_prefix + suffix)

    # Benchmark detection
    new_sequence = base_prefix + [99999, 99998]

    start = time.time()
    best_match, lcp_length = detector.find_best_prefix_match(new_sequence, sequences)
    elapsed = time.time() - start

    print(f"Prefix Detection Benchmark:")
    print(f"  Sequences checked: {len(sequences)}")
    print(f"  Time: {elapsed * 1000:.2f} ms")
    print(f"  Best match LCP: {lcp_length} tokens")
    print(f"  Match ratio: {lcp_length / len(new_sequence) * 100:.1f}%")
```

### 2.2 Efficient Prefix Matching with Hash Tables

```python
class HashBasedPrefixMatcher:
    """
    Efficient prefix matching using hash-based indexing
    """

    def __init__(self):
        # Map from prefix hash to list of sequences with that prefix
        self.prefix_index = {}

    def _compute_prefix_hash(self, tokens, length):
        """
        Compute hash of prefix of given length

        Use rolling hash for efficiency
        """
        if length == 0:
            return 0

        # Simple hash (in practice, use better hash function)
        prefix = tuple(tokens[:length])
        return hash(prefix)

    def index_sequence(self, sequence, sequence_id):
        """
        Index all prefixes of a sequence

        Args:
            sequence: List of tokens
            sequence_id: Unique identifier for this sequence
        """
        # Index all prefix lengths
        for length in range(1, len(sequence) + 1):
            prefix_hash = self._compute_prefix_hash(sequence, length)

            if prefix_hash not in self.prefix_index:
                self.prefix_index[prefix_hash] = []

            self.prefix_index[prefix_hash].append({
                "sequence_id": sequence_id,
                "prefix_length": length,
                "full_sequence": sequence
            })

    def find_matching_prefixes(self, query_sequence):
        """
        Find all sequences that share prefixes with query

        Returns: List of (sequence_id, shared_prefix_length)
        """
        matches = []

        # Check each prefix length of query
        for length in range(1, len(query_sequence) + 1):
            prefix_hash = self._compute_prefix_hash(query_sequence, length)

            if prefix_hash in self.prefix_index:
                for entry in self.prefix_index[prefix_hash]:
                    matches.append({
                        "sequence_id": entry["sequence_id"],
                        "shared_prefix_length": length
                    })

        # Group by sequence_id and keep longest match
        best_matches = {}
        for match in matches:
            seq_id = match["sequence_id"]
            length = match["shared_prefix_length"]

            if seq_id not in best_matches or length > best_matches[seq_id]:
                best_matches[seq_id] = length

        return best_matches

def benchmark_hash_based_matching():
    """
    Benchmark hash-based prefix matching vs linear scan
    """
    import random
    import time

    # Generate test data
    num_sequences = 1000
    prefix_length = 500
    suffix_length = 100

    base_prefix = list(range(prefix_length))
    sequences = []

    for i in range(num_sequences):
        suffix = [random.randint(10000, 20000) for _ in range(suffix_length)]
        sequences.append((base_prefix + suffix, i))

    # Method 1: Linear scan
    detector = PrefixDetector()
    query = base_prefix + [99999]

    start = time.time()
    for seq, seq_id in sequences:
        lcp = detector.find_longest_common_prefix(query, seq)
    linear_time = time.time() - start

    # Method 2: Hash-based
    matcher = HashBasedPrefixMatcher()

    # Index all sequences
    start = time.time()
    for seq, seq_id in sequences:
        matcher.index_sequence(seq, seq_id)
    index_time = time.time() - start

    # Query
    start = time.time()
    matches = matcher.find_matching_prefixes(query)
    query_time = time.time() - start

    print("Prefix Matching Benchmark:")
    print(f"  Linear scan: {linear_time * 1000:.2f} ms")
    print(f"  Hash-based indexing: {index_time * 1000:.2f} ms")
    print(f"  Hash-based query: {query_time * 1000:.2f} ms")
    print(f"  Total hash-based: {(index_time + query_time) * 1000:.2f} ms")
    print(f"  Speedup (query only): {linear_time / query_time:.2f}x")
```

---

## 3. RadixAttention: vLLM's Prefix Caching

### 3.1 Radix Tree Structure

```python
class RadixTreeNode:
    """
    Node in Radix tree for efficient prefix storage

    Radix tree compresses common prefixes into shared nodes
    """

    def __init__(self, tokens=None, is_cached=False):
        """
        Args:
            tokens: Token sequence for this edge (not node!)
            is_cached: Whether KV cache exists for this node
        """
        self.tokens = tokens or []
        self.is_cached = is_cached
        self.children = {}  # Map from first token to child node
        self.kv_cache_ref = None  # Reference to actual KV cache
        self.access_count = 0  # For LRU eviction
        self.last_access_time = 0

    def __repr__(self):
        return f"RadixNode(tokens={self.tokens[:10]}..., cached={self.is_cached})"

class RadixTree:
    """
    Radix tree for prefix caching

    Efficiently stores and retrieves KV caches for sequences with shared prefixes
    """

    def __init__(self):
        self.root = RadixTreeNode(tokens=[], is_cached=True)
        self.total_nodes = 1
        self.total_cached_tokens = 0

    def insert(self, tokens, kv_cache):
        """
        Insert sequence and its KV cache into tree

        Args:
            tokens: List of token IDs
            kv_cache: KV cache for this sequence
        """
        current = self.root
        remaining_tokens = tokens[:]

        while remaining_tokens:
            # Find child with matching first token
            first_token = remaining_tokens[0]

            if first_token not in current.children:
                # No match - create new node
                new_node = RadixTreeNode(
                    tokens=remaining_tokens,
                    is_cached=True
                )
                new_node.kv_cache_ref = kv_cache
                current.children[first_token] = new_node

                self.total_nodes += 1
                self.total_cached_tokens += len(remaining_tokens)
                break

            # Found matching child
            child = current.children[first_token]

            # Find common prefix between remaining tokens and child's tokens
            common_len = self._find_common_prefix_length(
                remaining_tokens,
                child.tokens
            )

            if common_len == len(child.tokens):
                # Entire child edge matches - continue down tree
                remaining_tokens = remaining_tokens[common_len:]
                current = child

            else:
                # Partial match - need to split the edge
                self._split_edge(current, child, first_token, common_len, remaining_tokens, kv_cache)
                break

    def _find_common_prefix_length(self, tokens1, tokens2):
        """Find length of common prefix"""
        common_len = 0
        for i in range(min(len(tokens1), len(tokens2))):
            if tokens1[i] == tokens2[i]:
                common_len += 1
            else:
                break
        return common_len

    def _split_edge(self, parent, child, first_token, split_point, new_tokens, new_kv_cache):
        """
        Split an edge when there's a partial match

        Before:
        parent -> [A, B, C, D] (child)

        After (splitting at position 2):
        parent -> [A, B] (new_middle) -> [C, D] (child)
                            -> [E, F] (new_sibling)
        """
        # Create middle node with common prefix
        middle_node = RadixTreeNode(
            tokens=child.tokens[:split_point],
            is_cached=True  # Can cache the common part
        )

        # Update child to only have suffix
        child.tokens = child.tokens[split_point:]

        # Create new sibling for the diverging path
        new_suffix = new_tokens[split_point:]
        new_sibling = RadixTreeNode(
            tokens=new_suffix,
            is_cached=True
        )
        new_sibling.kv_cache_ref = new_kv_cache

        # Reconnect tree
        parent.children[first_token] = middle_node
        middle_node.children[child.tokens[0]] = child

        if new_suffix:
            middle_node.children[new_suffix[0]] = new_sibling

        self.total_nodes += 2  # Added middle and new_sibling

    def lookup(self, tokens):
        """
        Find longest cached prefix for given tokens

        Returns:
            (cached_tokens, kv_cache_ref, remaining_tokens)
        """
        current = self.root
        cached_tokens = []
        kv_cache_ref = None
        remaining_tokens = tokens[:]

        while remaining_tokens:
            first_token = remaining_tokens[0]

            if first_token not in current.children:
                # No more matches
                break

            child = current.children[first_token]

            # Check how much of child edge matches
            common_len = self._find_common_prefix_length(
                remaining_tokens,
                child.tokens
            )

            # Add matched portion to cached tokens
            cached_tokens.extend(child.tokens[:common_len])

            if child.is_cached and common_len == len(child.tokens):
                # Full match - can use this cache
                kv_cache_ref = child.kv_cache_ref
                remaining_tokens = remaining_tokens[common_len:]
                current = child

                # Update access statistics
                child.access_count += 1
                import time
                child.last_access_time = time.time()

            else:
                # Partial match or no cache
                remaining_tokens = remaining_tokens[common_len:]
                break

        return cached_tokens, kv_cache_ref, remaining_tokens

    def print_tree(self, node=None, prefix="", depth=0):
        """Print tree structure for debugging"""
        if node is None:
            node = self.root

        if depth == 0:
            print("Radix Tree Structure:")

        for first_token, child in node.children.items():
            cached_marker = "✓" if child.is_cached else "✗"
            print(f"{prefix}└─ {cached_marker} {child.tokens[:5]}... (len={len(child.tokens)})")
            self.print_tree(child, prefix + "   ", depth + 1)

def radix_tree_example():
    """
    Example of RadixTree usage
    """
    tree = RadixTree()

    # Insert sequences
    sequences = [
        ([1, 2, 3, 4, 5], "cache_1"),
        ([1, 2, 3, 6, 7], "cache_2"),
        ([1, 2, 8, 9], "cache_3"),
        ([1, 2, 3, 4, 10], "cache_4"),
    ]

    print("Inserting sequences:")
    for tokens, kv_cache in sequences:
        print(f"  {tokens}")
        tree.insert(tokens, kv_cache)

    print("\n")
    tree.print_tree()

    # Lookup examples
    print("\n\nLookup Examples:")

    test_sequences = [
        [1, 2, 3, 4, 5, 11],  # Exact match with suffix
        [1, 2, 3],             # Partial match
        [1, 2, 8, 9, 10],      # Match different branch
        [5, 6, 7],             # No match
    ]

    for test_seq in test_sequences:
        cached, kv_ref, remaining = tree.lookup(test_seq)
        print(f"\nQuery: {test_seq}")
        print(f"  Cached prefix: {cached}")
        print(f"  KV cache ref: {kv_ref}")
        print(f"  Remaining: {remaining}")
        print(f"  Hit rate: {len(cached) / len(test_seq) * 100:.1f}%")

radix_tree_example()
```

### 3.2 vLLM RadixAttention Implementation

```python
class RadixAttentionManager:
    """
    Manager for RadixAttention in vLLM

    Handles:
    - Automatic prefix detection
    - KV cache storage and retrieval
    - Cache eviction when memory is full
    """

    def __init__(self, max_cache_memory_gb=10):
        self.radix_tree = RadixTree()
        self.max_cache_memory = max_cache_memory_gb * 1024**3  # Convert to bytes
        self.current_cache_memory = 0

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0

    def process_request(self, tokens, kv_cache_generator):
        """
        Process request with prefix caching

        Args:
            tokens: Input token sequence
            kv_cache_generator: Function to generate KV cache for tokens

        Returns:
            (cached_kv, computed_kv, cache_hit_rate)
        """
        self.total_requests += 1

        # Step 1: Lookup in Radix tree
        cached_tokens, cached_kv, remaining_tokens = self.radix_tree.lookup(tokens)

        if cached_tokens:
            self.cache_hits += 1
            cache_hit_rate = len(cached_tokens) / len(tokens)
        else:
            self.cache_misses += 1
            cache_hit_rate = 0.0

        # Step 2: Compute KV cache for remaining tokens
        if remaining_tokens:
            computed_kv = kv_cache_generator(remaining_tokens)
        else:
            computed_kv = None

        # Step 3: Store new KV cache in tree
        if computed_kv is not None:
            # Check if we have space
            kv_size = self._estimate_kv_cache_size(computed_kv)

            if self.current_cache_memory + kv_size > self.max_cache_memory:
                # Evict old entries
                self._evict_cache_entries(kv_size)

            # Insert into tree
            self.radix_tree.insert(tokens, computed_kv)
            self.current_cache_memory += kv_size

        return cached_kv, computed_kv, cache_hit_rate

    def _estimate_kv_cache_size(self, kv_cache):
        """Estimate memory size of KV cache"""
        # Simplified estimation
        # In practice, would calculate based on actual tensor sizes
        return 1024 * 1024  # 1 MB placeholder

    def _evict_cache_entries(self, space_needed):
        """
        Evict cache entries to free space

        Strategy: LRU (Least Recently Used)
        """
        print(f"Evicting cache entries to free {space_needed / 1024**2:.2f} MB")

        # Collect all nodes with their access times
        nodes_to_evict = []

        def collect_nodes(node):
            if node.is_cached and node.kv_cache_ref is not None:
                nodes_to_evict.append(node)

            for child in node.children.values():
                collect_nodes(child)

        collect_nodes(self.radix_tree.root)

        # Sort by last access time (oldest first)
        nodes_to_evict.sort(key=lambda n: n.last_access_time)

        # Evict until we have enough space
        freed_space = 0

        for node in nodes_to_evict:
            if freed_space >= space_needed:
                break

            # Evict this node
            kv_size = self._estimate_kv_cache_size(node.kv_cache_ref)
            node.kv_cache_ref = None
            node.is_cached = False

            freed_space += kv_size
            self.current_cache_memory -= kv_size

        print(f"Evicted {len(nodes_to_evict)} entries, freed {freed_space / 1024**2:.2f} MB")

    def get_statistics(self):
        """Get caching statistics"""
        if self.total_requests == 0:
            return {
                "cache_hit_rate": 0.0,
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }

        return {
            "cache_hit_rate": self.cache_hits / self.total_requests,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cached_nodes": self.radix_tree.total_nodes,
            "memory_usage_gb": self.current_cache_memory / 1024**3
        }

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()

        print("\nPrefix Caching Statistics:")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"Hit Rate: {stats['cache_hit_rate'] * 100:.2f}%")
        print(f"Cached Nodes: {stats.get('cached_nodes', 0)}")
        print(f"Memory Usage: {stats.get('memory_usage_gb', 0):.2f} GB")
        print("=" * 50)
```

---

## 4. Optimization Strategies

### 4.1 Cache Eviction Policies

```python
class CacheEvictionPolicy:
    """
    Different cache eviction strategies
    """

    @staticmethod
    def lru_eviction(nodes, space_needed):
        """
        Least Recently Used (LRU) eviction

        Evict nodes that haven't been accessed recently
        """
        nodes.sort(key=lambda n: n.last_access_time)
        return nodes[:space_needed]

    @staticmethod
    def lfu_eviction(nodes, space_needed):
        """
        Least Frequently Used (LFU) eviction

        Evict nodes that are rarely accessed
        """
        nodes.sort(key=lambda n: n.access_count)
        return nodes[:space_needed]

    @staticmethod
    def size_aware_eviction(nodes, space_needed):
        """
        Size-aware eviction

        Evict large nodes first to free more space quickly
        """
        # Estimate size of each node
        nodes_with_size = [
            (node, len(node.tokens) * 1024)  # Rough estimate
            for node in nodes
        ]

        # Sort by size (largest first)
        nodes_with_size.sort(key=lambda x: x[1], reverse=True)

        evicted = []
        freed = 0

        for node, size in nodes_with_size:
            if freed >= space_needed:
                break
            evicted.append(node)
            freed += size

        return evicted

    @staticmethod
    def hybrid_eviction(nodes, space_needed):
        """
        Hybrid eviction combining multiple factors

        Score = (1 / access_count) * (time_since_access) * size
        Higher score = more likely to evict
        """
        import time

        current_time = time.time()

        scored_nodes = []
        for node in nodes:
            time_since_access = current_time - node.last_access_time
            access_frequency = node.access_count + 1  # Avoid division by zero
            size = len(node.tokens)

            score = (1.0 / access_frequency) * time_since_access * size
            scored_nodes.append((node, score))

        # Sort by score (highest first)
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        return [node for node, score in scored_nodes[:space_needed]]

def compare_eviction_policies():
    """
    Compare different eviction policies
    """
    import random
    import time

    # Create test nodes
    nodes = []
    for i in range(100):
        node = RadixTreeNode(tokens=list(range(i, i + random.randint(10, 100))))
        node.access_count = random.randint(1, 100)
        node.last_access_time = time.time() - random.uniform(0, 3600)  # Last hour
        nodes.append(node)

    space_needed = 20  # Need to evict 20 nodes

    policies = {
        "LRU": CacheEvictionPolicy.lru_eviction,
        "LFU": CacheEvictionPolicy.lfu_eviction,
        "Size-Aware": CacheEvictionPolicy.size_aware_eviction,
        "Hybrid": CacheEvictionPolicy.hybrid_eviction
    }

    print("Eviction Policy Comparison:")
    print("=" * 60)

    for name, policy_func in policies.items():
        start = time.time()
        evicted = policy_func(nodes[:], space_needed)
        elapsed = time.time() - start

        avg_access_count = sum(n.access_count for n in evicted) / len(evicted)
        avg_age = sum(time.time() - n.last_access_time for n in evicted) / len(evicted)

        print(f"\n{name}:")
        print(f"  Evicted: {len(evicted)} nodes")
        print(f"  Avg access count: {avg_access_count:.2f}")
        print(f"  Avg age: {avg_age:.2f}s")
        print(f"  Time: {elapsed * 1000:.2f} ms")
```

### 4.2 Adaptive Prefix Length

```python
class AdaptivePrefixCache:
    """
    Adaptively determine optimal prefix length to cache

    Not all prefixes are worth caching - adjust based on:
    - Prefix frequency
    - Computation cost
    - Available memory
    """

    def __init__(self, min_prefix_length=10, max_prefix_length=10000):
        self.min_prefix_length = min_prefix_length
        self.max_prefix_length = max_prefix_length

        # Track prefix reuse statistics
        self.prefix_reuse_counts = {}

    def should_cache_prefix(self, tokens):
        """
        Decide whether to cache this prefix

        Criteria:
        1. Length within acceptable range
        2. Expected to be reused (based on history)
        3. Sufficient memory available
        """
        prefix_length = len(tokens)

        # Check length
        if prefix_length < self.min_prefix_length:
            return False, "Too short"

        if prefix_length > self.max_prefix_length:
            return False, "Too long"

        # Check reuse potential
        prefix_hash = hash(tuple(tokens))

        if prefix_hash in self.prefix_reuse_counts:
            reuse_count = self.prefix_reuse_counts[prefix_hash]

            if reuse_count >= 2:
                # Has been reused before - worth caching
                return True, "High reuse potential"
            else:
                # Seen once but not reused yet
                return False, "Low reuse so far"
        else:
            # First time seeing this prefix
            self.prefix_reuse_counts[prefix_hash] = 1
            return False, "First occurrence"

    def update_reuse_stats(self, tokens):
        """Update statistics when prefix is reused"""
        prefix_hash = hash(tuple(tokens))
        self.prefix_reuse_counts[prefix_hash] = self.prefix_reuse_counts.get(prefix_hash, 0) + 1

    def optimize_prefix_length(self, tokens, computation_cost_per_token):
        """
        Find optimal prefix length to cache

        Balance:
        - Longer prefix = more computation saved
        - Longer prefix = more memory used
        - Longer prefix = less likely to be reused exactly
        """
        optimal_length = self.min_prefix_length
        max_value = 0

        for length in range(self.min_prefix_length, min(len(tokens), self.max_prefix_length)):
            prefix = tokens[:length]
            prefix_hash = hash(tuple(prefix))

            # Expected reuse count (based on history or prediction)
            expected_reuse = self.prefix_reuse_counts.get(prefix_hash, 0)

            # Value = (computation saved) × (expected reuse) - (memory cost)
            computation_saved = length * computation_cost_per_token
            memory_cost = length * 0.1  # Simplified cost
            value = computation_saved * expected_reuse - memory_cost

            if value > max_value:
                max_value = value
                optimal_length = length

        return optimal_length

def adaptive_caching_example():
    """
    Example of adaptive prefix caching
    """
    adaptive_cache = AdaptivePrefixCache(min_prefix_length=50, max_prefix_length=5000)

    # Simulate request stream
    common_doc = list(range(1000))  # Common document prefix

    requests = [
        common_doc + [10000 + i]
        for i in range(20)
    ]

    print("Adaptive Prefix Caching:")
    print("=" * 60)

    for i, request in enumerate(requests):
        should_cache, reason = adaptive_cache.should_cache_prefix(request[:1000])

        print(f"\nRequest {i + 1}:")
        print(f"  Prefix length: {len(request[:1000])}")
        print(f"  Should cache: {should_cache}")
        print(f"  Reason: {reason}")

        if should_cache:
            adaptive_cache.update_reuse_stats(request[:1000])

    # After processing all requests
    print("\n\nFinal Statistics:")
    total_prefixes = len(adaptive_cache.prefix_reuse_counts)
    highly_reused = sum(1 for count in adaptive_cache.prefix_reuse_counts.values() if count >= 5)

    print(f"  Total unique prefixes: {total_prefixes}")
    print(f"  Highly reused (≥5 times): {highly_reused}")
    print(f"  Reuse ratio: {highly_reused / total_prefixes * 100:.1f}%")
```

---

## 5. Production Deployment

### 5.1 Enabling Prefix Caching in vLLM

```python
from vllm import LLM, SamplingParams

def deploy_with_prefix_caching():
    """
    Deploy vLLM with prefix caching enabled
    """
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",

        # Enable prefix caching
        enable_prefix_caching=True,

        # Configuration
        gpu_memory_utilization=0.9,
        max_num_seqs=256,  # Larger batch with caching

        # Prefix cache settings
        max_num_batched_tokens=8192,  # Larger context with caching
    )

    return llm

def production_example():
    """
    Production example: Document Q&A system
    """
    llm = deploy_with_prefix_caching()

    # Long document (shared across many requests)
    long_document = """
    [Large document text - 10K tokens]
    """ * 100

    questions = [
        "What is the main topic?",
        "Who are the key people mentioned?",
        "What year was this written?",
        "What are the key findings?",
    ]

    # First request - no cache hit
    print("First request (cold start):")
    start = time.time()
    prompt = long_document + "\n\nQuestion: " + questions[0]
    output = llm.generate(prompt, SamplingParams(max_tokens=50))
    print(f"  Time: {time.time() - start:.2f}s")

    # Subsequent requests - cache hits
    for i, question in enumerate(questions[1:], 2):
        print(f"\nRequest {i} (cache hit expected):")
        start = time.time()
        prompt = long_document + "\n\nQuestion: " + question
        output = llm.generate(prompt, SamplingParams(max_tokens=50))
        print(f"  Time: {time.time() - start:.2f}s")

    # Expected output:
    # First request (cold start): 2.5s
    # Request 2 (cache hit expected): 0.3s
    # Request 3 (cache hit expected): 0.3s
    # Request 4 (cache hit expected): 0.3s
```

### 5.2 Monitoring Prefix Cache Performance

```python
class PrefixCacheMonitor:
    """
    Monitor prefix cache performance in production
    """

    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_processed": 0,
            "total_tokens_from_cache": 0,
            "latencies": [],
            "cache_hit_latencies": [],
            "cache_miss_latencies": []
        }

    def record_request(self, tokens_total, tokens_from_cache, latency):
        """Record metrics for a single request"""
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens_processed"] += tokens_total
        self.metrics["total_tokens_from_cache"] += tokens_from_cache
        self.metrics["latencies"].append(latency)

        if tokens_from_cache > 0:
            self.metrics["cache_hits"] += 1
            self.metrics["cache_hit_latencies"].append(latency)
        else:
            self.metrics["cache_misses"] += 1
            self.metrics["cache_miss_latencies"].append(latency)

    def get_summary(self):
        """Get summary statistics"""
        import numpy as np

        if self.metrics["total_requests"] == 0:
            return {}

        summary = {
            "total_requests": self.metrics["total_requests"],
            "cache_hit_rate": self.metrics["cache_hits"] / self.metrics["total_requests"],
            "token_cache_hit_rate": self.metrics["total_tokens_from_cache"] / self.metrics["total_tokens_processed"],

            "avg_latency": np.mean(self.metrics["latencies"]),
            "p50_latency": np.percentile(self.metrics["latencies"], 50),
            "p95_latency": np.percentile(self.metrics["latencies"], 95),
            "p99_latency": np.percentile(self.metrics["latencies"], 99),
        }

        if self.metrics["cache_hit_latencies"]:
            summary["avg_cache_hit_latency"] = np.mean(self.metrics["cache_hit_latencies"])

        if self.metrics["cache_miss_latencies"]:
            summary["avg_cache_miss_latency"] = np.mean(self.metrics["cache_miss_latencies"])

        return summary

    def print_report(self):
        """Print detailed monitoring report"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PREFIX CACHE MONITORING REPORT")
        print("=" * 60)

        print(f"\nRequests:")
        print(f"  Total: {summary.get('total_requests', 0)}")
        print(f"  Cache Hits: {self.metrics['cache_hits']}")
        print(f"  Cache Misses: {self.metrics['cache_misses']}")
        print(f"  Hit Rate: {summary.get('cache_hit_rate', 0) * 100:.2f}%")

        print(f"\nTokens:")
        print(f"  Total Processed: {self.metrics['total_tokens_processed']}")
        print(f"  From Cache: {self.metrics['total_tokens_from_cache']}")
        print(f"  Cache Hit Rate: {summary.get('token_cache_hit_rate', 0) * 100:.2f}%")

        print(f"\nLatency:")
        print(f"  Average: {summary.get('avg_latency', 0) * 1000:.2f} ms")
        print(f"  P50: {summary.get('p50_latency', 0) * 1000:.2f} ms")
        print(f"  P95: {summary.get('p95_latency', 0) * 1000:.2f} ms")
        print(f"  P99: {summary.get('p99_latency', 0) * 1000:.2f} ms")

        if "avg_cache_hit_latency" in summary:
            print(f"\nCache Hit Latency: {summary['avg_cache_hit_latency'] * 1000:.2f} ms")

        if "avg_cache_miss_latency" in summary:
            print(f"Cache Miss Latency: {summary['avg_cache_miss_latency'] * 1000:.2f} ms")

            if "avg_cache_hit_latency" in summary:
                speedup = summary["avg_cache_miss_latency"] / summary["avg_cache_hit_latency"]
                print(f"Speedup from cache: {speedup:.2f}x")

        print("=" * 60)

# Example usage in production
def monitor_production_deployment():
    """
    Monitor prefix caching in production
    """
    monitor = PrefixCacheMonitor()

    # Simulate requests
    for i in range(100):
        # Simulate: 70% of requests have cache hits
        import random
        import time

        tokens_total = random.randint(500, 2000)

        if random.random() < 0.7:
            # Cache hit
            tokens_from_cache = random.randint(400, tokens_total)
            latency = random.uniform(0.1, 0.3)  # Faster with cache
        else:
            # Cache miss
            tokens_from_cache = 0
            latency = random.uniform(0.8, 1.5)  # Slower without cache

        monitor.record_request(tokens_total, tokens_from_cache, latency)

    monitor.print_report()

# monitor_production_deployment()
```

---

## 6. Summary and Best Practices

### 6.1 When to Use Prefix Caching

**✅ Excellent Use Cases:**
- Document Q&A (same document, multiple questions)
- Chat with system prompts (long prompt, many user messages)
- Few-shot learning (same examples, different queries)
- RAG systems (retrieved context shared across queries)

**❌ Poor Use Cases:**
- Unique prompts for every request
- Short prompts (<100 tokens)
- Extremely diverse request patterns
- Memory-constrained deployments

### 6.2 Configuration Best Practices

```python
def get_prefix_cache_config(use_case):
    """
    Recommended configurations for different use cases
    """
    configs = {
        "document_qa": {
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 16384,  # Long documents
            "gpu_memory_utilization": 0.85,   # Leave room for cache
            "rationale": "Long documents reused across many questions"
        },

        "chatbot": {
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.9,
            "rationale": "System prompt cached across conversations"
        },

        "code_completion": {
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.9,
            "rationale": "Code context often shared"
        },

        "diverse_queries": {
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.95,
            "rationale": "Little prefix reuse, caching adds overhead"
        }
    }

    return configs.get(use_case, configs["document_qa"])
```

---

## 7. Exercises

**Exercise 8.4.1:** Implement Radix tree insertion and lookup

**Exercise 8.4.2:** Compare different cache eviction policies

**Exercise 8.4.3:** Measure prefix cache hit rates for your workload

**Exercise 8.4.4:** Optimize prefix matching performance

**Exercise 8.4.5:** Deploy and monitor prefix caching in production

---

## 8. Further Reading

1. **vLLM Documentation:**
   - Automatic Prefix Caching guide
   - RadixAttention implementation

2. **Research Papers:**
   - "FlexGen: High-Throughput Generative Inference"
   - "Efficient Memory Management for LLM Serving"

3. **Related Topics:**
   - KV cache optimization
   - Memory management strategies

---

**Next:** [Module 8.5: RadixAttention](05_radix_attention.md)
