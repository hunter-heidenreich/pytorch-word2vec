"""Performance benchmarks for pair generation utilities."""

import pytest
import random
import time
from typing import List

from src.modern_word2vec.pairs import PairGenerator


class TestPairGeneratorPerformance:
    """Performance benchmarks for PairGenerator."""

    @pytest.mark.slow
    @pytest.mark.parametrize("vocab_size", [1000, 5000, 10000])
    @pytest.mark.parametrize("window_size", [2, 5, 10])
    def test_initialization_performance(self, vocab_size: int, window_size: int):
        """Benchmark initialization performance with different parameters."""
        tokens = list(range(vocab_size))

        start_time = time.time()
        generator = PairGenerator(
            token_ids=tokens, window_size=window_size, model_type="skipgram"
        )
        end_time = time.time()

        init_time = end_time - start_time
        print(
            f"Initialization time for vocab_size={vocab_size}, window_size={window_size}: {init_time:.4f}s"
        )

        # Should initialize reasonably quickly
        expected_max_time = 0.1 + (vocab_size / 10000)  # Scale with vocab size
        assert init_time < expected_max_time, (
            f"Initialization too slow: {init_time:.4f}s"
        )

        # Verify correctness
        assert len(generator) > 0
        assert len(generator.center_word_pair_starts) == vocab_size

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["skipgram", "cbow"])
    def test_random_access_performance(self, model_type: str):
        """Benchmark random access performance."""
        vocab_size = 10000
        tokens = list(range(vocab_size))
        generator = PairGenerator(
            token_ids=tokens, window_size=5, model_type=model_type
        )

        # Generate random indices
        num_accesses = 1000
        random_indices = [
            random.randint(0, len(generator) - 1) for _ in range(num_accesses)
        ]

        # Benchmark random access
        start_time = time.time()
        for idx in random_indices:
            pair = generator.generate_pair_at_index(idx)
            assert isinstance(pair, tuple)  # Basic correctness check
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_access = total_time / num_accesses

        print(
            f"Random access performance ({model_type}): {avg_time_per_access * 1000:.4f}ms per access"
        )

        # Should be very fast (O(log n) expected)
        assert avg_time_per_access < 0.001, (
            f"Random access too slow: {avg_time_per_access:.6f}s"
        )

    @pytest.mark.slow
    def test_sequential_access_performance(self):
        """Benchmark sequential access performance."""
        vocab_size = 10000
        tokens = list(range(vocab_size))
        generator = PairGenerator(
            token_ids=tokens, window_size=5, model_type="skipgram"
        )

        # Benchmark sequential access of first N pairs
        num_accesses = min(5000, len(generator))

        start_time = time.time()
        for i in range(num_accesses):
            pair = generator.generate_pair_at_index(i)
            assert isinstance(pair, tuple)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_access = total_time / num_accesses

        print(
            f"Sequential access performance: {avg_time_per_access * 1000:.4f}ms per access"
        )

        # Sequential access should be very fast
        assert avg_time_per_access < 0.0005, (
            f"Sequential access too slow: {avg_time_per_access:.6f}s"
        )

    @pytest.mark.slow
    def test_memory_efficiency_comparison(self):
        """Compare memory efficiency against naive approaches."""
        import sys

        vocab_size = 5000
        window_size = 5
        tokens = list(range(vocab_size))

        # Our efficient generator
        generator = PairGenerator(
            token_ids=tokens, window_size=window_size, model_type="skipgram"
        )

        # Measure generator memory usage
        generator_memory = (
            sys.getsizeof(generator)
            + sys.getsizeof(generator.token_ids)
            + sys.getsizeof(generator.center_word_pair_starts)
            + sys.getsizeof(generator.dynamic_window_sizes)
        )

        # Estimate memory for storing all pairs (naive approach)
        total_pairs = len(generator)
        naive_memory = total_pairs * 2 * 8  # 2 integers per pair, 8 bytes each

        efficiency_ratio = naive_memory / generator_memory
        print(
            f"Memory efficiency: Generator uses {generator_memory} bytes vs {naive_memory} bytes for naive storage"
        )
        print(f"Efficiency ratio: {efficiency_ratio:.2f}x")

        # Generator should be much more memory efficient
        assert efficiency_ratio > 5, (
            f"Generator not memory efficient enough: {efficiency_ratio:.2f}x"
        )

    @pytest.mark.slow
    def test_dynamic_window_performance_impact(self):
        """Test performance impact of dynamic window sizing."""
        vocab_size = 5000
        tokens = list(range(vocab_size))

        # Static window
        start_time = time.time()
        static_generator = PairGenerator(
            token_ids=tokens, window_size=5, model_type="skipgram", dynamic_window=False
        )
        static_init_time = time.time() - start_time

        # Dynamic window
        start_time = time.time()
        dynamic_generator = PairGenerator(
            token_ids=tokens,
            window_size=5,
            model_type="skipgram",
            dynamic_window=True,
            rng=random.Random(42),
        )
        dynamic_init_time = time.time() - start_time

        print(f"Static window init time: {static_init_time:.4f}s")
        print(f"Dynamic window init time: {dynamic_init_time:.4f}s")

        # Dynamic should not be significantly slower
        slowdown_factor = (
            dynamic_init_time / static_init_time if static_init_time > 0 else 1
        )
        assert slowdown_factor < 3, (
            f"Dynamic window too slow: {slowdown_factor:.2f}x slower"
        )

        # Test access performance
        num_accesses = 1000
        static_indices = [
            random.randint(0, len(static_generator) - 1) for _ in range(num_accesses)
        ]
        dynamic_indices = [
            random.randint(0, len(dynamic_generator) - 1) for _ in range(num_accesses)
        ]

        # Static access
        start_time = time.time()
        for idx in static_indices:
            static_generator.generate_pair_at_index(idx)
        static_access_time = time.time() - start_time

        # Dynamic access
        start_time = time.time()
        for idx in dynamic_indices:
            dynamic_generator.generate_pair_at_index(idx)
        dynamic_access_time = time.time() - start_time

        print(f"Static access time: {static_access_time:.4f}s")
        print(f"Dynamic access time: {dynamic_access_time:.4f}s")

    @pytest.mark.slow
    def test_scaling_behavior(self):
        """Test how performance scales with vocabulary size."""
        vocab_sizes = [1000, 2000, 5000, 10000]
        init_times = []
        access_times = []

        for vocab_size in vocab_sizes:
            tokens = list(range(vocab_size))

            # Measure initialization time
            start_time = time.time()
            generator = PairGenerator(
                token_ids=tokens, window_size=5, model_type="skipgram"
            )
            init_time = time.time() - start_time
            init_times.append(init_time)

            # Measure access time
            num_accesses = 100
            indices = [
                random.randint(0, len(generator) - 1) for _ in range(num_accesses)
            ]

            start_time = time.time()
            for idx in indices:
                generator.generate_pair_at_index(idx)
            access_time = (time.time() - start_time) / num_accesses
            access_times.append(access_time)

            print(
                f"Vocab size {vocab_size}: init={init_time:.4f}s, avg_access={access_time * 1000:.4f}ms"
            )

        # Check that scaling is reasonable (not exponential)
        for i in range(1, len(vocab_sizes)):
            size_ratio = vocab_sizes[i] / vocab_sizes[i - 1]
            init_time_ratio = (
                init_times[i] / init_times[i - 1] if init_times[i - 1] > 0 else 1
            )
            access_time_ratio = (
                access_times[i] / access_times[i - 1] if access_times[i - 1] > 0 else 1
            )

            # Initialization should scale roughly linearly
            assert init_time_ratio < size_ratio * 2, (
                f"Initialization scaling too poor: {init_time_ratio:.2f}x"
            )

            # Access should scale logarithmically (much better than linear)
            assert access_time_ratio < 2, (
                f"Access time scaling too poor: {access_time_ratio:.2f}x"
            )


class TestComparisonBenchmarks:
    """Comparison benchmarks against alternative implementations."""

    def naive_pair_generator(
        self, tokens: List[int], window_size: int, model_type: str
    ) -> List:
        """Naive implementation for comparison."""
        pairs = []
        for center_idx, center_token in enumerate(tokens):
            for offset in range(-window_size, window_size + 1):
                context_idx = center_idx + offset
                if 0 <= context_idx < len(tokens) and context_idx != center_idx:
                    if model_type == "skipgram":
                        pairs.append((center_token, tokens[context_idx]))
                    elif model_type == "cbow":
                        # For CBOW, collect context for each center word
                        context = []
                        for ctx_offset in range(-window_size, window_size + 1):
                            ctx_idx = center_idx + ctx_offset
                            if 0 <= ctx_idx < len(tokens) and ctx_idx != center_idx:
                                context.append(tokens[ctx_idx])
                        if context:
                            pairs.append((context, center_token))
                        break  # Only one CBOW pair per center word
        return pairs

    @pytest.mark.slow
    def test_correctness_vs_naive_implementation(self):
        """Verify our implementation produces same results as naive approach."""
        tokens = list(range(100))
        window_size = 2

        for model_type in ["skipgram", "cbow"]:
            # Our implementation
            generator = PairGenerator(
                token_ids=tokens, window_size=window_size, model_type=model_type
            )

            our_pairs = []
            for i in range(len(generator)):
                pair = generator.generate_pair_at_index(i)
                if model_type == "cbow":
                    our_pairs.append((tuple(pair[0]), pair[1]))  # Convert list to tuple
                else:
                    our_pairs.append(pair)

            # Naive implementation
            naive_pairs = self.naive_pair_generator(tokens, window_size, model_type)
            if model_type == "cbow":
                naive_pairs = [
                    (tuple(context), target) for context, target in naive_pairs
                ]

            # Should produce identical results
            assert set(our_pairs) == set(naive_pairs), (
                f"Results differ for {model_type}"
            )
            print(f"✓ {model_type} correctness verified against naive implementation")

    @pytest.mark.slow
    def test_performance_vs_naive_implementation(self):
        """Compare performance against naive implementation."""
        vocab_size = 1000
        tokens = list(range(vocab_size))
        window_size = 3

        # Naive approach timing
        start_time = time.time()
        naive_pairs = self.naive_pair_generator(tokens, window_size, "skipgram")
        naive_time = time.time() - start_time

        # Our approach timing
        start_time = time.time()
        generator = PairGenerator(
            token_ids=tokens, window_size=window_size, model_type="skipgram"
        )

        # Generate same number of pairs
        our_pairs = []
        for i in range(min(len(naive_pairs), len(generator))):
            our_pairs.append(generator.generate_pair_at_index(i))
        our_time = time.time() - start_time

        speedup = naive_time / our_time if our_time > 0 else float("inf")

        print(f"Naive implementation: {naive_time:.4f}s")
        print(f"Our implementation: {our_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Our implementation should be faster for generation
        # (Note: naive generates all pairs upfront, ours generates on-demand)
        assert len(our_pairs) == len(naive_pairs), (
            "Should generate same number of pairs"
        )
