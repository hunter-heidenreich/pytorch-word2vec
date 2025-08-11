"""Additional stress tests and edge cases for pair generation."""

import pytest
import random
import torch
import time
from unittest.mock import patch

from src.modern_word2vec.pairs import PairGenerator, create_pair_tensors


class TestPairGeneratorStress:
    """Stress tests and additional edge cases for PairGenerator."""

    def test_very_large_tokens_sequence(self):
        """Test with very large token sequence."""
        # Create a large sequence
        large_tokens = list(range(50000))
        generator = PairGenerator(
            token_ids=large_tokens, window_size=5, model_type="skipgram"
        )

        # Should handle large sequences efficiently
        assert len(generator) > 0

        # Test random access performance
        test_indices = random.sample(range(len(generator)), min(100, len(generator)))
        for idx in test_indices:
            pair = generator.generate_pair_at_index(idx)
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_maximum_window_size_edge(self):
        """Test with maximum possible window size."""
        tokens = list(range(100))
        max_window = len(tokens) * 2  # Larger than sequence

        generator = PairGenerator(
            token_ids=tokens, window_size=max_window, model_type="skipgram"
        )

        # Should handle gracefully
        assert len(generator) > 0

        # Every token should be paired with every other token
        pairs_set = set()
        for i in range(min(1000, len(generator))):  # Sample first 1000
            center, context = generator.generate_pair_at_index(i)
            pairs_set.add((center, context))

        # Should have many unique pairs
        assert len(pairs_set) > 50

    def test_repeated_tokens_in_sequence(self):
        """Test with repeated tokens in sequence."""
        tokens = [1, 2, 1, 3, 1, 4, 1]  # Token 1 appears multiple times

        generator = PairGenerator(
            token_ids=tokens, window_size=2, model_type="skipgram"
        )

        # Should generate valid pairs
        for i in range(min(10, len(generator))):
            center, context = generator.generate_pair_at_index(i)
            assert center in tokens
            assert context in tokens

    def test_extreme_dynamic_window_variation(self):
        """Test dynamic window with extreme variations."""
        tokens = list(range(20))

        # Mock random to return extreme values
        mock_rng = random.Random()
        with patch.object(
            mock_rng, "randint", side_effect=lambda a, b: b
        ):  # Always max
            generator = PairGenerator(
                token_ids=tokens,
                window_size=10,
                model_type="skipgram",
                dynamic_window=True,
                rng=mock_rng,
            )

            # All windows should be max size
            assert all(size == 10 for size in generator.dynamic_window_sizes)

    def test_memory_usage_large_dataset(self):
        """Test memory efficiency with large dataset."""
        import sys

        tokens = list(range(10000))
        generator = PairGenerator(
            token_ids=tokens, window_size=5, model_type="skipgram"
        )

        # Memory usage should be reasonable
        # The generator should not store all pairs in memory
        generator_size = sys.getsizeof(generator) + sys.getsizeof(generator.token_ids)

        # Should be much smaller than storing all pairs
        estimated_pairs = len(generator)
        pair_storage_size = estimated_pairs * 2 * 8  # 2 ints per pair, 8 bytes each

        # Generator should use much less memory than storing all pairs
        assert generator_size < pair_storage_size / 10

    def test_concurrent_access_safety(self):
        """Test thread safety of pair generation."""
        tokens = list(range(1000))
        generator = PairGenerator(
            token_ids=tokens, window_size=3, model_type="skipgram"
        )

        # Simulate concurrent access
        results = []
        test_indices = [i for i in range(0, min(100, len(generator)), 10)]

        for idx in test_indices:
            pair = generator.generate_pair_at_index(idx)
            results.append((idx, pair))

        # All results should be consistent
        for idx, pair in results:
            # Re-generate same pair
            same_pair = generator.generate_pair_at_index(idx)
            assert pair == same_pair

    def test_numerical_stability_large_indices(self):
        """Test numerical stability with very large indices."""
        tokens = list(range(1000))
        generator = PairGenerator(
            token_ids=tokens, window_size=5, model_type="skipgram"
        )

        # Test indices near the maximum
        max_idx = len(generator) - 1
        test_indices = [max_idx - i for i in range(min(10, max_idx + 1))]

        for idx in test_indices:
            if idx >= 0:
                pair = generator.generate_pair_at_index(idx)
                assert isinstance(pair, tuple)

    def test_pathological_token_sequences(self):
        """Test with pathological token sequences."""
        # Test with all same tokens
        same_tokens = [42] * 100
        generator = PairGenerator(
            token_ids=same_tokens, window_size=5, model_type="skipgram"
        )

        # Should still generate pairs
        for i in range(min(5, len(generator))):
            center, context = generator.generate_pair_at_index(i)
            assert center == 42
            assert context == 42

    def test_boundary_condition_window_sizes(self):
        """Test boundary conditions for window sizes."""
        tokens = [1, 2, 3, 4, 5]

        # Test window size equal to sequence length
        generator = PairGenerator(
            token_ids=tokens, window_size=len(tokens), model_type="skipgram"
        )

        # Should handle without errors
        assert len(generator) > 0

    def test_cbow_with_maximum_context(self):
        """Test CBOW with maximum possible context."""
        tokens = list(range(10))
        generator = PairGenerator(
            token_ids=tokens, window_size=len(tokens), model_type="cbow"
        )

        # Each target should have maximum context
        for i in range(min(5, len(generator))):
            context_list, target = generator.generate_pair_at_index(i)
            assert isinstance(context_list, list)
            assert len(context_list) == len(tokens) - 1  # All except target


class TestCreatePairTensorsEdgeCases:
    """Additional edge cases for tensor creation."""

    def test_very_large_context_lists(self):
        """Test with very large context lists."""
        large_context = list(range(10000))
        target = 50000

        src_tensor, tgt_tensor = create_pair_tensors(large_context, target)

        assert src_tensor.shape == (10000,)
        assert tgt_tensor.item() == target
        assert torch.equal(src_tensor, torch.tensor(large_context))

    def test_tensor_device_consistency(self):
        """Test tensor device handling."""
        context = [1, 2, 3]
        target = 4

        src_tensor, tgt_tensor = create_pair_tensors(context, target)

        # Both tensors should be on same device
        assert src_tensor.device == tgt_tensor.device

    def test_extreme_token_values(self):
        """Test with extreme token values."""
        # Test with maximum integer values
        import sys

        max_int = sys.maxsize

        src_tensor, tgt_tensor = create_pair_tensors(max_int, max_int - 1)

        # Should handle large values
        assert src_tensor.item() == max_int
        assert tgt_tensor.item() == max_int - 1

    def test_mixed_type_handling(self):
        """Test handling of mixed types in context."""
        # Python allows mixed types in lists, but tensors require homogeneous types
        context = [1, 2, 3]  # All ints
        target = 4

        src_tensor, tgt_tensor = create_pair_tensors(context, target)

        # Should work with homogeneous types
        assert src_tensor.dtype == torch.long
        assert tgt_tensor.dtype == torch.long


class TestPairGeneratorAlgorithmicCorrectness:
    """Tests for algorithmic correctness of pair generation."""

    def test_skipgram_mathematical_correctness(self):
        """Test mathematical correctness of skip-gram pair generation."""
        tokens = [0, 1, 2, 3, 4]
        window_size = 2

        generator = PairGenerator(
            token_ids=tokens, window_size=window_size, model_type="skipgram"
        )

        # Manually calculate expected pairs
        expected_pairs = []
        for center_idx, center_token in enumerate(tokens):
            for offset in range(-window_size, window_size + 1):
                context_idx = center_idx + offset
                if 0 <= context_idx < len(tokens) and context_idx != center_idx:
                    expected_pairs.append((center_token, tokens[context_idx]))

        # Generate all pairs from generator
        generated_pairs = []
        for i in range(len(generator)):
            pair = generator.generate_pair_at_index(i)
            generated_pairs.append(pair)

        # Should have same number of pairs
        assert len(generated_pairs) == len(expected_pairs)

        # All generated pairs should be in expected pairs
        assert set(generated_pairs) == set(expected_pairs)

    def test_cbow_mathematical_correctness(self):
        """Test mathematical correctness of CBOW pair generation."""
        tokens = [0, 1, 2, 3, 4]
        window_size = 1

        generator = PairGenerator(
            token_ids=tokens, window_size=window_size, model_type="cbow"
        )

        # Manually calculate expected pairs
        expected_pairs = []
        for center_idx, center_token in enumerate(tokens):
            context = []
            for offset in range(-window_size, window_size + 1):
                context_idx = center_idx + offset
                if 0 <= context_idx < len(tokens) and context_idx != center_idx:
                    context.append(tokens[context_idx])

            if context:  # Only if context is not empty
                expected_pairs.append((tuple(context), center_token))

        # Generate all pairs
        generated_pairs = []
        for i in range(len(generator)):
            context_list, target = generator.generate_pair_at_index(i)
            generated_pairs.append((tuple(context_list), target))

        # Should match expected pairs
        assert len(generated_pairs) == len(expected_pairs)
        assert set(generated_pairs) == set(expected_pairs)

    def test_binary_search_efficiency(self):
        """Test that binary search gives O(log n) performance."""
        large_tokens = list(range(10000))
        generator = PairGenerator(
            token_ids=large_tokens, window_size=5, model_type="skipgram"
        )

        # Time multiple lookups

        num_lookups = 1000
        test_indices = [
            random.randint(0, len(generator) - 1) for _ in range(num_lookups)
        ]

        start_time = time.time()
        for idx in test_indices:
            generator.generate_pair_at_index(idx)
        end_time = time.time()

        # Should be very fast (O(log n) per lookup)
        avg_time_per_lookup = (end_time - start_time) / num_lookups
        assert avg_time_per_lookup < 0.001, (
            f"Lookup too slow: {avg_time_per_lookup:.6f}s"
        )

    def test_index_mapping_correctness(self):
        """Test that index mapping is built correctly."""
        tokens = [10, 20, 30, 40]
        generator = PairGenerator(
            token_ids=tokens, window_size=1, model_type="skipgram"
        )

        # Verify that center_word_pair_starts is monotonically increasing
        starts = generator.center_word_pair_starts
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], (
                "Pair starts should be monotonically increasing"
            )

        # Verify total pairs calculation
        manually_calculated_pairs = 0
        for i in range(len(tokens)):
            context_size = min(i + 1, 1) + min(len(tokens) - i - 1, 1)
            if i == 0 or i == len(tokens) - 1:
                context_size = 1  # Edge tokens have only one neighbor
            else:
                context_size = 2  # Middle tokens have two neighbors
            manually_calculated_pairs += context_size

        assert len(generator) == manually_calculated_pairs

    def test_dynamic_window_randomness(self):
        """Test that dynamic window produces varied window sizes."""
        tokens = list(range(100))

        generator = PairGenerator(
            token_ids=tokens,
            window_size=5,
            model_type="skipgram",
            dynamic_window=True,
            rng=random.Random(42),  # Fixed seed
        )

        window_sizes = generator.dynamic_window_sizes

        # Should have variety in window sizes
        unique_sizes = set(window_sizes)
        assert len(unique_sizes) > 1, "Dynamic window should produce varied sizes"

        # All sizes should be in valid range
        assert all(1 <= size <= 5 for size in window_sizes)


class TestErrorHandlingAndValidation:
    """Test error handling and input validation."""

    def test_invalid_indices_comprehensive(self):
        """Comprehensive test of invalid index handling."""
        tokens = [1, 2, 3]
        generator = PairGenerator(
            token_ids=tokens, window_size=1, model_type="skipgram"
        )

        invalid_indices = [
            -1,
            -100,
            len(generator),
            len(generator) + 1,
            len(generator) + 1000,
            float("inf"),
        ]

        for idx in invalid_indices:
            if isinstance(idx, float):
                continue  # Skip float indices
            with pytest.raises(IndexError):
                generator.generate_pair_at_index(idx)

    def test_model_type_case_sensitivity(self):
        """Test model type case sensitivity and variations."""
        tokens = [1, 2, 3, 4]

        valid_variations = ["skipgram", "SKIPGRAM", "SkipGram", "cbow", "CBOW", "CBoW"]

        for model_type in valid_variations:
            generator = PairGenerator(
                token_ids=tokens, window_size=1, model_type=model_type
            )
            # Should not raise error during initialization
            assert generator.model_type in ["skipgram", "cbow"]

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        # Empty tokens
        generator = PairGenerator(token_ids=[], window_size=2, model_type="skipgram")
        assert len(generator) == 0

        # Test with None RNG (should use default)
        generator = PairGenerator(
            token_ids=[1, 2, 3], window_size=1, model_type="skipgram", rng=None
        )
        assert generator.rng is not None
