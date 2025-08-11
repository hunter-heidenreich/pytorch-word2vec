"""Comprehensive tests for training pair generation utilities."""

import pytest
import random
import torch

from src.modern_word2vec.pairs import PairGenerator, create_pair_tensors


class TestPairGenerator:
    """Test cases for PairGenerator class."""

    @pytest.fixture
    def sample_tokens(self):
        """Sample token sequence for testing."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    @pytest.fixture
    def short_tokens(self):
        """Short token sequence for edge case testing."""
        return [10, 20, 30]

    def test_init_skipgram(self, sample_tokens):
        """Test PairGenerator initialization for skip-gram."""
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=2, model_type="skipgram"
        )

        assert generator.token_ids == sample_tokens
        assert generator.window_size == 2
        assert generator.model_type == "skipgram"
        assert not generator.dynamic_window
        assert generator.total_pairs > 0
        assert len(generator.center_word_pair_starts) == len(sample_tokens)

    def test_init_cbow(self, sample_tokens):
        """Test PairGenerator initialization for CBOW."""
        generator = PairGenerator(
            token_ids=sample_tokens,
            window_size=2,
            model_type="CBOW",  # Test case insensitive
        )

        assert generator.model_type == "cbow"
        assert generator.total_pairs > 0

    def test_init_with_custom_rng(self, sample_tokens):
        """Test initialization with custom random number generator."""
        custom_rng = random.Random(42)
        generator = PairGenerator(
            token_ids=sample_tokens,
            window_size=2,
            model_type="skipgram",
            rng=custom_rng,
        )

        assert generator.rng is custom_rng

    def test_skipgram_pair_generation_basic(self):
        """Test basic skip-gram pair generation."""
        tokens = [0, 1, 2, 3, 4]
        generator = PairGenerator(
            token_ids=tokens, window_size=1, model_type="skipgram"
        )

        # Test a few specific pairs
        pairs = []
        for i in range(min(10, len(generator))):
            try:
                pair = generator.generate_pair_at_index(i)
                pairs.append(pair)
            except IndexError:
                break

        # Verify pairs are valid
        for center, context in pairs:
            assert isinstance(center, int)
            assert isinstance(context, int)
            assert center in tokens
            assert context in tokens
            assert center != context

    def test_cbow_pair_generation_basic(self):
        """Test basic CBOW pair generation."""
        tokens = [0, 1, 2, 3, 4]
        generator = PairGenerator(token_ids=tokens, window_size=2, model_type="cbow")

        # Test first few pairs
        for i in range(min(3, len(generator))):
            context_list, target = generator.generate_pair_at_index(i)

            assert isinstance(context_list, list)
            assert isinstance(target, int)
            assert target in tokens
            assert len(context_list) > 0
            assert all(token in tokens for token in context_list)
            assert target not in context_list

    def test_window_size_effects(self, sample_tokens):
        """Test how different window sizes affect pair generation."""
        generators = {}
        for window_size in [1, 2, 3]:
            generators[window_size] = PairGenerator(
                token_ids=sample_tokens, window_size=window_size, model_type="skipgram"
            )

        # Larger window sizes should generally produce more pairs
        assert len(generators[2]) >= len(generators[1])
        assert len(generators[3]) >= len(generators[2])

    def test_dynamic_window_skipgram(self, sample_tokens):
        """Test dynamic window sizing for skip-gram."""
        # Use fixed seed for reproducible results
        rng = random.Random(42)
        generator = PairGenerator(
            token_ids=sample_tokens,
            window_size=3,
            model_type="skipgram",
            dynamic_window=True,
            rng=rng,
        )

        # Check that dynamic window sizes were computed
        assert len(generator.dynamic_window_sizes) == len(sample_tokens)
        assert all(1 <= size <= 3 for size in generator.dynamic_window_sizes)

    def test_pair_generation_consistency(self, sample_tokens):
        """Test that the same index always generates the same pair."""
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=2, model_type="skipgram"
        )

        # Generate same pair multiple times
        if len(generator) > 0:
            pair1 = generator.generate_pair_at_index(0)
            pair2 = generator.generate_pair_at_index(0)
            assert pair1 == pair2

    def test_all_pairs_generation_skipgram(self, short_tokens):
        """Test generation of all pairs for skip-gram with small dataset."""
        generator = PairGenerator(
            token_ids=short_tokens, window_size=1, model_type="skipgram"
        )

        all_pairs = []
        for i in range(len(generator)):
            pair = generator.generate_pair_at_index(i)
            all_pairs.append(pair)

        # Verify we got expected number of unique pairs
        assert len(all_pairs) == len(generator)

        # All pairs should be unique
        assert len(set(all_pairs)) == len(all_pairs)

    def test_all_pairs_generation_cbow(self, short_tokens):
        """Test generation of all pairs for CBOW with small dataset."""
        generator = PairGenerator(
            token_ids=short_tokens, window_size=1, model_type="cbow"
        )

        all_pairs = []
        for i in range(len(generator)):
            context, target = generator.generate_pair_at_index(i)
            all_pairs.append((tuple(context), target))

        assert len(all_pairs) == len(generator)

    def test_index_out_of_bounds(self, sample_tokens):
        """Test error handling for out-of-bounds indices."""
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=2, model_type="skipgram"
        )

        # Test negative index
        with pytest.raises(IndexError):
            generator.generate_pair_at_index(-1)

        # Test index beyond dataset size
        with pytest.raises(IndexError):
            generator.generate_pair_at_index(len(generator))

        # Test very large index
        with pytest.raises(IndexError):
            generator.generate_pair_at_index(len(generator) + 1000)

    def test_single_token_dataset(self):
        """Test edge case with single token."""
        tokens = [42]

        # Skip-gram with single token should have no pairs
        generator_sg = PairGenerator(
            token_ids=tokens, window_size=2, model_type="skipgram"
        )
        assert len(generator_sg) == 0

        # CBOW with single token should have no pairs
        generator_cbow = PairGenerator(
            token_ids=tokens, window_size=2, model_type="cbow"
        )
        assert len(generator_cbow) == 0

    def test_two_token_dataset(self):
        """Test edge case with two tokens."""
        tokens = [1, 2]

        # Skip-gram should generate 2 pairs (1->2, 2->1)
        generator_sg = PairGenerator(
            token_ids=tokens, window_size=1, model_type="skipgram"
        )
        assert len(generator_sg) == 2

        pair1 = generator_sg.generate_pair_at_index(0)
        pair2 = generator_sg.generate_pair_at_index(1)
        assert pair1 != pair2

        # CBOW should generate 2 pairs
        generator_cbow = PairGenerator(
            token_ids=tokens, window_size=1, model_type="cbow"
        )
        assert len(generator_cbow) == 2

    def test_empty_token_dataset(self):
        """Test edge case with empty token list."""
        tokens = []

        generator = PairGenerator(
            token_ids=tokens, window_size=2, model_type="skipgram"
        )
        assert len(generator) == 0

    def test_large_window_size(self, sample_tokens):
        """Test with window size larger than dataset."""
        large_window = len(sample_tokens) + 5
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=large_window, model_type="skipgram"
        )

        # Should still work without errors
        assert len(generator) > 0
        if len(generator) > 0:
            pair = generator.generate_pair_at_index(0)
            assert isinstance(pair, tuple)

    def test_zero_window_size(self, sample_tokens):
        """Test with zero window size."""
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=0, model_type="skipgram"
        )

        # Zero window size should produce no pairs
        assert len(generator) == 0

    def test_invalid_model_type(self, sample_tokens):
        """Test with invalid model type."""
        # Should not raise error during initialization
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=2, model_type="invalid"
        )

        # But should handle gracefully during pair generation
        if len(generator) > 0:
            with pytest.raises(IndexError):
                generator.generate_pair_at_index(0)

    def test_deterministic_with_seed(self, sample_tokens):
        """Test deterministic behavior with fixed seed."""

        def create_generator_with_seed(seed):
            return PairGenerator(
                token_ids=sample_tokens,
                window_size=2,
                model_type="skipgram",
                dynamic_window=True,
                rng=random.Random(seed),
            )

        gen1 = create_generator_with_seed(42)
        gen2 = create_generator_with_seed(42)

        # Same seed should produce same results
        assert len(gen1) == len(gen2)
        assert gen1.dynamic_window_sizes == gen2.dynamic_window_sizes

    def test_binary_search_correctness(self, sample_tokens):
        """Test that binary search correctly identifies center words."""
        generator = PairGenerator(
            token_ids=sample_tokens, window_size=2, model_type="skipgram"
        )

        # Test several random indices
        test_indices = (
            [0, len(generator) // 4, len(generator) // 2, len(generator) - 1]
            if len(generator) > 0
            else []
        )

        for idx in test_indices:
            if idx < len(generator):
                center, context = generator.generate_pair_at_index(idx)
                # Verify center and context are valid tokens
                assert center in sample_tokens
                assert context in sample_tokens

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large token sequence
        large_tokens = list(range(10000))
        generator = PairGenerator(
            token_ids=large_tokens, window_size=5, model_type="skipgram"
        )

        # Should handle large datasets efficiently
        assert len(generator) > 0

        # Test random access
        import time

        start_time = time.time()
        for i in range(100):  # Test 100 random accesses
            idx = random.randint(0, len(generator) - 1)
            pair = generator.generate_pair_at_index(idx)
            assert isinstance(pair, tuple)
        end_time = time.time()

        # Should be fast (adjust threshold as needed)
        assert end_time - start_time < 1.0, "Random access too slow"

    def test_context_window_boundaries(self):
        """Test that context windows respect sequence boundaries."""
        tokens = [0, 1, 2, 3, 4]
        generator = PairGenerator(
            token_ids=tokens, window_size=3, model_type="skipgram"
        )

        # Generate all pairs and verify context is within bounds
        for i in range(len(generator)):
            center, context = generator.generate_pair_at_index(i)

            # Find center position in original sequence
            center_pos = tokens.index(center)
            context_pos = tokens.index(context)

            # Context should be within window distance
            distance = abs(center_pos - context_pos)
            assert 1 <= distance <= 3, f"Invalid distance {distance} for window size 3"


class TestCreatePairTensors:
    """Test cases for create_pair_tensors function."""

    def test_skipgram_tensor_creation(self):
        """Test tensor creation for skip-gram pairs."""
        center = 5
        context = 10

        src_tensor, tgt_tensor = create_pair_tensors(center, context)

        assert isinstance(src_tensor, torch.Tensor)
        assert isinstance(tgt_tensor, torch.Tensor)
        assert src_tensor.item() == center
        assert tgt_tensor.item() == context

    def test_cbow_tensor_creation(self):
        """Test tensor creation for CBOW pairs."""
        context = [1, 2, 4, 5]
        target = 3

        src_tensor, tgt_tensor = create_pair_tensors(context, target)

        assert isinstance(src_tensor, torch.Tensor)
        assert isinstance(tgt_tensor, torch.Tensor)
        assert src_tensor.tolist() == context
        assert tgt_tensor.item() == target

    def test_empty_context_cbow(self):
        """Test tensor creation with empty context."""
        context = []
        target = 5

        src_tensor, tgt_tensor = create_pair_tensors(context, target)

        assert isinstance(src_tensor, torch.Tensor)
        assert isinstance(tgt_tensor, torch.Tensor)
        assert src_tensor.tolist() == []
        assert tgt_tensor.item() == target

    def test_single_context_cbow(self):
        """Test tensor creation with single context word."""
        context = [7]
        target = 3

        src_tensor, tgt_tensor = create_pair_tensors(context, target)

        assert src_tensor.tolist() == [7]
        assert tgt_tensor.item() == 3

    def test_tensor_dtypes(self):
        """Test that tensors have correct data types."""
        # Test skip-gram
        src_tensor, tgt_tensor = create_pair_tensors(1, 2)
        assert src_tensor.dtype == torch.long
        assert tgt_tensor.dtype == torch.long

        # Test CBOW
        src_tensor, tgt_tensor = create_pair_tensors([1, 2, 3], 4)
        assert src_tensor.dtype == torch.long
        assert tgt_tensor.dtype == torch.long

    def test_tensor_shapes(self):
        """Test tensor shapes for different input types."""
        # Skip-gram: scalar inputs should produce scalar tensors
        src_tensor, tgt_tensor = create_pair_tensors(1, 2)
        assert src_tensor.shape == torch.Size([])
        assert tgt_tensor.shape == torch.Size([])

        # CBOW: list input should produce 1D tensor
        src_tensor, tgt_tensor = create_pair_tensors([1, 2, 3], 4)
        assert src_tensor.shape == torch.Size([3])
        assert tgt_tensor.shape == torch.Size([])

    def test_large_values(self):
        """Test with large token ID values."""
        large_id = 1000000
        src_tensor, tgt_tensor = create_pair_tensors(large_id, large_id + 1)

        assert src_tensor.item() == large_id
        assert tgt_tensor.item() == large_id + 1

    def test_negative_values(self):
        """Test with negative token IDs (edge case)."""
        src_tensor, tgt_tensor = create_pair_tensors(-1, -2)

        assert src_tensor.item() == -1
        assert tgt_tensor.item() == -2


class TestPairGeneratorIntegration:
    """Integration tests combining PairGenerator with tensor creation."""

    def test_skipgram_end_to_end(self):
        """Test complete skip-gram pipeline."""
        tokens = [10, 20, 30, 40, 50]
        generator = PairGenerator(
            token_ids=tokens, window_size=2, model_type="skipgram"
        )

        # Generate several pairs and convert to tensors
        tensor_pairs = []
        for i in range(min(5, len(generator))):
            center, context = generator.generate_pair_at_index(i)
            src_tensor, tgt_tensor = create_pair_tensors(center, context)
            tensor_pairs.append((src_tensor, tgt_tensor))

        # Verify all tensors are valid
        for src, tgt in tensor_pairs:
            assert isinstance(src, torch.Tensor)
            assert isinstance(tgt, torch.Tensor)
            assert src.item() in tokens
            assert tgt.item() in tokens

    def test_cbow_end_to_end(self):
        """Test complete CBOW pipeline."""
        tokens = [10, 20, 30, 40, 50]
        generator = PairGenerator(token_ids=tokens, window_size=2, model_type="cbow")

        # Generate pairs and convert to tensors
        for i in range(min(3, len(generator))):
            context_list, target = generator.generate_pair_at_index(i)
            src_tensor, tgt_tensor = create_pair_tensors(context_list, target)

            assert isinstance(src_tensor, torch.Tensor)
            assert isinstance(tgt_tensor, torch.Tensor)
            assert tgt_tensor.item() in tokens
            assert all(token in tokens for token in src_tensor.tolist())

    def test_batch_generation(self):
        """Test generating batches of pairs."""
        tokens = list(range(20))
        generator = PairGenerator(
            token_ids=tokens, window_size=2, model_type="skipgram"
        )

        batch_size = 8
        batch_indices = random.sample(
            range(len(generator)), min(batch_size, len(generator))
        )

        batch_src = []
        batch_tgt = []

        for idx in batch_indices:
            center, context = generator.generate_pair_at_index(idx)
            src_tensor, tgt_tensor = create_pair_tensors(center, context)
            batch_src.append(src_tensor)
            batch_tgt.append(tgt_tensor)

        # Stack into batch tensors
        if batch_src:
            batch_src_tensor = torch.stack(batch_src)
            batch_tgt_tensor = torch.stack(batch_tgt)

            assert batch_src_tensor.shape == (len(batch_indices),)
            assert batch_tgt_tensor.shape == (len(batch_indices),)

    def test_reproducibility_integration(self):
        """Test reproducibility of entire pipeline."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        def generate_pairs_with_seed(seed):
            generator = PairGenerator(
                token_ids=tokens,
                window_size=2,
                model_type="skipgram",
                dynamic_window=True,
                rng=random.Random(seed),
            )

            pairs = []
            for i in range(min(5, len(generator))):
                pair = generator.generate_pair_at_index(i)
                pairs.append(pair)
            return pairs

        pairs1 = generate_pairs_with_seed(42)
        pairs2 = generate_pairs_with_seed(42)

        assert pairs1 == pairs2
