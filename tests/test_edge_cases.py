"""Additional edge case tests for hierarchical softmax implementation."""

import pytest
import torch
import math

from src.modern_word2vec.hierarchical_softmax import HierarchicalSoftmax, HuffmanTree


class TestHierarchicalSoftmaxEdgeCases:
    """Additional edge case tests for hierarchical softmax."""

    def test_zero_frequency_words(self):
        """Test handling of words with zero frequency."""
        word_to_idx = {"common": 0, "rare": 1, "zero_freq": 2}
        word_counts = {"common": 100, "rare": 1, "zero_freq": 0}

        # Should handle zero frequency gracefully (treated as frequency 1)
        tree = HuffmanTree(word_counts, word_to_idx)

        # All words should have valid codes
        for word_idx in word_to_idx.values():
            assert word_idx in tree.word_codes
            assert word_idx in tree.word_paths

    def test_identical_frequencies(self):
        """Test handling when all words have identical frequencies."""
        word_to_idx = {"word1": 0, "word2": 1, "word3": 2, "word4": 3}
        word_counts = {"word1": 50, "word2": 50, "word3": 50, "word4": 50}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=4,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        input_embeddings = torch.randn(2, 16)
        target_words = torch.tensor([0, 1])

        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

    def test_extreme_frequency_distribution(self):
        """Test with extremely skewed frequency distribution."""
        word_to_idx = {"very_common": 0, "rare1": 1, "rare2": 2, "rare3": 3}
        word_counts = {"very_common": 1000000, "rare1": 1, "rare2": 1, "rare3": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=4,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        # Very common word should have shorter path
        tree = hsoftmax.huffman_tree
        common_path_len = len(tree.word_codes[0])
        rare_path_len = len(tree.word_codes[1])

        # Common word should have shorter or equal path
        assert common_path_len <= rare_path_len

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        word_to_idx = {"a": 0, "b": 1, "c": 2}
        word_counts = {"a": 10, "b": 5, "c": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=4,
            vocab_size=3,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        input_embeddings = torch.randn(1, 4)
        target_words = torch.tensor([1])

        loss = hsoftmax(input_embeddings, target_words)
        assert loss.shape == torch.Size([])  # Scalar
        assert torch.isfinite(loss)

    def test_very_large_batch(self):
        """Test with large batch size."""
        word_to_idx = {"x": 0, "y": 1, "z": 2}
        word_counts = {"x": 5, "y": 3, "z": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=3,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        batch_size = 1000
        input_embeddings = torch.randn(batch_size, 8)
        target_words = torch.randint(0, 3, (batch_size,))

        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

    def test_nan_input_handling(self):
        """Test robustness to NaN inputs."""
        word_to_idx = {"a": 0, "b": 1}
        word_counts = {"a": 5, "b": 3}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=4,
            vocab_size=2,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        # Create input with NaN
        input_embeddings = torch.tensor(
            [[1.0, 2.0, float("nan"), 4.0], [5.0, 6.0, 7.0, 8.0]]
        )
        target_words = torch.tensor([0, 1])

        loss = hsoftmax(input_embeddings, target_words)
        # Loss should be NaN when input contains NaN
        assert torch.isnan(loss)

    def test_inf_input_handling(self):
        """Test robustness to infinite inputs."""
        word_to_idx = {"a": 0, "b": 1}
        word_counts = {"a": 5, "b": 3}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=4,
            vocab_size=2,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        # Create input with infinity
        input_embeddings = torch.tensor(
            [[float("inf"), 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )
        target_words = torch.tensor([0, 1])

        loss = hsoftmax(input_embeddings, target_words)
        # Should handle infinite values (may result in NaN or inf)
        assert torch.isnan(loss) or torch.isinf(loss)

    def test_target_word_out_of_bounds(self):
        """Test error handling for out-of-bounds target words."""
        word_to_idx = {"a": 0, "b": 1, "c": 2}
        word_counts = {"a": 5, "b": 3, "c": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=4,
            vocab_size=3,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        input_embeddings = torch.randn(2, 4)
        target_words = torch.tensor([0, 5])  # Index 5 is out of bounds

        # This should raise an error or handle gracefully
        with pytest.raises((IndexError, RuntimeError)):
            loss = hsoftmax(input_embeddings, target_words)

    def test_empty_paths_handling(self):
        """Test handling when some words have empty paths (single word case)."""
        # Single word vocabulary results in empty paths
        word_to_idx = {"only": 0}
        word_counts = {"only": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=4,
            vocab_size=1,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        input_embeddings = torch.randn(2, 4)
        target_words = torch.tensor([0, 0])

        # Should handle empty paths gracefully
        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

    def test_probability_numerical_precision(self):
        """Test numerical precision in probability computation."""
        word_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
        word_counts = {"a": 10, "b": 8, "c": 6, "d": 4}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=32,
            vocab_size=4,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        # Use very small embeddings to test precision
        input_embeddings = torch.full((1, 32), 1e-7)

        probs = hsoftmax.predict_probabilities(input_embeddings)

        # Probabilities should sum to 1 within numerical precision
        prob_sum = probs.sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(1), atol=1e-6)

        # All probabilities should be positive
        assert torch.all(probs > 0)

    def test_gradient_accumulation_compatibility(self):
        """Test compatibility with gradient accumulation."""
        word_to_idx = {"word1": 0, "word2": 1, "word3": 2}
        word_counts = {"word1": 5, "word2": 3, "word3": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=3,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        optimizer = torch.optim.SGD(hsoftmax.parameters(), lr=0.01)

        # Simulate gradient accumulation
        accumulation_steps = 3
        total_loss = 0

        for step in range(accumulation_steps):
            input_embeddings = torch.randn(2, 8)
            target_words = torch.randint(0, 3, (2,))

            loss = hsoftmax(input_embeddings, target_words) / accumulation_steps
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        assert math.isfinite(total_loss)

    def test_mixed_precision_compatibility(self):
        """Test compatibility with mixed precision training."""
        word_to_idx = {"a": 0, "b": 1, "c": 2}
        word_counts = {"a": 5, "b": 3, "c": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=3,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        # Test with half precision
        if torch.cuda.is_available():
            hsoftmax = hsoftmax.half().cuda()
            input_embeddings = torch.randn(2, 8, dtype=torch.half, device="cuda")
            target_words = torch.tensor([0, 1], device="cuda")
        else:
            # CPU doesn't support half precision for all operations, use float32
            hsoftmax = hsoftmax.float()
            input_embeddings = torch.randn(2, 8, dtype=torch.float32)
            target_words = torch.tensor([0, 1])

        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

    def test_reproducibility_with_seeds(self):
        """Test that results are reproducible with fixed seeds."""
        word_to_idx = {"word1": 0, "word2": 1, "word3": 2}
        word_counts = {"word1": 10, "word2": 5, "word3": 1}

        def run_forward_pass():
            torch.manual_seed(42)
            hsoftmax = HierarchicalSoftmax(
                embedding_dim=16,
                vocab_size=3,
                word_counts=word_counts,
                word_to_idx=word_to_idx,
            )

            torch.manual_seed(123)
            input_embeddings = torch.randn(3, 16)
            target_words = torch.tensor([0, 1, 2])

            return hsoftmax(input_embeddings, target_words)

        # Run twice with same seeds
        loss1 = run_forward_pass()
        loss2 = run_forward_pass()

        # Results should be identical
        assert torch.allclose(loss1, loss2)

    def test_memory_efficiency_large_vocab(self):
        """Test memory efficiency with large vocabulary."""
        vocab_size = 5000
        word_to_idx = {f"word_{i:05d}": i for i in range(vocab_size)}

        # Create Zipf-like distribution
        word_counts = {}
        for i, word in enumerate(word_to_idx.keys()):
            word_counts[word] = max(1, 10000 // (i + 1))

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=128,
            vocab_size=vocab_size,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        # Memory usage should be roughly (vocab_size - 1) * embedding_dim parameters
        total_params = sum(p.numel() for p in hsoftmax.parameters())
        expected_params_approx = (vocab_size - 1) * 128

        # Allow 10% tolerance
        assert abs(total_params - expected_params_approx) / expected_params_approx < 0.1
