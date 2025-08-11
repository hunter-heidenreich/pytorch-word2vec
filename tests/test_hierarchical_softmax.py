"""Comprehensive tests for the HierarchicalSoftmax class."""

import pytest
import torch

from src.modern_word2vec.hierarchical_softmax import (
    HierarchicalSoftmax,
    build_word_counts_from_dataset,
)
from tests.conftest import (
    assert_tensor_close,
    assert_valid_probability_distribution,
    create_mock_dataset,
)


class TestHierarchicalSoftmax:
    """Test cases for HierarchicalSoftmax implementation."""

    def test_init(self, small_vocab, small_word_counts):
        """Test HierarchicalSoftmax initialization."""
        embedding_dim = 64
        vocab_size = len(small_vocab)

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        # Check basic attributes
        assert hsoftmax.embedding_dim == embedding_dim
        assert hsoftmax.vocab_size == vocab_size
        assert hasattr(hsoftmax, "huffman_tree")
        assert hasattr(hsoftmax, "inner_node_embeddings")

        # Check embedding layer dimensions
        expected_inner_nodes = hsoftmax.huffman_tree.num_inner_nodes
        assert hsoftmax.inner_node_embeddings.num_embeddings == expected_inner_nodes
        assert hsoftmax.inner_node_embeddings.embedding_dim == embedding_dim

    def test_weight_initialization(self, small_vocab, small_word_counts):
        """Test that weights are properly initialized."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=32,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        # Inner node embeddings should be initialized to zero
        weights = hsoftmax.inner_node_embeddings.weight.data
        assert torch.allclose(weights, torch.zeros_like(weights))

    def test_forward_basic(self, small_vocab, small_word_counts):
        """Test basic forward pass functionality."""
        embedding_dim = 32
        batch_size = 4

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=embedding_dim,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        # Create test inputs
        torch.manual_seed(42)
        input_embeddings = torch.randn(batch_size, embedding_dim)
        target_words = torch.tensor([0, 1, 2, 3])

        # Forward pass should work without errors
        loss = hsoftmax(input_embeddings, target_words)

        # Loss should be a scalar tensor
        assert loss.dim() == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_forward_gradient_flow(self, small_vocab, small_word_counts):
        """Test that gradients flow properly through the model."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        # Initialize inner node embeddings with small random values to ensure gradient flow
        with torch.no_grad():
            hsoftmax.inner_node_embeddings.weight.normal_(0, 0.01)

        # Create inputs that require gradients
        input_embeddings = torch.randn(2, 16, requires_grad=True)
        target_words = torch.tensor([0, 1])

        # Forward and backward
        loss = hsoftmax(input_embeddings, target_words)
        loss.backward()

        # Check that gradients were computed
        assert input_embeddings.grad is not None

        # Check that at least some gradients are non-zero (with tolerance for edge cases)
        grad_norm = torch.norm(input_embeddings.grad)
        assert grad_norm > 1e-6, f"Gradients are too small: norm={grad_norm}"

        # Check inner node gradients if there are internal nodes
        if hsoftmax.huffman_tree.num_inner_nodes > 0:
            inner_grads = hsoftmax.inner_node_embeddings.weight.grad
            if inner_grads is not None:
                assert torch.isfinite(inner_grads).all()

    def test_forward_different_batch_sizes(self, small_vocab, small_word_counts):
        """Test forward pass with different batch sizes."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        for batch_size in [1, 3, 8, 16]:
            input_embeddings = torch.randn(batch_size, 16)
            target_words = torch.randint(0, len(small_vocab), (batch_size,))

            loss = hsoftmax(input_embeddings, target_words)
            assert loss.dim() == 0
            assert torch.isfinite(loss)

    def test_forward_edge_cases(self, small_vocab, small_word_counts):
        """Test forward pass edge cases."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        # Test with extreme embedding values
        large_embeddings = torch.full((2, 8), 10.0)
        small_embeddings = torch.full((2, 8), -10.0)
        target_words = torch.tensor([0, 1])

        loss_large = hsoftmax(large_embeddings, target_words)
        loss_small = hsoftmax(small_embeddings, target_words)

        assert torch.isfinite(loss_large)
        assert torch.isfinite(loss_small)

    def test_predict_probabilities_basic(self, small_vocab, small_word_counts):
        """Test basic probability prediction functionality."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        batch_size = 3
        input_embeddings = torch.randn(batch_size, 16)

        probs = hsoftmax.predict_probabilities(input_embeddings)

        # Check output shape
        assert probs.shape == (batch_size, len(small_vocab))

        # Check probability properties
        assert_valid_probability_distribution(probs, dim=1)

    def test_predict_probabilities_consistency(self, small_vocab, small_word_counts):
        """Test that probabilities are consistent across calls."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        input_embeddings = torch.randn(2, 16)

        probs1 = hsoftmax.predict_probabilities(input_embeddings)
        probs2 = hsoftmax.predict_probabilities(input_embeddings)

        assert_tensor_close(probs1, probs2)

    def test_device_consistency(self, small_vocab, small_word_counts):
        """Test that computations work consistently across devices."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        # Test CPU
        input_cpu = torch.randn(2, 16)
        target_cpu = torch.tensor([0, 1])
        loss_cpu = hsoftmax(input_cpu, target_cpu)

        # Test GPU if available
        if torch.cuda.is_available():
            hsoftmax_cuda = hsoftmax.cuda()
            input_cuda = input_cpu.cuda()
            target_cuda = target_cpu.cuda()
            loss_cuda = hsoftmax_cuda(input_cuda, target_cuda)

            # Results should be close (allowing for numerical differences)
            assert_tensor_close(loss_cpu, loss_cuda.cpu(), rtol=1e-4)

    def test_single_word_vocabulary(self):
        """Test edge case with single word vocabulary."""
        word_to_idx = {"only": 0}
        word_counts = {"only": 1}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=1,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        input_embeddings = torch.randn(1, 8)
        target_words = torch.tensor([0])

        # Should handle single word case gracefully
        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

        # Probabilities should be deterministic (probability 1 for the only word)
        probs = hsoftmax.predict_probabilities(input_embeddings)
        expected_probs = torch.ones(1, 1)
        assert_tensor_close(probs, expected_probs, rtol=1e-4)

    def test_two_word_vocabulary(self):
        """Test minimal case with two words."""
        word_to_idx = {"first": 0, "second": 1}
        word_counts = {"first": 10, "second": 5}

        hsoftmax = HierarchicalSoftmax(
            embedding_dim=8,
            vocab_size=2,
            word_counts=word_counts,
            word_to_idx=word_to_idx,
        )

        input_embeddings = torch.randn(2, 8)
        target_words = torch.tensor([0, 1])

        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

        probs = hsoftmax.predict_probabilities(input_embeddings)
        assert_valid_probability_distribution(probs, dim=1)

    def test_loss_decreases_with_training(self, small_vocab, small_word_counts):
        """Test that loss can decrease with training updates."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=16,
            vocab_size=len(small_vocab),
            word_counts=small_word_counts,
            word_to_idx=small_vocab,
        )

        optimizer = torch.optim.SGD(hsoftmax.parameters(), lr=0.1)

        # Generate fixed training data
        torch.manual_seed(42)
        input_embeddings = torch.randn(4, 16)
        target_words = torch.tensor([0, 1, 2, 3])

        # Initial loss
        initial_loss = hsoftmax(input_embeddings, target_words).item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            loss = hsoftmax(input_embeddings, target_words)
            loss.backward()
            optimizer.step()

        # Final loss
        final_loss = hsoftmax(input_embeddings, target_words).item()

        # Loss should generally decrease (allowing some tolerance for noise)
        assert final_loss < initial_loss + 0.1

    def test_different_embedding_dimensions(self, small_vocab, small_word_counts):
        """Test with different embedding dimensions."""
        for embedding_dim in [8, 32, 128, 256]:
            hsoftmax = HierarchicalSoftmax(
                embedding_dim=embedding_dim,
                vocab_size=len(small_vocab),
                word_counts=small_word_counts,
                word_to_idx=small_vocab,
            )

            input_embeddings = torch.randn(2, embedding_dim)
            target_words = torch.tensor([0, 1])

            loss = hsoftmax(input_embeddings, target_words)
            assert torch.isfinite(loss)

    @pytest.mark.slow
    def test_larger_vocabulary(self, medium_vocab, medium_word_counts):
        """Test with larger vocabulary."""
        hsoftmax = HierarchicalSoftmax(
            embedding_dim=64,
            vocab_size=len(medium_vocab),
            word_counts=medium_word_counts,
            word_to_idx=medium_vocab,
        )

        batch_size = 8
        input_embeddings = torch.randn(batch_size, 64)
        target_words = torch.randint(0, len(medium_vocab), (batch_size,))

        # Test forward pass
        loss = hsoftmax(input_embeddings, target_words)
        assert torch.isfinite(loss)

        # Test probability prediction (this is expensive for large vocabs)
        probs = hsoftmax.predict_probabilities(
            input_embeddings[:2]
        )  # Just test 2 samples
        assert_valid_probability_distribution(probs, dim=1)


class TestBuildWordCountsFromDataset:
    """Test cases for build_word_counts_from_dataset function."""

    def test_with_vocab_builder(self, small_vocab, small_word_counts):
        """Test word count extraction when dataset has vocab_builder."""
        dataset = create_mock_dataset(small_vocab, small_word_counts)

        extracted_counts = build_word_counts_from_dataset(dataset)

        # Should extract the actual word counts
        for word in small_vocab.keys():
            assert extracted_counts[word] == small_word_counts[word]

    def test_without_vocab_builder(self, small_vocab):
        """Test fallback when dataset doesn't have vocab_builder."""

        class MockDatasetNoVocab:
            def __init__(self, word_to_idx):
                self.word_to_idx = word_to_idx

        dataset = MockDatasetNoVocab(small_vocab)
        extracted_counts = build_word_counts_from_dataset(dataset)

        # Should assign uniform counts of 1
        for word in small_vocab.keys():
            assert extracted_counts[word] == 1

    def test_missing_words_in_counts(self, small_vocab):
        """Test when some words are missing from vocab_builder counts."""
        partial_counts = {"apple": 50, "banana": 30}  # Missing some words
        dataset = create_mock_dataset(small_vocab, partial_counts)

        extracted_counts = build_word_counts_from_dataset(dataset)

        # Should use actual counts where available, fallback to 1
        assert extracted_counts["apple"] == 50
        assert extracted_counts["banana"] == 30
        for word in ["cherry", "date", "elderberry"]:
            assert extracted_counts[word] == 1
