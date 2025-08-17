"""Comprehensive tests for Word2Vec model definitions.

This test suite provides extensive coverage of the model classes,
including:

- Word2VecBase initialization and configuration
- SkipGram model functionality
- CBOW model functionality
- Full softmax and hierarchical softmax integration
- Weight initialization verification
- Forward pass functionality
- Loss computation
- Error handling and edge cases
- Device compatibility

Coverage: Aiming for >95% of models/__init__.py module
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.modern_word2vec.models import Word2VecBase, SkipGramModel, CBOWModel


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.word_to_idx = {f"word_{i}": i for i in range(vocab_size)}
        self.idx_to_word = {i: f"word_{i}" for i in range(vocab_size)}


class MockVocabBuilder:
    """Mock vocabulary builder for testing."""

    def __init__(self, word_counts):
        self.word_counts = word_counts


def create_mock_dataset_with_vocab_builder(vocab_size=100):
    """Create a mock dataset with vocabulary builder."""
    dataset = MockDataset(vocab_size)
    word_counts = {
        f"word_{i}": vocab_size - i for i in range(vocab_size)
    }  # Decreasing frequency
    dataset.vocab_builder = MockVocabBuilder(word_counts)
    return dataset


class TestWord2VecBase:
    """Test cases for Word2VecBase class."""

    def test_init_full_softmax_default(self):
        """Test Word2VecBase initialization with full softmax (default)."""
        vocab_size = 100
        embedding_dim = 50

        model = Word2VecBase(vocab_size, embedding_dim)

        assert model.vocab_size == vocab_size
        assert model.embedding_dim == embedding_dim
        assert model.output_layer_type == "full_softmax"
        assert isinstance(model.in_embeddings, nn.Embedding)
        assert model.in_embeddings.num_embeddings == vocab_size
        assert model.in_embeddings.embedding_dim == embedding_dim
        assert isinstance(model.out_embeddings, nn.Embedding)
        assert model.out_embeddings.num_embeddings == vocab_size
        assert model.out_embeddings.embedding_dim == embedding_dim
        assert model.hierarchical_softmax is None

    def test_init_hierarchical_softmax_with_dataset(self):
        """Test Word2VecBase initialization with hierarchical softmax."""
        vocab_size = 100
        embedding_dim = 50
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax") as mock_hs:
            model = Word2VecBase(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            assert model.vocab_size == vocab_size
            assert model.embedding_dim == embedding_dim
            assert model.output_layer_type == "hierarchical_softmax"
            assert isinstance(model.in_embeddings, nn.Embedding)
            assert model.out_embeddings is None
            mock_hs.assert_called_once()

    def test_init_negative_sampling_with_dataset(self):
        """Test Word2VecBase initialization with negative sampling."""
        vocab_size = 100
        embedding_dim = 50
        num_negative = 10
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.NegativeSampling") as mock_ns:
            model = Word2VecBase(
                vocab_size,
                embedding_dim,
                output_layer_type="negative_sampling",
                dataset=dataset,
                num_negative=num_negative,
            )

            assert model.vocab_size == vocab_size
            assert model.embedding_dim == embedding_dim
            assert model.output_layer_type == "negative_sampling"
            assert isinstance(model.in_embeddings, nn.Embedding)
            assert model.out_embeddings is None
            assert model.hierarchical_softmax is None
            mock_ns.assert_called_once()

    def test_init_negative_sampling_without_dataset(self):
        """Test Word2VecBase initialization with negative sampling but no dataset."""
        vocab_size = 100
        embedding_dim = 50

        with pytest.raises(
            ValueError, match="Dataset is required for negative sampling"
        ):
            Word2VecBase(
                vocab_size, embedding_dim, output_layer_type="negative_sampling"
            )

    def test_weight_initialization_full_softmax(self):
        """Test that weights are initialized correctly for full softmax."""
        vocab_size = 10
        embedding_dim = 5

        model = Word2VecBase(vocab_size, embedding_dim)

        # Check input embeddings initialization
        in_weights = model.in_embeddings.weight.data
        expected_bound = 0.5 / embedding_dim
        assert torch.all(in_weights >= -expected_bound)
        assert torch.all(in_weights <= expected_bound)

        # Check output embeddings initialization (should be zero)
        out_weights = model.out_embeddings.weight.data
        assert torch.allclose(out_weights, torch.zeros_like(out_weights))

    def test_weight_initialization_hierarchical_softmax(self):
        """Test that weights are initialized correctly for hierarchical softmax."""
        vocab_size = 10
        embedding_dim = 5
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = Word2VecBase(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            # Check input embeddings initialization
            in_weights = model.in_embeddings.weight.data
            expected_bound = 0.5 / embedding_dim
            assert torch.all(in_weights >= -expected_bound)
            assert torch.all(in_weights <= expected_bound)

            # No output embeddings for hierarchical softmax
            assert model.out_embeddings is None

    def test_compute_scores_full_softmax(self):
        """Test score computation for full softmax."""
        vocab_size = 10
        embedding_dim = 5
        batch_size = 3

        model = Word2VecBase(vocab_size, embedding_dim)
        input_embeddings = torch.randn(batch_size, embedding_dim)

        scores = model._compute_scores(input_embeddings)

        assert scores.shape == (batch_size, vocab_size)
        # Scores should be real numbers
        assert torch.all(torch.isfinite(scores))

    def test_compute_scores_negative_sampling(self):
        """Test score computation for negative sampling."""
        vocab_size = 10
        embedding_dim = 5
        batch_size = 3
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.NegativeSampling") as mock_ns:
            mock_ns_instance = Mock()
            mock_ns.return_value = mock_ns_instance
            mock_scores = torch.randn(batch_size, vocab_size)
            mock_ns_instance.predict_scores.return_value = mock_scores

            model = Word2VecBase(
                vocab_size,
                embedding_dim,
                output_layer_type="negative_sampling",
                dataset=dataset,
            )
            model.negative_sampling = mock_ns_instance

            input_embeddings = torch.randn(batch_size, embedding_dim)
            scores = model._compute_scores(input_embeddings)

            assert torch.allclose(scores, mock_scores)
            mock_ns_instance.predict_scores.assert_called_once_with(input_embeddings)

    def test_compute_loss_full_softmax(self):
        """Test loss computation for full softmax."""
        vocab_size = 10
        embedding_dim = 5
        batch_size = 3

        model = Word2VecBase(vocab_size, embedding_dim)
        input_embeddings = torch.randn(batch_size, embedding_dim)
        targets = torch.randint(0, vocab_size, (batch_size,))

        loss = model.compute_loss(input_embeddings, targets)

        assert loss.shape == ()  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
        assert torch.isfinite(loss)

    def test_compute_loss_negative_sampling(self):
        """Test loss computation for negative sampling."""
        vocab_size = 10
        embedding_dim = 5
        batch_size = 3
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        mock_ns = Mock()
        mock_ns.return_value = torch.tensor(1.8)

        with patch(
            "src.modern_word2vec.models.NegativeSampling", return_value=mock_ns
        ):
            model = Word2VecBase(
                vocab_size,
                embedding_dim,
                output_layer_type="negative_sampling",
                dataset=dataset,
            )

            input_embeddings = torch.randn(batch_size, embedding_dim)
            targets = torch.randint(0, vocab_size, (batch_size,))

            loss = model.compute_loss(input_embeddings, targets)

            mock_ns.assert_called_once_with(input_embeddings, targets)
            assert loss == 1.8

    def test_different_vocab_sizes(self):
        """Test model with different vocabulary sizes."""
        embedding_dim = 64

        for vocab_size in [1, 10, 100, 1000]:
            model = Word2VecBase(vocab_size, embedding_dim)
            assert model.vocab_size == vocab_size
            assert model.in_embeddings.num_embeddings == vocab_size
            assert model.out_embeddings.num_embeddings == vocab_size

    def test_different_embedding_dimensions(self):
        """Test model with different embedding dimensions."""
        vocab_size = 100

        for embedding_dim in [1, 32, 64, 128, 300]:
            model = Word2VecBase(vocab_size, embedding_dim)
            assert model.embedding_dim == embedding_dim
            assert model.in_embeddings.embedding_dim == embedding_dim
            assert model.out_embeddings.embedding_dim == embedding_dim

    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 4

        model = Word2VecBase(vocab_size, embedding_dim)

        # Test CPU
        input_embeddings = torch.randn(batch_size, embedding_dim)
        targets = torch.randint(0, vocab_size, (batch_size,))

        scores = model._compute_scores(input_embeddings)
        loss = model.compute_loss(input_embeddings, targets)

        assert scores.device.type == "cpu"
        assert loss.device.type == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            input_embeddings = input_embeddings.cuda()
            targets = targets.cuda()

            scores = model._compute_scores(input_embeddings)
            loss = model.compute_loss(input_embeddings, targets)

            assert scores.device.type == "cuda"
            assert loss.device.type == "cuda"


class TestSkipGramModel:
    """Test cases for SkipGramModel class."""

    def test_init_inherits_from_base(self):
        """Test that SkipGramModel properly inherits from Word2VecBase."""
        vocab_size = 100
        embedding_dim = 50

        model = SkipGramModel(vocab_size, embedding_dim)

        assert isinstance(model, Word2VecBase)
        assert model.vocab_size == vocab_size
        assert model.embedding_dim == embedding_dim
        assert model.output_layer_type == "full_softmax"

    def test_init_with_hierarchical_softmax(self):
        """Test SkipGramModel initialization with hierarchical softmax."""
        vocab_size = 100
        embedding_dim = 50
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = SkipGramModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            assert model.output_layer_type == "hierarchical_softmax"
            assert model.out_embeddings is None

    def test_forward_full_softmax(self):
        """Test forward pass with full softmax."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 8

        model = SkipGramModel(vocab_size, embedding_dim)
        target_words = torch.randint(0, vocab_size, (batch_size,))

        output = model.forward(target_words)

        assert output.shape == (batch_size, vocab_size)
        assert torch.all(torch.isfinite(output))

    def test_forward_hierarchical_softmax_with_context(self):
        """Test forward pass with hierarchical softmax and context targets."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 8
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        mock_loss = torch.tensor(1.5)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = SkipGramModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )
            model.hierarchical_softmax = Mock(return_value=mock_loss)

            target_words = torch.randint(0, vocab_size, (batch_size,))
            context_targets = torch.randint(0, vocab_size, (batch_size,))

            output = model.forward(target_words, context_targets)

            assert output == mock_loss

    def test_forward_hierarchical_softmax_without_context(self):
        """Test forward pass with hierarchical softmax but no context targets."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 8
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = SkipGramModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            target_words = torch.randint(0, vocab_size, (batch_size,))

            with pytest.raises(
                ValueError, match="context_targets required for hierarchical softmax"
            ):
                model.forward(target_words)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        vocab_size = 30
        embedding_dim = 16

        model = SkipGramModel(vocab_size, embedding_dim)

        for batch_size in [1, 5, 10, 32]:
            target_words = torch.randint(0, vocab_size, (batch_size,))
            output = model.forward(target_words)
            assert output.shape == (batch_size, vocab_size)

    def test_forward_edge_case_single_word(self):
        """Test forward pass with single word batch."""
        vocab_size = 10
        embedding_dim = 8

        model = SkipGramModel(vocab_size, embedding_dim)
        target_words = torch.tensor([3])

        output = model.forward(target_words)

        assert output.shape == (1, vocab_size)
        assert torch.all(torch.isfinite(output))

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        vocab_size = 20
        embedding_dim = 10

        model = SkipGramModel(vocab_size, embedding_dim)

        # Use specific indices that exist in vocabulary
        target_words = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([4, 5, 6, 7])

        # Forward pass
        scores = model.forward(target_words)
        loss = torch.nn.functional.cross_entropy(scores, targets)

        # Backward pass
        loss.backward()

        # Check gradients exist for output embeddings (used in loss computation)
        assert model.out_embeddings.weight.grad is not None
        # Check that gradients are non-zero
        assert torch.sum(torch.abs(model.out_embeddings.weight.grad)) > 0


class TestCBOWModel:
    """Test cases for CBOWModel class."""

    def test_init_inherits_from_base(self):
        """Test that CBOWModel properly inherits from Word2VecBase."""
        vocab_size = 100
        embedding_dim = 50

        model = CBOWModel(vocab_size, embedding_dim)

        assert isinstance(model, Word2VecBase)
        assert model.vocab_size == vocab_size
        assert model.embedding_dim == embedding_dim
        assert model.output_layer_type == "full_softmax"

    def test_init_with_hierarchical_softmax(self):
        """Test CBOWModel initialization with hierarchical softmax."""
        vocab_size = 100
        embedding_dim = 50
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = CBOWModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            assert model.output_layer_type == "hierarchical_softmax"
            assert model.out_embeddings is None

    def test_forward_full_softmax(self):
        """Test forward pass with full softmax."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 8
        window_size = 2
        context_size = 2 * window_size

        model = CBOWModel(vocab_size, embedding_dim)
        context_words = torch.randint(0, vocab_size, (batch_size, context_size))

        output = model.forward(context_words)

        assert output.shape == (batch_size, vocab_size)
        assert torch.all(torch.isfinite(output))

    def test_forward_hierarchical_softmax_with_targets(self):
        """Test forward pass with hierarchical softmax and center targets."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 8
        context_size = 4
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        mock_loss = torch.tensor(2.0)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = CBOWModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )
            model.hierarchical_softmax = Mock(return_value=mock_loss)

            context_words = torch.randint(0, vocab_size, (batch_size, context_size))
            center_targets = torch.randint(0, vocab_size, (batch_size,))

            output = model.forward(context_words, center_targets)

            assert output == mock_loss

    def test_forward_hierarchical_softmax_without_targets(self):
        """Test forward pass with hierarchical softmax but no center targets."""
        vocab_size = 50
        embedding_dim = 32
        batch_size = 8
        context_size = 4
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax"):
            model = CBOWModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            context_words = torch.randint(0, vocab_size, (batch_size, context_size))

            with pytest.raises(
                ValueError, match="center_targets required for hierarchical softmax"
            ):
                model.forward(context_words)

    def test_forward_context_averaging(self):
        """Test that context words are properly averaged."""
        vocab_size = 10
        embedding_dim = 4

        model = CBOWModel(vocab_size, embedding_dim)

        # Create known context words
        context_words = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert context_words.shape == (2, 4)  # batch_size=2, context_size=4

        # Forward pass
        output = model.forward(context_words)

        # Manually compute expected result
        context_embeds = model.in_embeddings(context_words)  # (2, 4, 4)
        expected_context_vector = torch.mean(context_embeds, dim=1)  # (2, 4)
        expected_scores = torch.matmul(
            expected_context_vector, model.out_embeddings.weight.t()
        )

        assert torch.allclose(output, expected_scores, atol=1e-6)

    def test_forward_different_context_sizes(self):
        """Test forward pass with different context sizes."""
        vocab_size = 30
        embedding_dim = 16
        batch_size = 4

        model = CBOWModel(vocab_size, embedding_dim)

        for context_size in [1, 2, 4, 8]:
            context_words = torch.randint(0, vocab_size, (batch_size, context_size))
            output = model.forward(context_words)
            assert output.shape == (batch_size, vocab_size)

    def test_forward_edge_case_single_context(self):
        """Test forward pass with single context word."""
        vocab_size = 10
        embedding_dim = 8

        model = CBOWModel(vocab_size, embedding_dim)
        context_words = torch.randint(0, vocab_size, (2, 1))

        output = model.forward(context_words)

        assert output.shape == (2, vocab_size)
        assert torch.all(torch.isfinite(output))

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        vocab_size = 20
        embedding_dim = 10

        model = CBOWModel(vocab_size, embedding_dim)
        context_words = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
        targets = torch.tensor([16, 17, 18, 19])

        # Forward pass
        scores = model.forward(context_words)
        loss = torch.nn.functional.cross_entropy(scores, targets)

        # Backward pass
        loss.backward()

        # Check gradients exist for embeddings used in computation
        assert model.in_embeddings.weight.grad is not None
        assert model.out_embeddings.weight.grad is not None
        # Check that output embedding gradients are non-zero (they're used in loss)
        assert torch.sum(torch.abs(model.out_embeddings.weight.grad)) > 0


class TestModelIntegration:
    """Integration tests for model functionality."""

    def test_skipgram_vs_cbow_different_architectures(self):
        """Test that SkipGram and CBOW have different architectures."""
        vocab_size = 20
        embedding_dim = 16

        skipgram = SkipGramModel(vocab_size, embedding_dim)
        cbow = CBOWModel(vocab_size, embedding_dim)

        # Both models should have the same basic structure
        assert hasattr(skipgram, "in_embeddings")
        assert hasattr(skipgram, "out_embeddings")
        assert hasattr(cbow, "in_embeddings")
        assert hasattr(cbow, "out_embeddings")

        # Both should accept different input shapes
        target_words = torch.tensor([0, 1, 2, 3])
        context_words = torch.tensor(
            [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
        )

        skipgram_output = skipgram.forward(target_words)
        cbow_output = cbow.forward(context_words)

        # Both should have same output shape for same batch size
        assert skipgram_output.shape == cbow_output.shape
        assert skipgram_output.shape == (4, vocab_size)

    def test_model_state_dict_compatibility(self):
        """Test that models can save and load state dictionaries."""
        vocab_size = 50
        embedding_dim = 32

        # Test SkipGram
        skipgram = SkipGramModel(vocab_size, embedding_dim)
        state_dict = skipgram.state_dict()

        new_skipgram = SkipGramModel(vocab_size, embedding_dim)
        new_skipgram.load_state_dict(state_dict)

        # Test CBOW
        cbow = CBOWModel(vocab_size, embedding_dim)
        state_dict = cbow.state_dict()

        new_cbow = CBOWModel(vocab_size, embedding_dim)
        new_cbow.load_state_dict(state_dict)

    def test_model_parameter_count(self):
        """Test that models have expected parameter counts."""
        vocab_size = 100
        embedding_dim = 50

        # Full softmax models should have 2 * vocab_size * embedding_dim parameters
        expected_params = 2 * vocab_size * embedding_dim

        skipgram = SkipGramModel(vocab_size, embedding_dim)
        cbow = CBOWModel(vocab_size, embedding_dim)

        skipgram_params = sum(p.numel() for p in skipgram.parameters())
        cbow_params = sum(p.numel() for p in cbow.parameters())

        assert skipgram_params == expected_params
        assert cbow_params == expected_params

    def test_training_mode_compatibility(self):
        """Test models work in both training and evaluation modes."""
        vocab_size = 30
        embedding_dim = 16
        batch_size = 4

        skipgram = SkipGramModel(vocab_size, embedding_dim)
        cbow = CBOWModel(vocab_size, embedding_dim)

        target_words = torch.randint(0, vocab_size, (batch_size,))
        context_words = torch.randint(0, vocab_size, (batch_size, 4))

        # Test training mode
        skipgram.train()
        cbow.train()

        skipgram_train_output = skipgram.forward(target_words)
        cbow_train_output = cbow.forward(context_words)

        # Test evaluation mode
        skipgram.eval()
        cbow.eval()

        skipgram_eval_output = skipgram.forward(target_words)
        cbow_eval_output = cbow.forward(context_words)

        # Outputs should be the same (no dropout or batch norm)
        assert torch.allclose(skipgram_train_output, skipgram_eval_output)
        assert torch.allclose(cbow_train_output, cbow_eval_output)

    def test_large_model_creation(self):
        """Test creation of larger models."""
        vocab_size = 10000
        embedding_dim = 300

        # Should not raise memory errors for reasonable sizes
        skipgram = SkipGramModel(vocab_size, embedding_dim)
        cbow = CBOWModel(vocab_size, embedding_dim)

        assert skipgram.vocab_size == vocab_size
        assert cbow.vocab_size == vocab_size
        assert skipgram.embedding_dim == embedding_dim
        assert cbow.embedding_dim == embedding_dim


class TestModelEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_vocab_size(self):
        """Test behavior with zero vocabulary size."""
        # PyTorch doesn't prevent creating embedding with 0 vocab size,
        # but forward pass should fail
        model = SkipGramModel(0, 50)
        # Forward pass with empty input should fail
        with pytest.raises((RuntimeError, IndexError)):
            empty_input = torch.tensor([])
            model.forward(empty_input)

    def test_zero_embedding_dim(self):
        """Test behavior with zero embedding dimension."""
        # PyTorch allows 0 embedding dim, but operations may fail
        model = SkipGramModel(100, 0)
        input_tensor = torch.tensor([0, 1])
        output = model.forward(input_tensor)
        # Should produce empty output
        assert output.shape == (2, 100)  # batch_size x vocab_size

    def test_negative_parameters(self):
        """Test behavior with negative parameters."""
        with pytest.raises((ValueError, RuntimeError)):
            SkipGramModel(-10, 50)

        with pytest.raises((ValueError, RuntimeError)):
            SkipGramModel(100, -50)

    def test_very_small_models(self):
        """Test models with minimal valid parameters."""
        skipgram = SkipGramModel(1, 1)
        cbow = CBOWModel(1, 1)

        assert skipgram.vocab_size == 1
        assert skipgram.embedding_dim == 1
        assert cbow.vocab_size == 1
        assert cbow.embedding_dim == 1

    def test_out_of_bounds_indices(self):
        """Test behavior with out-of-bounds word indices."""
        vocab_size = 10
        embedding_dim = 8

        model = SkipGramModel(vocab_size, embedding_dim)

        # This should raise an error
        with pytest.raises(IndexError):
            out_of_bounds_words = torch.tensor([vocab_size])  # Index out of bounds
            model.forward(out_of_bounds_words)

    def test_empty_tensor_inputs(self):
        """Test behavior with empty tensor inputs."""
        vocab_size = 10
        embedding_dim = 8

        skipgram = SkipGramModel(vocab_size, embedding_dim)
        cbow = CBOWModel(vocab_size, embedding_dim)

        # Empty tensors produce empty outputs, not errors
        empty_words = torch.tensor([], dtype=torch.long)
        skipgram_output = skipgram.forward(empty_words)
        assert skipgram_output.shape == (0, vocab_size)

        empty_context = torch.tensor([], dtype=torch.long).reshape(0, 4)
        cbow_output = cbow.forward(empty_context)
        assert cbow_output.shape == (0, vocab_size)

    def test_mismatched_tensor_types(self):
        """Test behavior with incorrect tensor types."""
        vocab_size = 10
        embedding_dim = 8

        model = SkipGramModel(vocab_size, embedding_dim)

        # Float indices should cause issues
        with pytest.raises((RuntimeError, TypeError)):
            float_words = torch.tensor([1.5, 2.7])
            model.forward(float_words)


class TestModelWithHierarchicalSoftmax:
    """Specific tests for models with hierarchical softmax integration."""

    def test_build_word_counts_called_correctly(self):
        """Test that build_word_counts_from_dataset is called correctly."""
        vocab_size = 20
        embedding_dim = 16
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with (
            patch(
                "src.modern_word2vec.models.build_word_counts_from_dataset"
            ) as mock_build,
            patch("src.modern_word2vec.models.HierarchicalSoftmax") as mock_hs,
        ):
            mock_build.return_value = {"word_0": 20, "word_1": 19}  # Mock word counts

            model = SkipGramModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            mock_build.assert_called_once_with(dataset)
            mock_hs.assert_called_once()
            assert model.output_layer_type == "hierarchical_softmax"

    def test_hierarchical_softmax_parameters_passed_correctly(self):
        """Test that correct parameters are passed to HierarchicalSoftmax."""
        vocab_size = 20
        embedding_dim = 16
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with (
            patch(
                "src.modern_word2vec.models.build_word_counts_from_dataset"
            ) as mock_build,
            patch("src.modern_word2vec.models.HierarchicalSoftmax") as mock_hs,
        ):
            expected_word_counts = {"word_0": 20, "word_1": 19}
            mock_build.return_value = expected_word_counts

            model = SkipGramModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            mock_hs.assert_called_once_with(
                embedding_dim, vocab_size, expected_word_counts, dataset.word_to_idx
            )
            assert model.output_layer_type == "hierarchical_softmax"

    def test_hierarchical_softmax_different_model_types(self):
        """Test hierarchical softmax works with both SkipGram and CBOW."""
        vocab_size = 15
        embedding_dim = 12
        dataset = create_mock_dataset_with_vocab_builder(vocab_size)

        with patch("src.modern_word2vec.models.HierarchicalSoftmax") as mock_hs:
            skipgram = SkipGramModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            cbow = CBOWModel(
                vocab_size,
                embedding_dim,
                output_layer_type="hierarchical_softmax",
                dataset=dataset,
            )

            # Both should create hierarchical softmax
            assert mock_hs.call_count == 2
            assert skipgram.hierarchical_softmax is not None
            assert cbow.hierarchical_softmax is not None
            assert skipgram.out_embeddings is None
            assert cbow.out_embeddings is None
