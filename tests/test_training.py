"""Comprehensive tests for the training module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader, TensorDataset

from modern_word2vec.training import Trainer
from modern_word2vec.config import TrainConfig
from modern_word2vec.models import SkipGramModel, CBOWModel
from tests.conftest import create_mock_dataset


class TestTrainer:
    """Test cases for the Trainer class."""

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cpu")  # Use CPU for consistent testing

    @pytest.fixture
    def basic_config(self):
        """Basic training configuration."""
        return TrainConfig(
            embedding_dim=50,
            batch_size=4,
            epochs=2,
            learning_rate=0.01,
            optimizer="adam",
            weight_decay=0.1,
            grad_clip=1.0,
            compile=False,
            mixed_precision=False,
            num_workers=0,
            pin_memory=False,
            seed=42,
        )

    @pytest.fixture
    def sgd_config(self):
        """SGD training configuration."""
        return TrainConfig(
            embedding_dim=50,
            batch_size=4,
            epochs=2,
            learning_rate=0.01,
            optimizer="sgd",
            weight_decay=0.1,
            grad_clip=1.0,
        )

    @pytest.fixture
    def simple_model(self):
        """Simple model for testing (not hierarchical softmax)."""
        vocab_size = 10
        embedding_dim = 50
        model = SkipGramModel(
            vocab_size, embedding_dim, output_layer_type="full_softmax"
        )
        return model

    @pytest.fixture
    def hierarchical_model(self, small_vocab, small_word_counts):
        """Model with hierarchical softmax."""
        vocab_size = len(small_vocab)
        embedding_dim = 50
        dataset = create_mock_dataset(small_vocab, small_word_counts)
        model = SkipGramModel(
            vocab_size,
            embedding_dim,
            output_layer_type="hierarchical_softmax",
            dataset=dataset,
        )
        return model

    @pytest.fixture
    def cbow_hierarchical_model(self, small_vocab, small_word_counts):
        """CBOW model with hierarchical softmax."""
        vocab_size = len(small_vocab)
        embedding_dim = 50
        dataset = create_mock_dataset(small_vocab, small_word_counts)
        model = CBOWModel(
            vocab_size,
            embedding_dim,
            output_layer_type="hierarchical_softmax",
            dataset=dataset,
        )
        return model

    @pytest.fixture
    def sample_dataloader(self):
        """Sample dataloader for testing."""
        # Create simple input/target pairs
        inputs = torch.randint(
            0, 10, (20,)
        )  # 20 samples, single word input (shape: batch)
        targets = torch.randint(0, 10, (20,))  # 20 target words
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    @pytest.fixture
    def cbow_dataloader(self):
        """Sample dataloader for CBOW testing."""
        # Create context words input and center word targets
        inputs = torch.randint(0, 5, (20, 4))  # 20 samples, 4 context words each
        targets = torch.randint(0, 5, (20,))  # 20 center words
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_trainer_initialization_simple_model(
        self, simple_model, device, basic_config
    ):
        """Test trainer initialization with simple model."""
        trainer = Trainer(simple_model, device, basic_config)

        assert trainer.model == simple_model
        assert trainer.device == device
        assert trainer.config == basic_config
        assert trainer.criterion is not None
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
        assert trainer.optimizer is not None
        assert trainer.scheduler is None  # Not set up until training starts

    def test_trainer_initialization_hierarchical_model(
        self, hierarchical_model, device, basic_config
    ):
        """Test trainer initialization with hierarchical softmax model."""
        trainer = Trainer(hierarchical_model, device, basic_config)

        assert trainer.model == hierarchical_model
        assert trainer.criterion is None  # No criterion for hierarchical softmax
        assert trainer.optimizer is not None

    def test_build_optimizer_adam(self, simple_model, device, basic_config):
        """Test building Adam optimizer."""
        trainer = Trainer(simple_model, device, basic_config)

        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]["lr"] == basic_config.learning_rate
        assert (
            trainer.optimizer.param_groups[0]["weight_decay"]
            == basic_config.weight_decay
        )

    def test_build_optimizer_sgd(self, simple_model, device, sgd_config):
        """Test building SGD optimizer."""
        trainer = Trainer(simple_model, device, sgd_config)

        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.param_groups[0]["lr"] == sgd_config.learning_rate
        assert (
            trainer.optimizer.param_groups[0]["weight_decay"] == sgd_config.weight_decay
        )

    def test_setup_scheduler_sgd(self, simple_model, device, sgd_config):
        """Test setting up scheduler for SGD."""
        trainer = Trainer(simple_model, device, sgd_config)
        trainer._setup_scheduler(100)

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LinearLR)

    def test_setup_scheduler_adam(self, simple_model, device, basic_config):
        """Test that no scheduler is set up for Adam."""
        trainer = Trainer(simple_model, device, basic_config)
        trainer._setup_scheduler(100)

        assert trainer.scheduler is None

    def test_train_step_simple_model(self, simple_model, device, basic_config):
        """Test single training step with simple model."""
        trainer = Trainer(simple_model, device, basic_config)

        inputs = torch.randint(0, 10, (4,))
        targets = torch.randint(0, 10, (4,))

        loss = trainer._train_step(inputs, targets)

        assert isinstance(loss, float)
        assert loss > 0  # Loss should be positive

    def test_train_step_hierarchical_model(
        self, hierarchical_model, device, basic_config
    ):
        """Test single training step with hierarchical softmax model."""
        trainer = Trainer(hierarchical_model, device, basic_config)

        inputs = torch.randint(0, 5, (4,))
        targets = torch.randint(0, 5, (4,))

        loss = trainer._train_step(inputs, targets)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_cbow_hierarchical(
        self, cbow_hierarchical_model, device, basic_config
    ):
        """Test single training step with CBOW hierarchical softmax model."""
        trainer = Trainer(cbow_hierarchical_model, device, basic_config)

        # CBOW takes multiple context words as input
        inputs = torch.randint(0, 5, (4, 4))  # 4 samples, 4 context words each
        targets = torch.randint(0, 5, (4,))

        loss = trainer._train_step(inputs, targets)

        assert isinstance(loss, float)
        assert loss > 0

    def test_compute_loss_simple_model(self, simple_model, device, basic_config):
        """Test loss computation for simple model."""
        trainer = Trainer(simple_model, device, basic_config)

        inputs = torch.randint(0, 10, (4,))
        targets = torch.randint(0, 10, (4,))

        inputs = inputs.to(device)
        targets = targets.to(device)

        loss = trainer._compute_loss(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() > 0

    def test_compute_loss_hierarchical_skipgram(
        self, hierarchical_model, device, basic_config
    ):
        """Test loss computation for hierarchical softmax Skip-gram model."""
        trainer = Trainer(hierarchical_model, device, basic_config)

        inputs = torch.randint(0, 5, (4,))
        targets = torch.randint(0, 5, (4,))

        inputs = inputs.to(device)
        targets = targets.to(device)

        loss = trainer._compute_loss(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

    def test_compute_loss_hierarchical_cbow(
        self, cbow_hierarchical_model, device, basic_config
    ):
        """Test loss computation for hierarchical softmax CBOW model."""
        trainer = Trainer(cbow_hierarchical_model, device, basic_config)

        # Test with context words that include padding (zeros)
        inputs = torch.tensor([[1, 2, 3, 0], [2, 3, 4, 1], [0, 1, 2, 3], [4, 0, 1, 2]])
        targets = torch.randint(0, 5, (4,))

        inputs = inputs.to(device)
        targets = targets.to(device)

        loss = trainer._compute_loss(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

    def test_train_full_simple_model(
        self, simple_model, device, basic_config, sample_dataloader
    ):
        """Test full training loop with simple model."""
        trainer = Trainer(simple_model, device, basic_config)

        stats = trainer.train(sample_dataloader)

        assert isinstance(stats, dict)
        assert "avg_loss" in stats
        assert "steps" in stats
        assert "samples" in stats
        assert "time_sec" in stats
        assert "steps_per_sec" in stats
        assert "samples_per_sec" in stats
        assert "final_lr" in stats

        assert stats["steps"] > 0
        assert stats["samples"] > 0
        assert stats["avg_loss"] > 0
        assert stats["time_sec"] > 0

    def test_train_full_hierarchical_model(
        self, hierarchical_model, device, basic_config
    ):
        """Test full training loop with hierarchical model."""
        trainer = Trainer(hierarchical_model, device, basic_config)

        # Create a compatible dataloader for 5-word vocab
        inputs = torch.randint(0, 5, (20,))  # 20 samples, indices 0-4 for 5-word vocab
        targets = torch.randint(0, 5, (20,))  # 20 target words
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        stats = trainer.train(dataloader)

        assert isinstance(stats, dict)
        assert stats["steps"] > 0
        assert stats["avg_loss"] > 0

    def test_train_full_cbow_hierarchical(
        self, cbow_hierarchical_model, device, basic_config, cbow_dataloader
    ):
        """Test full training loop with CBOW hierarchical model."""
        trainer = Trainer(cbow_hierarchical_model, device, basic_config)

        stats = trainer.train(cbow_dataloader)

        assert isinstance(stats, dict)
        assert stats["steps"] > 0
        assert stats["avg_loss"] > 0

    def test_train_with_sgd_scheduler(
        self, simple_model, device, sgd_config, sample_dataloader
    ):
        """Test training with SGD and learning rate scheduler."""
        trainer = Trainer(simple_model, device, sgd_config)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        stats = trainer.train(sample_dataloader)
        final_lr = trainer.optimizer.param_groups[0]["lr"]

        # Learning rate should have decreased
        assert final_lr < initial_lr
        assert stats["final_lr"] == final_lr

    def test_train_with_grad_clipping(
        self, simple_model, device, basic_config, sample_dataloader
    ):
        """Test training with gradient clipping."""
        basic_config.grad_clip = 0.5  # Enable gradient clipping
        trainer = Trainer(simple_model, device, basic_config)

        stats = trainer.train(sample_dataloader)

        assert stats["avg_loss"] > 0  # Training should still work

    def test_mixed_precision_initialization(self, simple_model, device, basic_config):
        """Test mixed precision initialization."""
        basic_config.mixed_precision = True
        trainer = Trainer(simple_model, device, basic_config)

        assert trainer.scaler is not None
        # On CPU, mixed precision should be disabled
        if device.type == "cpu":
            assert not trainer.scaler.is_enabled()

    @patch("torch.compile")
    def test_model_compilation(self, mock_compile, simple_model, device, basic_config):
        """Test model compilation when enabled."""
        basic_config.compile = True
        mock_compile.return_value = simple_model

        _ = Trainer(simple_model, device, basic_config)

        mock_compile.assert_called_once_with(simple_model)

    @patch("torch.compile")
    def test_model_compilation_failure(
        self, mock_compile, simple_model, device, basic_config
    ):
        """Test that compilation failure is handled gracefully."""
        basic_config.compile = True
        mock_compile.side_effect = Exception("Compilation failed")

        # Should not raise exception
        trainer = Trainer(simple_model, device, basic_config)

        assert trainer.model == simple_model

    def test_train_step_with_mixed_precision(self, simple_model, device, basic_config):
        """Test training step with mixed precision (mocked)."""
        basic_config.mixed_precision = True
        trainer = Trainer(simple_model, device, basic_config)

        # Mock the scaler to simulate CUDA availability
        trainer.scaler = Mock()
        trainer.scaler.is_enabled.return_value = True
        trainer.scaler.scale.return_value = Mock()
        trainer.scaler.step = Mock()
        trainer.scaler.update = Mock()
        trainer.scaler.unscale_ = Mock()

        # Mock autocast context manager and _compute_loss to avoid shape issues
        with (
            patch("torch.cuda.amp.autocast") as mock_autocast,
            patch.object(
                trainer, "_compute_loss", return_value=torch.tensor(0.5)
            ) as mock_compute_loss,
        ):
            mock_autocast.return_value.__enter__ = Mock()
            mock_autocast.return_value.__exit__ = Mock()

            inputs = torch.randint(0, 10, (4,))
            targets = torch.randint(0, 10, (4,))

            loss = trainer._train_step(inputs, targets)

            assert isinstance(loss, float)
            trainer.scaler.scale.assert_called()
            trainer.scaler.step.assert_called()
            trainer.scaler.update.assert_called()
            mock_compute_loss.assert_called_once()

    def test_train_step_grad_clipping_with_mixed_precision(
        self, simple_model, device, basic_config
    ):
        """Test training step with both gradient clipping and mixed precision."""
        basic_config.mixed_precision = True
        basic_config.grad_clip = 1.0
        trainer = Trainer(simple_model, device, basic_config)

        # Mock the scaler
        trainer.scaler = Mock()
        trainer.scaler.is_enabled.return_value = True
        trainer.scaler.scale.return_value = Mock()
        trainer.scaler.step = Mock()
        trainer.scaler.update = Mock()
        trainer.scaler.unscale_ = Mock()

        with (
            patch("torch.cuda.amp.autocast") as mock_autocast,
            patch("torch.nn.utils.clip_grad_norm_") as mock_clip,
            patch.object(
                trainer, "_compute_loss", return_value=torch.tensor(0.5)
            ) as mock_compute_loss,
        ):
            mock_autocast.return_value.__enter__ = Mock()
            mock_autocast.return_value.__exit__ = Mock()

            inputs = torch.randint(0, 10, (4,))
            targets = torch.randint(0, 10, (4,))

            loss = trainer._train_step(inputs, targets)

            assert isinstance(loss, float)
            trainer.scaler.unscale_.assert_called()
            mock_clip.assert_called()
            mock_compute_loss.assert_called_once()

    def test_cbow_model_detection(self, cbow_hierarchical_model, device, basic_config):
        """Test that CBOW models are correctly detected in hierarchical softmax."""
        trainer = Trainer(cbow_hierarchical_model, device, basic_config)

        # Verify model class name detection works
        assert "CBOW" in cbow_hierarchical_model.__class__.__name__

        # Test compute loss with CBOW-specific logic
        inputs = torch.randint(0, 5, (4, 4))  # Context words
        targets = torch.randint(0, 5, (4,))

        inputs = inputs.to(device)
        targets = targets.to(device)

        loss = trainer._compute_loss(inputs, targets)
        assert isinstance(loss, torch.Tensor)

    def test_empty_dataloader(self, simple_model, device, basic_config):
        """Test training with empty dataloader."""
        trainer = Trainer(simple_model, device, basic_config)

        # Create empty dataloader
        empty_dataset = TensorDataset(
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        )
        empty_dataloader = DataLoader(empty_dataset, batch_size=4)

        stats = trainer.train(empty_dataloader)

        assert stats["steps"] == 0
        assert stats["samples"] == 0
        assert stats["avg_loss"] == 0.0

    def test_scheduler_none_behavior(
        self, simple_model, device, basic_config, sample_dataloader
    ):
        """Test that training works when scheduler is None (Adam optimizer)."""
        trainer = Trainer(simple_model, device, basic_config)

        # Scheduler should be None for Adam
        assert trainer.scheduler is None

        stats = trainer.train(sample_dataloader)

        # Training should complete successfully
        assert stats["steps"] > 0
        assert stats["avg_loss"] > 0

    def test_device_movement(self, simple_model, basic_config):
        """Test that inputs are properly moved to device."""
        device = torch.device("cpu")
        trainer = Trainer(simple_model, device, basic_config)

        # Create inputs on different device (if possible)
        inputs = torch.randint(0, 10, (4,))
        targets = torch.randint(0, 10, (4,))

        # This should work regardless of input device
        loss = trainer._train_step(inputs, targets)
        assert isinstance(loss, float)

    def test_zero_grad_optimization(self, simple_model, device, basic_config):
        """Test that zero_grad is called with set_to_none=True for efficiency."""
        trainer = Trainer(simple_model, device, basic_config)

        # Mock the optimizer to verify zero_grad call
        trainer.optimizer.zero_grad = Mock()

        inputs = torch.randint(0, 10, (4,))
        targets = torch.randint(0, 10, (4,))

        trainer._train_step(inputs, targets)

        trainer.optimizer.zero_grad.assert_called_with(set_to_none=True)

    def test_non_blocking_transfer(self, simple_model, device, basic_config):
        """Test that tensors are moved to device with non_blocking=True."""
        trainer = Trainer(simple_model, device, basic_config)

        inputs = torch.randint(0, 10, (4,))
        targets = torch.randint(0, 10, (4,))

        # Mock tensor .to() method to verify non_blocking parameter
        with (
            patch.object(
                inputs, "to", return_value=inputs.to(device)
            ) as mock_inputs_to,
            patch.object(
                targets, "to", return_value=targets.to(device)
            ) as mock_targets_to,
        ):
            trainer._train_step(inputs, targets)

            mock_inputs_to.assert_called_with(device, non_blocking=True)
            mock_targets_to.assert_called_with(device, non_blocking=True)
