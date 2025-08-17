"""Tests for the collate function in train_streaming.py's create_streaming_components."""

import torch
from unittest.mock import MagicMock

from modern_word2vec.cli.train_streaming import create_streaming_components


class TestTrainStreamingCollate:
    """Test cases for the collate function in create_streaming_components."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock args for CBOW model
        self.args_cbow = MagicMock()
        self.args_cbow.model_type = "cbow"
        self.args_cbow.vocab_size = 100
        self.args_cbow.window_size = 2
        self.args_cbow.lower = True
        self.args_cbow.tokenizer = "basic"
        self.args_cbow.subsample = 0.0
        self.args_cbow.dynamic_window = False
        self.args_cbow.buffer_size = 1000
        self.args_cbow.vocab_sample_size = 10000
        self.args_cbow.shuffle_buffer_size = 100
        self.args_cbow.batch_size = 32
        self.args_cbow.vocab_file = None
        self.args_cbow.seed = 42

        # Create mock args for skip-gram model
        self.args_skipgram = MagicMock()
        self.args_skipgram.model_type = "skipgram"
        self.args_skipgram.vocab_size = 100
        self.args_skipgram.window_size = 2
        self.args_skipgram.lower = True
        self.args_skipgram.tokenizer = "basic"
        self.args_skipgram.subsample = 0.0
        self.args_skipgram.dynamic_window = False
        self.args_skipgram.buffer_size = 1000
        self.args_skipgram.vocab_sample_size = 10000
        self.args_skipgram.shuffle_buffer_size = 100
        self.args_skipgram.batch_size = 32
        self.args_skipgram.vocab_file = None
        self.args_skipgram.seed = 42

    def test_collate_fn_empty_batch_cbow(self):
        """Test CBOW collate function with empty batch."""
        # Use a simple text source that will be mocked away
        text_source = ["test text"]
        
        # Mock the vocabulary building to avoid actual processing
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_cbow, text_source)
        
        # Test empty batch
        result = dataloader.collate_fn([])
        assert result[0].numel() == 0
        assert result[1].numel() == 0
        assert result[0].dtype == torch.long
        assert result[1].dtype == torch.long

    def test_collate_fn_empty_batch_skipgram(self):
        """Test skip-gram collate function with empty batch."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_skipgram, text_source)
        
        # Test empty batch
        result = dataloader.collate_fn([])
        assert result[0].numel() == 0
        assert result[1].numel() == 0

    def test_collate_fn_cbow_valid_tensors(self):
        """Test CBOW collate function with valid tensors."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_cbow, text_source)
        
        # Test batch with context tensors of different sizes
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor(4)),
            (torch.tensor([5, 6]), torch.tensor(7)),
            (torch.tensor([8]), torch.tensor(9)),
        ]
        
        result = dataloader.collate_fn(batch)
        inputs, targets = result
        
        # Check shapes - should be padded to max length (3)
        assert inputs.shape == (3, 3)
        assert targets.shape == (3,)
        
        # Check padding - first row should be unchanged
        assert torch.equal(inputs[0], torch.tensor([1, 2, 3]))
        # Second row should have one zero padding
        assert torch.equal(inputs[1], torch.tensor([5, 6, 0]))
        # Third row should have two zero paddings
        assert torch.equal(inputs[2], torch.tensor([8, 0, 0]))
        
        # Check targets
        assert torch.equal(targets, torch.tensor([4, 7, 9]))

    def test_collate_fn_cbow_empty_tensors(self):
        """Test CBOW collate function with empty tensors in batch."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_cbow, text_source)
        
        # Test batch with some empty tensors
        batch = [
            (torch.empty(0, dtype=torch.long), torch.tensor(1)),
            (torch.tensor([2, 3]), torch.tensor(4)),
            (torch.empty(0, dtype=torch.long), torch.tensor(5)),
        ]
        
        result = dataloader.collate_fn(batch)
        inputs, targets = result
        
        # Should only include the non-empty tensor
        assert inputs.shape == (1, 2)
        assert targets.shape == (1,)
        assert torch.equal(inputs[0], torch.tensor([2, 3]))
        assert torch.equal(targets, torch.tensor([4]))

    def test_collate_fn_cbow_all_empty_tensors(self):
        """Test CBOW collate function when all tensors are empty."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_cbow, text_source)
        
        # Test batch with all empty tensors
        batch = [
            (torch.empty(0, dtype=torch.long), torch.tensor(1)),
            (torch.empty(0, dtype=torch.long), torch.tensor(2)),
        ]
        
        result = dataloader.collate_fn(batch)
        inputs, targets = result
        
        # Should return empty tensors
        assert inputs.numel() == 0
        assert targets.numel() == 0

    def test_collate_fn_cbow_non_tensor_inputs(self):
        """Test CBOW collate function with non-tensor inputs."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_cbow, text_source)
        
        # Test batch with non-tensor inputs (should be filtered out)
        batch = [
            ("not a tensor", torch.tensor(1)),
            (torch.tensor([2, 3]), torch.tensor(4)),
            (None, torch.tensor(5)),
        ]
        
        result = dataloader.collate_fn(batch)
        inputs, targets = result
        
        # Should only include the valid tensor
        assert inputs.shape == (1, 2)
        assert targets.shape == (1,)
        assert torch.equal(inputs[0], torch.tensor([2, 3]))
        assert torch.equal(targets, torch.tensor([4]))

    def test_collate_fn_skipgram_valid_tensors(self):
        """Test skip-gram collate function with valid tensors."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_skipgram, text_source)
        
        # Test batch for skip-gram
        batch = [
            (torch.tensor(1), torch.tensor(2)),
            (torch.tensor(3), torch.tensor(4)),
        ]
        
        result = dataloader.collate_fn(batch)
        inputs, targets = result
        
        assert inputs.shape == (2,)
        assert targets.shape == (2,)
        assert torch.equal(inputs, torch.tensor([1, 3]))
        assert torch.equal(targets, torch.tensor([2, 4]))

    def test_collate_fn_preserves_dtype(self):
        """Test that collate function preserves tensor data types."""
        text_source = ["test text"]
        
        with patch_streaming_components():
            dataloader, _ = create_streaming_components(self.args_cbow, text_source)
        
        # Test with specific dtype
        batch = [
            (torch.tensor([1, 2], dtype=torch.long), torch.tensor(3, dtype=torch.long)),
            (torch.tensor([4], dtype=torch.long), torch.tensor(5, dtype=torch.long)),
        ]
        
        result = dataloader.collate_fn(batch)
        inputs, targets = result
        
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long


def patch_streaming_components():
    """Context manager to patch the streaming components creation."""
    from unittest.mock import patch
    
    # Mock the streaming dataset and vocabulary building
    mock_vocab = {"test": 0, "word": 1, "<UNK>": 2}
    mock_dataset = MagicMock()
    mock_dataset.vocab_size = len(mock_vocab)
    
    return patch.multiple(
        "modern_word2vec.data.streaming",
        StreamingWord2VecDataset=MagicMock(return_value=mock_dataset),
        build_vocabulary_from_stream=MagicMock(return_value=mock_vocab),
        load_vocabulary=MagicMock(return_value=mock_vocab),
    )
