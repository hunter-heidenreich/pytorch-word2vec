"""Comprehensive tests for the data module."""

import random
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from modern_word2vec.config import DataConfig
from modern_word2vec.data import (
    Word2VecDataset,
    cbow_collate,
    load_texts_from_hf,
    generate_synthetic_texts,
)


class TestWord2VecDataset:
    """Test cases for Word2VecDataset class."""

    def test_init_basic(self):
        """Test basic dataset initialization."""
        texts = ["hello world", "this is a test"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        assert dataset.cfg == config
        assert hasattr(dataset, "vocab_builder")
        assert hasattr(dataset, "word_to_idx")
        assert hasattr(dataset, "idx_to_word")
        assert hasattr(dataset, "pair_generator")
        assert isinstance(dataset.vocab_size, int)
        assert dataset.vocab_size > 0

    def test_init_with_custom_rng(self):
        """Test dataset initialization with custom RNG."""
        texts = ["hello world", "this is a test"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")
        rng = random.Random(42)

        dataset = Word2VecDataset(texts, config, rng=rng)

        assert dataset.rng == rng

    def test_init_with_subsampling(self):
        """Test dataset initialization with subsampling enabled."""
        texts = ["hello world hello", "world hello test"]
        config = DataConfig(
            vocab_size=1000, window_size=2, model_type="skipgram", subsample_t=1e-5
        )

        dataset = Word2VecDataset(texts, config)

        assert hasattr(dataset, "vocab_builder")
        assert dataset.vocab_builder.subsample_t == 1e-5

    def test_init_cbow_model(self):
        """Test dataset initialization with CBOW model type."""
        texts = ["hello world", "this is a test"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="cbow")

        dataset = Word2VecDataset(texts, config)

        assert dataset.pair_generator.model_type == "cbow"

    def test_len(self):
        """Test dataset length method."""
        texts = ["hello world", "this is a test"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        length = len(dataset)
        assert isinstance(length, int)
        assert length > 0

    def test_getitem_skipgram(self):
        """Test dataset getitem for skipgram."""
        texts = ["hello world test"]
        config = DataConfig(vocab_size=1000, window_size=1, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        # Get first item
        input_tensor, target_tensor = dataset[0]

        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        assert input_tensor.dtype == torch.long
        assert target_tensor.dtype == torch.long

    def test_getitem_cbow(self):
        """Test dataset getitem for CBOW."""
        texts = ["hello world test example"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="cbow")

        dataset = Word2VecDataset(texts, config)

        # Get first item
        input_tensor, target_tensor = dataset[0]

        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        assert input_tensor.dtype == torch.long
        assert target_tensor.dtype == torch.long

    def test_empty_texts(self):
        """Test dataset with empty texts."""
        texts = []
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")

        # Should handle empty texts gracefully
        dataset = Word2VecDataset(texts, config)
        assert len(dataset) == 0

    def test_single_word_texts(self):
        """Test dataset with single word texts."""
        texts = ["hello", "world"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)
        # Should work but might have limited pairs
        assert isinstance(len(dataset), int)

    def test_reproducibility_with_seed(self):
        """Test that dataset produces consistent results with same seed."""
        texts = ["hello world test example"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        dataset1 = Word2VecDataset(texts, config, rng=rng1)
        dataset2 = Word2VecDataset(texts, config, rng=rng2)

        # Same vocabulary should be built
        assert dataset1.word_to_idx == dataset2.word_to_idx
        assert dataset1.vocab_size == dataset2.vocab_size

    def test_dynamic_window(self):
        """Test dataset with dynamic window enabled."""
        texts = ["hello world test example"]
        config = DataConfig(
            vocab_size=1000, window_size=3, model_type="skipgram", dynamic_window=True
        )

        dataset = Word2VecDataset(texts, config)

        assert dataset.pair_generator.dynamic_window is True


class TestCbowCollate:
    """Test cases for CBOW collate function."""

    def test_basic_collation(self):
        """Test basic CBOW collation."""
        # Create sample batch data
        batch = [
            (torch.tensor([1, 2]), torch.tensor(3)),
            (torch.tensor([4, 5]), torch.tensor(6)),
        ]

        inputs, targets = cbow_collate(batch)

        assert inputs.shape == (2, 2)
        assert targets.shape == (2,)
        assert torch.equal(inputs[0], torch.tensor([1, 2]))
        assert torch.equal(targets, torch.tensor([3, 6]))

    def test_padding_different_lengths(self):
        """Test collation with different input lengths."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor(4)),
            (torch.tensor([5, 6]), torch.tensor(7)),
            (torch.tensor([8]), torch.tensor(9)),
        ]

        inputs, targets = cbow_collate(batch)

        assert inputs.shape == (3, 3)  # Padded to max length
        assert targets.shape == (3,)
        # Check first row (no padding needed)
        assert torch.equal(inputs[0], torch.tensor([1, 2, 3]))
        # Check second row (one padding)
        assert torch.equal(inputs[1], torch.tensor([5, 6, 0]))
        # Check third row (two paddings)
        assert torch.equal(inputs[2], torch.tensor([8, 0, 0]))

    def test_scalar_input_handling(self):
        """Test handling of scalar inputs."""
        batch = [
            (torch.tensor(1), torch.tensor(2)),
            (torch.tensor(3), torch.tensor(4)),
        ]

        inputs, targets = cbow_collate(batch)

        assert inputs.shape == (2, 1)
        assert targets.shape == (2,)
        assert torch.equal(inputs, torch.tensor([[1], [3]]))

    def test_empty_batch(self):
        """Test collation with empty batch."""
        batch = []

        # Should handle empty batch gracefully
        with pytest.raises((IndexError, ValueError)):
            cbow_collate(batch)

    def test_single_item_batch(self):
        """Test collation with single item."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor(4)),
        ]

        inputs, targets = cbow_collate(batch)

        assert inputs.shape == (1, 3)
        assert targets.shape == (1,)

    def test_preserve_data_types(self):
        """Test that collation preserves tensor data types."""
        batch = [
            (torch.tensor([1, 2], dtype=torch.long), torch.tensor(3, dtype=torch.long)),
            (torch.tensor([4, 5], dtype=torch.long), torch.tensor(6, dtype=torch.long)),
        ]

        inputs, targets = cbow_collate(batch)

        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long


class TestLoadTextsFromHf:
    """Test cases for loading texts from Hugging Face datasets."""

    @patch("modern_word2vec.data.load_dataset")
    def test_load_with_text_column(self, mock_load_dataset):
        """Test loading dataset with 'text' column."""
        # Mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["text", "label"]
        mock_ds.__getitem__.return_value = ["Hello world", "Test sentence"]
        mock_load_dataset.return_value = mock_ds

        result = load_texts_from_hf("test_dataset", None, "train")

        mock_load_dataset.assert_called_once_with("test_dataset", None, split="train")
        assert result == ["Hello world", "Test sentence"]

    @patch("modern_word2vec.data.load_dataset")
    def test_load_with_content_column(self, mock_load_dataset):
        """Test loading dataset with 'content' column."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["content", "label"]
        mock_ds.__getitem__.return_value = ["Content text"]
        mock_load_dataset.return_value = mock_ds

        result = load_texts_from_hf("test_dataset", None, "train")

        assert result == ["Content text"]

    @patch("modern_word2vec.data.load_dataset")
    def test_load_with_sentence_column(self, mock_load_dataset):
        """Test loading dataset with 'sentence' column."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["sentence", "label"]
        mock_ds.__getitem__.return_value = ["Sentence text"]
        mock_load_dataset.return_value = mock_ds

        result = load_texts_from_hf("test_dataset", None, "train")

        assert result == ["Sentence text"]

    @patch("modern_word2vec.data.load_dataset")
    def test_load_with_document_column(self, mock_load_dataset):
        """Test loading dataset with 'document' column."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["document", "label"]
        mock_ds.__getitem__.return_value = ["Document text"]
        mock_load_dataset.return_value = mock_ds

        result = load_texts_from_hf("test_dataset", None, "train")

        assert result == ["Document text"]

    @patch("modern_word2vec.data.load_dataset")
    def test_load_with_raw_column(self, mock_load_dataset):
        """Test loading dataset with 'raw' column."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["raw", "label"]
        mock_ds.__getitem__.return_value = ["Raw text"]
        mock_load_dataset.return_value = mock_ds

        result = load_texts_from_hf("test_dataset", None, "train")

        assert result == ["Raw text"]

    @patch("modern_word2vec.data.load_dataset")
    def test_load_no_text_column_error(self, mock_load_dataset):
        """Test error when no suitable text column is found."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["label", "id", "category"]
        mock_load_dataset.return_value = mock_ds

        with pytest.raises(ValueError) as exc_info:
            load_texts_from_hf("test_dataset", None, "train")

        assert "Could not find a text column" in str(exc_info.value)
        assert "Available columns: ['label', 'id', 'category']" in str(exc_info.value)

    @patch("modern_word2vec.data.load_dataset")
    def test_load_filters_non_string_and_empty(self, mock_load_dataset):
        """Test that loading filters out non-string and empty texts."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["text"]
        mock_ds.__getitem__.return_value = [
            "Valid text",
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
            "Another valid text",
            123,  # Non-string
        ]
        mock_load_dataset.return_value = mock_ds

        result = load_texts_from_hf("test_dataset", None, "train")

        assert result == ["Valid text", "Another valid text"]

    @patch("modern_word2vec.data.load_dataset")
    def test_load_with_config_parameter(self, mock_load_dataset):
        """Test loading with configuration parameter."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["text"]
        mock_ds.__getitem__.return_value = ["Test text"]
        mock_load_dataset.return_value = mock_ds

        load_texts_from_hf("test_dataset", "config_name", "validation")

        mock_load_dataset.assert_called_once_with(
            "test_dataset", "config_name", split="validation"
        )


class TestGenerateSyntheticTexts:
    """Test cases for synthetic text generation."""

    def test_basic_generation(self):
        """Test basic synthetic text generation."""
        rng = random.Random(42)

        texts = generate_synthetic_texts(n_sentences=5, vocab_size=10, rng=rng)

        assert len(texts) == 5
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text.split()) >= 5 for text in texts)  # At least 5 words
        assert all(len(text.split()) <= 20 for text in texts)  # At most 20 words

    def test_vocab_size_respected(self):
        """Test that vocabulary size is respected."""
        rng = random.Random(42)
        vocab_size = 5

        texts = generate_synthetic_texts(n_sentences=10, vocab_size=vocab_size, rng=rng)

        # Extract all unique words
        all_words = set()
        for text in texts:
            all_words.update(text.split())

        # Should only have words from tok0 to tok4
        expected_words = {f"tok{i}" for i in range(vocab_size)}
        assert all_words.issubset(expected_words)

    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        texts1 = generate_synthetic_texts(n_sentences=5, vocab_size=10, rng=rng1)
        texts2 = generate_synthetic_texts(n_sentences=5, vocab_size=10, rng=rng2)

        assert texts1 == texts2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        rng1 = random.Random(42)
        rng2 = random.Random(123)

        texts1 = generate_synthetic_texts(n_sentences=5, vocab_size=10, rng=rng1)
        texts2 = generate_synthetic_texts(n_sentences=5, vocab_size=10, rng=rng2)

        assert texts1 != texts2

    def test_zero_sentences(self):
        """Test generation with zero sentences."""
        rng = random.Random(42)

        texts = generate_synthetic_texts(n_sentences=0, vocab_size=10, rng=rng)

        assert texts == []

    def test_single_vocab_word(self):
        """Test generation with single vocabulary word."""
        rng = random.Random(42)

        texts = generate_synthetic_texts(n_sentences=3, vocab_size=1, rng=rng)

        assert len(texts) == 3
        # All texts should only contain 'tok0'
        for text in texts:
            words = text.split()
            assert all(word == "tok0" for word in words)

    def test_large_generation(self):
        """Test generation with large parameters."""
        rng = random.Random(42)

        texts = generate_synthetic_texts(n_sentences=100, vocab_size=50, rng=rng)

        assert len(texts) == 100
        assert all(isinstance(text, str) for text in texts)


class TestDataIntegration:
    """Integration tests for data module components."""

    def test_dataset_with_dataloader(self):
        """Test dataset integration with PyTorch DataLoader."""
        texts = ["hello world test", "another example sentence"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Test that we can iterate through the dataloader
        batch = next(iter(dataloader))
        inputs, targets = batch

        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert inputs.shape[0] <= 2  # Batch size
        assert targets.shape[0] <= 2

    def test_cbow_dataset_with_collate(self):
        """Test CBOW dataset with custom collate function."""
        texts = ["hello world test example sentence"]
        config = DataConfig(vocab_size=1000, window_size=2, model_type="cbow")

        dataset = Word2VecDataset(texts, config)
        dataloader = DataLoader(
            dataset, batch_size=3, shuffle=False, collate_fn=cbow_collate
        )

        batch = next(iter(dataloader))
        inputs, targets = batch

        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert len(inputs.shape) == 2  # Batched and padded
        assert inputs.shape[0] <= 3  # Batch size

    def test_end_to_end_synthetic_workflow(self):
        """Test end-to-end workflow with synthetic data."""
        rng = random.Random(42)

        # Generate synthetic texts
        texts = generate_synthetic_texts(n_sentences=10, vocab_size=20, rng=rng)

        # Create dataset
        config = DataConfig(vocab_size=50, window_size=3, model_type="skipgram")
        dataset = Word2VecDataset(texts, config)

        # Test that everything works together
        assert len(dataset) > 0

        # Get a sample
        input_tensor, target_tensor = dataset[0]
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with larger dataset."""
        rng = random.Random(42)

        # Generate a larger synthetic dataset
        texts = generate_synthetic_texts(n_sentences=1000, vocab_size=100, rng=rng)
        config = DataConfig(vocab_size=200, window_size=5, model_type="cbow")

        dataset = Word2VecDataset(texts, config)

        # Should be able to create without memory issues
        assert len(dataset) > 0
        assert dataset.vocab_size <= 200

    def test_different_tokenization_strategies(self):
        """Test dataset with different tokenization strategies."""
        texts = ["Hello, World! This is a test.", "Another example with numbers 123."]

        # Test with basic tokenization
        config1 = DataConfig(
            vocab_size=1000, window_size=2, model_type="skipgram", tokenizer="basic"
        )
        dataset1 = Word2VecDataset(texts, config1)

        # Test with enhanced tokenization
        config2 = DataConfig(
            vocab_size=1000, window_size=2, model_type="skipgram", tokenizer="enhanced"
        )
        dataset2 = Word2VecDataset(texts, config2)

        # Both should work but may have different vocabularies
        assert len(dataset1) >= 0
        assert len(dataset2) >= 0


class TestDataEdgeCases:
    """Test edge cases for data module."""

    def test_very_short_texts(self):
        """Test with very short texts."""
        texts = ["a", "b", "c"]
        config = DataConfig(vocab_size=10, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        # Should handle gracefully
        assert isinstance(len(dataset), int)

    def test_very_long_texts(self):
        """Test with very long texts."""
        long_text = " ".join(f"word{i}" for i in range(1000))
        texts = [long_text]
        config = DataConfig(vocab_size=500, window_size=5, model_type="cbow")

        dataset = Word2VecDataset(texts, config)

        assert len(dataset) > 0

    def test_unicode_texts(self):
        """Test with unicode texts."""
        texts = ["Hello 世界", "Café résumé", "Москва"]
        config = DataConfig(vocab_size=100, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        # Should handle unicode gracefully
        assert isinstance(len(dataset), int)

    def test_special_characters_texts(self):
        """Test with texts containing special characters."""
        texts = ["@#$%^&*()", "!!!???", "---===+++"]
        config = DataConfig(vocab_size=100, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        # Should handle special characters
        assert isinstance(len(dataset), int)

    def test_mixed_language_texts(self):
        """Test with mixed language texts."""
        texts = [
            "Hello world and 你好世界",
            "English français español",
            "Mixed languages in one sentence",
        ]
        config = DataConfig(vocab_size=200, window_size=3, model_type="cbow")

        dataset = Word2VecDataset(texts, config)

        assert isinstance(len(dataset), int)

    def test_extremely_large_vocabulary_request(self):
        """Test with extremely large vocabulary size request."""
        texts = ["hello world test"]
        config = DataConfig(vocab_size=1000000, window_size=2, model_type="skipgram")

        dataset = Word2VecDataset(texts, config)

        # Actual vocabulary should be much smaller than requested
        assert dataset.vocab_size < 1000000
        assert dataset.vocab_size > 0
