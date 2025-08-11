"""Common test fixtures and utilities for testing."""

import pytest
import torch
from typing import Dict


@pytest.fixture
def small_vocab():
    """Small vocabulary for testing."""
    return {
        "apple": 0,
        "banana": 1,
        "cherry": 2,
        "date": 3,
        "elderberry": 4,
    }


@pytest.fixture
def small_word_counts():
    """Word counts for small vocabulary."""
    return {
        "apple": 100,
        "banana": 80,
        "cherry": 60,
        "date": 40,
        "elderberry": 20,
    }


@pytest.fixture
def medium_vocab():
    """Medium vocabulary for testing."""
    words = [f"word_{i:03d}" for i in range(50)]
    return {word: i for i, word in enumerate(words)}


@pytest.fixture
def medium_word_counts(medium_vocab):
    """Word counts for medium vocabulary with varied frequencies."""
    import random

    random.seed(42)  # For reproducible tests
    return {word: random.randint(1, 1000) for word in medium_vocab.keys()}


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    torch.manual_seed(42)
    return torch.randn(10, 128)  # 10 samples, 128 dimensions


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_tensor_close(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8
):
    """Assert that two tensors are close within tolerance."""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Tensors not close:\nActual: {actual}\nExpected: {expected}\n"
        f"Max difference: {torch.max(torch.abs(actual - expected))}"
    )


def assert_valid_probability_distribution(
    probs: torch.Tensor, dim: int = -1, rtol: float = 1e-5
):
    """Assert that tensor represents valid probability distributions."""
    # Check non-negative
    assert torch.all(probs >= 0), f"Found negative probabilities: {probs[probs < 0]}"

    # Check sums to 1 along specified dimension
    sums = torch.sum(probs, dim=dim)
    expected_sums = torch.ones_like(sums)
    (
        assert_tensor_close(sums, expected_sums, rtol=rtol),
        (f"Probability distributions don't sum to 1: {sums}"),
    )


def create_mock_dataset(word_to_idx: Dict[str, int], word_counts: Dict[str, int]):
    """Create a mock dataset object for testing."""

    class MockVocabBuilder:
        def __init__(self, word_counts):
            self.word_counts = word_counts

    class MockDataset:
        def __init__(self, word_to_idx, word_counts):
            self.word_to_idx = word_to_idx
            self.vocab_builder = MockVocabBuilder(word_counts)

    return MockDataset(word_to_idx, word_counts)
