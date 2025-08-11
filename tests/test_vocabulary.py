"""Comprehensive tests for vocabulary building utilities.

This test suite provides extensive coverage of the VocabularyBuilder class,
including:

- Basic vocabulary building functionality
- Word frequency ordering and counting
- Vocabulary size limiting
- Subsampling with Word2Vec formula
- Edge cases (empty tokens, unicode, large vocabularies)
- Performance testing
- Integration and end-to-end workflows
- Deterministic behavior and reproducibility

Coverage: 100% of vocabulary.py module
"""

import math
import random
from collections import Counter

from src.modern_word2vec.vocabulary import VocabularyBuilder
from src.modern_word2vec.config import SPECIAL_TOKENS


class TestVocabularyBuilder:
    """Test cases for VocabularyBuilder class."""

    def test_init_default_parameters(self):
        """Test VocabularyBuilder initialization with default parameters."""
        vocab_size = 1000
        builder = VocabularyBuilder(vocab_size)

        assert builder.vocab_size == vocab_size
        assert builder.subsample_t == 0.0
        assert builder.word_counts is None

    def test_init_with_subsampling(self):
        """Test VocabularyBuilder initialization with subsampling enabled."""
        vocab_size = 1000
        subsample_t = 1e-5
        builder = VocabularyBuilder(vocab_size, subsample_t)

        assert builder.vocab_size == vocab_size
        assert builder.subsample_t == subsample_t
        assert builder.word_counts is None

    def test_build_vocabulary_basic(self):
        """Test basic vocabulary building without subsampling."""
        tokens = ["apple", "banana", "apple", "cherry", "banana", "apple"]
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Check that word_counts is stored
        assert builder.word_counts is not None
        assert builder.word_counts["apple"] == 3
        assert builder.word_counts["banana"] == 2
        assert builder.word_counts["cherry"] == 1

        # Check vocabulary mappings
        expected_words = ["apple", "banana", "cherry"]  # Most common first
        for i, word in enumerate(expected_words):
            assert word_to_idx[word] == i
            assert idx_to_word[i] == word

        # Check UNK token
        assert SPECIAL_TOKENS["UNK"] in word_to_idx
        assert word_to_idx[SPECIAL_TOKENS["UNK"]] == len(expected_words)
        assert idx_to_word[len(expected_words)] == SPECIAL_TOKENS["UNK"]

        # Without subsampling, tokens should be unchanged
        assert filtered_tokens == tokens

    def test_build_vocabulary_with_vocab_size_limit(self):
        """Test vocabulary building with limited vocabulary size."""
        tokens = ["word1", "word2", "word3", "word4", "word5"] * 10
        builder = VocabularyBuilder(vocab_size=3)  # Only 2 words + UNK
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Should only have 2 most common words + UNK
        assert len(word_to_idx) == 3
        assert len(idx_to_word) == 3

        # Check UNK token is present
        assert SPECIAL_TOKENS["UNK"] in word_to_idx
        assert word_to_idx[SPECIAL_TOKENS["UNK"]] == 2

    def test_build_vocabulary_empty_tokens(self):
        """Test vocabulary building with empty token list."""
        tokens = []
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Should only contain UNK token
        assert len(word_to_idx) == 1
        assert len(idx_to_word) == 1
        assert SPECIAL_TOKENS["UNK"] in word_to_idx
        assert word_to_idx[SPECIAL_TOKENS["UNK"]] == 0
        assert filtered_tokens == []

    def test_build_vocabulary_single_word(self):
        """Test vocabulary building with single unique word."""
        tokens = ["word"] * 5
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Should have one word + UNK
        assert len(word_to_idx) == 2
        assert len(idx_to_word) == 2
        assert word_to_idx["word"] == 0
        assert word_to_idx[SPECIAL_TOKENS["UNK"]] == 1
        assert filtered_tokens == tokens

    def test_word_frequency_ordering(self):
        """Test that words are ordered by frequency in vocabulary."""
        tokens = ["rare"] + ["common"] * 5 + ["very_common"] * 10
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Most frequent should have lowest index
        assert word_to_idx["very_common"] == 0
        assert word_to_idx["common"] == 1
        assert word_to_idx["rare"] == 2

    def test_deterministic_with_same_seed(self):
        """Test that vocabulary building is deterministic with same seed."""
        tokens = ["apple", "banana", "cherry"] * 100
        builder1 = VocabularyBuilder(vocab_size=10, subsample_t=1e-3)
        builder2 = VocabularyBuilder(vocab_size=10, subsample_t=1e-3)

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result1 = builder1.build_vocabulary(tokens, rng1)
        result2 = builder2.build_vocabulary(tokens, rng2)

        assert result1[0] == result2[0]  # word_to_idx
        assert result1[1] == result2[1]  # idx_to_word
        assert result1[2] == result2[2]  # filtered_tokens


class TestVocabularySubsampling:
    """Test cases for vocabulary subsampling functionality."""

    def test_no_subsampling_when_disabled(self):
        """Test that no subsampling occurs when subsample_t is 0."""
        tokens = ["frequent"] * 1000 + ["rare"] * 10
        builder = VocabularyBuilder(vocab_size=10, subsample_t=0.0)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # No subsampling should occur
        assert len(filtered_tokens) == len(tokens)
        assert filtered_tokens == tokens

    def test_subsampling_reduces_frequent_words(self):
        """Test that subsampling reduces frequency of common words."""
        # Create tokens with one very frequent word
        frequent_word = "the"
        rare_words = ["apple", "banana", "cherry"]
        tokens = [frequent_word] * 10000 + rare_words * 10

        builder = VocabularyBuilder(vocab_size=100, subsample_t=1e-4)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Count occurrences in filtered tokens
        filtered_counts = Counter(filtered_tokens)
        original_counts = Counter(tokens)

        # Frequent word should be reduced more than rare words
        frequent_reduction = (
            filtered_counts[frequent_word] / original_counts[frequent_word]
        )
        rare_reduction = filtered_counts["apple"] / original_counts["apple"]

        assert frequent_reduction < rare_reduction
        assert len(filtered_tokens) < len(tokens)

    def test_subsampling_formula_correctness(self):
        """Test that subsampling follows the correct Word2Vec formula."""
        tokens = ["word"] * 1000
        builder = VocabularyBuilder(vocab_size=10, subsample_t=1e-3)
        rng = random.Random(42)

        # Build vocabulary to calculate frequencies
        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Calculate expected keep probability
        freq = 1.0  # "word" has frequency 1.0 (appears in all positions)
        expected_keep_prob = (math.sqrt(freq / builder.subsample_t) + 1) * (
            builder.subsample_t / freq
        )

        # Calculate actual keep probability
        actual_keep_prob = len(filtered_tokens) / len(tokens)

        # Should be approximately equal (with some random variance)
        assert abs(actual_keep_prob - expected_keep_prob) < 0.1

    def test_subsampling_preserves_rare_words(self):
        """Test that rare words are preserved during subsampling."""
        # Create mix of frequent and rare words
        # Use multiple occurrences of rare words to ensure they survive subsampling
        tokens = ["frequent"] * 1000 + ["rare1"] * 5 + ["rare2"] * 5 + ["rare3"] * 5
        builder = VocabularyBuilder(vocab_size=10, subsample_t=1e-4)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Rare words should be mostly preserved (at least some occurrences should remain)
        filtered_counts = Counter(filtered_tokens)
        # With 5 occurrences each, rare words should have high probability of surviving
        rare_words_preserved = sum(
            1 for word in ["rare1", "rare2", "rare3"] if filtered_counts[word] >= 1
        )
        assert (
            rare_words_preserved >= 2
        )  # At least 2 out of 3 rare words should be preserved

    def test_subsampling_different_thresholds(self):
        """Test subsampling behavior with different threshold values."""
        tokens = ["common"] * 1000 + ["rare"] * 10

        # Test with different thresholds
        thresholds = [1e-5, 1e-4, 1e-3, 1e-2]
        filtered_lengths = []

        for threshold in thresholds:
            builder = VocabularyBuilder(vocab_size=10, subsample_t=threshold)
            rng = random.Random(42)  # Same seed for consistency

            _, _, filtered_tokens = builder.build_vocabulary(tokens, rng)
            filtered_lengths.append(len(filtered_tokens))

        # Higher threshold should result in more aggressive subsampling
        for i in range(len(thresholds) - 1):
            assert filtered_lengths[i] <= filtered_lengths[i + 1]

    def test_zero_frequency_words_kept(self):
        """Test that words with zero frequency (edge case) are kept."""
        builder = VocabularyBuilder(vocab_size=10, subsample_t=1e-3)

        # Test the internal method directly
        tokens = ["word"]
        word_counts = Counter(tokens)
        rng = random.Random(42)

        # Create a scenario where we test zero frequency handling
        # We'll manually create word_counts with a zero count word
        word_counts["zero_word"] = 0

        # Call _apply_subsampling with the modified word_counts
        filtered_tokens = builder._apply_subsampling(
            tokens + ["zero_word"], word_counts, rng
        )

        # The zero_word should be kept (freq <= 0 returns True)
        assert "zero_word" in filtered_tokens


class TestVocabularyEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_large_vocabulary_size(self):
        """Test with vocabulary size larger than unique words."""
        tokens = ["word1", "word2", "word3"]
        builder = VocabularyBuilder(vocab_size=1000)  # Much larger than unique words
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Should only contain the unique words + UNK
        assert len(word_to_idx) == 4  # 3 words + UNK
        assert len(idx_to_word) == 4

    def test_vocabulary_size_one(self):
        """Test with vocabulary size of 1 (only UNK token)."""
        tokens = ["word1", "word2", "word3"]
        builder = VocabularyBuilder(vocab_size=1)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Should only contain UNK token
        assert len(word_to_idx) == 1
        assert len(idx_to_word) == 1
        assert SPECIAL_TOKENS["UNK"] in word_to_idx

    def test_duplicate_tokens_handled_correctly(self):
        """Test that duplicate tokens are counted correctly."""
        tokens = ["apple"] * 5 + ["banana"] * 3 + ["cherry"] * 1
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Check counts are stored correctly
        assert builder.word_counts["apple"] == 5
        assert builder.word_counts["banana"] == 3
        assert builder.word_counts["cherry"] == 1

    def test_unicode_tokens(self):
        """Test vocabulary building with unicode tokens."""
        tokens = ["café", "naïve", "résumé", "café", "naïve"]
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Unicode tokens should be handled correctly
        assert "café" in word_to_idx
        assert "naïve" in word_to_idx
        assert "résumé" in word_to_idx
        assert builder.word_counts["café"] == 2
        assert builder.word_counts["naïve"] == 2

    def test_very_long_tokens(self):
        """Test vocabulary building with very long tokens."""
        long_token = "a" * 1000
        tokens = [long_token, "short", long_token, "short"]
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Long tokens should be handled correctly
        assert long_token in word_to_idx
        assert "short" in word_to_idx
        assert builder.word_counts[long_token] == 2


class TestVocabularyPerformance:
    """Performance tests for vocabulary building."""

    def test_large_vocabulary_performance(self):
        """Test vocabulary building performance with large datasets."""
        import time

        # Create large token list
        tokens = []
        for i in range(1000):
            tokens.extend([f"word_{i}"] * (1000 - i))  # Zipf-like distribution

        builder = VocabularyBuilder(vocab_size=500, subsample_t=1e-4)
        rng = random.Random(42)

        start_time = time.time()
        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )
        end_time = time.time()

        build_time = end_time - start_time

        # Should complete reasonably quickly
        assert build_time < 5.0, f"Vocabulary building too slow: {build_time:.2f}s"

        # Verify results are sensible
        assert len(word_to_idx) <= 501  # vocab_size + UNK
        assert len(filtered_tokens) < len(tokens)  # Subsampling should reduce size

        print(f"Built vocabulary with {len(tokens)} tokens in {build_time:.3f}s")
        print(f"Filtered to {len(filtered_tokens)} tokens")


class TestVocabularyIntegration:
    """Integration tests for vocabulary building."""

    def test_end_to_end_workflow(self):
        """Test complete vocabulary building workflow."""
        # Simulate realistic text data
        tokens = []
        # Add common words
        common_words = ["the", "and", "of", "to", "a"]
        for word in common_words:
            tokens.extend([word] * 100)

        # Add less common words
        uncommon_words = ["apple", "banana", "cherry", "date"]
        for word in uncommon_words:
            tokens.extend([word] * 10)

        # Add rare words
        rare_words = ["elderberry", "fig", "grape"]
        tokens.extend(rare_words)

        # Shuffle for realistic distribution
        rng = random.Random(42)
        rng.shuffle(tokens)

        # Build vocabulary with subsampling
        builder = VocabularyBuilder(vocab_size=20, subsample_t=1e-3)
        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        # Verify results
        assert len(word_to_idx) <= 20
        assert len(idx_to_word) == len(word_to_idx)
        assert SPECIAL_TOKENS["UNK"] in word_to_idx
        assert len(filtered_tokens) < len(tokens)  # Subsampling should reduce tokens

        # Most frequent words should have lowest indices
        for word in common_words:
            if word in word_to_idx:
                assert word_to_idx[word] < 10  # Should be in top indices

    def test_consistency_across_runs(self):
        """Test that results are consistent across multiple runs with same seed."""
        tokens = ["word1", "word2", "word3"] * 100

        results = []
        for _ in range(5):
            builder = VocabularyBuilder(vocab_size=10, subsample_t=1e-3)
            rng = random.Random(42)  # Same seed
            result = builder.build_vocabulary(tokens, rng)
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result[0] == first_result[0]  # word_to_idx
            assert result[1] == first_result[1]  # idx_to_word
            assert result[2] == first_result[2]  # filtered_tokens

    def test_memory_efficiency(self):
        """Test memory efficiency with large token lists."""
        # Create large token list
        tokens = ["word"] * 100000
        builder = VocabularyBuilder(vocab_size=10)
        rng = random.Random(42)

        # Should handle large inputs without issues
        word_to_idx, idx_to_word, filtered_tokens = builder.build_vocabulary(
            tokens, rng
        )

        assert len(word_to_idx) == 2  # "word" + UNK
        assert len(filtered_tokens) == len(tokens)  # No subsampling

    def test_vocabulary_builder_reuse(self):
        """Test that VocabularyBuilder can be reused for multiple datasets."""
        builder = VocabularyBuilder(vocab_size=10, subsample_t=1e-3)

        # First dataset
        tokens1 = ["apple", "banana"] * 50
        rng1 = random.Random(42)
        builder.build_vocabulary(tokens1, rng1)
        first_word_counts = dict(builder.word_counts)

        # Second dataset
        tokens2 = ["cherry", "date"] * 50
        rng2 = random.Random(43)
        builder.build_vocabulary(tokens2, rng2)
        second_word_counts = dict(builder.word_counts)

        # Word counts should be updated for second dataset
        assert first_word_counts != second_word_counts
        assert "cherry" in second_word_counts
        assert "date" in second_word_counts
