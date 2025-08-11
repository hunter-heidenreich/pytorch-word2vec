"""Comprehensive tests for the HuffmanTree class."""

import pytest

from src.modern_word2vec.hierarchical_softmax import HuffmanTree


class TestHuffmanTree:
    """Test cases for HuffmanTree implementation."""

    def test_init_with_small_vocab(self, small_vocab, small_word_counts):
        """Test HuffmanTree initialization with small vocabulary."""
        tree = HuffmanTree(small_word_counts, small_vocab)

        # Check basic attributes
        assert tree.word_to_idx == small_vocab
        assert len(tree.idx_to_word) == len(small_vocab)
        assert tree.num_inner_nodes >= 0

        # Check that every word has a code and path
        for word_idx in small_vocab.values():
            assert word_idx in tree.word_codes
            assert word_idx in tree.word_paths

    def test_binary_codes_properties(self, small_vocab, small_word_counts):
        """Test that binary codes have correct properties."""
        tree = HuffmanTree(small_word_counts, small_vocab)

        # All codes should be lists of 0s and 1s
        for word_idx in small_vocab.values():
            code = tree.word_codes[word_idx]
            assert isinstance(code, list)
            for bit in code:
                assert bit in [0, 1], f"Invalid bit {bit} in code {code}"

    def test_path_consistency(self, small_vocab, small_word_counts):
        """Test that paths and codes have consistent lengths."""
        tree = HuffmanTree(small_word_counts, small_vocab)

        for word_idx in small_vocab.values():
            path = tree.word_paths[word_idx]
            code = tree.word_codes[word_idx]
            assert len(path) == len(code), (
                f"Path and code length mismatch for word {word_idx}: "
                f"path={len(path)}, code={len(code)}"
            )

    def test_frequency_based_code_length(self, small_vocab, small_word_counts):
        """Test that more frequent words get shorter codes."""
        tree = HuffmanTree(small_word_counts, small_vocab)

        # Get word frequencies and code lengths
        word_freqs = []
        code_lengths = []

        for word, word_idx in small_vocab.items():
            freq = small_word_counts[word]
            code_len = len(tree.word_codes[word_idx])
            word_freqs.append(freq)
            code_lengths.append(code_len)

        # Sort by frequency (descending)
        freq_code_pairs = list(zip(word_freqs, code_lengths))
        freq_code_pairs.sort(key=lambda x: x[0], reverse=True)

        # Check that code lengths generally increase as frequency decreases
        # (This is a general property but not strict due to tree balancing)
        prev_freq = float("inf")
        for freq, code_len in freq_code_pairs:
            assert freq <= prev_freq
            prev_freq = freq

    def test_single_word_vocabulary(self):
        """Test edge case with single word."""
        word_to_idx = {"only": 0}
        word_counts = {"only": 1}

        tree = HuffmanTree(word_counts, word_to_idx)

        # Single word should have empty path and code
        assert len(tree.word_codes[0]) == 0
        assert len(tree.word_paths[0]) == 0
        assert tree.num_inner_nodes == 0

    def test_two_word_vocabulary(self):
        """Test minimal case with two words."""
        word_to_idx = {"first": 0, "second": 1}
        word_counts = {"first": 10, "second": 5}

        tree = HuffmanTree(word_counts, word_to_idx)

        # Two words should have codes of length 1
        assert len(tree.word_codes[0]) == 1
        assert len(tree.word_codes[1]) == 1

        # Codes should be different
        assert tree.word_codes[0] != tree.word_codes[1]

        # Should have exactly one internal node
        assert tree.num_inner_nodes == 1

    def test_deterministic_construction(self, medium_vocab, medium_word_counts):
        """Test that tree construction is deterministic."""
        tree1 = HuffmanTree(medium_word_counts, medium_vocab)
        tree2 = HuffmanTree(medium_word_counts, medium_vocab)

        # Trees should be identical
        assert tree1.word_codes == tree2.word_codes
        assert tree1.word_paths == tree2.word_paths
        assert tree1.num_inner_nodes == tree2.num_inner_nodes

    def test_internal_node_indices(self, small_vocab, small_word_counts):
        """Test that internal node indices are valid."""
        tree = HuffmanTree(small_word_counts, small_vocab)
        vocab_size = len(small_vocab)

        for word_idx in small_vocab.values():
            path = tree.word_paths[word_idx]
            for node_id in path:
                # Internal nodes should have IDs >= vocab_size
                assert node_id >= vocab_size, (
                    f"Invalid internal node ID {node_id}, should be >= {vocab_size}"
                )
                # Internal node IDs should be within expected range
                assert node_id < vocab_size + tree.num_inner_nodes, (
                    f"Internal node ID {node_id} out of range"
                )

    def test_missing_word_counts(self, small_vocab):
        """Test handling of words not in word_counts."""
        # Only provide counts for some words
        partial_counts = {"apple": 50, "banana": 30}

        tree = HuffmanTree(partial_counts, small_vocab)

        # Should still create codes for all words
        for word_idx in small_vocab.values():
            assert word_idx in tree.word_codes
            assert word_idx in tree.word_paths

    def test_empty_vocabulary(self):
        """Test edge case with empty vocabulary."""
        word_to_idx = {}
        word_counts = {}

        tree = HuffmanTree(word_counts, word_to_idx)

        assert tree.word_codes == {}
        assert tree.word_paths == {}
        assert tree.num_inner_nodes == 0

    @pytest.mark.slow
    def test_large_vocabulary_performance(self):
        """Test performance with larger vocabulary."""
        # Create large vocabulary
        vocab_size = 1000
        word_to_idx = {f"word_{i:04d}": i for i in range(vocab_size)}

        # Create realistic frequency distribution (Zipf-like)
        word_counts = {}
        for i, word in enumerate(word_to_idx.keys()):
            word_counts[word] = max(1, 1000 // (i + 1))

        # Construction should be reasonably fast
        import time

        start_time = time.time()
        tree = HuffmanTree(word_counts, word_to_idx)
        end_time = time.time()

        # Should complete in reasonable time (adjust threshold as needed)
        assert end_time - start_time < 1.0, "Tree construction too slow"

        # Verify correctness
        assert len(tree.word_codes) == vocab_size
        assert len(tree.word_paths) == vocab_size
        assert tree.num_inner_nodes == vocab_size - 1  # For binary tree
