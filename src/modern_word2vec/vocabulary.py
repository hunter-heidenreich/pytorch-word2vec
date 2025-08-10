"""Vocabulary building utilities."""

import math
import random
from collections import Counter
from typing import Dict, List

from modern_word2vec.config import SPECIAL_TOKENS


class VocabularyBuilder:
    """Builder for Word2Vec vocabulary with subsampling support."""

    def __init__(self, vocab_size: int, subsample_t: float = 0.0):
        """Initialize vocabulary builder.

        Args:
            vocab_size: Maximum vocabulary size
            subsample_t: Subsampling threshold (0.0 to disable)
        """
        self.vocab_size = vocab_size
        self.subsample_t = subsample_t
        self.word_counts = None  # Will be set during vocabulary building

    def build_vocabulary(
        self, tokens: List[str], rng: random.Random
    ) -> tuple[Dict[str, int], Dict[int, str], List[str]]:
        """Build vocabulary from tokens with optional subsampling.

        Args:
            tokens: List of all tokens
            rng: Random number generator for subsampling

        Returns:
            Tuple of (word_to_idx, idx_to_word, filtered_tokens)
        """
        # Count words and build vocabulary
        word_counts = Counter(tokens)
        self.word_counts = word_counts  # Store for later use
        vocab_words = [word for word, _ in word_counts.most_common(self.vocab_size - 1)]

        # Create mappings
        word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        word_to_idx[SPECIAL_TOKENS["UNK"]] = len(vocab_words)
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        # Apply subsampling if enabled
        filtered_tokens = (
            self._apply_subsampling(tokens, word_counts, rng)
            if self.subsample_t > 0.0
            else tokens
        )

        return word_to_idx, idx_to_word, filtered_tokens

    def _apply_subsampling(
        self, tokens: List[str], word_counts: Counter, rng: random.Random
    ) -> List[str]:
        """Apply word subsampling to reduce frequent words.

        Args:
            tokens: Original tokens
            word_counts: Word frequency counts
            rng: Random number generator

        Returns:
            Filtered tokens after subsampling
        """
        total_tokens = len(tokens)
        token_freq = {word: count / total_tokens for word, count in word_counts.items()}

        def should_keep_word(word: str) -> bool:
            """Determine if word should be kept based on subsampling."""
            freq = token_freq.get(word, 0.0)
            if freq <= 0:
                return True

            # Word2Vec subsampling formula
            keep_prob = (math.sqrt(freq / self.subsample_t) + 1) * (
                self.subsample_t / freq
            )
            return rng.random() < keep_prob

        return [word for word in tokens if should_keep_word(word)]
