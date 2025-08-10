"""Enhanced tokenization utilities."""

import re
from typing import List

from modern_word2vec.config import CONTRACTIONS, SPECIAL_TOKENS, DataConfig


class Tokenizer:
    """Tokenizer with multiple strategies."""

    def __init__(self, config: DataConfig):
        """Initialize tokenizer with configuration.

        Args:
            config: Data configuration containing tokenizer settings
        """
        self.config = config

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text based on configuration.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        text = text.lower() if self.config.lowercase else text

        tokenizer_map = {
            "basic": self._basic_tokenize,
            "enhanced": self._enhanced_tokenize,
            "simple": self._simple_tokenize,
            "split": self._split_tokenize,
        }

        tokenize_fn = tokenizer_map.get(self.config.tokenizer, self._basic_tokenize)
        return tokenize_fn(text)

    def tokenize_corpus(self, texts: List[str]) -> List[str]:
        """Tokenize a corpus of texts.

        Args:
            texts: List of text documents

        Returns:
            List of all tokens
        """
        tokens = []
        for text in texts:
            tokens.extend(self.tokenize(text))
        return tokens

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization - alphanumeric words only."""
        return re.findall(r"\b\w+\b", text)

    def _enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with preprocessing."""
        # Handle contractions
        for pattern, replacement in CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Normalize special patterns
        text = re.sub(r"https?://\S+|www\.\S+", SPECIAL_TOKENS["URL"], text)
        text = re.sub(r"\S+@\S+\.\S+", SPECIAL_TOKENS["EMAIL"], text)

        # Normalize numbers if enabled
        if self.config.normalize_numbers:
            text = re.sub(r"\b\d{4}\b", SPECIAL_TOKENS["YEAR"], text)
            text = re.sub(r"\b\d+\.\d+\b", SPECIAL_TOKENS["DECIMAL"], text)
            text = re.sub(r"\b\d+\b", SPECIAL_TOKENS["NUMBER"], text)

        # Handle repeated punctuation
        text = re.sub(r"[!]{2,}", "!!", text)
        text = re.sub(r"[?]{2,}", "??", text)
        text = re.sub(r"[.]{3,}", "...", text)

        # Extract tokens and filter by length
        tokens = re.findall(r"\b\w+\b|[.,!?;]", text)
        return self._filter_tokens_by_length(tokens)

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization preserving some punctuation."""
        return re.findall(r"\b\w+\b|[.,!?;]", text)

    def _split_tokenize(self, text: str) -> List[str]:
        """Simple whitespace splitting."""
        return text.split()

    def _filter_tokens_by_length(self, tokens: List[str]) -> List[str]:
        """Filter tokens by length constraints."""
        return [
            token
            for token in tokens
            if self.config.min_token_length
            <= len(token)
            <= self.config.max_token_length
        ]
