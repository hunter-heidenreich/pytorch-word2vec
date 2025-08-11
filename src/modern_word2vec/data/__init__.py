"""Data handling and configuration for Word2Vec."""

import random
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from modern_word2vec.config import DataConfig, SPECIAL_TOKENS
from modern_word2vec.tokenization import Tokenizer
from modern_word2vec.vocabulary import VocabularyBuilder
from modern_word2vec.pairs import PairGenerator, create_pair_tensors

# Import streaming functionality
try:
    from modern_word2vec.data.streaming import (
        StreamingWord2VecDataset,
        StreamingDataConfig,
        create_streaming_dataloader,
    )

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False


class Word2VecDataset(Dataset):
    """Dataset for Word2Vec training."""

    def __init__(
        self,
        texts: List[str],
        config: DataConfig,
        rng: Optional[random.Random] = None,
    ):
        """Initialize dataset.

        Args:
            texts: List of text documents
            config: Data configuration
            rng: Random number generator for reproducibility
        """
        self.cfg = config
        self.rng = rng or random.Random()

        # Tokenize corpus
        tokenizer = Tokenizer(config)
        tokens = tokenizer.tokenize_corpus(texts)

        # Build vocabulary with optional subsampling
        self.vocab_builder = VocabularyBuilder(config.vocab_size, config.subsample_t)
        self.word_to_idx, self.idx_to_word, filtered_tokens = (
            self.vocab_builder.build_vocabulary(tokens, self.rng)
        )
        self.vocab_size = len(self.word_to_idx)

        # Convert tokens to IDs
        token_ids = [
            self.word_to_idx.get(word, self.word_to_idx[SPECIAL_TOKENS["UNK"]])
            for word in filtered_tokens
        ]

        # Initialize pair generator
        self.pair_generator = PairGenerator(
            token_ids=token_ids,
            window_size=config.window_size,
            model_type=config.model_type,
            dynamic_window=config.dynamic_window,
            rng=self.rng,
        )

    def __len__(self) -> int:
        """Return number of training pairs."""
        return len(self.pair_generator)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training pair at index.

        Args:
            idx: Index of training pair

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        src, tgt = self.pair_generator.generate_pair_at_index(idx)
        return create_pair_tensors(src, tgt)


def cbow_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for CBOW training with padding for variable-length contexts.

    Args:
        batch: List of (input, target) tensor pairs where inputs can have variable lengths

    Returns:
        Tuple of (batched_inputs, batched_targets)
    """
    inputs = []
    targets = []

    for item in batch:
        inp, tgt = item
        # Ensure inputs are 1D tensors
        if inp.dim() == 0:
            inp = inp.unsqueeze(0)
        inputs.append(inp)
        targets.append(tgt)

    # Find maximum context length in this batch
    max_len = max(inp.size(0) for inp in inputs)

    # Pad all inputs to the same length
    padded_inputs = []
    for inp in inputs:
        if inp.size(0) < max_len:
            # Pad with zeros (could use a special padding token if needed)
            padding = torch.zeros(max_len - inp.size(0), dtype=inp.dtype)
            padded_inp = torch.cat([inp, padding])
        else:
            padded_inp = inp
        padded_inputs.append(padded_inp)

    return torch.stack(padded_inputs), torch.stack(targets)


def load_texts_from_hf(dataset: str, config: Optional[str], split: str) -> List[str]:
    """Load texts from Hugging Face datasets.

    Args:
        dataset: Dataset name
        config: Dataset configuration
        split: Dataset split

    Returns:
        List of text strings

    Raises:
        ValueError: If no suitable text column is found
    """
    ds = load_dataset(dataset, config, split=split)

    # Search for text column
    text_columns = ["text", "content", "sentence", "document", "raw"]
    text_col = next((col for col in text_columns if col in ds.column_names), None)

    if text_col is None:
        raise ValueError(
            f"Could not find a text column in dataset {dataset}. "
            f"Available columns: {ds.column_names}"
        )

    return [text for text in ds[text_col] if isinstance(text, str) and text.strip()]


def generate_synthetic_texts(
    n_sentences: int, vocab_size: int, rng: random.Random
) -> List[str]:
    """Generate synthetic texts for testing.

    Args:
        n_sentences: Number of sentences to generate
        vocab_size: Size of vocabulary to use
        rng: Random number generator

    Returns:
        List of synthetic text strings
    """
    words = [f"tok{i}" for i in range(vocab_size)]
    texts = []

    for _ in range(n_sentences):
        sentence_length = rng.randint(5, 20)
        sentence = " ".join(rng.choice(words) for _ in range(sentence_length))
        texts.append(sentence)

    return texts
