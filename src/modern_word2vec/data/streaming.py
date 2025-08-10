"""Streaming dataset implementation for Word2Vec training."""

import math
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union, Dict

import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import IterableDataset as HFIterableDataset, load_dataset

from modern_word2vec.tokenization import Tokenizer


@dataclass
class StreamingDataConfig:
    """Configuration for streaming data processing."""

    vocab_size: int = 10000
    window_size: int = 2
    model_type: str = "skipgram"  # or cbow
    lowercase: bool = True
    tokenizer: str = "basic"  # basic|enhanced|simple|split
    min_token_length: int = 1
    max_token_length: int = 50
    normalize_numbers: bool = False
    subsample_t: float = 0.0
    dynamic_window: bool = False

    # Streaming-specific configs
    buffer_size: int = 10000  # Token buffer for context generation
    vocab_sample_size: int = 1000000  # Tokens to sample for vocab building
    shuffle_buffer_size: int = 1000  # Training pairs shuffle buffer
    min_pairs_per_batch: int = 100  # Minimum pairs to accumulate before yielding


class StreamingWord2VecDataset(IterableDataset):
    """True one-pass streaming dataset for Word2Vec training.

    This implementation:
    1. Requires a pre-built vocabulary (no vocabulary building during training)
    2. Makes a single pass over the data source
    3. Uses a true sliding window for pair generation
    4. Handles multi-worker scenarios correctly

    For billion-word datasets, vocabulary should be built offline in a separate process.
    """

    def __init__(
        self,
        text_source: Union[List[str], Iterator[str], HFIterableDataset],
        vocab: Dict[str, int],
        config: StreamingDataConfig,
        rng: Optional[random.Random] = None,
    ):
        """Initialize streaming dataset with pre-built vocabulary.

        Args:
            text_source: Source of text data (list, iterator, or HF dataset)
            vocab: Pre-built vocabulary mapping words to indices (REQUIRED)
            config: Streaming data configuration
            rng: Random number generator for reproducibility

        Raises:
            ValueError: If vocab is None (vocabulary is required for streaming)
        """
        if vocab is None:
            raise ValueError(
                "StreamingWord2VecDataset requires a pre-built vocabulary. "
                "For large datasets, build vocabulary offline using build_vocabulary_from_stream(). "
                "For small datasets, use the non-streaming Word2VecDataset instead."
            )

        self.cfg = config
        self.text_source = text_source
        self.rng = rng or random.Random()

        # Use provided vocabulary
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.idx_to_word = {i: w for w, i in self.vocab.items()}

        # Precompute subsampling probabilities if enabled
        self.subsample_probs = None
        if config.subsample_t > 0:
            print("Computing subsampling probabilities from vocabulary...")
            self.subsample_probs = self._compute_subsample_probs_from_vocab()

    def _compute_subsample_probs_from_vocab(self) -> Dict[str, float]:
        """Compute subsampling probabilities using zipfian distribution assumption.

        Since we can't make multiple passes, we estimate word frequencies
        using a zipfian distribution based on vocabulary rank.
        """
        subsample_probs = {}

        # Estimate frequencies using zipfian distribution (common in natural language)
        # Most frequent word gets frequency 0.1, then decreases as 1/rank
        for word, rank in self.vocab.items():
            if word == "<UNK>":
                freq = 0.01  # Low frequency for unknown words
            else:
                freq = 0.1 / (rank + 1)  # Zipfian-like distribution

            if freq > self.cfg.subsample_t:
                prob = (math.sqrt(freq / self.cfg.subsample_t) + 1) * (
                    self.cfg.subsample_t / freq
                )
                subsample_probs[word] = min(1.0, prob)
            else:
                subsample_probs[word] = 1.0

        return subsample_probs

    def _create_text_iterator(self) -> Iterator[str]:
        """Create a single-use iterator over the text source.

        This iterator can only be consumed once - perfect for true streaming.
        """
        if isinstance(self.text_source, list):
            return iter(self.text_source)
        elif hasattr(self.text_source, "__iter__"):
            return iter(self.text_source)
        elif isinstance(self.text_source, HFIterableDataset):
            # For HF datasets, extract the text field
            return (item.get("text", "") for item in self.text_source)
        else:
            raise ValueError(f"Unsupported text source type: {type(self.text_source)}")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize a single text."""
        if not text:
            return []

        text = text.lower() if self.cfg.lowercase else text

        if self.cfg.tokenizer == "basic":
            tokens = re.findall(r"\b\w+\b", text)
        elif self.cfg.tokenizer == "enhanced":
            tokens = self._enhanced_tokenize(text)
        elif self.cfg.tokenizer == "simple":
            tokens = re.findall(r"\b\w+\b|[.,!?;]", text)
        else:  # "split"
            tokens = text.split()

        # Filter by length
        tokens = [
            t
            for t in tokens
            if self.cfg.min_token_length <= len(t) <= self.cfg.max_token_length
        ]

        return tokens

    def _enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization (simplified version)."""
        # Handle contractions
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would",
            r"'m": " am",
        }

        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)

        # Normalize special tokens
        text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
        text = re.sub(r"\S+@\S+\.\S+", "<EMAIL>", text)

        if self.cfg.normalize_numbers:
            text = re.sub(r"\b\d{4}\b", "<YEAR>", text)
            text = re.sub(r"\b\d+\.\d+\b", "<DECIMAL>", text)
            text = re.sub(r"\b\d+\b", "<NUMBER>", text)

        return re.findall(r"\b\w+\b|[.,!?;]", text)

    def _should_keep_token(self, token: str) -> bool:
        """Check if token should be kept based on subsampling."""
        if not self.subsample_probs or token not in self.subsample_probs:
            return True
        return self.rng.random() < self.subsample_probs[token]

    def _generate_pairs_from_sliding_window(
        self, center_idx: int, window: List[int]
    ) -> Iterator[Tuple[Union[int, List[int]], int]]:
        """Generate training pairs for a single center word using a sliding window.

        This is called once per new token that enters the center of the window,
        avoiding the inefficient re-processing of the same tokens.
        """
        if center_idx >= len(window):
            return

        center = window[center_idx]

        # Dynamic window sizing
        if self.cfg.model_type == "skipgram" and self.cfg.dynamic_window:
            W = self.rng.randint(1, self.cfg.window_size)
        else:
            W = self.cfg.window_size

        # Extract context around center
        start = max(0, center_idx - W)
        end = min(len(window), center_idx + W + 1)
        context = [window[j] for j in range(start, end) if j != center_idx]

        if self.cfg.model_type == "skipgram":
            for c in context:
                yield (center, c)
        elif self.cfg.model_type == "cbow":
            # Fix: Accept any non-empty context, not just full windows
            if len(context) > 0:
                yield (context, center)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Single-pass iteration over training pairs using true streaming.

        This implementation:
        1. Makes exactly ONE pass over the data source
        2. Uses a true sliding window for efficient pair generation
        3. Generates pairs only for new tokens entering the center
        4. Handles multi-worker scenarios correctly
        """
        # Handle multi-worker case
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "StreamingWord2VecDataset does not support num_workers > 0. "
                "Streaming datasets with multiple workers require complex data "
                "partitioning logic that is not implemented. Use num_workers=0."
            )

        # Sliding window buffer for maintaining context
        # Size: 2 * window_size + 1 to always have full context for center
        window_size = 2 * self.cfg.window_size + 1
        sliding_window = deque(maxlen=window_size)

        # Local shuffle buffer for breaking up sequential patterns
        pair_buffer = []

        # Single pass over the data source - this is the only time we consume it
        print("Starting single-pass streaming over data source...")
        text_iter = self._create_text_iterator()

        for text in text_iter:
            # Tokenize and convert to IDs
            tokens = self._tokenize_text(text)

            # Apply subsampling
            if self.subsample_probs:
                tokens = [
                    t for t in tokens if t in self.vocab and self._should_keep_token(t)
                ]

            # Convert to IDs
            token_ids = [
                self.vocab.get(t, self.vocab["<UNK>"])
                for t in tokens
                if t in self.vocab or "<UNK>" in self.vocab
            ]

            # Process each token with true sliding window
            for token_id in token_ids:
                # Add new token to sliding window
                sliding_window.append(token_id)

                # Generate pairs only when window has enough context
                # and only for the token that just became the center
                if len(sliding_window) >= window_size:
                    center_idx = len(sliding_window) // 2  # Middle of the window

                    # Generate pairs for this center token
                    for pair in self._generate_pairs_from_sliding_window(
                        center_idx, list(sliding_window)
                    ):
                        pair_buffer.append(pair)

                        # Yield pairs when buffer reaches target size
                        if len(pair_buffer) >= self.cfg.min_pairs_per_batch:
                            # Limited local shuffling
                            if len(pair_buffer) >= self.cfg.shuffle_buffer_size:
                                self.rng.shuffle(pair_buffer)

                            # Yield a batch
                            for src, tgt in pair_buffer[: self.cfg.min_pairs_per_batch]:
                                yield torch.tensor(src), torch.tensor(tgt)

                            # Keep remaining pairs for next batch
                            pair_buffer = pair_buffer[self.cfg.min_pairs_per_batch :]

        # Yield any remaining pairs
        if pair_buffer:
            self.rng.shuffle(pair_buffer)
            for src, tgt in pair_buffer:
                yield torch.tensor(src), torch.tensor(tgt)

        print("Completed single-pass streaming.")


def build_vocabulary_from_stream(
    text_source: Union[List[str], Iterator[str], str],
    config: StreamingDataConfig,
    output_path: Optional[str] = None,
) -> Dict[str, int]:
    """Build vocabulary from a text stream - a separate offline process.

    This function should be run once as a preprocessing step for large datasets.
    It consumes the entire text source to build a comprehensive vocabulary.

    Args:
        text_source: Source of text data (list, iterator, HF dataset name, or file path)
        config: Configuration for tokenization and vocabulary size
        output_path: Optional path to save vocabulary as JSON file

    Returns:
        Dictionary mapping words to indices
    """
    print("Building vocabulary from text stream...")

    # Handle different text source types
    if isinstance(text_source, str):
        if text_source.endswith(".txt") or "/" in text_source:
            # File path
            def file_iterator():
                with open(text_source, "r") as f:
                    for line in f:
                        yield line.strip()

            text_iter = file_iterator()
        else:
            # HuggingFace dataset name
            dataset = load_dataset(text_source, streaming=True, split="train")
            text_iter = (item.get("text", "") for item in dataset)
    elif isinstance(text_source, list):
        text_iter = iter(text_source)
    else:
        text_iter = text_source

    # Use the dedicated Tokenizer class for clean, reusable tokenization
    tokenizer = Tokenizer(config)

    # Count all tokens
    from collections import Counter

    token_counts = Counter()
    tokens_seen = 0
    texts_processed = 0

    for text in text_iter:
        if not text:
            continue

        texts_processed += 1
        if texts_processed % 10000 == 0:
            print(f"Processed {texts_processed:,} texts, {tokens_seen:,} tokens...")

        tokens = tokenizer.tokenize(text)
        for token in tokens:
            token_counts[token] += 1
            tokens_seen += 1

    print(
        f"Vocabulary building complete: {tokens_seen:,} tokens from {texts_processed:,} texts"
    )

    # Build vocabulary mapping
    most_common = token_counts.most_common(config.vocab_size - 1)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    vocab["<UNK>"] = len(vocab)

    print(f"Built vocabulary with {len(vocab):,} words")

    # Save vocabulary if path provided
    if output_path:
        import json

        with open(output_path, "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"Vocabulary saved to {output_path}")

    return vocab


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """Load a pre-built vocabulary from a JSON file.

    Args:
        vocab_path: Path to vocabulary JSON file

    Returns:
        Dictionary mapping words to indices
    """
    import json

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab):,} words from {vocab_path}")
    return vocab


def create_streaming_dataloader(
    text_source: Union[List[str], Iterator[str], str],
    vocab: Dict[str, int],
    config: StreamingDataConfig,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create a streaming dataloader for Word2Vec training.

    Args:
        text_source: Source of text data (list, iterator, HF dataset name, or file path)
        vocab: Pre-built vocabulary (REQUIRED - use build_vocabulary_from_stream)
        config: Streaming configuration
        batch_size: Batch size
        num_workers: Number of worker processes (MUST be 0 for streaming datasets)
        seed: Random seed

    Returns:
        DataLoader for streaming training

    Raises:
        ValueError: If num_workers > 0 or vocab is None
    """
    if num_workers > 0:
        raise ValueError(
            "StreamingWord2VecDataset does not support num_workers > 0. "
            "Multi-worker streaming requires complex data partitioning that can "
            "lead to incorrect training. Use num_workers=0 for streaming datasets."
        )

    if vocab is None:
        raise ValueError(
            "vocab is required for streaming datasets. Use build_vocabulary_from_stream() "
            "to create a vocabulary from your data source first."
        )

    # Handle different text source types
    if isinstance(text_source, str):
        if text_source.endswith(".txt") or "/" in text_source:
            # File path
            def file_iterator():
                with open(text_source, "r") as f:
                    for line in f:
                        yield line.strip()

            text_iter = file_iterator()
        else:
            # HuggingFace dataset name
            dataset = load_dataset(text_source, streaming=True, split="train")
            text_iter = dataset
    else:
        text_iter = text_source

    # Create dataset
    dataset = StreamingWord2VecDataset(
        text_iter, vocab, config, rng=random.Random(seed)
    )

    # Improved collate function
    def collate_fn(batch):
        if not batch:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        if config.model_type == "cbow":
            # For CBOW, inputs are lists of context tokens
            inputs = []
            targets = []
            for src, tgt in batch:
                if isinstance(src, torch.Tensor) and src.numel() > 0:
                    inputs.append(src)
                    targets.append(tgt)

            if inputs:
                # Pad context sequences to same length for batching
                max_len = max(inp.numel() for inp in inputs)
                padded_inputs = []
                for inp in inputs:
                    if inp.numel() < max_len:
                        padding = torch.zeros(max_len - inp.numel(), dtype=inp.dtype)
                        padded_inputs.append(torch.cat([inp, padding]))
                    else:
                        padded_inputs.append(inp)
                return torch.stack(padded_inputs), torch.stack(targets)
            else:
                return torch.empty(0, dtype=torch.long), torch.empty(
                    0, dtype=torch.long
                )
        else:
            # For skip-gram, standard collation
            inputs, targets = zip(*batch)
            return torch.stack(inputs), torch.stack(targets)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Force single worker
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def load_streaming_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
) -> HFIterableDataset:
    """Load a streaming dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset
        config_name: Configuration name
        split: Dataset split

    Returns:
        Streaming HuggingFace dataset
    """
    return load_dataset(dataset_name, config_name, split=split, streaming=True)
