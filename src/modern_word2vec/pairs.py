"""Training pair generation utilities."""

import bisect
import random
from typing import List, Tuple, Union

import torch


class PairGenerator:
    """Efficient training pair generator for Word2Vec."""

    def __init__(
        self,
        token_ids: List[int],
        window_size: int,
        model_type: str,
        dynamic_window: bool = False,
        rng: random.Random = None,
    ):
        """Initialize pair generator.

        Args:
            token_ids: List of token IDs
            window_size: Maximum window size
            model_type: Either "skipgram" or "cbow"
            dynamic_window: Whether to use dynamic window sizes
            rng: Random number generator
        """
        self.token_ids = token_ids
        self.window_size = window_size
        self.model_type = model_type.lower()
        self.dynamic_window = dynamic_window
        self.rng = rng or random.Random()

        # Pre-compute window sizes and pair mappings
        self.dynamic_window_sizes = []
        self.center_word_pair_starts = []
        self.total_pairs = 0

        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build efficient O(log n) index mapping for pair lookup."""
        for i in range(len(self.token_ids)):
            self.center_word_pair_starts.append(self.total_pairs)

            # Determine window size for this center word
            if self.model_type == "skipgram" and self.dynamic_window:
                window_size = self.rng.randint(1, self.window_size)
            else:
                window_size = self.window_size

            self.dynamic_window_sizes.append(window_size)

            # Calculate context size and number of pairs
            start = max(0, i - window_size)
            end = min(len(self.token_ids), i + window_size + 1)
            context_size = end - start - 1  # -1 for center word

            if self.model_type == "skipgram":
                self.total_pairs += context_size
            elif self.model_type == "cbow" and context_size > 0:
                self.total_pairs += 1

    def generate_pair_at_index(self, idx: int) -> Tuple[Union[int, List[int]], int]:
        """Generate training pair at given index using O(log n) lookup.

        Args:
            idx: Index of the pair to generate

        Returns:
            Tuple of (input, target) where input is int for skipgram,
            List[int] for CBOW

        Raises:
            IndexError: If index is out of range
        """
        if idx >= self.total_pairs:
            raise IndexError(
                f"Index {idx} out of range for dataset size {self.total_pairs}"
            )

        # Binary search to find center word
        center_idx = bisect.bisect_right(self.center_word_pair_starts, idx) - 1
        local_idx = idx - self.center_word_pair_starts[center_idx]

        # Get center word and context
        center = self.token_ids[center_idx]
        window_size = self.dynamic_window_sizes[center_idx]

        start = max(0, center_idx - window_size)
        end = min(len(self.token_ids), center_idx + window_size + 1)
        context = [self.token_ids[j] for j in range(start, end) if j != center_idx]

        # Generate appropriate pair based on model type
        if self.model_type == "skipgram":
            if local_idx < len(context):
                return (center, context[local_idx])
        elif self.model_type == "cbow":
            if len(context) > 0 and local_idx == 0:
                return (context, center)

        raise IndexError(f"Failed to generate pair at index {idx}")

    def __len__(self) -> int:
        """Return total number of training pairs."""
        return self.total_pairs


def create_pair_tensors(
    src: Union[int, List[int]], tgt: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create tensors from training pairs.

    Args:
        src: Source (int for skipgram, List[int] for CBOW)
        tgt: Target token ID

    Returns:
        Tuple of (source_tensor, target_tensor)
    """
    return torch.tensor(src), torch.tensor(tgt)
