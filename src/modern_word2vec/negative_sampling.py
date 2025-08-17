"""Negative Sampling implementation for Word2Vec.

This module implements negative sampling as described in the Word2Vec papers:
- "Distributed Representations of Words and Phrases and their Compositionality" (Mikolov et al., 2013)
- "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)

Negative sampling reduces computational complexity from O(V) to O(k) where V is vocabulary 
size and k is the number of negative samples (typically 5-20).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseSampler:
    """Noise sampler for negative sampling using word frequency distribution.
    
    Implements the unigram distribution raised to the 3/4 power as described
    in the original Word2Vec paper. This gives less frequent words a higher
    probability of being selected as negative samples.
    """

    def __init__(
        self,
        word_counts: Dict[str, int],
        word_to_idx: Dict[str, int],
        power: float = 0.75,
    ):
        """Initialize noise sampler.

        Args:
            word_counts: Dictionary mapping words to their frequency counts
            word_to_idx: Dictionary mapping words to vocabulary indices
            power: Power to raise the unigram distribution to (default 0.75)
        """
        self.word_to_idx = word_to_idx
        self.vocab_size = len(word_to_idx)
        self.power = power

        # Build sampling probabilities
        self.sampling_probs = self._build_sampling_probabilities(word_counts)

    def _build_sampling_probabilities(self, word_counts: Dict[str, int]) -> torch.Tensor:
        """Build sampling probabilities for negative sampling.

        Args:
            word_counts: Word frequency counts

        Returns:
            Tensor of sampling probabilities for each word index
        """
        # Initialize probabilities tensor
        probs = torch.zeros(self.vocab_size)

        # Calculate total count with power transformation
        total_count = 0.0
        for word, word_idx in self.word_to_idx.items():
            count = word_counts.get(word, 1)  # Default count of 1 if not found
            powered_count = count ** self.power
            probs[word_idx] = powered_count
            total_count += powered_count

        # Normalize to get probabilities
        if total_count > 0:
            probs = probs / total_count
        else:
            # Uniform distribution fallback
            probs.fill_(1.0 / self.vocab_size)

        return probs

    def sample_negative(
        self, 
        batch_size: int, 
        num_negative: int, 
        positive_words: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample negative words for a batch.

        Args:
            batch_size: Number of samples in the batch
            num_negative: Number of negative samples per positive example
            positive_words: Positive word indices to avoid sampling (optional)
            device: Device to place the sampled indices on

        Returns:
            Tensor of shape (batch_size, num_negative) containing negative word indices
        """
        if device is None:
            device = torch.device('cpu')

        # Move sampling probabilities to the correct device
        if self.sampling_probs.device != device:
            self.sampling_probs = self.sampling_probs.to(device)

        # Sample negative words using multinomial distribution
        total_samples = batch_size * num_negative
        negative_samples = torch.multinomial(
            self.sampling_probs, 
            total_samples, 
            replacement=True
        )

        # Reshape to (batch_size, num_negative)
        negative_samples = negative_samples.view(batch_size, num_negative)

        # Optional: Replace any negative samples that match positive words
        # This is computationally expensive for large batches, so we make it optional
        if positive_words is not None and positive_words.numel() > 0:
            negative_samples = self._replace_positive_negatives(
                negative_samples, positive_words
            )

        return negative_samples

    def _replace_positive_negatives(
        self, 
        negative_samples: torch.Tensor, 
        positive_words: torch.Tensor
    ) -> torch.Tensor:
        """Replace negative samples that match positive words.

        Args:
            negative_samples: Tensor of shape (batch_size, num_negative)
            positive_words: Tensor of positive word indices

        Returns:
            Corrected negative samples tensor
        """
        batch_size, num_negative = negative_samples.shape

        # Expand positive words for comparison
        if positive_words.dim() == 1:
            positive_words = positive_words.unsqueeze(1)  # (batch_size, 1)

        # Find matches
        matches = negative_samples.unsqueeze(2) == positive_words.unsqueeze(1)
        matches = matches.any(dim=2)  # (batch_size, num_negative)

        # Resample where there are matches
        if matches.any():
            num_replacements = matches.sum().item()
            replacements = torch.multinomial(
                self.sampling_probs, 
                num_replacements, 
                replacement=True
            )
            negative_samples[matches] = replacements

        return negative_samples


class NegativeSampling(nn.Module):
    """Negative Sampling layer for efficient Word2Vec training.

    This replaces the standard softmax output layer with negative sampling,
    which reduces computational complexity from O(V) to O(k) where V is 
    vocabulary size and k is the number of negative samples.

    The approach uses the sigmoid function instead of softmax and trains
    the model to distinguish between actual context words (positive examples)
    and randomly sampled words (negative examples).
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        word_counts: Dict[str, int],
        word_to_idx: Dict[str, int],
        num_negative: int = 5,
        power: float = 0.75,
    ):
        """Initialize negative sampling layer.

        Args:
            embedding_dim: Dimension of input embeddings
            vocab_size: Size of vocabulary
            word_counts: Word frequency counts for building noise distribution
            word_to_idx: Mapping from words to vocabulary indices
            num_negative: Number of negative samples per positive example
            power: Power to raise the unigram distribution to (default 0.75)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_negative = num_negative

        # Create output embeddings (different from input embeddings)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights following original word2vec approach
        self._init_weights()

        # Create noise sampler
        self.noise_sampler = NoiseSampler(word_counts, word_to_idx, power)

    def _init_weights(self):
        """Initialize output embedding weights."""
        with torch.no_grad():
            # Initialize to zero as in original word2vec
            self.out_embeddings.weight.zero_()

    def forward(
        self, 
        input_embeddings: torch.Tensor, 
        target_words: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative sampling loss.

        Args:
            input_embeddings: Input embeddings of shape (batch_size, embedding_dim)
            target_words: Target word indices of shape (batch_size,)

        Returns:
            Loss tensor for the batch
        """
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device

        # Get positive target embeddings
        positive_out_embeds = self.out_embeddings(target_words)  # (batch_size, embedding_dim)

        # Compute positive scores (should be high)
        positive_scores = torch.sum(
            input_embeddings * positive_out_embeds, dim=1
        )  # (batch_size,)

        # Sample negative words
        negative_words = self.noise_sampler.sample_negative(
            batch_size, self.num_negative, target_words, device
        )  # (batch_size, num_negative)

        # Get negative target embeddings
        negative_out_embeds = self.out_embeddings(negative_words)  # (batch_size, num_negative, embedding_dim)

        # Compute negative scores (should be low)
        negative_scores = torch.bmm(
            negative_out_embeds, 
            input_embeddings.unsqueeze(2)
        ).squeeze(2)  # (batch_size, num_negative)

        # Apply sigmoid to get probabilities
        positive_loss = F.logsigmoid(positive_scores)  # log P(D=1|w,c)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)  # sum of log P(D=0|w',c)

        # Total loss is negative log likelihood
        total_loss = -(positive_loss + negative_loss)

        return total_loss.mean()

    def predict_scores(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute scores for all words given input embeddings.

        Note: This is computationally expensive O(V) and mainly for evaluation.
        For inference, consider using approximate methods or top-k prediction.

        Args:
            input_embeddings: Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            Score tensor of shape (batch_size, vocab_size)
        """
        # Compute scores using dot product with all output embeddings
        scores = torch.matmul(input_embeddings, self.out_embeddings.weight.t())
        return scores

    def predict_probabilities(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute probabilities for all words given input embeddings.

        Note: This computes sigmoid probabilities, not softmax probabilities.
        The probabilities don't sum to 1 across the vocabulary.

        Args:
            input_embeddings: Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            Probability tensor of shape (batch_size, vocab_size)
        """
        scores = self.predict_scores(input_embeddings)
        return torch.sigmoid(scores)


def build_word_counts_from_dataset(dataset) -> Dict[str, int]:
    """Build word frequency counts from a Word2Vec dataset.

    Args:
        dataset: Word2VecDataset instance

    Returns:
        Dictionary mapping words to their frequency counts
    """
    word_counts = {}

    # Get word counts from the dataset's vocabulary builder if available
    if hasattr(dataset, "vocab_builder") and hasattr(
        dataset.vocab_builder, "word_counts"
    ):
        # The vocab builder stores a Counter object with word strings as keys
        vocab_counts = dataset.vocab_builder.word_counts
        for word in dataset.word_to_idx.keys():
            word_counts[word] = vocab_counts.get(word, 1)
    else:
        # Fallback: assign uniform counts
        for word in dataset.word_to_idx.keys():
            word_counts[word] = 1

    return word_counts
