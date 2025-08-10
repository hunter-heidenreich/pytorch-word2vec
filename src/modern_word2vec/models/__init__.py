"""Model definitions for Word2Vec."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from modern_word2vec.hierarchical_softmax import HierarchicalSoftmax, build_word_counts_from_dataset


class Word2VecBase(nn.Module):
    """Base class for Word2Vec models with shared initialization logic."""

    def __init__(self, vocab_size: int, embedding_dim: int, output_layer_type: str = "full_softmax",
                 dataset=None):
        """Initialize base Word2Vec model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            output_layer_type: Type of output layer ("full_softmax" or "hierarchical_softmax")
            dataset: Dataset instance (required for hierarchical softmax to build tree)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_layer_type = output_layer_type

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Create appropriate output layer
        if output_layer_type == "hierarchical_softmax":
            if dataset is None:
                raise ValueError("Dataset is required for hierarchical softmax to build Huffman tree")
            
            word_counts = build_word_counts_from_dataset(dataset)
            self.hierarchical_softmax = HierarchicalSoftmax(
                embedding_dim, vocab_size, word_counts, dataset.word_to_idx
            )
            self.out_embeddings = None  # Not used in hierarchical softmax
        else:
            # Standard full softmax
            self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.hierarchical_softmax = None

        # Initialize weights following original word2vec.c approach:
        # Input embeddings: random uniform [-0.5, 0.5] / embedding_dim
        # Output embeddings: zero initialization (for full softmax)
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights following original word2vec.c approach."""
        # Input embeddings (syn0): uniform random [-0.5, 0.5] / embedding_dim
        with torch.no_grad():
            self.in_embeddings.weight.uniform_(-0.5, 0.5)
            self.in_embeddings.weight.div_(self.embedding_dim)

            # Output embeddings (syn1/syn1neg): zero initialization (only for full softmax)
            if self.out_embeddings is not None:
                self.out_embeddings.weight.zero_()

    def _compute_scores(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute output scores from input embeddings (full softmax only).

        Args:
            input_embeddings: Tensor of shape (batch, embedding_dim)

        Returns:
            Tensor of shape (batch, vocab_size) containing scores for all words
        """
        if self.output_layer_type == "hierarchical_softmax":
            raise ValueError("Cannot compute full scores with hierarchical softmax. Use compute_loss instead.")
        
        return torch.matmul(input_embeddings, self.out_embeddings.weight.t())
    
    def compute_loss(self, input_embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss using the appropriate output layer.

        Args:
            input_embeddings: Input embeddings tensor
            targets: Target word indices

        Returns:
            Loss tensor
        """
        if self.output_layer_type == "hierarchical_softmax":
            return self.hierarchical_softmax(input_embeddings, targets)
        else:
            # Standard cross-entropy loss with full softmax
            scores = self._compute_scores(input_embeddings)
            return torch.nn.functional.cross_entropy(scores, targets)


class SkipGramModel(Word2VecBase):
    """Skip-gram model for Word2Vec.

    In skip-gram, we predict context words given a center word.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, output_layer_type: str = "full_softmax",
                 dataset=None):
        """Initialize Skip-gram model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            output_layer_type: Type of output layer ("full_softmax" or "hierarchical_softmax")
            dataset: Dataset instance (required for hierarchical softmax)
        """
        super().__init__(vocab_size, embedding_dim, output_layer_type, dataset)

    def forward(self, target_word: torch.Tensor, context_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            target_word: Tensor of shape (batch,) containing target word indices
            context_targets: For hierarchical softmax, tensor of context word indices to predict

        Returns:
            If hierarchical softmax: loss tensor
            If full softmax: tensor of shape (batch, vocab_size) containing scores for all words
        """
        # target_word: (batch,)
        in_embeds = self.in_embeddings(target_word)  # (batch, D)
        
        if self.output_layer_type == "hierarchical_softmax":
            if context_targets is None:
                raise ValueError("context_targets required for hierarchical softmax forward pass")
            return self.compute_loss(in_embeds, context_targets)
        else:
            return self._compute_scores(in_embeds)


class CBOWModel(Word2VecBase):
    """CBOW (Continuous Bag of Words) model for Word2Vec.

    In CBOW, we predict a center word given its context words.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, output_layer_type: str = "full_softmax",
                 dataset=None):
        """Initialize CBOW model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            output_layer_type: Type of output layer ("full_softmax" or "hierarchical_softmax")
            dataset: Dataset instance (required for hierarchical softmax)
        """
        super().__init__(vocab_size, embedding_dim, output_layer_type, dataset)

    def forward(self, context_words: torch.Tensor, center_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            context_words: Tensor of shape (batch, 2*window) containing context word indices
            center_targets: For hierarchical softmax, tensor of center word indices to predict

        Returns:
            If hierarchical softmax: loss tensor
            If full softmax: tensor of shape (batch, vocab_size) containing scores for all words
        """
        # context_words: (batch, 2*window)
        in_embeds = self.in_embeddings(context_words)  # (batch, 2W, D)
        context_vector = torch.mean(in_embeds, dim=1)  # (batch, D)
        
        if self.output_layer_type == "hierarchical_softmax":
            if center_targets is None:
                raise ValueError("center_targets required for hierarchical softmax forward pass")
            return self.compute_loss(context_vector, center_targets)
        else:
            return self._compute_scores(context_vector)
