"""Modern Word2Vec implementation with PyTorch."""

__version__ = "0.1.0"

# Core models
from modern_word2vec.models import SkipGramModel, CBOWModel, Word2VecBase

# Configuration
from modern_word2vec.config import DataConfig, TrainConfig, ModelConfig

# Data handling
from modern_word2vec.data import Word2VecDataset
from modern_word2vec.tokenization import Tokenizer
from modern_word2vec.vocabulary import VocabularyBuilder
from modern_word2vec.pairs import PairGenerator

# Hierarchical Softmax
from modern_word2vec.hierarchical_softmax import (
    HierarchicalSoftmax,
    HuffmanTree,
    build_word_counts_from_dataset,
)

# Negative Sampling
from modern_word2vec.negative_sampling import (
    NegativeSampling,
    NoiseSampler,
)

# Training
from modern_word2vec.training import Trainer

# Utilities
from modern_word2vec.utils import (
    get_device,
    set_seed,
    setup_device_optimizations,
    export_embeddings,
    load_vocab_and_embeddings,
    find_similar,
    compute_cosine_similarities,
    TensorBoardLogger,
)

__all__ = [
    # Models
    "SkipGramModel",
    "CBOWModel",
    "Word2VecBase",
    # Configuration
    "DataConfig",
    "TrainConfig",
    "ModelConfig",
    # Data
    "Word2VecDataset",
    "Tokenizer",
    "VocabularyBuilder",
    "PairGenerator",
    # Hierarchical Softmax
    "HierarchicalSoftmax",
    "HuffmanTree",
    "build_word_counts_from_dataset",
    # Negative Sampling
    "NegativeSampling",
    "NoiseSampler",
    # Training
    "Trainer",
    # Utilities
    "get_device",
    "set_seed",
    "setup_device_optimizations",
    "export_embeddings",
    "load_vocab_and_embeddings",
    "find_similar",
    "compute_cosine_similarities",
    "TensorBoardLogger",
]
