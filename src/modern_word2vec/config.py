"""Centralized configuration management for Word2Vec."""

from dataclasses import dataclass
from typing import Dict, Literal


# Constants
DEFAULT_VOCAB_SIZE = 10000
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_WINDOW_SIZE = 2
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-3
DEFAULT_SEED = 42

MIN_TOKEN_LENGTH = 1
MAX_TOKEN_LENGTH = 50

# Type aliases
ModelType = Literal["skipgram", "cbow"]
TokenizerType = Literal["basic", "enhanced", "simple", "split"]
OptimizerType = Literal["adam", "sgd"]
DeviceType = Literal["cuda", "mps", "cpu"]
OutputLayerType = Literal["full_softmax", "hierarchical_softmax"]


@dataclass
class DataConfig:
    """Configuration for data processing."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    window_size: int = DEFAULT_WINDOW_SIZE
    model_type: ModelType = "skipgram"
    lowercase: bool = True
    tokenizer: TokenizerType = "basic"
    min_token_length: int = MIN_TOKEN_LENGTH
    max_token_length: int = MAX_TOKEN_LENGTH
    normalize_numbers: bool = False
    subsample_t: float = 0.0  # e.g., 1e-5 to enable subsampling
    dynamic_window: bool = False


@dataclass
class TrainConfig:
    """Configuration for training."""

    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: OptimizerType = "adam"
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    compile: bool = False
    mixed_precision: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = DEFAULT_SEED


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    model_type: ModelType = "skipgram"
    output_layer: OutputLayerType = "full_softmax"


# Contraction mappings for enhanced tokenization
CONTRACTIONS: Dict[str, str] = {
    r"won't": "will not",
    r"can't": "cannot",
    r"n't": " not",
    r"'re": " are",
    r"'ve": " have",
    r"'ll": " will",
    r"'d": " would",
    r"'m": " am",
}

# Special tokens for enhanced tokenization
SPECIAL_TOKENS = {
    "URL": "<URL>",
    "EMAIL": "<EMAIL>",
    "YEAR": "<YEAR>",
    "DECIMAL": "<DECIMAL>",
    "NUMBER": "<NUMBER>",
    "UNK": "<UNK>",
}
