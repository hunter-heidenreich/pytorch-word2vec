"""Command line argument parsing utilities."""

import argparse
from typing import Optional, List

from modern_word2vec.config import (
    DEFAULT_VOCAB_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
)


class ArgumentParser:
    """Centralized argument parser for Word2Vec CLI tools."""

    @staticmethod
    def create_train_parser() -> argparse.ArgumentParser:
        """Create argument parser for training CLI.

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Train Word2Vec (Skip-gram/CBOW) baseline"
        )

        # Add argument groups for better organization
        ArgumentParser._add_data_args(parser)
        ArgumentParser._add_model_args(parser)
        ArgumentParser._add_training_args(parser)
        ArgumentParser._add_output_args(parser)
        ArgumentParser._add_system_args(parser)

        return parser

    @staticmethod
    def create_query_parser() -> argparse.ArgumentParser:
        """Create argument parser for query CLI.

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Query saved Word2Vec embeddings for nearest neighbors"
        )

        parser.add_argument(
            "--run-dir",
            type=str,
            required=True,
            help="Path to directory containing embeddings.npy and vocab.json",
        )
        parser.add_argument("--word", type=str, required=True, help="Query word")
        parser.add_argument(
            "--topn", type=int, default=10, help="Number of similar words to return"
        )

        return parser

    @staticmethod
    def _add_data_args(parser: argparse.ArgumentParser) -> None:
        """Add data-related arguments."""
        data_group = parser.add_argument_group("Data Options")

        data_group.add_argument(
            "--dataset",
            type=str,
            default="wikitext",
            help="HF dataset name (default: wikitext)",
        )
        data_group.add_argument(
            "--dataset-config",
            type=str,
            default="wikitext-2-raw-v1",
            help="HF dataset config",
        )
        data_group.add_argument(
            "--split", type=str, default="train[:20%]", help="HF dataset split slice"
        )
        data_group.add_argument(
            "--text-file",
            type=str,
            help="Optional path to a local text file to train on",
        )
        data_group.add_argument(
            "--synthetic-sentences",
            type=int,
            default=0,
            help="Generate N synthetic sentences for benchmarking",
        )
        data_group.add_argument(
            "--synthetic-vocab",
            type=int,
            help="Synthetic vocab size (defaults to --vocab-size)",
        )

    @staticmethod
    def _add_model_args(parser: argparse.ArgumentParser) -> None:
        """Add model-related arguments."""
        model_group = parser.add_argument_group("Model Configuration")

        model_group.add_argument(
            "--model-type",
            type=str,
            default="skipgram",
            choices=["skipgram", "cbow"],
            help="Model type to train",
        )
        model_group.add_argument(
            "--output-layer",
            type=str,
            default="full_softmax",
            choices=["full_softmax", "hierarchical_softmax", "negative_sampling"],
            help="Output layer type (full_softmax, hierarchical_softmax, or negative_sampling)",
        )
        model_group.add_argument(
            "--num-negative",
            type=int,
            default=5,
            help="Number of negative samples per positive example (for negative sampling)",
        )
        model_group.add_argument(
            "--vocab-size",
            type=int,
            default=DEFAULT_VOCAB_SIZE,
            help="Maximum vocabulary size",
        )
        model_group.add_argument(
            "--embedding-dim",
            type=int,
            default=DEFAULT_EMBEDDING_DIM,
            help="Embedding dimension",
        )
        model_group.add_argument(
            "--window-size",
            type=int,
            default=DEFAULT_WINDOW_SIZE,
            help="Context window size",
        )

        # Text processing options
        model_group.add_argument(
            "--lower", action="store_true", dest="lowercase", help="Lowercase text"
        )
        model_group.add_argument(
            "--no-lower",
            action="store_false",
            dest="lowercase",
            help="Don't lowercase text",
        )
        parser.set_defaults(lowercase=True)

        model_group.add_argument(
            "--tokenizer",
            type=str,
            default="basic",
            choices=["basic", "enhanced", "simple", "split"],
            help="Tokenization strategy",
        )
        model_group.add_argument(
            "--subsample",
            type=float,
            default=0.0,
            help="Enable word subsampling with threshold (e.g., 1e-5)",
        )
        model_group.add_argument(
            "--dynamic-window", action="store_true", help="Use dynamic window sizes"
        )

    @staticmethod
    def _add_training_args(parser: argparse.ArgumentParser) -> None:
        """Add training-related arguments."""
        train_group = parser.add_argument_group("Training Configuration")

        train_group.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Training batch size",
        )
        train_group.add_argument(
            "--epochs",
            type=int,
            default=DEFAULT_EPOCHS,
            help="Number of training epochs",
        )
        train_group.add_argument(
            "--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate"
        )
        train_group.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "sgd"],
            help="Optimizer type",
        )
        train_group.add_argument(
            "--weight-decay", type=float, default=0.0, help="Weight decay coefficient"
        )
        train_group.add_argument(
            "--grad-clip",
            type=float,
            default=0.0,
            help="Gradient clipping threshold (0 to disable)",
        )

        # TensorBoard logging options
        tensorboard_group = parser.add_argument_group("TensorBoard Logging")
        tensorboard_group.add_argument(
            "--tensorboard", action="store_true", help="Enable TensorBoard logging"
        )
        tensorboard_group.add_argument(
            "--tensorboard-dir",
            type=str,
            default="runs/tensorboard",
            help="TensorBoard log directory",
        )
        tensorboard_group.add_argument(
            "--log-gradients",
            action="store_true",
            help="Log gradient statistics to TensorBoard",
        )
        tensorboard_group.add_argument(
            "--log-weights",
            action="store_true",
            help="Log model weights to TensorBoard",
        )
        tensorboard_group.add_argument(
            "--log-system-stats",
            action="store_true",
            help="Log system statistics (CPU, memory, GPU) to TensorBoard",
        )
        tensorboard_group.add_argument(
            "--log-interval",
            type=int,
            default=100,
            help="Log metrics every N training steps",
        )

        # Set default values for when TensorBoard is enabled
        parser.set_defaults(
            log_gradients=False,
            log_weights=False,
            log_system_stats=False,
        )

    @staticmethod
    def _add_system_args(parser: argparse.ArgumentParser) -> None:
        """Add system and optimization arguments."""
        system_group = parser.add_argument_group("System & Optimization")

        system_group.add_argument(
            "--compile", action="store_true", help="Use torch.compile if available"
        )
        system_group.add_argument(
            "--amp", action="store_true", help="Use mixed precision on CUDA"
        )
        system_group.add_argument(
            "--workers", type=int, default=0, help="Number of data loading workers"
        )
        system_group.add_argument(
            "--pin-memory",
            action="store_true",
            help="Pin memory for faster GPU transfer",
        )
        system_group.add_argument(
            "--seed",
            type=int,
            default=DEFAULT_SEED,
            help="Random seed for reproducibility",
        )
        system_group.add_argument(
            "--device",
            type=str,
            choices=["cuda", "mps", "cpu"],
            help="Device to use for training",
        )

    @staticmethod
    def _add_output_args(parser: argparse.ArgumentParser) -> None:
        """Add output-related arguments."""
        output_group = parser.add_argument_group("Output Options")

        output_group.add_argument(
            "--out-dir",
            type=str,
            default="runs/latest",
            help="Output directory for saved files",
        )
        output_group.add_argument(
            "--save",
            action="store_true",
            help="Save embeddings and vocab after training",
        )


def parse_train_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse training command line arguments.

    Args:
        argv: Optional command line arguments

    Returns:
        Parsed arguments namespace
    """
    parser = ArgumentParser.create_train_parser()
    return parser.parse_args(argv)


def parse_query_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse query command line arguments.

    Args:
        argv: Optional command line arguments

    Returns:
        Parsed arguments namespace
    """
    parser = ArgumentParser.create_query_parser()
    return parser.parse_args(argv)
