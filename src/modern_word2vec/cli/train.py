"""Training CLI for Word2Vec."""

import json
import os
import random
from dataclasses import asdict
from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader

from modern_word2vec import (
    CBOWModel,
    SkipGramModel,
    Trainer,
    Word2VecDataset,
    export_embeddings,
    get_device,
    set_seed,
)
from modern_word2vec.cli_args import parse_train_args
from modern_word2vec.config import DataConfig, TrainConfig
from modern_word2vec.data import (
    cbow_collate,
    generate_synthetic_texts,
    load_texts_from_hf,
)
from modern_word2vec.utils import setup_device_optimizations


def create_data_config(args) -> DataConfig:
    """Create data configuration from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Data configuration object
    """
    return DataConfig(
        vocab_size=args.vocab_size,
        window_size=args.window_size,
        model_type=args.model_type,
        lowercase=args.lowercase,
        tokenizer=args.tokenizer,
        subsample_t=args.subsample,
        dynamic_window=args.dynamic_window,
    )


def create_train_config(args) -> TrainConfig:
    """Create training configuration from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Training configuration object
    """
    return TrainConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        compile=args.compile,
        mixed_precision=args.amp,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        tensorboard=args.tensorboard,
        tensorboard_dir=args.tensorboard_dir,
        log_gradients=args.log_gradients,
        log_weights=args.log_weights,
        log_system_stats=args.log_system_stats,
        log_interval=args.log_interval,
    )


def load_training_texts(args) -> List[str]:
    """Load training texts based on arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of training text strings
    """
    if args.synthetic_sentences > 0:
        rng = random.Random(args.seed)
        vocab_size = args.synthetic_vocab or args.vocab_size
        texts = generate_synthetic_texts(args.synthetic_sentences, vocab_size, rng)
        print(f"Generated {len(texts)} synthetic sentences (vocab={vocab_size}).")
        return texts

    if args.text_file:
        with open(args.text_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} lines from local file.")
        return texts

    print("Loading dataset from Hugging Face...")
    texts = load_texts_from_hf(args.dataset, args.dataset_config, args.split)
    print(f"Loaded {len(texts)} lines from HF dataset.")
    return texts


def create_dataloader(dataset: Word2VecDataset, args) -> DataLoader:
    """Create data loader for training.

    Args:
        dataset: Word2Vec dataset
        args: Parsed command line arguments

    Returns:
        Configured data loader
    """

    def seed_worker(worker_id: int):
        """Initialize worker with deterministic seed."""
        base_seed = (args.seed + worker_id) % (2**32 - 1)
        random.seed(base_seed)
        np.random.seed(base_seed % (2**32 - 1))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        collate_fn=cbow_collate if args.model_type == "cbow" else None,
        drop_last=False,
        persistent_workers=bool(args.workers > 0),
        worker_init_fn=seed_worker if args.workers > 0 else None,
    )


def create_model(
    model_type: str,
    vocab_size: int,
    embedding_dim: int,
    output_layer_type: str = "full_softmax",
    dataset=None,
    num_negative: int = 5,
):
    """Create Word2Vec model based on type.

    Args:
        model_type: Type of model ("skipgram" or "cbow")
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        output_layer_type: Output layer type ("full_softmax", "hierarchical_softmax", or "negative_sampling")
        dataset: Dataset instance (required for hierarchical softmax and negative sampling)
        num_negative: Number of negative samples per positive example (for negative sampling)

    Returns:
        Word2Vec model instance

    Raises:
        ValueError: If model type is not supported
    """
    if model_type == "skipgram":
        return SkipGramModel(vocab_size, embedding_dim, output_layer_type, dataset, num_negative)
    elif model_type == "cbow":
        return CBOWModel(vocab_size, embedding_dim, output_layer_type, dataset, num_negative)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_training_artifacts(
    model,
    dataset: Word2VecDataset,
    data_config: DataConfig,
    train_config: TrainConfig,
    out_dir: str,
) -> None:
    """Save training artifacts to disk.

    Args:
        model: Trained model
        dataset: Training dataset
        data_config: Data configuration
        train_config: Training configuration
        out_dir: Output directory
    """
    export_embeddings(model, out_dir, dataset)

    config_data = {"data": asdict(data_config), "train": asdict(train_config)}

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)


def main(argv: Optional[List[str]] = None) -> None:
    """Main training function.

    Args:
        argv: Optional command line arguments
    """
    # Parse arguments and create configurations
    args = parse_train_args(argv)
    data_config = create_data_config(args)
    train_config = create_train_config(args)

    # Setup reproducibility and device
    set_seed(args.seed)
    device = get_device(args.device)
    setup_device_optimizations(device)

    # Load training data
    raw_texts = load_training_texts(args)

    # Create dataset and dataloader
    dataset = Word2VecDataset(raw_texts, data_config, rng=random.Random(args.seed))
    dataloader = create_dataloader(dataset, args)

    # Create and configure model
    model = create_model(
        args.model_type,
        dataset.vocab_size,
        args.embedding_dim,
        args.output_layer,
        dataset,
        args.num_negative,
    )

    # Train model
    trainer = Trainer(model, device, train_config)
    print("Starting training...")

    train_stats = trainer.train(dataloader)
    print(json.dumps(train_stats, indent=2))

    # Save artifacts if requested
    if args.save:
        save_training_artifacts(model, dataset, data_config, train_config, args.out_dir)
        print(f"Training artifacts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
